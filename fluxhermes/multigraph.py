"""Compile independent named graphs and let the source package own orchestration."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from .adapters.base import PreparedModel
from .adapters.fluxvla_export import tensor_specs_from_sdk
from .backend import get_backend_manager
from .bundle import create_bundle
from .contracts import BundleManifest
from .core import FUNCTION_REWRITER
from .errors import ArtifactError, ConfigurationError
from .exporters import export_onnx
from .optimization import optimize_onnx
from .plugins import load_symbol
from .utils import sha256_file, write_json
from .validation import validate_backend

_GRAPH_NAME = re.compile(r'[A-Za-z][A-Za-z0-9_-]*\Z')


def _graph_name(name):
    if type(name) is not str or _GRAPH_NAME.fullmatch(name) is None:
        raise ConfigurationError(f'Unsafe graph name: {name!r}')
    return name


def _export_model(graph):
    axes = {}
    for spec in (*graph.input_specs, *graph.output_specs):
        dynamic = {
            index: f'{spec.name}_dim_{index}'
            for index, dim in enumerate(spec.shape) if dim == -1
        }
        if dynamic:
            axes[spec.name] = dynamic
    return PreparedModel(
        module=graph.module,
        inputs=graph.inputs,
        output_names=tuple(spec.name for spec in graph.output_specs),
        dynamic_axes=axes,
    )


def execute_multigraph(pipeline, prepared, task_processor, device):
    """Use ordinary backend builders per graph, then validate the full policy."""
    config = pipeline.config
    run = pipeline.pipeline_manager.run
    if config.precision != 'fp32' or config.quantization.get('mode', 'none') != 'none':
        raise ConfigurationError(
            'Multi-graph source precision must be preserved; global FP16/INT8 '
            'conversion is not supported.')
    if config.optimization.get('fp16_conversion', False):
        raise ConfigurationError(
            'Multi-graph export cannot globally cast protected layers.')
    runtime = dict(config.package.get('runtime', {}))
    executor_metadata = prepared.metadata.get('runtime')
    if not isinstance(executor_metadata, dict) or not callable(
            load_symbol(str(executor_metadata.get('entrypoint', '')))):
        raise ConfigurationError(
            'Prepared deployment requires a callable runtime entrypoint.')
    if 'executor' in runtime and runtime['executor'] != executor_metadata:
        raise ConfigurationError(
            'package.runtime.executor conflicts with the source recipe.')
    runtime['executor'] = dict(executor_metadata)
    runtime.setdefault('action_output', config.outputs[0].name)

    manager = get_backend_manager(config.backend['type'])
    records, engines, graph_reports, bundle_files = {}, {}, {}, {}
    validation_enabled = bool(config.validation.get('enabled', True))
    for name, graph in prepared.graphs.items():
        _graph_name(name)
        graph_dir = config.work_dir / 'graphs' / name
        graph_dir.mkdir(parents=True, exist_ok=True)
        for option, required in graph.export_options.items():
            if option in config.ir and (type(config.ir[option]) is not type(required)
                                        or config.ir[option] != required):
                raise ConfigurationError(
                    f'Graph {name} requires ir_config.{option}={required!r}; '
                    f'got {config.ir[option]!r}. Use the recipe export profile.')
        graph_config = replace(
            config,
            name=f'{config.name}.{name}',
            work_dir=graph_dir,
            inputs=tensor_specs_from_sdk(graph.input_specs),
            outputs=tensor_specs_from_sdk(graph.output_specs),
            backend={
                **graph.build_options,
                **config.backend
            },
            ir={
                **graph.export_options,
                **config.ir
            },
        )
        export_model = _export_model(graph)
        with FUNCTION_REWRITER.context(
                codebase=config.codebase['type'],
                backend=config.backend['type'],
                ir=config.ir['type'],
                config=graph_config):
            ir_path = run('export_ir', export_onnx, export_model,
                          graph_dir / 'model.onnx', graph_config.ir)
        ir_path = run('optimize_ir', optimize_onnx, ir_path,
                      graph_dir / 'model.optimized.onnx', config.optimization)
        result = run('build_backend', manager.to_backend, [ir_path], graph_dir,
                     graph_config)
        relative_files = []
        for source in result.files:
            relative = f'backend/{name}/{source.name}'
            if relative in bundle_files:
                raise ArtifactError(f'Backend artifact basename collision: {relative}')
            bundle_files[relative] = source
            relative_files.append(relative)
        if not relative_files:
            raise ArtifactError(f'Backend returned no files for graph {name}.')
        records[name] = {
            'files': relative_files,
            'metadata': result.metadata,
            'inputs': [spec.to_dict() for spec in graph_config.inputs],
            'outputs': [spec.to_dict() for spec in graph_config.outputs],
            'export_options': graph_config.ir,
        }
        if validation_enabled:
            engines[name] = manager.build_wrapper(
                result.files,
                device=device,
                target_metadata=result.metadata,
                require_target_match=True)
            graph_reports[name] = run('validate_graph', validate_backend, export_model,
                                      engines[name], config.validation)
        if config.package.get('include_onnx', False):
            # Copy declared external data, not a filename heuristic. The same
            # checked staging handles initializers and tensor-valued attributes.
            from .exporters.artifacts import stage_graph

            staged, _ = stage_graph(ir_path, graph_dir / 'packaged_onnx')
            for source in staged:
                bundle_files[f'model/{name}/{source.name}'] = source

    if validation_enabled:
        executor = prepared.create_executor(engines)
        policy_reference = SimpleNamespace(
            inputs=prepared.policy_inputs, run_reference=prepared.run_reference)
        validation = run('validate', validate_backend, policy_reference, executor,
                         config.validation)
        validation['passed'] = validation['passed'] and all(
            report['passed'] for report in graph_reports.values())
        validation.update(enabled=True, graphs=graph_reports)
    else:
        validation = {
            'enabled': False,
            'passed': False,
            'reason': 'validation was disabled'
        }
    source_metadata = dict(prepared.metadata)
    source_metadata.update(task_processor.runtime_metadata())
    if config.config_path is not None:
        source_metadata['deployment_config_sha256'] = sha256_file(config.config_path)
        bundle_files['build/deployment.py'] = config.config_path
    bundle_files[
        'build/resolved_config.json'] = config.work_dir / 'resolved_config.json'
    manifest = BundleManifest(
        schema_version=2,
        name=config.name,
        created_at=datetime.now(timezone.utc).isoformat(),
        backend=config.backend['type'],
        precision=config.precision,
        backend_files=tuple(path for record in records.values()
                            for path in record['files']),
        backend_metadata={
            'graph_schema_version': 1,
            'graphs': records
        },
        source=source_metadata,
        inputs=config.inputs,
        outputs=config.outputs,
        runtime=runtime,
        validation=validation,
        files={},
        metadata={
            'codebase_config': config.codebase,
            'backend_config': config.backend,
            'ir_config': config.ir,
            'optimization': config.optimization,
            'quantization': config.quantization,
            'pipeline': pipeline.pipeline_manager.report(),
        },
    )
    bundle = run(
        'package',
        create_bundle,
        pipeline._bundle_dir(),
        manifest,
        bundle_files,
        assets=pipeline._package_assets())
    write_json(config.work_dir / 'result.json', {
        'bundle': str(bundle),
        'pipeline': pipeline.pipeline_manager.report()
    })
    return bundle


def build_multigraph_wrapper(manifest, bundle_dir, manager, *, device, profile_index,
                             require_target_match, **kwargs):
    """Restore graph engines and the installed package executor, without weights."""
    root = Path(bundle_dir).expanduser().resolve()
    records = manifest.backend_metadata.get('graphs')
    if manifest.backend_metadata.get('graph_schema_version') != 1 or not isinstance(
            records, dict) or not records:
        raise ArtifactError('Invalid multi-graph backend metadata.')
    declared = []
    for name, record in records.items():
        _graph_name(name)
        if not isinstance(record, dict) or not isinstance(record.get('files'), list):
            raise ArtifactError(f'Invalid backend graph file record: {name}')
        declared.extend(record['files'])
    if (any(type(path) is not str for path in declared)
            or len(declared) != len(set(declared))
            or set(declared) != set(manifest.backend_files)):
        raise ArtifactError(
            'Graph file groups must partition manifest.backend_files exactly.')
    executor_metadata = manifest.runtime.get('executor')
    if not isinstance(executor_metadata, dict):
        raise ArtifactError('Multi-graph bundle has no source executor metadata.')
    factory = load_symbol(str(executor_metadata.get('entrypoint', '')))
    if not callable(factory):
        raise ArtifactError('Multi-graph executor entrypoint is not callable.')
    engines = {}
    for name, record in records.items():
        files = []
        for relative in record['files']:
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or relative not in manifest.files:
                raise ArtifactError(f'Unsafe or unchecked graph artifact: {relative}')
            files.append(path)
        engines[name] = manager.build_wrapper(
            files,
            device=device,
            profile_index=profile_index,
            target_metadata=record.get('metadata', {}),
            require_target_match=require_target_match,
            **kwargs)
    executor = factory(engines, metadata=executor_metadata)
    if not callable(getattr(executor, 'infer', None)):
        raise ArtifactError('Source executor factory must return callable infer().')
    return executor
