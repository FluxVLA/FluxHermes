"""Model, rewrite, ONNX, backend, validation, and packaging transforms."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from ..apis import build_task_processor
from ..backend import get_backend_manager
from ..bundle import create_bundle
from ..contracts import BundleManifest
from ..core import FUNCTION_REWRITER, StageRecord
from ..errors import ConfigurationError
from ..exporters import export_onnx
from ..optimization import lower_onnx_precision, optimize_onnx
from ..quantization import quantize_onnx_qdq
from ..utils import sha256_file, write_json
from ..validation import validate_backend
from .base import TRANSFORMS, BaseTransform, Compose


def _path(data, value, *, artifact=False):
    path = Path(value).expanduser()
    if not path.is_absolute():
        config = data['config']
        if not config.pipeline_explicit and not artifact:
            base = config.config_path.parent if config.config_path else Path.cwd()
        else:
            base = (
                config.work_dir if artifact or config.config_path is None else
                config.config_path.parent)
        path = base / path
    return path.resolve()


@TRANSFORMS.register_module()
class PrepareModel(BaseTransform):
    """Initialize the source model, load weights, and prepare export inputs."""
    name = 'prepare_model'
    requires = ('config', 'device')
    provides = ('task_processor', 'prepared')
    invalidates = ('validation', 'bundle')

    def validate(self, available, config):
        if not config.codebase or not config.source:
            raise ConfigurationError('PrepareModel requires codebase_config.source.')
        return super().validate(available, config)

    def __call__(self, data):
        processor = self.run(
            data,
            build_task_processor,
            data['config'],
            data['device'],
            stage='build_task_processor')
        data['task_processor'] = processor
        data['prepared'] = self.run(data, processor.prepare_model)
        return data


@TRANSFORMS.register_module()
class Rewrite(BaseTransform):
    """Apply temporary operator rewrites only inside the nested transforms."""
    name = 'rewrite'

    def __init__(self, transforms):
        self.transforms = Compose(transforms)

    def validate(self, available, config):
        return self.transforms.validate(available, config)

    def plan(self):
        return {**super().plan(), 'transforms': self.transforms.plan()}

    def __call__(self, data):
        config = data['config']
        with FUNCTION_REWRITER.context(
                codebase=config.codebase.get('type', 'default'),
                backend=config.backend['type'],
                ir=config.ir['type'],
                config=config):
            return self.transforms(data)


class GraphTransform(BaseTransform):
    """Shared file and option handling for graph-producing transforms."""
    requires = ('graph_path', )
    provides = ('graph_path', )
    invalidates = ('backend_result', 'validation', 'bundle')
    section = 'optimization'
    default_output = 'model.onnx'

    def __init__(self, output=None, options=None):
        self.output = output
        self.options = dict(options or {})

    def settings(self, data):
        config = data['config']
        if not config.pipeline_explicit:
            # Historical before hooks share/mutate the actual config option dict.
            return config.work_dir / self.default_output, getattr(config, self.section)
        options = {**getattr(config, self.section), **self.options}
        output = _path(data, self.output or self.default_output, artifact=True)
        return output, options


@TRANSFORMS.register_module()
class ExportONNX(GraphTransform):
    """Export the prepared model to a named ONNX graph."""
    name = 'export_ir'
    requires = ('prepared', )
    invalidates = GraphTransform.invalidates + ('precision_ready', 'quantized',
                                                'source_metadata', 'export_state')
    section = 'ir'

    def __call__(self, data):
        output, options = self.settings(data)
        data['graph_path'] = data['output'] = self.run(data, export_onnx,
                                                       data['prepared'], output,
                                                       options)
        return data


@TRANSFORMS.register_module()
class LoadONNX(BaseTransform):
    """Reuse an existing ONNX artifact without initializing a source model."""
    name = 'load_ir'
    provides = ('graph_path', 'source_metadata')
    invalidates = GraphTransform.invalidates + ('precision_ready', 'quantized',
                                                'export_state')

    def __init__(self, path, precision=None):
        self.path, self.precision = path, precision
        if precision == 'fp16':
            self.provides += ('precision_ready', )
        elif precision == 'int8':
            self.provides += ('quantized', )

    def validate(self, available, config):
        if self.precision is not None and self.precision != config.precision:
            raise ConfigurationError(
                'LoadONNX.precision must match backend_config.precision.')
        return super().validate(available, config)

    def __call__(self, data):
        path = _path(data, self.path)
        if not path.is_file():
            raise ConfigurationError(f'ONNX artifact not found: {path}')
        data['graph_path'] = data['output'] = path
        data['source_metadata'] = dict(
            onnx_path=str(path), onnx_sha256=sha256_file(path))
        if self.precision == 'fp16':
            data['precision_ready'] = True
        elif self.precision == 'int8':
            data['quantized'] = True
        return data


@TRANSFORMS.register_module()
class OptimizeONNX(GraphTransform):
    """Apply configured ONNX graph optimizations."""
    name = 'optimize_ir'
    default_output = 'model.optimized.onnx'

    def __call__(self, data):
        output, options = self.settings(data)
        if data['config'].pipeline_explicit:
            options.setdefault('simplify', True)
        data['graph_path'] = data['output'] = self.run(data, optimize_onnx,
                                                       data['graph_path'], output,
                                                       options)
        return data


@TRANSFORMS.register_module()
class LowerPrecision(GraphTransform):
    """Explicitly lower the graph precision before backend conversion."""
    name = 'lower_precision'
    provides = ('graph_path', 'precision_ready')

    def validate(self, available, config):
        options = {**config.optimization, **self.options}
        if (config.precision == 'fp16' and not options.get('fp16_conversion', True)
                and not options.get('graph_is_explicitly_typed', False)):
            raise ConfigurationError(
                'FP16 lowering requires fp16_conversion or an explicitly typed graph.')
        return super().validate(available, config)

    def __call__(self, data):
        output, options = self.settings(data)
        if self.output is None:
            output = data['config'].work_dir / f'model.{data["config"].precision}.onnx'
        data['graph_path'] = data['output'] = self.run(data, lower_onnx_precision,
                                                       data['graph_path'], output,
                                                       data['config'].precision,
                                                       options)
        data['precision_ready'] = True
        return data


@TRANSFORMS.register_module()
class QuantizeONNX(GraphTransform):
    """Quantize ONNX to INT8 Q/DQ with representative calibration samples."""
    name = 'quantize'
    requires = ('graph_path', 'prepared')
    provides = ('graph_path', 'quantized')
    section = 'quantization'
    default_output = 'model.int8.onnx'

    def validate(self, available, config):
        if config.precision != 'int8':
            raise ConfigurationError(
                'QuantizeONNX requires backend_config.precision=int8.')
        return super().validate(available, config)

    def __call__(self, data):
        output, options = self.settings(data)
        data['graph_path'] = data['output'] = self.run(
            data, quantize_onnx_qdq, data['graph_path'], output,
            data['prepared'].calibration_source, options)
        data['quantized'] = True
        return data


@TRANSFORMS.register_module()
class ConvertBackend(BaseTransform):
    """Convert the current ONNX graph into backend files."""
    name = 'build_backend'
    requires = ('graph_path', )
    provides = ('backend_result', )
    invalidates = ('validation', 'bundle')

    def validate(self, available, config):
        if (config.precision == 'fp16' and 'precision_ready' not in available
                and not config.optimization.get('graph_is_explicitly_typed', False)):
            raise ConfigurationError(
                'FP16 conversion requires LowerPrecision or LoadONNX(precision="fp16").'
            )
        if config.precision == 'int8' and 'quantized' not in available:
            raise ConfigurationError(
                'INT8 conversion requires QuantizeONNX or LoadONNX(precision="int8").')
        return super().validate(available, config)

    def __call__(self, data):
        config = data['config']
        manager = get_backend_manager(config.backend['type'])
        result = self.run(data, manager.to_backend, [data['graph_path']],
                          config.work_dir, config)
        data['backend_result'] = result
        if result.files:
            data['output'] = result.files[0]
        return data


@TRANSFORMS.register_module()
class IsolatedExport(BaseTransform):
    """Run the explicitly configured export transforms in a subprocess."""
    name = 'isolated_export'
    provides = ('graph_path', 'source_metadata', 'export_state')
    invalidates = GraphTransform.invalidates + ('precision_ready', 'quantized',
                                                'prepared', 'task_processor')

    def __init__(self, transforms):
        self.definitions = transforms
        self.transforms = Compose(transforms)

    def validate(self, available, config):
        produced = self.transforms.validate(
            {'config', 'manager', 'device', 'run_id', 'output', 'validation'}, config)
        if not {'graph_path', 'prepared', 'task_processor'} <= produced:
            raise ConfigurationError(
                'IsolatedExport requires model preparation and ONNX export.')
        return super().validate(available, config) | (
            produced & {'precision_ready', 'quantized'})

    def plan(self):
        return {**super().plan(), 'transforms': self.transforms.plan()}

    def __call__(self, data):
        config = data['config']
        worker_config = replace(config, pipeline=tuple(self.definitions))
        config_path = config.work_dir / 'export-config.json'
        write_json(config_path, worker_config.to_dict())
        process = subprocess.run([
            sys.executable, '-B', '-m', 'fluxhermes.export_worker',
            str(config_path), data['run_id']
        ],
                                 check=False)
        report_path = config.work_dir / 'export-stage.json'
        if not report_path.is_file():
            raise ConfigurationError(
                f'Export worker exited {process.returncode} without a stage report.')
        state = json.loads(report_path.read_text(encoding='utf-8'))
        if state['run_id'] != data['run_id']:
            raise ConfigurationError('Export worker report belongs to a different run.')
        data['manager'].records.extend(
            StageRecord(**record) for record in state['stages'])
        if process.returncode != 0 or not state['passed']:
            raise ConfigurationError(
                f'Export worker failed ({process.returncode}): {state.get("error")}')
        graph = Path(state['graph_path']).resolve()
        if not graph.is_relative_to(config.work_dir.resolve()):
            raise ConfigurationError('Export worker graph escaped its work directory.')
        data.update(
            graph_path=graph,
            output=graph,
            source_metadata=state['source_metadata'],
            export_state=state)
        for key in ('precision_ready', 'quantized'):
            if state.get(key):
                data[key] = True
        return data


@TRANSFORMS.register_module()
class PrepareReference(PrepareModel):
    """Reload the source model with the exact inputs captured by isolated export."""
    name = 'prepare_reference'
    requires = ('config', 'device', 'export_state')

    def __call__(self, data):
        from ..validation.parity import _load_sample

        data = super().__call__(data)
        prepared = data['prepared']
        if {
                name: str(value.dtype)
                for name, value in prepared.inputs.items()
        } != data['export_state']['input_dtypes']:
            raise ConfigurationError(
                'Reference reload changed the representative input contract.')
        prepared.inputs, _ = _load_sample(data['config'].work_dir / 'export-inputs.npz',
                                          prepared.inputs)
        return data


@TRANSFORMS.register_module()
class ValidateBackend(BaseTransform):
    """Compare source-model and backend outputs using the configured thresholds."""
    name = 'validate'
    requires = ('prepared', 'backend_result')
    provides = ('validation', )
    invalidates = ('bundle', )

    def __call__(self, data):
        config, result = data['config'], data['backend_result']
        if not config.pipeline_explicit and not config.validation.get('enabled', True):
            data['validation'] = dict(
                enabled=False, passed=False, reason='validation was disabled')
            return data
        wrapper = get_backend_manager(config.backend['type']).build_wrapper(
            result.files, device=data['device'], require_target_match=False)
        data['validation'] = self.run(data, validate_backend, data['prepared'], wrapper,
                                      config.validation)
        data['validation']['enabled'] = True
        return data


@TRANSFORMS.register_module()
class PackageBundle(BaseTransform):
    """Package backend files, source metadata, validation, and runtime assets."""
    name = 'package'
    requires = ('backend_result', )
    provides = ('bundle', )

    def validate(self, available, config):
        if not ('source_metadata' in available
                or {'prepared', 'task_processor'} <= available):
            raise ConfigurationError(
                'PackageBundle requires PrepareModel or LoadONNX source metadata.')
        if config.package.get('include_onnx', False) and 'graph_path' not in available:
            raise ConfigurationError('PackageBundle(include_onnx) requires graph_path.')
        return super().validate(available, config)

    def __call__(self, data):
        config, result = data['config'], data['backend_result']
        if 'source_metadata' in data:
            source = dict(data['source_metadata'])
        else:
            source = {
                **data['prepared'].metadata,
                **data['task_processor'].runtime_metadata()
            }
        if config.config_path is not None:
            source['deployment_config_sha256'] = sha256_file(config.config_path)
        runtime = dict(config.package.get('runtime', {}))
        runtime.setdefault('action_output', config.outputs[0].name)
        validation = data.get('validation') or dict(
            enabled=False,
            passed=False,
            reason=('No validation for the current artifacts'
                    if config.pipeline_explicit else 'validation was disabled'))
        relative_files = [f'backend/{path.name}' for path in result.files]
        manifest = BundleManifest(
            schema_version=2,
            name=config.name,
            created_at=datetime.now(timezone.utc).isoformat(),
            backend=config.backend['type'],
            precision=config.precision,
            backend_files=tuple(relative_files),
            backend_metadata=dict(result.metadata),
            source=source,
            inputs=config.inputs,
            outputs=config.outputs,
            runtime=runtime,
            validation=validation,
            files={},
            metadata=dict(
                codebase_config=config.codebase,
                backend_config=config.backend,
                ir_config=config.ir,
                quantization=config.quantization,
                optimization=config.optimization,
                transforms=config.pipeline,
                pipeline=data['manager'].report()))
        if not config.pipeline_explicit:
            manifest.metadata.pop('transforms')
        files = dict(zip(relative_files, result.files))
        files['build/resolved_config.json'] = config.work_dir / 'resolved_config.json'
        if config.config_path is not None:
            files['build/deployment.py'] = config.config_path
        if config.package.get('include_onnx', False):
            graph = data['graph_path']
            files['model/model.onnx'] = graph
        destination = config.package.get('output_dir')
        output = _path(data, destination) if destination else config.work_dir / 'bundle'
        if not config.pipeline_explicit:
            output = output.resolve()
        assets = [_path(data, value) for value in config.package.get('assets', ())]
        data['bundle'] = data['output'] = self.run(
            data, create_bundle, output, manifest, files, assets=assets)
        return data
