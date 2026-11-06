from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .apis import build_task_processor, import_config_plugins
from .backend import get_backend_manager
from .bundle import create_bundle, validate_bundle_destination
from .contracts import BundleManifest, DeploymentConfig
from .core import FUNCTION_REWRITER, PipelineManager
from .errors import ConfigurationError
from .exporters import export_onnx
from .optimization import lower_onnx_precision, optimize_onnx
from .plugins import add_python_paths, load_symbol
from .quantization import quantize_onnx_qdq
from .utils import sha256_file, write_json
from .validation import validate_backend


@dataclass(frozen=True)
class StagePlan:
    name: str
    enabled: bool
    description: str


class CompilePipeline:
    """Codebase -> IR -> backend pipeline modeled after MMDeploy boundaries."""

    def __init__(self, config: DeploymentConfig) -> None:
        config.validate_source_boundary()
        self.config = config
        self.pipeline_manager = PipelineManager()
        self._register_hooks()

    def _register_hooks(self) -> None:
        import_config_plugins(self.config)
        python_paths = list(self.config.source.get('python_paths', ()))
        python_paths.extend(self.config.codebase.get('python_paths', ()))
        add_python_paths(python_paths)
        for hook_config in self.config.pipeline_hooks:
            hook = load_symbol(str(hook_config['hook']))
            self.pipeline_manager.add_hook(
                stage=str(hook_config['stage']),
                hook=hook,
                when=str(hook_config.get('when', 'after')),
            )

    def plan(self) -> List[StagePlan]:
        quant_mode = str(self.config.quantization.get('mode', 'none')).lower()
        codebase = self.config.codebase['type']
        task = self.config.codebase['task']
        backend = self.config.backend['type']
        ir = self.config.ir['type']
        return [
            StagePlan(
                'prepare_model',
                True,
                f'build {codebase}:{task} task processor and export wrapper',
            ),
            StagePlan('export_ir', True, f'export a named {ir} graph'),
            StagePlan(
                'optimize_ir',
                bool(self.config.optimization.get('simplify', False))
                or self.config.precision == 'fp16',
                f'optimize and explicitly type the {ir} graph',
            ),
            StagePlan(
                'quantize',
                quant_mode != 'none',
                f'apply {quant_mode} quantization with representative data',
            ),
            StagePlan(
                'build_backend',
                True,
                f'build {backend} files at {self.config.precision} precision',
            ),
            StagePlan(
                'validate',
                bool(self.config.validation.get('enabled', True)),
                'compare source and backend-wrapper outputs',
            ),
            StagePlan('package', True, 'write an integrity-checked robot bundle'),
        ]

    def plan_dict(self) -> Dict[str, Any]:
        return {
            'name': self.config.name,
            'work_dir': str(self.config.work_dir),
            'codebase': self.config.codebase['type'],
            'task': self.config.codebase['task'],
            'ir': self.config.ir['type'],
            'backend': self.config.backend['type'],
            'precision': self.config.precision,
            'stages': [stage.__dict__ for stage in self.plan()],
            'inputs': [item.to_dict() for item in self.config.inputs],
            'outputs': [item.to_dict() for item in self.config.outputs],
        }

    def _bundle_dir(self) -> Path:
        configured = self.config.package.get('output_dir')
        if configured:
            path = Path(str(configured)).expanduser()
            if not path.is_absolute() and self.config.config_path is not None:
                path = self.config.config_path.parent / path
            return path.resolve()
        return (self.config.work_dir / 'bundle').resolve()

    def _package_assets(self) -> List[Path]:
        resolved = []
        for value in self.config.package.get('assets', ()):
            path = Path(str(value)).expanduser()
            if not path.is_absolute() and self.config.config_path is not None:
                path = self.config.config_path.parent / path
            resolved.append(path.resolve())
        return resolved

    def _validate_precision_configuration(self) -> None:
        if (self.config.precision == 'fp16'
                and not bool(self.config.optimization.get('fp16_conversion', True))
                and not bool(
                    self.config.optimization.get('graph_is_explicitly_typed', False))):
            raise ConfigurationError(
                'FP16 requires optimization.fp16_conversion=True or a task '
                'processor that declares graph_is_explicitly_typed=True.')

    def execute(self) -> Path:
        validate_bundle_destination(self._bundle_dir())
        work_dir = self.config.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        write_json(work_dir / 'resolved_config.json', self.config.to_dict())
        self._validate_precision_configuration()

        device = str(
            self.config.codebase.get('device',
                                     self.config.source.get('device', 'cuda:0')))
        task_processor = self.pipeline_manager.run(
            'build_task_processor',
            build_task_processor,
            self.config,
            device,
        )
        prepared = self.pipeline_manager.run('prepare_model',
                                             task_processor.prepare_model)
        if hasattr(prepared, 'graphs'):
            from .multigraph import execute_multigraph

            return execute_multigraph(self, prepared, task_processor, device)

        codebase_name = str(self.config.codebase['type'])
        backend_name = str(self.config.backend['type'])
        ir_name = str(self.config.ir['type'])
        with FUNCTION_REWRITER.context(
                codebase=codebase_name,
                backend=backend_name,
                ir=ir_name,
                config=self.config,
        ):
            ir_path = self.pipeline_manager.run(
                'export_ir',
                export_onnx,
                prepared,
                work_dir / 'model.onnx',
                self.config.ir,
            )

        graph_path = self.pipeline_manager.run(
            'optimize_ir',
            optimize_onnx,
            ir_path,
            work_dir / 'model.optimized.onnx',
            self.config.optimization,
        )
        graph_path = self.pipeline_manager.run(
            'lower_precision',
            lower_onnx_precision,
            graph_path,
            work_dir / f'model.{self.config.precision}.onnx',
            self.config.precision,
            self.config.optimization,
        )

        quant_mode = str(self.config.quantization.get('mode', 'none')).lower()
        if quant_mode not in {'none', 'int8'}:
            raise ConfigurationError(f'Unsupported quantization mode: {quant_mode}. '
                                     'Initial support is none or int8 Q/DQ.')
        if self.config.precision == 'int8' and quant_mode != 'int8':
            raise ConfigurationError(
                'Backend precision=int8 requires quantization.mode=int8.')
        if quant_mode == 'int8':
            if self.config.precision != 'int8':
                raise ConfigurationError(
                    'INT8 Q/DQ quantization requires backend precision=int8.')
            graph_path = self.pipeline_manager.run(
                'quantize',
                quantize_onnx_qdq,
                graph_path,
                work_dir / 'model.int8.onnx',
                prepared.calibration_source,
                self.config.quantization,
            )

        backend_manager = get_backend_manager(backend_name)
        backend_result = self.pipeline_manager.run(
            'build_backend',
            backend_manager.to_backend,
            [graph_path],
            work_dir,
            self.config,
        )

        if bool(self.config.validation.get('enabled', True)):
            wrapper = backend_manager.build_wrapper(
                backend_result.files,
                device=device,
                require_target_match=False,
            )
            validation = self.pipeline_manager.run(
                'validate',
                validate_backend,
                prepared,
                wrapper,
                self.config.validation,
            )
            validation['enabled'] = True
        else:
            validation = {
                'enabled': False,
                'passed': False,
                'reason': 'validation was disabled',
            }

        source_metadata = dict(prepared.metadata)
        source_metadata.update(task_processor.runtime_metadata())
        if self.config.config_path is not None:
            source_metadata['deployment_config_sha256'] = sha256_file(
                self.config.config_path)
        runtime = dict(self.config.package.get('runtime', {}))
        runtime.setdefault('action_output', self.config.outputs[0].name)

        backend_relative_files = [
            f'backend/{path.name}' for path in backend_result.files
        ]
        manifest = BundleManifest(
            schema_version=2,
            name=self.config.name,
            created_at=datetime.now(timezone.utc).isoformat(),
            backend=backend_name,
            precision=self.config.precision,
            backend_files=tuple(backend_relative_files),
            backend_metadata=dict(backend_result.metadata),
            source=source_metadata,
            inputs=self.config.inputs,
            outputs=self.config.outputs,
            runtime=runtime,
            validation=validation,
            files={},
            metadata={
                'codebase_config': self.config.codebase,
                'backend_config': self.config.backend,
                'ir_config': self.config.ir,
                'quantization': self.config.quantization,
                'optimization': self.config.optimization,
                'pipeline': self.pipeline_manager.report(),
            },
        )

        bundle_files: Dict[str, str | Path] = {
            relative: source
            for relative, source in zip(backend_relative_files, backend_result.files)
        }
        bundle_files['build/resolved_config.json'] = (work_dir / 'resolved_config.json')
        if self.config.config_path is not None:
            bundle_files['build/deployment.py'] = self.config.config_path
        if bool(self.config.package.get('include_onnx', False)):
            bundle_files['model/model.onnx'] = graph_path
            candidates = set(graph_path.parent.glob(graph_path.name + '*'))
            candidates.update(graph_path.parent.glob(graph_path.stem + '*.data'))
            for external in sorted(candidates):
                if external.is_file() and external != graph_path:
                    bundle_files[f'model/{external.name}'] = external

        bundle = self.pipeline_manager.run(
            'package',
            create_bundle,
            self._bundle_dir(),
            manifest,
            bundle_files,
            assets=self._package_assets(),
        )
        (work_dir / 'result.json').write_text(
            json.dumps(
                {
                    'bundle': str(bundle),
                    'pipeline': self.pipeline_manager.report(),
                },
                indent=2,
            ) + '\n',
            encoding='utf-8',
        )
        return bundle
