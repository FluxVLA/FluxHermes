from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .errors import ConfigurationError

Shape = Tuple[int, ...]


def _shape(value: Optional[Iterable[int]]) -> Optional[Shape]:
    if value is None:
        return None
    result = tuple(int(dim) for dim in value)
    if not result:
        raise ConfigurationError('Tensor shapes must contain at least one dimension.')
    if any(dim == 0 or dim < -1 for dim in result):
        raise ConfigurationError(f'Invalid tensor shape: {result}')
    return result


@dataclass(frozen=True)
class TensorSpec:
    """Stable input/output contract stored with every deployment bundle."""

    name: str
    dtype: str
    shape: Shape
    min_shape: Optional[Shape] = None
    opt_shape: Optional[Shape] = None
    max_shape: Optional[Shape] = None
    semantic: str = ''

    def __post_init__(self) -> None:
        object.__setattr__(self, 'shape', _shape(self.shape))
        object.__setattr__(self, 'min_shape', _shape(self.min_shape))
        object.__setattr__(self, 'opt_shape', _shape(self.opt_shape))
        object.__setattr__(self, 'max_shape', _shape(self.max_shape))
        if not self.name:
            raise ConfigurationError('TensorSpec.name cannot be empty.')
        if not self.dtype:
            raise ConfigurationError(f'{self.name}: dtype cannot be empty.')
        profiles = (self.min_shape, self.opt_shape, self.max_shape)
        if any(item is not None for item in profiles) and not all(item is not None
                                                                  for item in profiles):
            raise ConfigurationError(
                f'{self.name}: min/opt/max shapes must be provided together.')
        if all(item is not None for item in profiles):
            rank = len(self.shape)
            if any(len(item) != rank for item in profiles if item is not None):
                raise ConfigurationError(
                    f'{self.name}: optimization profile rank does not match shape.')
            assert self.min_shape is not None
            assert self.opt_shape is not None
            assert self.max_shape is not None
            for minimum, optimum, maximum in zip(self.min_shape, self.opt_shape,
                                                 self.max_shape):
                if not minimum <= optimum <= maximum:
                    raise ConfigurationError(
                        f'{self.name}: expected min <= opt <= max, got '
                        f'{minimum}, {optimum}, {maximum}.')
        if -1 in self.shape and self.min_shape is None:
            raise ConfigurationError(
                f'{self.name}: dynamic dimensions require min/opt/max shapes.')

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'TensorSpec':
        return cls(
            name=str(data['name']),
            dtype=str(data['dtype']),
            shape=tuple(data['shape']),
            min_shape=data.get('min_shape'),
            opt_shape=data.get('opt_shape'),
            max_shape=data.get('max_shape'),
            semantic=str(data.get('semantic', '')),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeploymentConfig:
    """Normalized top-level configuration for the compile pipeline."""

    name: str
    work_dir: Path
    codebase: Dict[str, Any]
    backend: Dict[str, Any]
    ir: Dict[str, Any]
    inputs: Tuple[TensorSpec, ...]
    outputs: Tuple[TensorSpec, ...]
    optimization: Dict[str, Any] = field(default_factory=dict)
    quantization: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    package: Dict[str, Any] = field(default_factory=dict)
    pipeline: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    pipeline_hooks: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    config_path: Optional[Path] = None

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        config_path: Optional[Path] = None,
    ) -> 'DeploymentConfig':
        required = ('name', 'work_dir', 'inputs', 'outputs')
        missing = [key for key in required if key not in data]
        if missing:
            raise ConfigurationError(
                f'Missing deployment configuration keys: {", ".join(missing)}')
        pipeline = data.get('pipeline', ())
        if not isinstance(pipeline, (list, tuple)):
            raise ConfigurationError(
                'deployment.pipeline must be a list of transforms.')
        inputs = tuple(TensorSpec.from_dict(item) for item in data['inputs'])
        outputs = tuple(TensorSpec.from_dict(item) for item in data['outputs'])
        if not inputs or not outputs:
            raise ConfigurationError('At least one input and one output are required.')
        names = [spec.name for spec in inputs + outputs]
        if len(names) != len(set(names)):
            raise ConfigurationError('Input/output tensor names must be unique.')
        work_dir = Path(str(data['work_dir'])).expanduser()
        if not work_dir.is_absolute() and config_path is not None:
            work_dir = (config_path.parent / work_dir).resolve()
        codebase = cls._normalize_codebase(data)
        backend = cls._normalize_backend(data)
        ir = cls._normalize_ir(data)
        default_precision = 'fp16' if backend.get('type') == 'tensorrt' else 'fp32'
        precision = str(backend.get('precision', default_precision))
        if backend.get('type') == 'tensorrt':
            if precision not in {'fp32', 'fp16', 'bf16', 'int8'}:
                raise ConfigurationError(f'Unsupported TensorRT precision: {precision}')
            if precision == 'bf16' and backend.get('strongly_typed') is False:
                raise ConfigurationError('TensorRT bf16 requires strongly_typed=True.')
            if 'allow_tf32' in backend and type(backend['allow_tf32']) is not bool:
                raise ConfigurationError('TensorRT allow_tf32 must be true or false.')
        if 'pipeline' not in data:
            pipeline = cls._legacy_pipeline(data, precision)
        timing_cache = backend.get('timing_cache')
        if timing_cache and config_path is not None:
            timing_path = Path(str(timing_cache)).expanduser()
            if not timing_path.is_absolute():
                backend['timing_cache'] = str(
                    (config_path.parent / timing_path).resolve())
        validation = dict(data.get('validation', {}))
        samples_manifest = validation.get('samples_manifest')
        if samples_manifest and config_path is not None:
            samples_path = Path(str(samples_manifest)).expanduser()
            if not samples_path.is_absolute():
                samples_path = config_path.parent / samples_path
            validation['samples_manifest'] = str(samples_path.resolve())
        return cls(
            name=str(data['name']),
            work_dir=work_dir,
            codebase=codebase,
            backend=backend,
            ir=ir,
            inputs=inputs,
            outputs=outputs,
            optimization=dict(data.get('optimization', {})),
            quantization=dict(data.get('quantization', {})),
            validation=validation,
            package=dict(data.get('package', {})),
            pipeline=tuple(pipeline),
            pipeline_hooks=tuple(data.get('pipeline_hooks', ())),
            config_path=config_path,
        )

    @staticmethod
    def _legacy_pipeline(data: Mapping[str, Any],
                         precision: str) -> list[Dict[str, Any]]:
        """Translate the original stages only when pipeline is absent."""
        mode = str(data.get('quantization', {}).get('mode', 'none')).lower()
        if mode not in {'none', 'int8'}:
            raise ConfigurationError(f'Unsupported quantization mode: {mode}. '
                                     'Initial support is none or int8 Q/DQ.')
        if precision == 'int8' and mode != 'int8':
            raise ConfigurationError(
                'Backend precision=int8 requires quantization.mode=int8.')
        if mode == 'int8' and precision != 'int8':
            raise ConfigurationError(
                'INT8 Q/DQ quantization requires backend precision=int8.')
        pipeline = [
            dict(type='PrepareModel'),
            dict(type='Rewrite', transforms=[dict(type='ExportONNX')]),
            dict(
                type='OptimizeONNX',
                options=dict(
                    simplify=bool(data.get('optimization', {}).get('simplify',
                                                                   False)))),
            dict(type='LowerPrecision'),
        ]
        if mode == 'int8':
            pipeline.append(dict(type='QuantizeONNX'))
        pipeline.append(dict(type='ConvertBackend'))
        if bool(data.get('validation', {}).get('enabled', True)):
            pipeline.append(dict(type='ValidateBackend'))
        pipeline.append(dict(type='PackageBundle'))
        return pipeline

    @staticmethod
    def _normalize_codebase(data: Mapping[str, Any]) -> Dict[str, Any]:
        if 'codebase_config' in data:
            codebase = dict(data['codebase_config'])
        elif 'source' in data:
            source = dict(data['source'])
            source_type = str(source.get('type', ''))
            codebase = {
                'type': source_type,
                'task': ('predict_action' if source_type == 'fluxvla' else 'custom'),
                'source': source,
            }
        else:
            return {}
        if not codebase:
            return {}
        if not codebase.get('type') or not codebase.get('task'):
            raise ConfigurationError(
                'codebase_config requires non-empty type and task.')
        source = dict(codebase.get('source', {}))
        if not source:
            raise ConfigurationError('codebase_config.source cannot be empty.')
        codebase['source'] = source
        return codebase

    @staticmethod
    def _normalize_backend(data: Mapping[str, Any]) -> Dict[str, Any]:
        backend = dict(data.get('backend_config', data.get('tensorrt', {})))
        backend.setdefault('type', 'tensorrt')
        if not backend.get('type'):
            raise ConfigurationError('backend_config.type cannot be empty.')
        return backend

    @staticmethod
    def _normalize_ir(data: Mapping[str, Any]) -> Dict[str, Any]:
        ir = dict(data.get('ir_config', data.get('export', {})))
        ir.setdefault('type', 'onnx')
        if ir.get('type') != 'onnx':
            raise ConfigurationError('The initial FluxHermes IR must be onnx.')
        return ir

    @property
    def source(self) -> Dict[str, Any]:
        return self.codebase.get('source', {})

    @property
    def export(self) -> Dict[str, Any]:
        return self.ir

    @property
    def tensorrt(self) -> Dict[str, Any]:
        """Compatibility alias for the initial TensorRT-only configuration."""
        return self.backend

    @property
    def precision(self) -> str:
        default = 'fp16' if self.backend.get('type') == 'tensorrt' else 'fp32'
        return str(self.backend.get('precision', default))

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['codebase_config'] = result.pop('codebase')
        result['backend_config'] = result.pop('backend')
        result['ir_config'] = result.pop('ir')
        result['work_dir'] = str(self.work_dir)
        result['config_path'] = str(self.config_path) if self.config_path else None
        return result


@dataclass(frozen=True)
class BundleManifest:
    """Versioned, integrity-checked deployment bundle manifest."""

    schema_version: int
    name: str
    created_at: str
    backend: str
    precision: str
    backend_files: Tuple[str, ...]
    backend_metadata: Dict[str, Any]
    source: Dict[str, Any]
    inputs: Tuple[TensorSpec, ...]
    outputs: Tuple[TensorSpec, ...]
    runtime: Dict[str, Any]
    validation: Dict[str, Any]
    files: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'BundleManifest':
        schema_version = int(data.get('schema_version', 0))
        if schema_version not in {1, 2}:
            raise ConfigurationError(
                f"Unsupported bundle schema: {data.get('schema_version')}")
        if schema_version == 1:
            legacy = dict(data.get('engine', {}))
            legacy_files = legacy.pop('files', ())
            legacy_path = legacy.pop('path', None)
            backend_files = tuple(
                legacy_files or ([legacy_path] if legacy_path else ()))
            backend_metadata = legacy
        else:
            backend_files = tuple(data.get('backend_files', ()))
            backend_metadata = dict(data.get('backend_metadata', {}))
        if not backend_files:
            raise ConfigurationError('Bundle manifest has no backend files.')
        return cls(
            schema_version=schema_version,
            name=str(data['name']),
            created_at=str(data['created_at']),
            backend=str(data['backend']),
            precision=str(data['precision']),
            backend_files=backend_files,
            backend_metadata=backend_metadata,
            source=dict(data.get('source', {})),
            inputs=tuple(TensorSpec.from_dict(item) for item in data['inputs']),
            outputs=tuple(TensorSpec.from_dict(item) for item in data['outputs']),
            runtime=dict(data.get('runtime', {})),
            validation=dict(data.get('validation', {})),
            files=dict(data.get('files', {})),
            metadata=dict(data.get('metadata', {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['backend_files'] = list(self.backend_files)
        result['inputs'] = [spec.to_dict() for spec in self.inputs]
        result['outputs'] = [spec.to_dict() for spec in self.outputs]
        return result
