from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

from ..contracts import TensorSpec
from ..errors import ConfigurationError, DependencyError


def _logger_severity(trt: Any, value: str) -> Any:
    levels = {
        'verbose': trt.Logger.VERBOSE,
        'info': trt.Logger.INFO,
        'warning': trt.Logger.WARNING,
        'error': trt.Logger.ERROR,
        'internal_error': trt.Logger.INTERNAL_ERROR,
    }
    try:
        return levels[value.lower()]
    except KeyError as error:
        raise ConfigurationError(f'Unknown TensorRT log level: {value}') from error


def _set_weak_precision(
    config: Any,
    builder: Any,
    trt: Any,
    precision: str,
) -> None:
    flag_names = {
        'fp16': 'FP16',
        'int8': 'INT8',
    }
    if precision == 'fp32':
        return
    name = flag_names[precision]
    flag = getattr(trt.BuilderFlag, name, None)
    if flag is None:
        raise ConfigurationError(
            f'This TensorRT build does not expose {name} precision.')
    config.set_flag(flag)
    fast_property = {
        'fp16': 'platform_has_fast_fp16',
        'int8': 'platform_has_fast_int8',
    }.get(precision)
    if fast_property and not bool(getattr(builder, fast_property, True)):
        print(f'[FluxHermes] warning: target reports no fast {precision} support.')


def _add_profile(
    builder: Any,
    network: Any,
    config: Any,
    inputs: Iterable[TensorSpec],
) -> None:
    dynamic = [spec for spec in inputs if spec.min_shape is not None]
    if not dynamic:
        return
    profile = builder.create_optimization_profile()
    network_names = {
        network.get_input(index).name
        for index in range(network.num_inputs)
    }
    for spec in dynamic:
        if spec.name not in network_names:
            raise ConfigurationError(
                f'Optimization profile input not found in ONNX graph: {spec.name}')
        assert spec.min_shape is not None
        assert spec.opt_shape is not None
        assert spec.max_shape is not None
        profile.set_shape(spec.name, spec.min_shape, spec.opt_shape, spec.max_shape)
        actual = tuple(tuple(shape) for shape in profile.get_shape(spec.name))
        expected = (spec.min_shape, spec.opt_shape, spec.max_shape)
        if actual != expected:
            raise ConfigurationError(
                f'TensorRT optimization profile readback mismatch for {spec.name}.')
    if not profile or config.add_optimization_profile(profile) < 0:
        raise ConfigurationError('TensorRT rejected the optimization profile.')


def _resolve_strongly_typed(trt_major: int, precision: str, option: Any) -> bool:
    if option == 'auto':
        strongly_typed = precision == 'bf16' or trt_major >= 11
    elif isinstance(option, bool):
        strongly_typed = option
    else:
        raise ConfigurationError(
            "tensorrt.strongly_typed must be true, false, or 'auto'.")
    if precision == 'bf16' and not strongly_typed:
        raise ConfigurationError('TensorRT bf16 requires strongly_typed=True.')
    return strongly_typed


def build_tensorrt_engine(
    onnx_path: str | Path,
    output_path: str | Path,
    inputs: Iterable[TensorSpec],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """Build an engine; file_descriptor_limit raises the process's soft limit."""
    descriptor_limit = options.get('file_descriptor_limit')
    if descriptor_limit is not None:
        if type(descriptor_limit) is not int or descriptor_limit < 1:
            raise ConfigurationError(
                'file_descriptor_limit must be a positive integer.')
        try:
            import resource
        except ImportError as error:
            raise ConfigurationError(
                'file_descriptor_limit requires POSIX resource support.') from error
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard != resource.RLIM_INFINITY and descriptor_limit > hard:
            raise ConfigurationError(
                'file_descriptor_limit exceeds the process hard limit.')
        if soft != resource.RLIM_INFINITY and descriptor_limit > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (descriptor_limit, hard))
        descriptor_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    if 'allow_tf32' in options and type(options['allow_tf32']) is not bool:
        raise ConfigurationError('TensorRT allow_tf32 must be true or false.')
    try:
        import tensorrt as trt
    except Exception as error:
        raise DependencyError(
            'TensorRT engine building requires the TensorRT Python bindings.'
        ) from error
    source = Path(onnx_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    severity = _logger_severity(trt, str(options.get('log_level', 'warning')))
    logger = trt.Logger(severity)
    trt.init_libnvinfer_plugins(logger, '')
    builder = trt.Builder(logger)
    trt_major = int(str(trt.__version__).split('.', 1)[0])
    precision = str(options.get('precision', 'fp16')).lower()
    strongly_typed = _resolve_strongly_typed(trt_major, precision,
                                             options.get('strongly_typed', 'auto'))
    network_flags = 0
    explicit_flag = getattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH', None)
    if explicit_flag is not None:
        network_flags |= 1 << int(explicit_flag)
    if strongly_typed:
        strong_flag = getattr(trt.NetworkDefinitionCreationFlag, 'STRONGLY_TYPED', None)
        if strong_flag is None:
            raise ConfigurationError(
                'This TensorRT version does not support strongly typed networks.')
        network_flags |= 1 << int(strong_flag)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(source)):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError('TensorRT ONNX parse failed:\n' + '\n'.join(errors))
    build_config = builder.create_builder_config()
    if 'allow_tf32' in options:
        tf32_flag = getattr(trt.BuilderFlag, 'TF32', None)
        if tf32_flag is None:
            raise ConfigurationError('This TensorRT build does not expose TF32.')
        if options['allow_tf32']:
            build_config.set_flag(tf32_flag)
        else:
            build_config.clear_flag(tf32_flag)
        if bool(build_config.get_flag(tf32_flag)) != options['allow_tf32']:
            raise ConfigurationError('TensorRT did not apply allow_tf32 as requested.')
    workspace_gib = float(options.get('workspace_size_gib', 8))
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                       int(workspace_gib * (1024**3)))
    if not strongly_typed:
        _set_weak_precision(build_config, builder, trt, precision)
    if not strongly_typed and bool(options.get('strict_types', False)):
        flag = getattr(trt.BuilderFlag, 'OBEY_PRECISION_CONSTRAINTS', None)
        if flag is not None:
            build_config.set_flag(flag)
    _add_profile(builder, network, build_config, inputs)
    timing_cache_path = Path(
        options.get('timing_cache',
                    output.with_suffix('.timing'))).expanduser().resolve()
    cache_bytes = timing_cache_path.read_bytes() if timing_cache_path.is_file() else b''
    timing_cache = build_config.create_timing_cache(cache_bytes)
    build_config.set_timing_cache(timing_cache, ignore_mismatch=False)
    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise RuntimeError('TensorRT failed to build a serialized engine.')
    output.write_bytes(memoryview(serialized))
    updated_cache = build_config.get_timing_cache()
    if updated_cache is not None:
        timing_cache_path.parent.mkdir(parents=True, exist_ok=True)
        timing_cache_path.write_bytes(bytes(updated_cache.serialize()))
    details: Dict[str, Any] = {
        'file':
        output.name,
        'size_bytes':
        output.stat().st_size,
        'tensorrt_version':
        str(trt.__version__),
        'file_descriptor_limit':
        descriptor_limit,
        'precision':
        precision,
        'strongly_typed':
        strongly_typed,
        'allow_tf32': (bool(build_config.get_flag(trt.BuilderFlag.TF32)) if hasattr(
            trt.BuilderFlag, 'TF32') else None),
        'workspace_size_gib':
        workspace_gib,
        'timing_cache':
        str(timing_cache_path),
    }
    try:
        import torch
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            details['device'] = torch.cuda.get_device_name(index)
            details['compute_capability'] = list(
                torch.cuda.get_device_capability(index))
            details['cuda_version'] = torch.version.cuda
    except Exception:
        pass
    metadata_path = output.with_suffix(output.suffix + '.json')
    metadata_path.write_text(
        json.dumps(details, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return details
