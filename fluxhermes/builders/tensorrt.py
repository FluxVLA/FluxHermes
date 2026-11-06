from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

from ..contracts import TensorSpec
from ..errors import ConfigurationError, DependencyError
from ..runtime.tensorrt_logger import get_tensorrt_logger


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
        print('[FluxHermes] warning: target reports '
              f'no fast {precision} support.', file=sys.stderr)


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
                'Optimization profile input not found in ONNX graph: '
                f'{spec.name}')
        assert spec.min_shape is not None
        assert spec.opt_shape is not None
        assert spec.max_shape is not None
        requested = tuple(tuple(shape) for shape in (
            spec.min_shape, spec.opt_shape, spec.max_shape))
        # TensorRT 10.13.3 returns None on a successful set_shape call.
        # Check the mutated profile rather than interpreting that return value.
        profile.set_shape(spec.name, *requested)
        observed = tuple(tuple(int(dim) for dim in shape)
                         for shape in profile.get_shape(spec.name))
        if observed != requested:
            raise ConfigurationError(
                f'TensorRT profile bounds differ for {spec.name}: '
                f'requested={requested}, observed={observed}.')
    if not profile:
        raise ConfigurationError('TensorRT optimization profile is invalid.')
    if config.add_optimization_profile(profile) < 0:
        raise ConfigurationError('TensorRT rejected the optimization profile.')


def build_tensorrt_engine(
    onnx_path: str | Path,
    output_path: str | Path,
    inputs: Iterable[TensorSpec],
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """Build on the selected CUDA device without changing caller context."""
    from ..runtime.tensorrt import _cuda_target_metadata, _resolve_cuda_device

    device = _resolve_cuda_device(options.get('device', 'cuda:0'))
    try:
        import tensorrt as trt
        import torch
    except ImportError as error:
        raise DependencyError('TensorRT building requires the TensorRT Python bindings '
                              'and torch.') from error
    with torch.cuda.device(device):
        target = _cuda_target_metadata(device, tensorrt_version=str(trt.__version__))
        return _build_on_device(
            onnx_path, output_path, inputs, options, trt=trt, target=target)


def _build_on_device(
    onnx_path: str | Path,
    output_path: str | Path,
    inputs: Iterable[TensorSpec],
    options: Dict[str, Any],
    *,
    trt: Any,
    target: Dict[str, Any],
) -> Dict[str, Any]:
    # Called only inside the public entry point's fixed CUDA device context.
    source = Path(onnx_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    severity = _logger_severity(trt, str(options.get('log_level', 'warning')))
    logger = get_tensorrt_logger(severity)
    trt.init_libnvinfer_plugins(logger, '')
    builder = trt.Builder(logger)
    trt_major = int(str(trt.__version__).split('.', 1)[0])
    strong_option = options.get('strongly_typed', 'auto')
    if strong_option == 'auto':
        strongly_typed = trt_major >= 11
    elif isinstance(strong_option, bool):
        strongly_typed = strong_option
    else:
        raise ConfigurationError(
            "tensorrt.strongly_typed must be true, false, or 'auto'.")
    network_flags = 0
    explicit_flag = getattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH', None)
    if explicit_flag is not None:
        network_flags |= 1 << int(explicit_flag)
    if strongly_typed:
        strong_flag = getattr(trt.NetworkDefinitionCreationFlag, 'STRONGLY_TYPED', None)
        if strong_flag is None:
            raise ConfigurationError('This TensorRT version does not support '
                                     'strongly typed networks.')
        network_flags |= 1 << int(strong_flag)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(source)):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError('TensorRT ONNX parse failed:\n' + '\n'.join(errors))
    build_config = builder.create_builder_config()
    if 'tf32' in options:
        if type(options['tf32']) is not bool:
            raise ConfigurationError('tensorrt.tf32 must be true or false.')
        tf32_flag = getattr(trt.BuilderFlag, 'TF32', None)
        if tf32_flag is None:
            raise ConfigurationError('This TensorRT build cannot configure TF32 math.')
        if options['tf32']:
            build_config.set_flag(tf32_flag)
        else:
            build_config.clear_flag(tf32_flag)
    workspace_gib = float(options.get('workspace_size_gib', 8))
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                       int(workspace_gib * (1024**3)))
    precision = str(options.get('precision', 'fp16')).lower()
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
    cache_bytes = (
        timing_cache_path.read_bytes() if timing_cache_path.is_file() else b'')
    timing_cache = build_config.create_timing_cache(cache_bytes)
    if timing_cache is None:
        raise ConfigurationError(
            f'TensorRT could not create timing cache: {timing_cache_path}')
    if not build_config.set_timing_cache(timing_cache, ignore_mismatch=False):
        raise ConfigurationError(
            f'TensorRT rejected timing cache: {timing_cache_path}')
    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise RuntimeError('TensorRT failed to build a serialized engine.')
    output.write_bytes(bytes(serialized))
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
        'precision':
        precision,
        'strongly_typed':
        strongly_typed,
        'tf32': (bool(build_config.get_flag(trt.BuilderFlag.TF32)) if hasattr(
            trt.BuilderFlag, 'TF32') else None),
        'workspace_size_gib':
        workspace_gib,
        'timing_cache':
        str(timing_cache_path),
    }
    details.update(target)
    metadata_path = output.with_suffix(output.suffix + '.json')
    metadata_path.write_text(
        json.dumps(details, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return details
