from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from zipfile import BadZipFile

from ..apis import build_backend_wrapper
from ..bundle import describe_bundle, verify_bundle
from ..config import load_config
from ..errors import FluxHermesError
from ..pipeline import CompilePipeline
from ..utils import environment_report


def _load_npz(path: str) -> Dict[str, Any]:
    """Read named NumPy arrays without permitting pickle or duplicate names."""
    try:
        import numpy as np
    except ImportError as error:
        raise FluxHermesError(
            'NPZ input handling requires the optional dependency numpy.') from error
    source = Path(path).expanduser()
    try:
        archive = np.load(source, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise FluxHermesError(f'{source}: expected a named NPZ archive.')
        with archive:
            if len(archive.files) != len(set(archive.files)):
                raise FluxHermesError(f'{source}: NPZ tensor names must be unique.')
            return {name: archive[name] for name in archive.files}
    except (OSError, ValueError, BadZipFile) as error:
        raise FluxHermesError(f'{source}: cannot load NPZ tensors: {error}') from error


def _load_inputs(path: str, *, device: str) -> Dict[str, Any]:
    """Preload real named tensors on the selected device without dtype casts."""
    source = Path(path).expanduser()
    suffix = source.suffix.lower()
    if suffix not in ('.npz', '.safetensors'):
        raise FluxHermesError(f'{source}: inputs must use .npz or .safetensors format.')
    try:
        import torch
    except ImportError as error:
        raise FluxHermesError(
            'CLI tensor input handling requires the optional dependency torch.'
        ) from error
    try:
        target = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as error:
        raise FluxHermesError(f'Invalid input device {device!r}: {error}') from error
    if target.type not in ('cpu', 'cuda'):
        raise FluxHermesError(
            f'Unsupported input device {device!r}; choose cpu or cuda:<index>.')
    if target.type == 'cuda' and not torch.cuda.is_available():
        raise FluxHermesError(
            f'Input device {device!r} requires available CUDA; no CPU fallback is used.'
        )
    if suffix == '.npz':
        tensors = {}
        for name, value in _load_npz(str(source)).items():
            try:
                tensors[name] = torch.from_numpy(value)
            except (TypeError, ValueError, RuntimeError) as error:
                raise FluxHermesError(
                    f'{source}: tensor {name!r} cannot preserve NumPy dtype '
                    f'{getattr(value, "dtype", None)} in torch: {error}') from error
    else:
        try:
            from safetensors import SafetensorError
            from safetensors.torch import load_file
        except ImportError as error:
            raise FluxHermesError(
                'Safetensors input handling requires the optional dependency '
                'safetensors.') from error
        try:
            tensors = load_file(str(source), device='cpu')
        except (OSError, ValueError, SafetensorError) as error:
            raise FluxHermesError(
                f'{source}: cannot load safetensors inputs: {error}') from error
    if not tensors:
        raise FluxHermesError(f'{source}: input archive contains no tensors.')
    result = {}
    for name, value in tensors.items():
        try:
            result[name] = value.to(device=target)
        except (RuntimeError, ValueError) as error:
            raise FluxHermesError(
                f'{source}: cannot place tensor {name!r} on {device!r}: {error}'
            ) from error
    return result


def _save_npz(path: str, outputs: Dict[str, Any]) -> Path:
    try:
        import numpy as np
    except ImportError as error:
        raise FluxHermesError('NPZ output handling requires numpy.') from error
    converted: Dict[str, Any] = {}
    for name, value in outputs.items():
        try:
            if hasattr(value, 'detach'):
                value = value.detach()
            if hasattr(value, 'cpu'):
                value = value.cpu()
            if hasattr(value, 'numpy'):
                value = value.numpy()
        except (RuntimeError, TypeError, ValueError) as error:
            raise FluxHermesError(
                f'Output {name!r} cannot be represented in NPZ without '
                f'changing dtype: {error}') from error
        converted[name] = value
    destination = Path(path).expanduser().resolve()
    if not str(destination).endswith('.npz'):
        destination = destination.with_name(destination.name + '.npz')
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez(destination, **converted)
    except (OSError, ValueError) as error:
        raise FluxHermesError(
            f'Cannot save NPZ outputs to {destination}: {error}') from error
    return destination


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return ordered[index]


def _compile(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    pipeline = CompilePipeline(config)
    if not args.execute:
        print(json.dumps(pipeline.plan_dict(), indent=2, sort_keys=True))
        print('\nDry run only. Add --execute to build the deployment bundle.')
        return 0
    bundle = pipeline.execute()
    print(f'Bundle created: {bundle}')
    return 0


def _inspect(args: argparse.Namespace) -> int:
    print(describe_bundle(args.bundle))
    return 0


def _doctor(args: argparse.Namespace) -> int:
    print(json.dumps(environment_report(), indent=2, sort_keys=True))
    return 0


def _run(args: argparse.Namespace) -> int:
    manifest = verify_bundle(args.bundle)
    if not args.allow_unvalidated and not manifest.validation.get('passed', False):
        raise FluxHermesError(
            'Bundle is not validated. Use --allow-unvalidated only for development.')
    inputs = _load_inputs(args.inputs, device=args.device)
    engine = build_backend_wrapper(
        manifest,
        args.bundle,
        profile_index=args.profile,
        device=args.device,
        require_target_match=not args.allow_target_mismatch,
    )
    outputs = engine.infer(inputs, synchronize=True)
    destination = _save_npz(args.output, outputs)
    print(f'Wrote {len(outputs)} output tensor(s) to {destination}')
    return 0


def _measure_latency(runner, *, warmup: int, repeats: int, device: str,
                     synchronize_device: bool) -> Dict[str, Any]:
    """Measure real callable execution; never include setup or serialization."""
    if synchronize_device:
        import torch
        target = torch.device(device)
    else:
        target = None
    for _ in range(warmup):
        runner()
        if target is not None and target.type == 'cuda':
            torch.cuda.synchronize(target)
    samples = []
    for _ in range(repeats):
        if target is not None and target.type == 'cuda':
            torch.cuda.synchronize(target)
        start = time.perf_counter()
        runner()
        if target is not None and target.type == 'cuda':
            torch.cuda.synchronize(target)
        samples.append((time.perf_counter() - start) * 1000.0)
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise FluxHermesError('Benchmark produced no finite, positive latency samples.')
    return {
        'mean_ms': statistics.fmean(samples),
        'p50_ms': _percentile(samples, 0.50),
        'p90_ms': _percentile(samples, 0.90),
        'p99_ms': _percentile(samples, 0.99),
        'min_ms': min(samples),
        'max_ms': max(samples),
    }


def _benchmark_reference(args: argparse.Namespace, manifest: Any, inputs: Dict[str,
                                                                               Any]):
    """Benchmark the actual public SDK reference, retaining CPU outputs only."""
    from ..apis import build_task_processor

    provenance = {}
    for name in ('recipe', 'source_config_sha256', 'checkpoint_sha256',
                 'inputs_sha256'):
        value = manifest.source.get(name)
        if not isinstance(value, str) or not value:
            raise FluxHermesError(f'Bundle source provenance requires {name}.')
        if name != 'recipe' and (len(value) != 64
                                 or any(character not in '0123456789abcdef'
                                        for character in value)):
            raise FluxHermesError(f'Bundle source provenance has invalid {name}.')
        provenance[name] = value
    # Use the bundle's recorded parity policy, not weaker reference-config
    # overrides or new default tolerances chosen by the benchmark command.
    tolerances = {}
    for name in ('atol', 'rtol', 'min_cosine_similarity'):
        value = manifest.validation.get(name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise FluxHermesError(
                f'Reference comparison requires recorded finite bundle '
                f'validation.{name}.')
        if (name != 'min_cosine_similarity'
                and value < 0) or (name == 'min_cosine_similarity'
                                   and not -1 <= value <= 1):
            raise FluxHermesError(f'Invalid bundle parity tolerance {name}={value}.')
        tolerances[name] = float(value)
    config = load_config(args.reference_config)
    if config.codebase.get('type') != 'fluxvla' or config.codebase.get(
            'task') != 'export_package':
        raise FluxHermesError(
            '--reference-config requires the native fluxvla export_package task.')
    try:
        import torch
        from fluxvla_export import PreparedDeployment

        processor = build_task_processor(config, device=args.device)
        prepared = processor.prepare_model()
    except ImportError as error:
        raise FluxHermesError(
            f'Reference benchmarking requires the installed fluxvla-export SDK '
            f'and its model dependencies: {error}') from error
    if not isinstance(prepared, PreparedDeployment):
        raise FluxHermesError(
            'Reference task must return native SDK PreparedDeployment.')
    actual_provenance = {
        'recipe': prepared.recipe,
        **{
            name: prepared.metadata.get(name)
            for name in provenance if name != 'recipe'
        }
    }
    for name, expected in provenance.items():
        if actual_provenance[name] != expected:
            raise FluxHermesError(
                f'Reference source provenance mismatch for {name}: '
                f'bundle={expected!r}, reference={actual_provenance[name]!r}.')
    if set(inputs) != set(prepared.policy_inputs):
        raise FluxHermesError(
            'CLI input names differ from the bound reference fixture.')
    for name, expected in prepared.policy_inputs.items():
        actual = inputs[name]
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise FluxHermesError(
                f'CLI input {name!r} dtype/shape differs from the bound '
                f'reference fixture.')
        if actual.device != expected.device or not torch.equal(actual, expected):
            raise FluxHermesError(
                f'CLI input {name!r} values/device differ from the bound '
                f'reference fixture.')
    # This is a genuine, untimed source execution. Only its detached CPU data
    # survives preparation; no fake reference callable is created for parity.
    outputs = prepared.run_reference()
    frozen_outputs = {
        name: value.detach().cpu().clone()
        for name, value in outputs.items()
    }
    del outputs
    statistics_report = _measure_latency(
        prepared.run_reference,
        warmup=args.warmup,
        repeats=args.repeats,
        device=args.device,
        synchronize_device=True)
    report = {
        'backend': 'pytorch',
        'torch_version': str(torch.__version__),
        'torch_num_threads': torch.get_num_threads(),
        'torch_num_interop_threads': torch.get_num_interop_threads(),
        'config': str(Path(args.reference_config).expanduser().resolve()),
        'device': args.device,
        'warmup': args.warmup,
        'repeats': args.repeats,
        'inputs_preloaded': True,
        'input_loading_included': False,
        'host_to_device_transfer_included': False,
        'model_loading_included': False,
        'timing_scope': 'synchronized_policy_inference',
        'synchronize': True,
        'provenance': provenance,
        **statistics_report,
    }
    del prepared, processor
    return report, frozen_outputs, tolerances


def _release_reference_memory(device: str) -> None:
    # Called after _benchmark_reference returns, so no model/prepared/task
    # references remain on its stack. CLI inputs stay alive for backend use.
    import gc

    import torch

    gc.collect()
    target = torch.device(device)
    if target.type == 'cuda':
        torch.cuda.synchronize(target)
        with torch.cuda.device(target):
            torch.cuda.empty_cache()


def _benchmark(args: argparse.Namespace) -> int:
    if args.warmup < 0:
        raise FluxHermesError('--warmup must be greater than or equal to 0.')
    if args.repeats <= 0:
        raise FluxHermesError('--repeats must be greater than 0.')
    manifest = verify_bundle(args.bundle)
    if not args.allow_unvalidated and not manifest.validation.get('passed', False):
        raise FluxHermesError('Refusing to benchmark an unvalidated robot bundle.')
    inputs = _load_inputs(args.inputs, device=args.device)
    reference_report = None
    if args.reference_config is not None:
        reference_report, reference_outputs, tolerances = _benchmark_reference(
            args, manifest, inputs)
        _release_reference_memory(args.device)
    engine = build_backend_wrapper(
        manifest,
        args.bundle,
        profile_index=args.profile,
        device=args.device,
        require_target_match=not args.allow_target_mismatch,
    )
    if reference_report is not None:
        from ..validation.parity import compare_outputs

        candidate = engine.infer(inputs, synchronize=True)
        parity = compare_outputs(reference_outputs, candidate, **tolerances)
        del reference_outputs, candidate
        if not parity['passed']:
            raise FluxHermesError(f'Reference/backend policy parity failed: {parity}')
    statistics_report = _measure_latency(
        partial(engine.infer, inputs, synchronize=True),
        warmup=args.warmup,
        repeats=args.repeats,
        device=args.device,
        synchronize_device=reference_report is not None)
    result = {
        'backend': manifest.backend,
        'device': args.device,
        'profile_index': args.profile,
        'input_format': Path(args.inputs).suffix.lower().lstrip('.'),
        'inputs_preloaded': True,
        'input_loading_included': False,
        'host_to_device_transfer_included': False,
        'backend_loading_included': False,
        'timing_scope': 'synchronized_policy_inference',
        'synchronize': True,
        'warmup': args.warmup,
        'repeats': args.repeats,
        **statistics_report,
    }
    if reference_report is not None:
        result['reference'] = reference_report
        result['comparison'] = {
            'parity':
            parity,
            'reference_over_backend_mean_ratio':
            reference_report['mean_ms'] / result['mean_ms'],
            'ratio_definition':
            'reference.mean_ms / mean_ms; greater than 1 means lower backend latency',
            'measurement_order': ['reference', 'backend'],
            'reference_released_before_backend_load':
            True,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='fluxhermes',
        description='Compile and run FluxVLA TensorRT deployment bundles.',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    compile_parser = subparsers.add_parser(
        'compile', help='plan or execute an ONNX/TensorRT compile pipeline')
    compile_parser.add_argument('config', help='deployment Python config')
    compile_parser.add_argument(
        '--execute', action='store_true', help='perform GPU build instead of dry run')
    compile_parser.set_defaults(handler=_compile)

    inspect_parser = subparsers.add_parser(
        'inspect', help='verify checksums and print bundle metadata')
    inspect_parser.add_argument('bundle')
    inspect_parser.set_defaults(handler=_inspect)

    doctor_parser = subparsers.add_parser(
        'doctor', help='report CUDA, TensorRT, ONNX, and PyTorch availability')
    doctor_parser.set_defaults(handler=_doctor)

    run_parser = subparsers.add_parser(
        'run', help='run one preprocessed NPZ or safetensors fixture through a bundle')
    run_parser.add_argument('bundle')
    run_parser.add_argument(
        '--inputs', required=True, help='named .npz or .safetensors tensors')
    run_parser.add_argument('--output', default='outputs.npz')
    run_parser.add_argument('--profile', type=int, default=0)
    run_parser.add_argument('--device', default='cuda:0')
    run_parser.add_argument('--allow-unvalidated', action='store_true')
    run_parser.add_argument('--allow-target-mismatch', action='store_true')
    run_parser.set_defaults(handler=_run)

    benchmark_parser = subparsers.add_parser(
        'benchmark', help='measure synchronized policy latency with preloaded inputs')
    benchmark_parser.add_argument('bundle')
    benchmark_parser.add_argument(
        '--inputs', required=True, help='named .npz or .safetensors tensors')
    benchmark_parser.add_argument(
        '--reference-config',
        metavar='HERMES_CONFIG',
        help='compare against the bound SDK PyTorch policy before loading the backend')
    benchmark_parser.add_argument('--warmup', type=int, default=10)
    benchmark_parser.add_argument('--repeats', type=int, default=100)
    benchmark_parser.add_argument('--profile', type=int, default=0)
    benchmark_parser.add_argument('--device', default='cuda:0')
    benchmark_parser.add_argument('--allow-unvalidated', action='store_true')
    benchmark_parser.add_argument('--allow-target-mismatch', action='store_true')
    benchmark_parser.set_defaults(handler=_benchmark)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.handler(args))
    except (FluxHermesError, FileNotFoundError, ValueError) as error:
        print(f'fluxhermes: error: {error}', file=sys.stderr)
        return 2


def _subcommand_main(command: str) -> None:
    raise SystemExit(main([command, *sys.argv[1:]]))


def compile_main() -> None:
    _subcommand_main('compile')


def inspect_main() -> None:
    _subcommand_main('inspect')


def run_main() -> None:
    _subcommand_main('run')


if __name__ == '__main__':
    raise SystemExit(main())
