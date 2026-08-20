from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ..apis import build_backend_wrapper
from ..bundle import describe_bundle, verify_bundle
from ..config import load_config
from ..errors import FluxHermesError
from ..pipeline import CompilePipeline
from ..utils import environment_report


def _load_npz(path: str) -> Dict[str, Any]:
    try:
        import numpy as np
    except Exception as error:
        raise FluxHermesError('NPZ input handling requires numpy.') from error
    with np.load(Path(path).expanduser(), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _save_npz(path: str, outputs: Dict[str, Any]) -> None:
    try:
        import numpy as np
    except Exception as error:
        raise FluxHermesError('NPZ output handling requires numpy.') from error
    converted: Dict[str, Any] = {}
    for name, value in outputs.items():
        if hasattr(value, 'detach'):
            value = value.detach()
        if hasattr(value, 'cpu'):
            value = value.cpu()
        if hasattr(value, 'numpy'):
            value = value.numpy()
        converted[name] = value
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **converted)


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
    inputs = _load_npz(args.inputs)
    engine = build_backend_wrapper(
        manifest,
        args.bundle,
        profile_index=args.profile,
        device=args.device,
        require_target_match=not args.allow_target_mismatch,
    )
    outputs = engine.infer(inputs, synchronize=True)
    _save_npz(args.output, outputs)
    print(f'Wrote {len(outputs)} output tensor(s) to {Path(args.output).resolve()}')
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    manifest = verify_bundle(args.bundle)
    if not args.allow_unvalidated and not manifest.validation.get('passed', False):
        raise FluxHermesError('Refusing to benchmark an unvalidated robot bundle.')
    inputs = _load_npz(args.inputs)
    engine = build_backend_wrapper(
        manifest,
        args.bundle,
        profile_index=args.profile,
        device=args.device,
        require_target_match=not args.allow_target_mismatch,
    )
    for _ in range(args.warmup):
        engine.infer(inputs, synchronize=True)
    samples = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        engine.infer(inputs, synchronize=True)
        samples.append((time.perf_counter() - start) * 1000.0)
    result = {
        'warmup': args.warmup,
        'repeats': args.repeats,
        'mean_ms': statistics.fmean(samples),
        'p50_ms': _percentile(samples, 0.50),
        'p90_ms': _percentile(samples, 0.90),
        'p99_ms': _percentile(samples, 0.99),
        'min_ms': min(samples),
        'max_ms': max(samples),
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
        'run', help='run one preprocessed NPZ sample through a bundle')
    run_parser.add_argument('bundle')
    run_parser.add_argument('--inputs', required=True)
    run_parser.add_argument('--output', default='outputs.npz')
    run_parser.add_argument('--profile', type=int, default=0)
    run_parser.add_argument('--device', default='cuda:0')
    run_parser.add_argument('--allow-unvalidated', action='store_true')
    run_parser.add_argument('--allow-target-mismatch', action='store_true')
    run_parser.set_defaults(handler=_run)

    benchmark_parser = subparsers.add_parser(
        'benchmark', help='measure synchronized end-to-end engine latency')
    benchmark_parser.add_argument('bundle')
    benchmark_parser.add_argument('--inputs', required=True)
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
