"""Measure synchronized engine latency using one representative sample."""
from __future__ import annotations

import argparse
import json
import statistics
import time

from .common import add_runtime_arguments, load_runtime, run_command


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return ordered[index]


def _execute(args: argparse.Namespace) -> int:
    engine, inputs = load_runtime(args)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='fluxhermes-benchmark', description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=100)
    return run_command(parser, _execute, argv)


if __name__ == '__main__':
    raise SystemExit(main())
