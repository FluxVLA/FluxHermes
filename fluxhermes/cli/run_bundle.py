"""Run one preprocessed NPZ sample and save the named output tensors."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import add_runtime_arguments, load_runtime, run_command, save_npz


def _execute(args: argparse.Namespace) -> int:
    engine, inputs = load_runtime(args)
    outputs = engine.infer(inputs, synchronize=True)
    save_npz(args.output, outputs)
    print(f'Wrote {len(outputs)} output tensor(s) to {Path(args.output).resolve()}')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='fluxhermes-run', description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument('--output', default='outputs.npz')
    return run_command(parser, _execute, argv)


if __name__ == '__main__':
    raise SystemExit(main())
