"""Compatibility entry point for the original fluxhermes/fluxdeploy CLI."""
from __future__ import annotations

import argparse
import importlib
import sys
from typing import Iterable

_COMMANDS = {
    'compile': 'compile',
    'inspect': 'inspect_bundle',
    'doctor': 'doctor',
    'run': 'run_bundle',
    'benchmark': 'benchmark',
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='fluxhermes', description='Compile and run FluxVLA deployment bundles.')
    parser.add_argument('command', choices=_COMMANDS)
    parser.add_argument('args', nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    module = importlib.import_module(f'.{_COMMANDS[args.command]}', __package__)
    return module.main(args.args)


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
