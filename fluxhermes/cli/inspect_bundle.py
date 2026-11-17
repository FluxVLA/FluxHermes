"""Verify deployment bundle integrity and display its metadata."""
from __future__ import annotations

import argparse

from .common import run_command


def _execute(args: argparse.Namespace) -> int:
    from ..bundle import describe_bundle

    print(describe_bundle(args.bundle))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='fluxhermes-inspect', description=__doc__)
    parser.add_argument('bundle')
    return run_command(parser, _execute, argv)


if __name__ == '__main__':
    raise SystemExit(main())
