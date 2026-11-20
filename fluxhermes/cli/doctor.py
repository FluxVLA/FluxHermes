"""Report the available deployment dependencies and GPU environment."""
from __future__ import annotations

import argparse
import json

from .common import run_command


def _execute(args: argparse.Namespace) -> int:
    from ..utils import environment_report

    print(json.dumps(environment_report(), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='fluxhermes-doctor', description=__doc__)
    return run_command(parser, _execute, argv)


if __name__ == '__main__':
    raise SystemExit(main())
