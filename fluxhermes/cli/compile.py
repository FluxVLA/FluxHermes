"""Plan or execute the complete deployment compile pipeline."""
from __future__ import annotations

import argparse
import json

from .common import run_command


def _execute(args: argparse.Namespace) -> int:
    from ..config import load_config
    from ..pipeline import CompilePipeline

    config = load_config(args.config)
    pipeline = CompilePipeline(config)
    if not args.execute:
        print(json.dumps(pipeline.plan_dict(), indent=2, sort_keys=True))
        print('\nDry run only. Add --execute to build the deployment bundle.')
        return 0
    bundle = pipeline.execute()
    print(f'Bundle created: {bundle}')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='fluxhermes-compile', description=__doc__)
    parser.add_argument('config', help='deployment Python config')
    parser.add_argument(
        '--execute', action='store_true', help='perform GPU build instead of dry run')
    return run_command(parser, _execute, argv)


if __name__ == '__main__':
    raise SystemExit(main())
