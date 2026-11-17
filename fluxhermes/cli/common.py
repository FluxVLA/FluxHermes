"""Shared argument handling, tensor I/O, and runtime loading for CLI tools."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..errors import FluxHermesError


def run_command(
    parser: argparse.ArgumentParser,
    handler: Callable[[argparse.Namespace], int],
    argv: list[str] | None = None,
) -> int:
    args = parser.parse_args(argv)
    try:
        return handler(args)
    except (FluxHermesError, FileNotFoundError, ValueError) as error:
        print(f'{parser.prog}: error: {error}', file=sys.stderr)
        return 2


def load_npz(path: str) -> dict[str, Any]:
    try:
        import numpy as np
    except Exception as error:
        raise FluxHermesError('NPZ input handling requires numpy.') from error
    with np.load(Path(path).expanduser(), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def save_npz(path: str, outputs: dict[str, Any]) -> None:
    try:
        import numpy as np
    except Exception as error:
        raise FluxHermesError('NPZ output handling requires numpy.') from error
    converted: dict[str, Any] = {}
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


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('bundle')
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--profile', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--allow-unvalidated', action='store_true')
    parser.add_argument('--allow-target-mismatch', action='store_true')


def load_runtime(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    from ..apis import build_backend_wrapper
    from ..bundle import verify_bundle

    manifest = verify_bundle(args.bundle)
    if not args.allow_unvalidated and not manifest.validation.get('passed', False):
        raise FluxHermesError(
            'Bundle is not validated. Use --allow-unvalidated only for development.')
    inputs = load_npz(args.inputs)
    engine = build_backend_wrapper(
        manifest,
        args.bundle,
        profile_index=args.profile,
        device=args.device,
        require_target_match=not args.allow_target_mismatch,
    )
    return engine, inputs
