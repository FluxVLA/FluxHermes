from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, output)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def copy_files(paths: Iterable[str | Path], destination: Path) -> Dict[str, str]:
    copied: Dict[str, str] = {}
    destination.mkdir(parents=True, exist_ok=True)
    for raw_path in paths:
        source = Path(raw_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f'Runtime asset not found: {source}')
        target = destination / source.name
        shutil.copy2(source, target)
        copied[str(target)] = sha256_file(target)
    return copied


def module_version(name: str) -> Optional[str]:
    try:
        module = __import__(name)
    except Exception:
        return None
    return str(getattr(module, '__version__', 'installed'))


def command_version(
    command: str,
    args: Iterable[str] = ('--version',),
) -> Optional[str]:
    executable = shutil.which(command)
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return 'installed'
    output = (result.stdout or result.stderr).strip().splitlines()
    return output[0] if output else 'installed'


def environment_report() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'machine': platform.machine(),
        'modules': {
            name: module_version(name)
            for name in ('numpy', 'torch', 'onnx', 'onnxruntime', 'tensorrt')
        },
        'commands': {
            'nvidia-smi': command_version('nvidia-smi'),
            'trtexec': command_version('trtexec'),
        },
    }
    try:
        import torch
        report['cuda'] = {
            'available': bool(torch.cuda.is_available()),
            'torch_cuda': torch.version.cuda,
            'device_count': torch.cuda.device_count(),
            'devices': [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception:
        report['cuda'] = {'available': False}
    try:
        from .backend import get_backend_manager
        report['backends'] = {
            'tensorrt': get_backend_manager('tensorrt').check_env(),
        }
    except Exception as error:
        report['backends'] = {'error': f'{type(error).__name__}: {error}'}
    try:
        from .codebase import CODEBASES, register_builtin_codebases
        register_builtin_codebases()
        report['codebases'] = sorted(CODEBASES.module_dict)
    except Exception as error:
        report['codebases'] = {'error': f'{type(error).__name__}: {error}'}
    return report
