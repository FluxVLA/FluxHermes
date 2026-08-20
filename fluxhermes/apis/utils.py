from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, List

from ..backend import get_backend_manager
from ..codebase import get_codebase
from ..contracts import BundleManifest, DeploymentConfig
from ..errors import ArtifactError
from ..plugins import add_python_paths


def import_config_plugins(config: DeploymentConfig) -> None:
    sections = (config.codebase, config.backend, config.ir)
    for section in sections:
        add_python_paths(section.get('python_paths', ()))
        for module in section.get('custom_modules', ()):
            importlib.import_module(str(module))


def build_task_processor(
    config: DeploymentConfig,
    device: str,
):
    import_config_plugins(config)
    codebase_name = str(config.codebase['type'])
    codebase = get_codebase(codebase_name)
    codebase.register_all_modules()
    return codebase.build_task_processor(config=config, device=device)


def get_backend_files_from_manifest(
    manifest: BundleManifest,
    bundle_dir: str | Path,
) -> List[Path]:
    root = Path(bundle_dir).expanduser().resolve()
    files = [(root / relative).resolve() for relative in manifest.backend_files]
    for path in files:
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ArtifactError(f'Unsafe backend file path: {path}') from error
    return files


def build_backend_wrapper(
    manifest: BundleManifest,
    bundle_dir: str | Path,
    device: str = 'cuda:0',
    profile_index: int = 0,
    require_target_match: bool = True,
    **kwargs: Any,
):
    backend_config = manifest.metadata.get('backend_config', {})
    add_python_paths(backend_config.get('python_paths', ()))
    for module in backend_config.get('custom_modules', ()):
        importlib.import_module(str(module))
    manager = get_backend_manager(manifest.backend)
    files = get_backend_files_from_manifest(manifest, bundle_dir)
    return manager.build_wrapper(
        files,
        device=device,
        profile_index=profile_index,
        target_metadata=manifest.backend_metadata,
        require_target_match=require_target_match,
        **kwargs,
    )
