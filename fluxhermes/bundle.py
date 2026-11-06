from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable

from .contracts import BundleManifest
from .errors import ArtifactError
from .utils import read_json, sha256_file, write_json

MANIFEST_NAME = 'manifest.json'


def load_manifest(bundle_dir: str | Path) -> BundleManifest:
    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ArtifactError(f'Bundle manifest not found: {manifest_path}')
    try:
        return BundleManifest.from_dict(read_json(manifest_path))
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactError(f'Invalid bundle manifest: {error}') from error


def verify_bundle(bundle_dir: str | Path) -> BundleManifest:
    root = Path(bundle_dir).expanduser().resolve()
    manifest = load_manifest(root)
    unchecked_backend_files = [
        path for path in manifest.backend_files if path not in manifest.files
    ]
    if unchecked_backend_files:
        raise ArtifactError('Backend files are not covered by bundle checksums: '
                            f'{unchecked_backend_files}')
    for relative, expected in manifest.files.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ArtifactError(
                f'Bundle manifest contains an unsafe path: {relative}') from error
        if not path.is_file():
            raise ArtifactError(f'Bundle file is missing: {relative}')
        actual = sha256_file(path)
        if actual != expected:
            raise ArtifactError(
                f'Bundle checksum mismatch for {relative}: {actual} != {expected}')
    return manifest


def validate_bundle_destination(output_dir: str | Path) -> tuple[Path, Path]:
    """Reject existing user data; compile and packaging share this preflight."""
    requested = Path(output_dir).expanduser()
    root = requested.resolve()
    if root.exists() or requested.is_symlink():
        raise ArtifactError(
            f'Bundle output must be a new path; refusing to replace {root}.')
    staging = root.with_name(root.name + '.staging')
    if staging.exists() or staging.is_symlink():
        raise ArtifactError(
            f'Bundle staging path already exists; refusing to remove {staging}.')
    return root, staging


def create_bundle(
        output_dir: str | Path,
        manifest: BundleManifest,
        files: Dict[str, str | Path],
        assets: Iterable[str | Path] = (),
) -> Path:
    root, staging = validate_bundle_destination(output_dir)
    staging.mkdir(parents=True)
    checksums: Dict[str, str] = {}
    for relative, raw_source in files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or '..' in relative_path.parts:
            raise ArtifactError(f'Unsafe bundle destination: {relative}')
        source = Path(raw_source).expanduser().resolve()
        if not source.is_file():
            raise ArtifactError(f'Bundle source file is missing: {source}')
        target = staging / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        checksums[relative_path.as_posix()] = sha256_file(target)
    for raw_asset in assets:
        source = Path(raw_asset).expanduser().resolve()
        if not source.exists():
            raise ArtifactError(f'Runtime asset is missing: {source}')
        if source.is_dir():
            target_root = staging / 'assets' / source.name
            shutil.copytree(source, target_root)
            for target in target_root.rglob('*'):
                if target.is_file():
                    relative = target.relative_to(staging)
                    checksums[relative.as_posix()] = sha256_file(target)
        else:
            relative = Path('assets') / source.name
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            checksums[relative.as_posix()] = sha256_file(target)
    manifest_data = manifest.to_dict()
    manifest_data['files'] = checksums
    write_json(staging / MANIFEST_NAME, manifest_data)
    if root.exists() or root.is_symlink():
        raise ArtifactError(
            f'Bundle output appeared during staging; refusing to replace {root}.')
    staging.rename(root)
    verify_bundle(root)
    return root


def describe_bundle(bundle_dir: str | Path) -> str:
    manifest = verify_bundle(bundle_dir)
    summary = {
        'name': manifest.name,
        'created_at': manifest.created_at,
        'backend': manifest.backend,
        'precision': manifest.precision,
        'validated': bool(manifest.validation.get('passed', False)),
        'backend_files': list(manifest.backend_files),
        'backend_metadata': manifest.backend_metadata,
        'inputs': [item.to_dict() for item in manifest.inputs],
        'outputs': [item.to_dict() for item in manifest.outputs],
        'files': manifest.files,
    }
    return json.dumps(summary, indent=2, sort_keys=True)
