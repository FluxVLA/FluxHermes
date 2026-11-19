from __future__ import annotations

import json
import os
import shutil
from pathlib import Path, PureWindowsPath
from typing import Dict, Iterable

from .contracts import BundleManifest
from .errors import ArtifactError, DependencyError
from .utils import read_json, sha256_file, write_json

MANIFEST_NAME = 'manifest.json'
BUNDLE_FILE_MODE_ENV = 'FLUXHERMES_BUNDLE_FILE_MODE'


def _stage_file(source: Path, target: Path) -> None:
    mode = os.getenv(BUNDLE_FILE_MODE_ENV, 'copy')
    if mode == 'copy':
        shutil.copy2(source, target)
        return
    if mode == 'hardlink':
        try:
            target.hardlink_to(source)
        except OSError as error:
            raise ArtifactError(
                f'Unable to hardlink bundle file {source} to {target}: {error}'
            ) from error
        shutil.copystat(source, target)
        return
    raise ArtifactError(
        f'Unsupported {BUNDLE_FILE_MODE_ENV} value: {mode!r}; '
        "expected 'copy' or 'hardlink'")


def _onnx_external_locations(graph: Path) -> set[Path]:
    """Read tensor references, never the potentially multi-GB external weights."""
    try:
        import onnx
    except ImportError as error:
        raise DependencyError('Bundling or verifying ONNX files requires onnx; '
                              'install fluxhermes[export].') from error
    from .exporters.onnx import _walk_tensor_protos

    try:
        model = onnx.load(str(graph), load_external_data=False)
    except Exception as error:
        raise ArtifactError(f'Invalid ONNX graph {graph}: {error}') from error
    locations = set()
    for tensor in _walk_tensor_protos(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        values = [item.value for item in tensor.external_data if item.key == 'location']
        if len(values) != 1 or not values[0]:
            raise ArtifactError(
                f'Invalid ONNX external data location in {graph}, '
                f'tensor {tensor.name!r}'
            )
        location = Path(values[0])
        windows_path = PureWindowsPath(values[0])
        if (not location.parts or location.is_absolute() or windows_path.drive
                or windows_path.root or '..' in location.parts
                or '..' in windows_path.parts):
            raise ArtifactError(
                f'Unsafe ONNX external data location {values[0]!r} in {graph}')
        locations.add(location)
    return locations


def _onnx_graph_locations(files: Dict[str, Path]) -> Dict[str, set[Path]]:
    """Distinguish graph candidates from referenced data even with .onnx names."""
    graphs: Dict[str, set[Path]] = {}
    errors = {}
    external_files = set()
    candidates = [(relative, source) for relative, source in files.items()
                  if Path(relative).suffix.lower() == '.onnx']
    # Small graph metadata should identify larger weight files before loading them.
    for relative, source in sorted(candidates, key=lambda item: item[1].stat().st_size):
        if relative in external_files:
            continue
        try:
            graphs[relative] = _onnx_external_locations(source)
        except ArtifactError as error:
            errors[relative] = error
        else:
            external_files.update((Path(relative).parent / location).as_posix()
                                  for location in graphs[relative])
    for relative, error in errors.items():
        if relative not in external_files:
            raise error
    return {
        relative: locations for relative, locations in graphs.items()
        if relative not in external_files
    }


def _collect_bundle_files(files: Dict[str, str | Path]) -> Dict[str, Path]:
    sources: Dict[str, Path] = {}

    def add(relative: Path, source: Path) -> None:
        if relative.is_absolute() or '..' in relative.parts:
            raise ArtifactError(f'Unsafe bundle destination: {relative}')
        key = relative.as_posix()
        if key in sources and sources[key] != source:
            raise ArtifactError(
                f'Bundle source conflict for {key}: {sources[key]} vs {source}')
        if not source.is_file():
            raise ArtifactError(f'Bundle source file is missing: {source}')
        sources[key] = source

    for relative, source in files.items():
        add(Path(relative), Path(source).expanduser().resolve())
    for relative, locations in _onnx_graph_locations(sources).items():
        graph = sources[relative]
        for location in sorted(locations):
            source = (graph.parent / location).resolve()
            if not source.is_relative_to(graph.parent):
                raise ArtifactError(
                    f'Unsafe ONNX external data location {location} in {graph}')
            add(Path(relative).parent / location, source)
    return sources


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
        raise ArtifactError(
            'Backend files are not covered by bundle checksums: '
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
    sources = {relative: root / relative for relative in manifest.files}
    for relative, locations in _onnx_graph_locations(sources).items():
        for location in sorted(locations):
            external = (Path(relative).parent / location).as_posix()
            if external not in manifest.files:
                raise ArtifactError(
                    'ONNX external data is not covered by bundle checksums: '
                    f'{external} (referenced by {relative})')
    return manifest


def create_bundle(
    output_dir: str | Path,
    manifest: BundleManifest,
    files: Dict[str, str | Path],
    assets: Iterable[str | Path] = (),
) -> Path:
    root = Path(output_dir).expanduser().resolve()
    sources = _collect_bundle_files(files)
    staging = root.with_name(root.name + '.staging')
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    checksums: Dict[str, str] = {}
    for relative, source in sources.items():
        relative_path = Path(relative)
        target = staging / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        _stage_file(source, target)
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
    if root.exists():
        shutil.rmtree(root)
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
