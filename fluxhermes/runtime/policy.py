from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..bundle import verify_bundle
from ..errors import ArtifactError
from ..plugins import add_python_paths, load_symbol


class IdentityHostAdapter:
    """Use when the robot loop already produces engine-ready tensors."""

    def __init__(self, manifest: Any = None, bundle_dir: Any = None) -> None:
        self.manifest = manifest
        self.bundle_dir = bundle_dir

    def preprocess(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        return observation

    def postprocess(self, outputs: Mapping[str, Any]) -> Any:
        action_name = self.manifest.runtime.get('action_output')
        if action_name:
            return outputs[action_name]
        if len(outputs) == 1:
            return next(iter(outputs.values()))
        return dict(outputs)

    def reset(self) -> None:
        return None


class RobotPolicy:
    """Validated deployment bundle exposed as a robot policy."""

    def __init__(
        self,
        bundle_dir: str | Path,
        require_validated: bool = True,
        profile_index: int = 0,
        device: str = 'cuda:0',
        require_target_match: bool = True,
    ) -> None:
        self.bundle_dir = Path(bundle_dir).expanduser().resolve()
        self.manifest = verify_bundle(self.bundle_dir)
        if require_validated and not self.manifest.validation.get('passed', False):
            raise ArtifactError(
                'Refusing to load an unvalidated backend for robot execution. '
                'Recompile with ValidateBackend in the pipeline or explicitly set '
                'require_validated=False for development only.')
        from ..apis import build_backend_wrapper
        self.engine = build_backend_wrapper(
            self.manifest,
            self.bundle_dir,
            profile_index=profile_index,
            device=device,
            require_target_match=require_target_match,
        )
        entrypoint = self.manifest.runtime.get('host_adapter')
        if entrypoint:
            add_python_paths([str(self.bundle_dir / 'assets')])
            adapter_type = load_symbol(str(entrypoint))
        else:
            adapter_type = IdentityHostAdapter
        self.host_adapter = adapter_type(
            manifest=self.manifest, bundle_dir=self.bundle_dir)

    @classmethod
    def from_bundle(cls, bundle_dir: str | Path, **kwargs: Any) -> 'RobotPolicy':
        return cls(bundle_dir, **kwargs)

    def predict(self, observation: Mapping[str, Any]) -> Any:
        inputs = self.host_adapter.preprocess(observation)
        outputs = self.engine.infer(inputs)
        return self.host_adapter.postprocess(outputs)

    def predict_action(self, **inputs: Any) -> Any:
        return self.predict(inputs)

    def reset(self) -> None:
        self.host_adapter.reset()
