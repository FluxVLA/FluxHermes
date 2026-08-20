# Real-robot runtime

## Bundle contents

A robot bundle contains:

```text
manifest.json
backend/model.plan
build/deployment.py
build/resolved_config.json
assets/                    optional tokenizer/stats/adapter files
```

The manifest includes checksums for every file. `fluxhermes inspect` and
`RobotPolicy` verify those checksums before asking the registered backend
manager to construct its wrapper.

## Preprocessed tensor mode

If the existing FluxVLA robot runner already performs tokenization,
normalization, padding, and action denormalization, use `IdentityHostAdapter`
and call `predict_action(**batch)`. This is the lowest-risk integration because
it preserves the existing FluxVLA semantics.

## Raw-observation mode

For a self-contained deployment, implement a host adapter:

```python
class AlohaHostAdapter:
    def __init__(self, manifest, bundle_dir):
        self.manifest = manifest
        self.bundle_dir = bundle_dir
        self.processor = load_processor(bundle_dir / 'assets' / 'processor')

    def preprocess(self, observation):
        return self.processor.encode(observation)

    def postprocess(self, outputs):
        normalized = outputs['actions']
        return self.processor.denormalize_and_crop(normalized)

    def reset(self):
        self.processor.reset()
```

Package the module and assets:

```python
package = dict(
    assets=[
        '/path/to/aloha_host_adapter.py',
        '/path/to/processor',
        '/path/to/dataset_statistics.json',
    ],
    runtime=dict(
        host_adapter='aloha_host_adapter:AlohaHostAdapter',
        action_output='actions',
    ),
)
```

## Release checklist

- Build on the actual device or a verified identical GPU/CUDA/TensorRT target.
- Run `fluxhermes doctor` and preserve its output with release metadata.
- Pass source-versus-engine parity on more than the single export sample.
- Confirm tokenizer, image order/color/layout, state order, masks, padding,
  normalization statistics, action cropping, and denormalization.
- Measure synchronized p50/p90/p99 latency at the intended shape profile.
- Verify control-loop deadline, stale-action timeout, watchdog, joint limits,
  workspace limits, collision handling, and emergency stop.
- Run simulation/task regression before guarded robot trials.
- Keep an immediate fallback to the previously validated bundle.

`--allow-unvalidated` and `require_validated=False` exist only for development;
they must not be used in a production robot launch.
