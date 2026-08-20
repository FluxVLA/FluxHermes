# MMDeploy concept alignment

FluxHermes references the architecture of
[OpenMMLab MMDeploy](https://github.com/open-mmlab/mmdeploy), especially its
codebase/task, backend manager/wrapper, rewrite, and pipeline boundaries. It is
not a fork and does not import MMDeploy at runtime.

| MMDeploy concept | FluxHermes equivalent | VLA-specific adaptation |
| --- | --- | --- |
| Codebase registry | `fluxhermes.codebase.CODEBASES` | FluxVLA and external recipes |
| Base task | `BaseTaskProcessor` | prepares `predict_action`, noise and robot state |
| IR config | `ir_config` | named ONNX tensor/action contract |
| Backend manager | `BaseBackendManager` | TensorRT build, timing cache, target metadata |
| Backend wrapper | `BaseBackendWrapper` | named tensor inference for robot policy |
| Function rewriter | `FUNCTION_REWRITER` | codebase/backend/IR-scoped export patches |
| Pipeline manager | `PipelineManager` | hooks plus auditable per-stage timing |

## Deliberate differences

- FluxHermes is deployment-only and does not reproduce OpenMMLab dataset,
  visualization, evaluator, or runner abstractions.
- The core registry and config normalization do not require MMEngine, so bundle
  inspection and dry-run planning remain lightweight.
- A bundle records SHA-256 checksums, validation metrics, build target, and
  multiple backend files.
- Robot runtime refuses unvalidated bundles by default and checks TensorRT/GPU
  target compatibility.
- Calibration samples must represent the real observation/action distribution;
  quantization is not treated as a generic image-only preprocessing problem.

## Extension rule

Add model semantics as a codebase task, backend conversion as a backend
manager, runtime execution as a backend wrapper, and export-only changes as
rewriters. Do not add model-family conditions directly to the generic compile
pipeline.
