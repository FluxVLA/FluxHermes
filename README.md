# FluxHermes

FluxHermes is the deployment codebase for FluxVLA policies. It compiles models
to ONNX and TensorRT, supports deployment optimization and post-training
quantization, validates source-versus-engine parity, and packages the required
artifacts for robot execution.

[中文说明](README_zh-CN.md)

> **Current status:** PI0.5 has an independent SDK and a two-graph
> compile/package/runtime path. Prior RTX 3090 / TensorRT 10.13.3.9 tests passed
> tiny-policy parity and native PyTorch-versus-SDK bitwise checks on real business
> weights. Full business-model TensorRT numerical validation **failed**
> (maximum absolute error 0.05605). No business deployment or speedup is claimed.

## Scope

- Export a FluxVLA `predict_action` graph with stable input/output names.
- Simplify ONNX graphs and reuse TensorRT timing caches.
- Build static or dynamic-shape TensorRT engines.
- Quantize with representative robot data through ONNX INT8 Q/DQ.
- Compare source and TensorRT outputs before declaring an artifact robot-ready.
- Package engines, preprocessing assets, interface metadata, and checksums.
- Run bundles through a PyTorch-buffered TensorRT runtime on x86 NVIDIA GPUs or
  Jetson devices.

FluxVLA remains the owner of model definitions, checkpoints, preprocessing,
normalization, and action semantics. FluxHermes owns graph compilation,
optimization, quantization, engine validation, and the deployment runtime.
FluxThemis remains the owner of rollout and evaluation orchestration.

## MMDeploy-inspired architecture

FluxHermes adopts MMDeploy's most useful separation of concerns without taking
a runtime dependency on MMDeploy:

- `codebase + task processor` owns FluxVLA model semantics and export inputs;
- `IR` owns the ONNX contract and export-time rewrites;
- `backend manager + wrapper` owns TensorRT conversion and named-tensor runtime;
- `function rewriter` applies temporary backend-specific graph changes;
- `pipeline manager` records stage timing, failures, and optional hooks.

FluxHermes adds robot-specific requirements on top: representative calibration
data, source/backend action parity, target-GPU checks, checksummed bundles, and
a default refusal to run unvalidated artifacts. See
[the MMDeploy alignment notes](docs/mmdeploy_alignment.md).

## Installation

The metadata and dry-run planner work in a lightweight Python 3.10-3.12
environment:

```bash
cd /root/projects/FluxHermes
python -m pip install -e .
fluxhermes doctor
```

For export and INT8 calibration:

```bash
python -m pip install -e '.[export,quantization,dev]'
```

Install TensorRT 10 or 11 from the NVIDIA distribution matching the target CUDA or
JetPack release. FluxHermes intentionally does not pull an arbitrary TensorRT
wheel that may be incompatible with the target driver.

## PI0.5 independent export

Install the independent SDK as a wheel from the immutable source revision
used by CI. Pip builds only the package subdirectory, not the training project:

```bash
SDK_REVISION=55a464ae753f16dd19f444f63e321cea5535a6dc
python -m pip install \
  "fluxvla-export[test-onnx] @ git+https://github.com/ZaneZhang-67/FluxVLA.git@${SDK_REVISION}#subdirectory=packages/fluxvla-export" \
  '.[export,onnxruntime,dev]'
```

The SDK's `test-onnx` extra pins the verified export toolchain. For SDK development
only, installing `../FluxVLA/packages/fluxvla-export[test-onnx]` from a sibling
checkout is optional; source-directory imports are not a runtime requirement.

Training supplies deployment JSON and six preprocessed tensors through the SDK's
`produce_pi05_artifacts` API, plus the original safetensors checkpoint.
Checkpoint storage defaults to FP32; explicitly declared BF16 storage is also
supported. The SDK validates the stored tensors and loads them into FP32 model
parameters, matching ordinary PyTorch loading of the same checkpoint. BF16
storage and BF16 autocast computation are separate choices. The checkpoint file
is not rewritten. Architecture, dimensions, preprocessing identifiers and
protected precision come from these artifacts; Hermes does not reconstruct a
training configuration or implement checkpoint loading.

```bash
export PI05_DEPLOYMENT_JSON=/path/to/export-case/export_config.json
export PI05_CHECKPOINT=/path/to/model.safetensors
export PI05_EXPORT_INPUTS=/path/to/export-case/export_inputs.safetensors

fluxhermes compile configs/pi05/pi05_export_fp32.py          # plan only
fluxhermes compile configs/pi05/pi05_export_fp32.py --execute
fluxhermes inspect artifacts/pi05-export-fp32
fluxhermes run artifacts/pi05-export-fp32 --inputs "$PI05_EXPORT_INPUTS" \
  --device cuda:0 --output /tmp/actions.npz
fluxhermes benchmark artifacts/pi05-export-fp32 --inputs "$PI05_EXPORT_INPUTS" \
  --device cuda:0 --warmup 10 --repeats 100 \
  --reference-config configs/pi05/pi05_export_fp32.py
```

The recipe chooses FP32 legacy/opset18 or BF16 Dynamo/opset22 with `optimize=True`.
The standard BF16 optimizer materializes constant reduction axes required by
TensorRT; its export flags cannot be silently overridden. Backend `precision=fp32` disables global
retyping: it preserves explicit BF16 casts and protected FP32 operations.
TensorRT uses strong typing and TF32 is off. FP16 conversion, INT8 and RTC are
not supported by this recipe. Both graphs and complete Euler actions must pass
validation before the bundle is accepted by the robot runtime.

The optional `onnxruntime` backend is CPU-only and rejects BF16; it is not a
TensorRT substitute. CLI fixtures may be NPZ or safetensors. Benchmark timing
covers synchronized policy inference on preloaded tensors, excluding disk,
engine loading and host-to-device transfer. `--reference-config` additionally
checks the same checkpoint/config/fixture, measures PyTorch, frees that model
before loading engines, requires output parity, and reports the measured ratio.
A ratio below one means the backend was slower. Use new work/bundle paths per
build; existing bundles or staging directories are never automatically deleted.

### Validation boundary

The NVIDIA results above were obtained on 2026-11-05, not rerun by this publication.
With real business BF16-stored weights, native PyTorch and the SDK matched
bitwise. Both ONNX graph exports and TensorRT builds passed, but full TensorRT
numerical validation failed at unchanged `atol=0.02`, `rtol=0.02`, cosine `0.999`
thresholds. A buildable engine is not a validated policy.

The 3.62B-parameter run used a phased harness. Memory viability of the ordinary
full `CompilePipeline` remains unproven. Do not relax tolerances or bypass
validation to promote these failed artifacts.

## Legacy training-environment adapters

Inspect the supplied legacy PI0.5 plan without loading a checkpoint or GPU library:

```bash
export FLUXVLA_ROOT=/root/projects/FluxVLA
fluxhermes compile configs/pi05/pi05_aloha_fp16.py
```

Prepare a representative NPZ with the exact input keys declared in the config,
set the checkpoint and sample paths, then execute the build in the matching
FluxVLA/TensorRT environment:

```bash
export PI05_CHECKPOINT=/path/to/model.safetensors
export PI05_DEPLOY_SAMPLE=/path/to/pi05_sample.npz

fluxhermes compile configs/pi05/pi05_aloha_fp16.py --execute
fluxhermes inspect artifacts/pi05-aloha-fp16
```

INT8 additionally requires a directory of representative NPZ samples:

```bash
export PI05_CALIBRATION_NPZ_DIR=/path/to/calibration_npz
fluxhermes compile configs/pi05/pi05_aloha_int8.py --execute
```

## Robot runtime

The default runtime refuses a bundle whose source-versus-TensorRT parity check
did not pass. For an independent-SDK bundle, install the SDK executor package
without model/export extras and the selected backend. TensorRT still comes from
the target-compatible NVIDIA distribution:

```bash
SDK_REVISION=55a464ae753f16dd19f444f63e321cea5535a6dc
python -m pip install \
  "fluxvla-export @ git+https://github.com/ZaneZhang-67/FluxVLA.git@${SDK_REVISION}#subdirectory=packages/fluxvla-export" \
  '.[runtime]' 'safetensors>=0.4,<1'
```

CLI `run`/`benchmark` need Torch; safetensors fixtures additionally need the
`safetensors` package shown above. NPZ uses the base NumPy dependency. A restored
bundle needs no training checkout, model construction, source checkpoint or
export-time deployment JSON. Installing packages and importing model modules
are distinct: the fresh-process integration tests verify model-free imports.

```python
from fluxhermes.runtime import RobotPolicy

policy = RobotPolicy.from_bundle(
    '/root/projects/FluxHermes/artifacts/pi05-aloha-fp16')

# Inputs are already tokenized/normalized according to the bundle contract.
actions = policy.predict_action(
    images=images,
    lang_tokens=lang_tokens,
    states=states,
    lang_masks=lang_masks,
    noise=noise,
)
```

For raw camera/state observations, package a host adapter and all tokenizer,
normalization, and embodiment assets under `package.assets`; set
`package.runtime.host_adapter` to `module:ClassName`. See
[the robot runtime guide](docs/robot_runtime.md).

One-sample and latency smoke tests are available from the CLI:

```bash
fluxhermes run artifacts/pi05-aloha-fp16 \
  --inputs /path/to/pi05_sample.npz --output /tmp/actions.npz

fluxhermes benchmark artifacts/pi05-aloha-fp16 \
  --inputs /path/to/pi05_sample.npz --warmup 20 --repeats 200
```

## Repository map

```text
configs/                 deployment templates and TensorRT profiles
docs/                    architecture, adapters, quantization, and robot guide
fluxhermes/apis/         public task/backend construction APIs
fluxhermes/codebase/     codebase and task-processor integrations
fluxhermes/backend/      backend managers and runtime wrappers
fluxhermes/core/         registry, pipeline manager, and function rewriters
fluxhermes/adapters/     low-level FluxVLA/custom source adapters
fluxhermes/exporters/    named ONNX export
fluxhermes/optimization/ ONNX graph optimization
fluxhermes/quantization/ representative-data INT8 Q/DQ
fluxhermes/builders/     TensorRT engine builder and timing cache
fluxhermes/runtime/      engine execution and robot policy API
fluxhermes/validation/   source-versus-engine parity gate
test/                    dependency-light contract and bundle tests
artifacts/               ignored robot bundles
work_dirs/               ignored intermediate graphs and build caches
```

## Documentation

- [Architecture and ownership](docs/architecture.md)
- [MMDeploy concept alignment](docs/mmdeploy_alignment.md)
- [Adding a model-family adapter](docs/model_adapter.md)
- [Quantization contract](docs/quantization.md)
- [Real-robot runtime and release checklist](docs/robot_runtime.md)

## Development

```bash
python -m unittest discover -s test -v             # dependency-light
python -m unittest discover -s test_integration -v # verified export profile above
# Explicit GPU check: absence of CUDA/TensorRT is an error, never a skip.
FLUXHERMES_CUDA_DEVICE=cuda:0 python -m unittest discover -s test_gpu -v
flake8 fluxhermes test test_integration test_gpu scripts
isort --check-only fluxhermes test test_integration test_gpu scripts
```

CI keeps dependency-light tests separate from backend and pinned-SDK CPU
integration tests. The two explicit GPU test modules exercise Add-graph runtime,
stream/profile safety and logger/JSON behavior; they do not reproduce the
separate tiny full-policy or business-model GPU evidence.

## License

Apache License 2.0.
