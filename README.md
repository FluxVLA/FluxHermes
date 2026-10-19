# FluxHermes

FluxHermes is the deployment codebase in the Flux ecosystem. Named after the
ancient Greek messenger god, it compiles
FluxVLA policies to ONNX and TensorRT, applies deployment-time optimization and
post-training quantization, validates source-versus-engine parity, and packages
the result as an integrity-checked bundle for robot execution.

[中文说明](README_zh-CN.md)

> **Current status:** the repository provides the complete framework, generic
> FluxVLA adapter, FP32/FP16 TensorRT build paths, INT8 Q/DQ quantization, a
> TensorRT 10/11 named-tensor runtime, bundle validation, templates, and
> CPU-only contract tests. Each model family still needs an exportable wrapper
> and representative samples; GPU builds must be verified on the intended
> NVIDIA target before robot use.

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

## Quick start

Inspect the supplied PI0.5 plan without loading a checkpoint or GPU library:

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
did not pass:

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
python -m unittest discover -s test -v
pytest
flake8 fluxhermes test scripts
isort --check-only fluxhermes test scripts
```

## License

Apache License 2.0.
