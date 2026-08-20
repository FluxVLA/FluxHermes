# Architecture and ownership

## Layered pipeline

```text
DeploymentConfig
  codebase_config ──> Codebase registry ──> TaskProcessor
                                              |
                                              v
                                        PreparedModel
                                              |
                              FunctionRewriter context
                                              |
  ir_config ─────────────────────────────> named ONNX
                                              |
                                  optimize / quantize
                                              |
  backend_config ─> BackendManager ─────> backend files
                                              |
                                        BackendWrapper
                                              |
                                  source/backend parity
                                              |
                                checksummed robot bundle
```

This follows the useful MMDeploy distinction between a source codebase, a task
processor, an intermediate representation, and a backend. FluxHermes keeps the
implementation small and dependency-light while making those boundaries
explicit.

## Core abstractions

### Codebase

`BaseCodebase` owns a task registry. The built-in `FluxVLACodebase` registers
`predict_action`; the generic `PythonCodebase` registers `custom` for external
model-family recipes. A codebase registers model-specific export rewrites but
does not build TensorRT engines.

### Task processor

`BaseTaskProcessor` converts a codebase config into `PreparedModel`: an
exportable module, ordered representative inputs, named outputs, dynamic axes,
calibration source, and provenance. This is the sole model-semantic boundary.

### IR and rewriter

The initial IR is ONNX. `FunctionRewriter` temporarily replaces selected
functions only within export, filtered by codebase, backend, and IR. Original
functions are restored even if export raises an exception.

### Backend manager and wrapper

`BaseBackendManager` converts one or more IR files into backend files and
constructs a `BaseBackendWrapper`. The TensorRT manager owns engine building;
its wrapper exposes the same named-input/named-output `infer` contract used by
validation and robot runtime. The manifest accepts multiple backend files so a
future VLA recipe can split vision, language/cache, and action-head engines.
Bundle schema 2 stores these as `backend_files` plus backend-specific
`backend_metadata`; schema 1 single-engine manifests remain readable.

### Pipeline manager

Every execution stage is timed and recorded with completed/failed status.
Optional Python hooks can run before or after a named stage. The build report is
written to `work_dirs/.../result.json` and the pre-package trace is embedded in
the bundle manifest.

## Configuration contract

```python
deployment = dict(
    name='pi05-aloha-fp16',
    work_dir='work_dirs/pi05-aloha-fp16',
    codebase_config=dict(
        type='fluxvla',
        task='predict_action',
        device='cuda:0',
        source=dict(...),
    ),
    ir_config=dict(type='onnx', opset=18),
    backend_config=dict(
        type='tensorrt',
        precision='fp16',
        strongly_typed='auto',
    ),
    inputs=[...],
    outputs=[...],
)
```

Legacy `source`, `export`, and `tensorrt` sections remain readable during the
initial rename, but new configs should use the three explicit sections above.

## Tensor and precision contract

Every input and output has a name, dtype, shape, semantic description, and—if
dynamic—a min/opt/max profile. The same contract drives ONNX export, backend
building, wrapper binding, validation, and bundle inspection.

FP16 compilation lowers the ONNX graph to explicit mixed precision before
TensorRT. The builder uses the weakly typed compatibility path on TensorRT 10
and switches to a strongly typed network automatically on TensorRT 11. INT8
uses an explicit Q/DQ graph in both cases.

## Ownership

| Repository | Responsibility |
| --- | --- |
| FluxVLA | model, checkpoint, preprocessing, statistics, action semantics |
| FluxHermes | export, rewrite, optimization, quantization, backend, bundle/runtime |
| FluxThemis | simulation/robot rollout, evaluation lifecycle, metrics |

## Safety boundary

Compilation success is not equivalent to robot readiness. By default,
`RobotPolicy` loads only bundles whose source-versus-backend parity check
passed. Task regression, control deadlines, joint/workspace limits, watchdogs,
collision handling, and emergency stop remain mandatory at integration time.
