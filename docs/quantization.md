# Quantization contract

## INT8 Q/DQ

`quantization.mode='int8'` inserts explicit quantize/dequantize nodes into the
ONNX graph with ONNX Runtime. TensorRT then consumes the Q/DQ graph. The
calibration source must yield the same named, preprocessed inputs used by the
engine.

Large VLA graphs use ONNX external tensor data by default so quantized models
are not limited to a single small protobuf file.

The generic FluxVLA adapter accepts either:

- `source.calibration_inputs`: a directory of `.npz` files; or
- `source.calibration_factory`: a `module:function` generator.

Each sample must contain every declared input. Calibration data should cover
camera exposure, scenes, language lengths, robot states, task families, and
embodiments expected in deployment. Training batches are not automatically a
valid replacement for deployment observations.

## Exclusions and sensitive layers

Use `nodes_to_exclude` for unsupported operators or numerically sensitive
normalization, softmax, sampling, or output layers. Begin with conservative
exclusions and use parity plus task rollouts to decide whether to expand INT8
coverage.

## Acceptance

Quantized artifacts must pass all of the following before release:

1. finite outputs for every validation sample;
2. configured absolute/relative/cosine parity thresholds;
3. latency and memory targets on the deployment device;
4. task-level simulation regression;
5. guarded real-robot regression at the intended control rate.

Weight-only INT4/FP4 and model-specific NVIDIA Model Optimizer recipes are not
yet implemented in the initial framework; add them as separate quantizer
plugins rather than overloading the INT8 path.
