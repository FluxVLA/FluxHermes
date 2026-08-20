# Codebase task processors and model-family adapters

## Generic FluxVLA adapter

Use `codebase_config.type='fluxvla'` and `task='predict_action'` when
`model.predict_action` can be represented by a flat collection of tensor
inputs. `source.bindings` maps flat backend inputs back to keyword arguments
and may group multiple tensors into a list:

```python
codebase_config = dict(
    type='fluxvla',
    task='predict_action',
    source=dict(
        type='fluxvla',
        fluxvla_root='/root/projects/FluxVLA',
        config='configs/pi05/example.py',
        checkpoint='/path/to/model.safetensors',
        model_section='model',
        sample_inputs='/path/to/sample.npz',
        bindings=dict(
            images=dict(inputs=['image_0', 'image_1'], container='list'),
            lang_tokens='lang_tokens',
            states='states',
        ),
    ),
)
```

Prefer an exportable base model section over a CUDA Graph/Triton inference
variant. TensorRT must see the mathematical graph rather than opaque captured
kernels.

## Custom Python task

Use `codebase_config.type='python'` with a custom recipe when a model family
needs explicit diffusion noise,
cache/state flattening, graph partitioning, custom TensorRT plugins, or multiple
engines. The configured factory receives the normalized deployment config and
target device and returns `PreparedModel`:

```python
from collections import OrderedDict

from fluxhermes.adapters import PreparedModel


def build_prepared_model(deployment, device, **kwargs):
    model = load_source_model(**kwargs).eval().to(device)
    wrapper = ExportableActionWrapper(model).eval()
    samples = load_npz(kwargs['sample_inputs'], device=device)
    return PreparedModel(
        module=wrapper,
        inputs=OrderedDict(
            (spec.name, samples[spec.name]) for spec in deployment.inputs),
        output_names=tuple(spec.name for spec in deployment.outputs),
        dynamic_axes={
            'lang_tokens': {1: 'prompt_length'},
            'lang_masks': {1: 'prompt_length'},
        },
        calibration_source=lambda: calibration_samples(**kwargs),
        metadata={'model_family': 'example'},
    )
```

The wrapper must make every stochastic input explicit so the reference and
TensorRT paths receive exactly the same values. It must return tensors in the
same order as `output_names`.

The recipe is constructed by `CustomTaskProcessor`; TensorRT-specific behavior
must remain in a backend manager or export rewriter rather than the recipe.

## Export rewriters

Register a temporary rewrite when the source function is mathematically valid
but cannot be exported or optimized for a particular backend:

```python
from fluxhermes.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'my_policy.ops.unsupported_function',
    codebase='fluxvla',
    backend='tensorrt',
    ir='onnx',
)
def rewrite_unsupported_function(context, tensor):
    # context.origin_func is available when the rewrite delegates to source.
    return exportable_equivalent(tensor)
```

The original function is restored immediately after the ONNX export context.

## Recommended model-family split

Large VLA policies often compile more reliably as several engines—vision,
language/cache, and iterative action head—rather than one monolithic graph.
The initial `PreparedModel` contract covers a single graph. Add a family recipe
and bundle-level executor before splitting a production model; do not silently
change preprocessing or action semantics to force one graph.
