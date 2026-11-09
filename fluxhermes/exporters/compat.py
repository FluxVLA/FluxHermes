"""Opt-in Torch 2.6 legacy symbolic corrections; no model implementation.

The scoped registry inspection is tied to Torch 2.6's registration API. Existing
custom handlers are rejected, never overwritten. The lock serializes Hermes
exports; callers must not mutate Torch's global registry concurrently outside
Hermes. Numerical BF16 runtime validation remains a separate backend concern.
"""
from __future__ import annotations

from contextlib import contextmanager
from threading import RLock

from packaging.version import Version

from ..errors import ConfigurationError

EXPORT_LOCK = RLock()


@contextmanager
def normalize_full_slices(torch):
    """Remove identity Tensor views before Torch 2.6's indexed-write lowering."""
    if Version(torch.__version__).release[:2] != (2, 6):
        raise ConfigurationError('Full-slice normalization requires Torch 2.6.')
    from torch.onnx import utils
    original = utils._optimize_graph
    if (original.__module__ != 'torch.onnx.utils'
            or original.__name__ != '_optimize_graph'):
        raise ConfigurationError('ONNX graph optimizer is already overridden.')
    stats = {'identity_slices_removed': 0}

    def visit(block):
        for node in block.nodes():
            yield node
            for child in node.blocks():
                yield from visit(child)

    def optimize(graph, *args, **kwargs):
        torch._C._jit_pass_inline(graph)
        for node in list(visit(graph)):
            if node.kind() != 'aten::slice':
                continue
            values = list(node.inputs())
            if (len(values) == 5 and values[2].toIValue() == 0
                    and values[3].toIValue() == 9223372036854775807
                    and values[4].toIValue() == 1):
                node.output().replaceAllUsesWith(values[0])
                node.destroy()
                stats['identity_slices_removed'] += 1
        return original(graph, *args, **kwargs)

    with EXPORT_LOCK:
        utils._optimize_graph = optimize
        try:
            yield stats
        finally:
            if utils._optimize_graph is not optimize:
                raise ConfigurationError('ONNX graph optimizer changed during export; retained external override.')
            utils._optimize_graph = original


@contextmanager
def legacy_bf16_symbolics(torch, opset: int):
    """Use public registration APIs and delegate all operations to Torch."""
    if Version(torch.__version__).release[:2] != (2, 6) or opset != 18:
        raise ConfigurationError(
            'legacy_bf16_compat requires the validated Torch 2.6 / opset 18 path.')
    from torch.onnx import _type_utils, symbolic_helper
    from torch.onnx import symbolic_opset11, symbolic_opset13, symbolic_opset14
    from torch.onnx._internal import registration

    original_sdpa = symbolic_opset14.scaled_dot_product_attention
    original_cumsum = symbolic_opset11.cumsum

    def sdpa(g, query, key, value, attn_mask=None, dropout_p=0.0,
             is_causal=False, scale=None, enable_gqa=False):
        if scale is not None and not symbolic_helper._is_none(scale):
            scale = g.op(
                'Cast', scale,
                to_i=_type_utils.JitScalarType.from_value(query).onnx_type())
        if symbolic_helper._maybe_get_const(enable_gqa, 'b'):
            query_heads = symbolic_helper._get_tensor_dim_size(query, -3)
            key_heads = symbolic_helper._get_tensor_dim_size(key, -3)
            value_heads = symbolic_helper._get_tensor_dim_size(value, -3)
            if (query_heads is None or key_heads is None or value_heads is None
                    or key_heads <= 0 or value_heads != key_heads
                    or query_heads % key_heads):
                raise ConfigurationError(
                    'GQA export requires static positive head counts with '
                    'query_heads divisible by equal key/value head counts.')
            repeats = g.op('Constant', value_t=torch.tensor(
                query_heads // key_heads, dtype=torch.int64))
            dim = g.op('Constant', value_t=torch.tensor(-3, dtype=torch.int64))
            key = symbolic_opset13.repeat_interleave(g, key, repeats, dim)
            value = symbolic_opset13.repeat_interleave(g, value, repeats, dim)
            enable_gqa = False
        return original_sdpa(g, query, key, value, attn_mask, dropout_p,
                             is_causal, scale, enable_gqa)

    def cumsum(g, value, dim, dtype=None):
        if _type_utils.JitScalarType.from_value(value) == _type_utils.JitScalarType.BOOL:
            value = g.op('Cast', value,
                         to_i=_type_utils.JitScalarType.INT64.onnx_type())
        return original_cumsum(g, value, dim, dtype)

    handlers = (
        ('aten::scaled_dot_product_attention', original_sdpa, sdpa),
        ('aten::cumsum', original_cumsum, cumsum),
    )
    with EXPORT_LOCK:
        for name, original, _ in handlers:
            group = registration.registry.get_function_group(name)
            if (group is None or group.get(opset) is not original
                    or group._functions.overridden(opset)):
                raise ConfigurationError(
                    f'legacy_bf16_compat registry conflict for {name} at opset {opset}; '
                    'existing registration is retained.')
        registered = []
        try:
            for name, _, handler in handlers:
                torch.onnx.register_custom_op_symbolic(name, handler, opset)
                registered.append((name, handler))
            yield
        finally:
            conflicts = []
            for name, handler in reversed(registered):
                group = registration.registry.get_function_group(name)
                if group.get(opset) is handler and group._functions.overridden(opset):
                    torch.onnx.unregister_custom_op_symbolic(name, opset)
                else:
                    conflicts.append(name)
            if conflicts:
                raise ConfigurationError(
                    'Torch registrations changed during export; external handlers '
                    f'were retained: {conflicts}')
