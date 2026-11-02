# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end PI0.5 execution over the Vision, Prefix and Denoise graphs."""

from __future__ import annotations

import struct
from typing import Any, Dict, Mapping

import torch

from .contract import PI05Contract, cache_names

_POLICY_INPUT_NAMES = (
    'images',
    'lang_tokens',
    'states',
    'img_masks',
    'lang_masks',
    'noise',
)


def _float32(value: float) -> float:
    return struct.unpack('f', struct.pack('f', value))[0]


def euler_timesteps(steps: int) -> tuple[float, ...]:
    """Match PI0.5's iterative FP32 time update exactly."""
    value = _float32(1.0)
    delta = _float32(-1.0 / steps)
    result = []
    for _ in range(steps):
        result.append(value)
        value = _float32(value + delta)
    return tuple(result)


def prepare_policy_inputs(
    inputs: Mapping[str, Any],
    contract: PI05Contract,
    require_cuda: bool = False,
) -> Dict[str, torch.Tensor]:
    """Validate policy inputs and return tensors for the three partitions."""
    missing = [name for name in _POLICY_INPUT_NAMES if name not in inputs]
    unknown = [name for name in inputs if name not in _POLICY_INPUT_NAMES]
    if missing or unknown:
        raise ValueError(
            f'PI0.5 inputs mismatch: missing={missing}, unknown={unknown}.')

    images = inputs['images']
    image_tensors: tuple[torch.Tensor, ...]
    if isinstance(images, (list, tuple)):
        image_tensors = tuple(images)
        if len(image_tensors) != contract.image_views:
            raise ValueError(f'images must contain {contract.image_views} views.')
        view_shape = (
            contract.batch,
            3,
            contract.image_height,
            contract.image_width,
        )
        for index, image in enumerate(image_tensors):
            if not isinstance(image, torch.Tensor):
                raise TypeError(f'images[{index}] must be a torch.Tensor.')
            if tuple(image.shape) != view_shape:
                raise ValueError(f'images[{index}] shape must be {view_shape}, got '
                                 f'{tuple(image.shape)}.')
        flat_images = torch.cat(image_tensors, dim=1)
    else:
        if not isinstance(images, torch.Tensor):
            raise TypeError('images must be a tensor or a list of tensors.')
        image_tensors = (images, )
        flat_images = images

    flat_shape = (
        contract.batch,
        contract.image_views * 3,
        contract.image_height,
        contract.image_width,
    )
    if tuple(flat_images.shape) != flat_shape:
        raise ValueError(
            f'images shape must be {flat_shape}, got {tuple(flat_images.shape)}.')
    if flat_images.dtype != torch.float32:
        raise TypeError('images must contain normalized FP32 model inputs.')

    tensors = {
        name: inputs[name]
        for name in _POLICY_INPUT_NAMES if name not in {'images'}
    }
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'{name} must be a torch.Tensor.')

    expected_shapes = {
        'lang_tokens': (contract.batch, contract.language_tokens),
        'states': (contract.batch, contract.action_dim),
        'img_masks': (contract.batch, contract.image_views),
        'lang_masks': (contract.batch, contract.language_tokens),
        'noise': (
            contract.batch,
            contract.action_horizon,
            contract.action_dim,
        ),
    }
    for name, shape in expected_shapes.items():
        if tuple(tensors[name].shape) != shape:
            raise ValueError(
                f'{name} shape must be {shape}, got {tuple(tensors[name].shape)}.')

    if tensors['lang_tokens'].dtype not in (torch.int32, torch.int64):
        raise TypeError('lang_tokens must be INT32 or INT64.')
    if tensors['img_masks'].dtype != torch.bool:
        raise TypeError('img_masks must be BOOL.')
    if not bool(torch.all(tensors['img_masks']).item()):
        raise ValueError('PI0.5 TensorRT requires every image view to be valid.')
    if tensors['lang_masks'].dtype != torch.bool:
        raise TypeError('lang_masks must be BOOL.')
    if tensors['noise'].dtype != torch.float32:
        raise TypeError('noise must be FP32.')

    device = flat_images.device
    all_tensors = (flat_images, *tensors.values())
    if any(value.device != device for value in all_tensors):
        raise ValueError('PI0.5 inputs must be on the same device.')
    if require_cuda and device.type != 'cuda':
        raise ValueError('PI0.5 TensorRT inputs must be on CUDA.')

    graph_images = flat_images.reshape(
        contract.image_views,
        3,
        contract.image_height,
        contract.image_width,
    ).to(torch.bfloat16).contiguous()
    return {
        'images': graph_images,
        'lang_tokens': tensors['lang_tokens'].to(torch.int32).contiguous(),
        'lang_mask': tensors['lang_masks'].contiguous(),
        'noise': tensors['noise'].contiguous(),
    }


class PI05SourceReferenceExecutor:
    """Execute the original FluxVLA PI0.5 policy for parity validation."""

    def __init__(
        self,
        model: Any,
        contract: PI05Contract,
    ) -> None:
        self.model = model
        self.contract = contract

    def infer(self, inputs: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        prepare_policy_inputs(inputs, self.contract)
        with torch.inference_mode():
            actions = self.model.predict_action(
                images=inputs['images'],
                lang_tokens=inputs['lang_tokens'],
                states=inputs['states'],
                img_masks=inputs['img_masks'],
                lang_masks=inputs['lang_masks'],
                noise=inputs['noise'].clone(),
            )
        return {'actions': actions.float()}

    def __call__(self, inputs: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        return self.infer(inputs)


class PI05TensorRTExecutor:
    """Execute Vision once, Prefix once and Denoise exactly ten times."""

    def __init__(
        self,
        engines: Mapping[str, Any],
        contract: PI05Contract,
    ) -> None:
        expected = ('vision', 'prefix', 'denoise')
        missing = [name for name in expected if name not in engines]
        unknown = [name for name in engines if name not in expected]
        if missing or unknown:
            raise ValueError(
                f'PI0.5 engines mismatch: missing={missing}, unknown={unknown}.')
        self.engines = {name: engines[name] for name in expected}
        self.contract = contract
        self.timesteps = euler_timesteps(contract.denoise_steps)
        self.cache_names = cache_names(contract)

    def infer(
        self,
        inputs: Mapping[str, Any],
        synchronize: bool = False,
    ) -> Dict[str, torch.Tensor]:
        prepared = prepare_policy_inputs(inputs, self.contract, require_cuda=True)
        device = prepared['noise'].device
        with torch.cuda.device(device):
            vision = self.engines['vision'].infer({
                'images': prepared['images'],
            })
            caches = self.engines['prefix'].infer({
                'image_prefix': vision['image_prefix'],
                'lang_tokens': prepared['lang_tokens'],
                'lang_mask': prepared['lang_mask'],
            })
            action = prepared['noise']
            for value in self.timesteps:
                timestep = torch.full(
                    (self.contract.batch, ),
                    value,
                    dtype=torch.float32,
                    device=device,
                )
                denoise_inputs = {
                    'x_t': action,
                    'timestep': timestep,
                    'lang_mask': prepared['lang_mask'],
                }
                denoise_inputs.update({name: caches[name] for name in self.cache_names})
                action = self.engines['denoise'].infer(denoise_inputs)['x_next']
            if synchronize:
                torch.cuda.current_stream(device).synchronize()
        return {'actions': action}

    def __call__(self, inputs: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        return self.infer(inputs)
