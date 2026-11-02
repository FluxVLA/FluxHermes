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
"""End-to-end GR00T-N1.5 execution over seven TensorRT graphs."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch

from .contract import GR00TN15Contract

GRAPH_NAMES = (
    'vit',
    'llm',
    'backbone_output',
    'state_encoder',
    'action_encoder',
    'dit',
    'action_decoder',
)

_POLICY_INPUT_NAMES = (
    'images',
    'lang_tokens',
    'states',
    'img_masks',
    'lang_masks',
    'embodiment_ids',
    'noise',
)


def prepare_policy_inputs(
    inputs: Mapping[str, Any],
    contract: GR00TN15Contract,
    require_cuda: bool = False,
) -> Dict[str, torch.Tensor]:
    missing = [name for name in _POLICY_INPUT_NAMES if name not in inputs]
    unknown = [name for name in inputs if name not in _POLICY_INPUT_NAMES]
    if missing or unknown:
        raise ValueError(
            f'GR00T-N1.5 inputs mismatch: missing={missing}, unknown={unknown}.')

    images = inputs['images']
    if isinstance(images, (list, tuple)):
        views = tuple(images)
        if len(views) != contract.image_views:
            raise ValueError(f'images must contain {contract.image_views} views.')
        view_shape = (
            contract.batch,
            3,
            contract.image_size,
            contract.image_size,
        )
        for index, image in enumerate(views):
            if not isinstance(image, torch.Tensor):
                raise TypeError(f'images[{index}] must be a torch.Tensor.')
            if tuple(image.shape) != view_shape:
                raise ValueError(f'images[{index}] shape must be {view_shape}, got '
                                 f'{tuple(image.shape)}.')
        flat_images = torch.cat(views, dim=1)
    else:
        if not isinstance(images, torch.Tensor):
            raise TypeError('images must be a tensor or a list of tensors.')
        flat_images = images

    flat_shape = (
        contract.batch,
        contract.image_views * 3,
        contract.image_size,
        contract.image_size,
    )
    if tuple(flat_images.shape) != flat_shape:
        raise ValueError(
            f'images shape must be {flat_shape}, got {tuple(flat_images.shape)}.')
    if flat_images.dtype != torch.float32:
        raise TypeError('images must contain normalized FP32 model inputs.')

    tensors = {name: inputs[name] for name in _POLICY_INPUT_NAMES if name != 'images'}
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'{name} must be a torch.Tensor.')

    expected_shapes = {
        'lang_tokens': (contract.batch, contract.sequence_length),
        'states': (contract.batch, contract.state_dim),
        'lang_masks': (contract.batch, contract.sequence_length),
        'embodiment_ids': (contract.batch, ),
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
    if tuple(tensors['img_masks'].shape) not in {
        (contract.image_views, ),
        (contract.batch, contract.image_views),
    }:
        raise ValueError(
            f'img_masks must describe exactly {contract.image_views} image views.')

    if tensors['lang_tokens'].dtype not in (torch.int32, torch.int64):
        raise TypeError('lang_tokens must be INT32 or INT64.')
    if tensors['states'].dtype != torch.float32:
        raise TypeError('states must be FP32.')
    if tensors['img_masks'].dtype != torch.bool:
        raise TypeError('img_masks must be BOOL.')
    if not bool(torch.all(tensors['img_masks']).item()):
        raise ValueError('GR00T-N1.5 requires every image view to be valid.')
    if tensors['lang_masks'].dtype not in (
            torch.bool,
            torch.int32,
            torch.int64,
    ):
        raise TypeError('lang_masks must be BOOL, INT32, or INT64.')
    if tensors['embodiment_ids'].dtype not in (torch.int32, torch.int64):
        raise TypeError('embodiment_ids must be INT32 or INT64.')
    if tensors['noise'].dtype != torch.float32:
        raise TypeError('noise must be FP32.')

    device = flat_images.device
    all_tensors = (flat_images, *tensors.values())
    if any(value.device != device for value in all_tensors):
        raise ValueError('GR00T-N1.5 inputs must be on the same device.')
    if require_cuda and device.type != 'cuda':
        raise ValueError('GR00T-N1.5 TensorRT inputs must be on CUDA.')

    pixel_values = flat_images.reshape(
        contract.image_views,
        3,
        contract.image_size,
        contract.image_size,
    ).to(torch.float16).contiguous()
    return {
        'images': flat_images,
        'pixel_values': pixel_values,
        'lang_tokens': tensors['lang_tokens'].to(torch.int64).contiguous(),
        'states': tensors['states'].contiguous(),
        'state': tensors['states'].to(torch.float16).unsqueeze(1).contiguous(),
        'img_masks': tensors['img_masks'].contiguous(),
        'attention_mask': tensors['lang_masks'].to(torch.int64).contiguous(),
        'embodiment_id': tensors['embodiment_ids'].to(torch.int64).contiguous(),
        'noise': tensors['noise'].contiguous(),
    }


def position_ids(
    contract: GR00TN15Contract,
    device: torch.device,
) -> torch.Tensor:
    return torch.arange(
        contract.vision_patches,
        dtype=torch.int64,
        device=device,
    )[None, :].expand(contract.image_views, -1).contiguous()


class GR00TReferenceExecutor:

    def __init__(self, model: Any, contract: GR00TN15Contract) -> None:
        self.model = model
        self.contract = contract

    def infer(self, inputs: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        prepared = prepare_policy_inputs(inputs, self.contract)
        with torch.inference_mode():
            features, attention_mask, _ = self.model.vlm_backbone(
                images=prepared['images'],
                lang_tokens=prepared['lang_tokens'],
                img_masks=prepared['img_masks'],
                lang_masks=prepared['attention_mask'],
            )
            head = self.model.vla_head
            features = head.vl_self_attention(head.vlln(features))
            state_features = head.state_encoder(
                prepared['state'], prepared['embodiment_id'])
            actions = prepared['noise'].to(features.dtype).clone()
            dt = 1.0 / self.contract.denoise_steps
            for step in range(self.contract.denoise_steps):
                discretized = int(
                    (step / float(self.contract.denoise_steps)) *
                    head.num_timestep_buckets)
                timestep = torch.full(
                    (self.contract.batch, ),
                    discretized,
                    dtype=torch.int64,
                    device=actions.device,
                )
                velocity = head.denoise_step(
                    actions,
                    features,
                    state_features,
                    attention_mask,
                    prepared['embodiment_id'],
                    timestep,
                )
                actions = actions + dt * velocity
            if head.ori_action_dim is not None:
                actions = actions[:, :, :head.ori_action_dim]
        return {'actions': actions.float()}

    def __call__(self, inputs: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        return self.infer(inputs)


class GR00TTensorRTExecutor:

    def __init__(
        self,
        engines: Mapping[str, Any],
        contract: GR00TN15Contract,
    ) -> None:
        missing = [name for name in GRAPH_NAMES if name not in engines]
        unknown = [name for name in engines if name not in GRAPH_NAMES]
        if missing or unknown:
            raise ValueError(f'GR00T-N1.5 engines mismatch: missing={missing}, '
                             f'unknown={unknown}.')
        self.engines = {name: engines[name] for name in GRAPH_NAMES}
        self.contract = contract

    def infer(
        self,
        inputs: Mapping[str, Any],
        synchronize: bool = False,
    ) -> Dict[str, torch.Tensor]:
        prepared = prepare_policy_inputs(inputs, self.contract, require_cuda=True)
        device = prepared['noise'].device
        contract = self.contract
        with torch.cuda.device(device), torch.inference_mode():
            positions = position_ids(contract, device)
            vit_embeds = self.engines['vit'].infer({
                'pixel_values':
                prepared['pixel_values'],
                'position_ids':
                positions,
            })['vit_embeds']
            features = self.engines['llm'].infer({
                'vit_embeds':
                vit_embeds,
                'lang_tokens':
                prepared['lang_tokens'],
                'attention_mask':
                prepared['attention_mask'],
            })['embeddings']
            features = self.engines['backbone_output'].infer({
                'backbone_features':
                features,
            })['output']
            state_features = self.engines['state_encoder'].infer({
                'state':
                prepared['state'],
                'embodiment_id':
                prepared['embodiment_id'],
            })['output']

            actions = prepared['noise'].to(torch.float16).contiguous()
            timestep = torch.empty(
                (contract.batch, ),
                dtype=torch.int64,
                device=device,
            )
            dt = 1.0 / contract.denoise_steps
            for step in range(contract.denoise_steps):
                discretized = int((step / float(contract.denoise_steps)) *
                                  contract.timestep_buckets)
                timestep.fill_(discretized)
                sa_embs = self.engines['action_encoder'].infer({
                    'actions':
                    actions,
                    'timesteps_tensor':
                    timestep,
                    'embodiment_id':
                    prepared['embodiment_id'],
                    'state_features':
                    state_features,
                })['sa_embs']
                model_output = self.engines['dit'].infer({
                    'sa_embs': sa_embs,
                    'vl_embs': features,
                    'timesteps_tensor': timestep,
                })['output']
                prediction = self.engines['action_decoder'].infer({
                    'model_output':
                    model_output,
                    'embodiment_id':
                    prepared['embodiment_id'],
                })['output']
                velocity = prediction[:, -contract.action_horizon:]
                actions = actions + dt * velocity

            actions = actions[:, :, :contract.output_action_dim]
            actions = actions.float()
            if synchronize:
                torch.cuda.current_stream(device).synchronize()
        return {'actions': actions}

    def __call__(self, inputs: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        return self.infer(inputs)
