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
        raise ValueError('img_masks must describe exactly two image views.')

    if tensors['lang_tokens'].dtype not in (torch.int32, torch.int64):
        raise TypeError('lang_tokens must be INT32 or INT64.')
    if tensors['states'].dtype != torch.float32:
        raise TypeError('states must be FP32.')
    if tensors['img_masks'].dtype != torch.bool:
        raise TypeError('img_masks must be BOOL.')
    if not bool(torch.all(tensors['img_masks']).item()):
        raise ValueError('GR00T-N1.5 requires both image views to be valid.')
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


def prepare_language_inputs(
    eagle: Any,
    vit_embeds: torch.Tensor,
    lang_tokens: torch.Tensor,
    contract: GR00TN15Contract,
) -> torch.Tensor:
    if eagle.vlm.use_pixel_shuffle:
        side = int(vit_embeds.shape[1]**0.5)
        if side * side != vit_embeds.shape[1]:
            raise ValueError('Eagle pixel shuffle requires a square token grid.')
        vit_embeds = vit_embeds.reshape(
            contract.image_views,
            side,
            side,
            contract.vision_hidden,
        )
        vit_embeds = eagle.vlm.pixel_shuffle(
            vit_embeds,
            scale_factor=eagle.vlm.downsample_ratio,
        )
        vit_embeds = vit_embeds.reshape(
            contract.image_views,
            -1,
            vit_embeds.shape[-1],
        )
    vit_embeds = eagle.vlm.mlp1(vit_embeds).to(torch.float16).contiguous()
    expected = (
        contract.image_views,
        contract.projected_image_tokens_per_view,
        contract.backbone_dim,
    )
    if tuple(vit_embeds.shape) != expected:
        raise ValueError(f'Eagle projected image tokens must be {expected}, got '
                         f'{tuple(vit_embeds.shape)}.')
    image_features = vit_embeds.reshape(
        contract.batch,
        contract.image_tokens,
        contract.backbone_dim,
    )

    embedding = eagle.vlm.language_model.get_input_embeddings()
    input_embeds = embedding(lang_tokens).to(torch.float16)
    flat_embeds = input_embeds.reshape(-1, contract.backbone_dim).contiguous()
    selected = lang_tokens.reshape(-1) == int(eagle.vlm.image_token_index)
    flat_embeds[selected] = image_features.reshape(-1, contract.backbone_dim)
    return flat_embeds.reshape(
        contract.batch,
        contract.sequence_length,
        contract.backbone_dim,
    ).contiguous()


class GR00TReferenceExecutor:

    def __init__(self, model: Any, contract: GR00TN15Contract) -> None:
        self.model = model
        self.contract = contract

    def infer(self, inputs: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        prepared = prepare_policy_inputs(inputs, self.contract)
        with torch.inference_mode():
            actions = self.model.predict_action(
                images=prepared['images'],
                lang_tokens=prepared['lang_tokens'],
                states=prepared['state'].squeeze(1),
                img_masks=prepared['img_masks'],
                lang_masks=prepared['attention_mask'],
                embodiment_ids=prepared['embodiment_id'],
                noise=prepared['noise'],
            )
        return {'actions': actions}

    def __call__(self, inputs: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        return self.infer(inputs)


class GR00TTensorRTExecutor:

    def __init__(
        self,
        engines: Mapping[str, Any],
        eagle: Any,
        action_head: Any,
        contract: GR00TN15Contract,
    ) -> None:
        missing = [name for name in GRAPH_NAMES if name not in engines]
        unknown = [name for name in engines if name not in GRAPH_NAMES]
        if missing or unknown:
            raise ValueError(f'GR00T-N1.5 engines mismatch: missing={missing}, '
                             f'unknown={unknown}.')
        if not action_head.add_positional_embeddings:
            raise ValueError('GR00T-N1.5 TensorRT requires action position embeddings.')
        self.engines = {name: engines[name] for name in GRAPH_NAMES}
        self.eagle = eagle
        self.action_head = action_head
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
            input_embeds = prepare_language_inputs(
                self.eagle,
                vit_embeds,
                prepared['lang_tokens'],
                contract,
            )
            features = self.engines['llm'].infer({
                'inputs_embeds':
                input_embeds,
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
            action_positions = torch.arange(
                contract.action_horizon,
                dtype=torch.int64,
                device=device,
            )
            action_positions = self.action_head.position_embedding(
                action_positions).unsqueeze(0).to(torch.float16)
            future = self.action_head.future_tokens.weight.unsqueeze(0).to(
                torch.float16)
            timestep = torch.empty(
                (contract.batch, ),
                dtype=torch.int64,
                device=device,
            )
            dt = 1.0 / contract.denoise_steps
            for step in range(contract.denoise_steps):
                discretized = int((step / float(contract.denoise_steps)) *
                                  self.action_head.num_timestep_buckets)
                timestep.fill_(discretized)
                action_features = self.engines['action_encoder'].infer({
                    'actions':
                    actions,
                    'timesteps_tensor':
                    timestep,
                    'embodiment_id':
                    prepared['embodiment_id'],
                })['output']
                action_features = action_features + action_positions
                sa_embs = torch.cat((state_features, future, action_features), dim=1)
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

            original_dim = self.action_head.ori_action_dim
            if original_dim is not None:
                actions = actions[:, :, :original_dim]
            actions = actions.float()
            if synchronize:
                torch.cuda.current_stream(device).synchronize()
        return {'actions': actions}

    def __call__(self, inputs: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        return self.infer(inputs)
