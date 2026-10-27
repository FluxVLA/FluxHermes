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
"""GR00T-N1.5 graph preparation and TensorRT orchestration."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from ...adapters.base import PreparedDeployment, PreparedGraph
from ...contracts import TensorSpec
from ..base import ModelFamily
from ..registry import MODEL_FAMILIES
from .contract import GR00TN15Contract
from .executor import (GR00TReferenceExecutor, GR00TTensorRTExecutor, position_ids,
                       prepare_language_inputs, prepare_policy_inputs)
from .export import GR00TBackboneOutputExport, GR00TLanguageExport, GR00TVisionExport


def _set_eager_attention(model: Any) -> None:
    for module in model.modules():
        config = getattr(module, 'config', None)
        if config is not None and hasattr(config, '_attn_implementation'):
            config._attn_implementation = 'eager'


@MODEL_FAMILIES.register_module('groot_n15')
class GR00TN15ModelFamily(ModelFamily):
    """Seven-graph TensorRT deployment for FluxVLA GR00T-N1.5."""

    name = 'groot_n15'
    export_dtype = 'float16'

    @classmethod
    def matches(cls, model_config: Any) -> bool:
        if isinstance(model_config, Mapping):
            model_type = model_config['type']
        else:
            model_type = model_config.type
        if model_type != 'LlavaVLA':
            return False
        if isinstance(model_config, Mapping):
            backbone = model_config['vlm_backbone']
            action_head = model_config['vla_head']
        else:
            backbone = model_config.vlm_backbone
            action_head = model_config.vla_head
        backbone_type = (
            backbone['type'] if isinstance(backbone, Mapping) else backbone.type)
        action_head_type = (
            action_head['type']
            if isinstance(action_head, Mapping) else action_head.type)
        return (backbone_type == 'EagleBackbone'
                and action_head_type == 'FlowMatchingHead')

    @classmethod
    def configure_model(cls, model_config: Any) -> None:
        if isinstance(model_config, Mapping):
            backbone = model_config['vlm_backbone']
        else:
            backbone = model_config.vlm_backbone
        if isinstance(backbone, Mapping):
            backbone['dtype'] = 'float16'
            backbone['attention_implementation'] = 'eager'
        else:
            backbone.dtype = 'float16'
            backbone.attention_implementation = 'eager'

    def make_representative_inputs(
        self,
        model: Any,
        model_config: Any,
        device: Any,
    ) -> Mapping[str, Any]:
        del model_config
        eagle = model.vlm_backbone
        action_head = model.vla_head
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        image_size = int(eagle.vlm.vision_model.config.image_size)
        image_views = 2
        sequence_length = 600
        image_tokens = image_views * int(eagle.vlm.num_image_token)
        state_dim = int(action_head.state_encoder.layer1.W.shape[1])
        lang_tokens = torch.zeros(1, sequence_length, dtype=torch.int64, device=device)
        lang_tokens[:, :image_tokens] = int(eagle.vlm.image_token_index)
        return {
            'images':
            torch.randn(
                1,
                image_views * 3,
                image_size,
                image_size,
                dtype=torch.float32,
                device=device,
                generator=generator,
            ),
            'lang_tokens':
            lang_tokens,
            'states':
            torch.zeros(
                1,
                state_dim,
                dtype=torch.float32,
                device=device,
            ),
            'img_masks':
            torch.ones(1, image_views, dtype=torch.bool, device=device),
            'lang_masks':
            torch.ones(1, sequence_length, dtype=torch.bool, device=device),
            'embodiment_ids':
            torch.zeros(1, dtype=torch.int64, device=device),
            'noise':
            torch.randn(
                1,
                int(action_head.num_steps),
                int(action_head.action_dim),
                dtype=torch.float32,
                device=device,
                generator=generator,
            ),
        }

    def prepare(
        self,
        model: Any,
        model_config: Any,
        representative_inputs: Mapping[str, Any],
    ) -> PreparedDeployment:
        if not self.matches(model_config):
            raise ValueError('GR00T-N1.5 requires LlavaVLA with EagleBackbone and '
                             'FlowMatchingHead.')
        eagle = model.vlm_backbone
        action_head = model.vla_head
        language_tokens = representative_inputs['lang_tokens']
        if not isinstance(language_tokens, torch.Tensor):
            raise TypeError('lang_tokens must be a torch.Tensor.')
        if language_tokens.ndim != 2:
            raise ValueError('lang_tokens must have shape (B, L).')

        vision_config = eagle.vlm.vision_model.config
        image_size = int(vision_config.image_size)
        patch_size = int(vision_config.patch_size)
        contract = GR00TN15Contract(
            image_views=2,
            image_size=image_size,
            vision_patches=(image_size // patch_size)**2,
            projected_image_tokens_per_view=int(eagle.vlm.num_image_token),
            sequence_length=int(language_tokens.shape[1]),
            state_dim=int(action_head.state_encoder.layer1.W.shape[1]),
            action_horizon=int(action_head.num_steps),
            action_dim=int(action_head.action_dim),
            future_tokens=int(action_head.future_tokens.num_embeddings),
            backbone_dim=int(action_head.vlln.normalized_shape[0]),
            vision_hidden=int(vision_config.hidden_size),
            input_embedding_dim=int(action_head.input_embedding_dim),
            hidden_size=int(action_head.hidden_size),
            denoise_steps=int(action_head.num_inference_timesteps),
        )
        if not action_head.add_positional_embeddings:
            raise ValueError('GR00T-N1.5 TensorRT requires action position embeddings.')
        non_fp16 = [
            name for name, parameter in model.named_parameters()
            if parameter.is_floating_point() and parameter.dtype != torch.float16
        ]
        if non_fp16:
            raise TypeError('GR00T-N1.5 TensorRT export requires an FP16 model; '
                            f'found {non_fp16[0]} with a different dtype.')

        _set_eager_attention(model)
        vision = GR00TVisionExport(eagle)
        language = GR00TLanguageExport(eagle)
        backbone_output = GR00TBackboneOutputExport(action_head)
        prepared = prepare_policy_inputs(representative_inputs, contract)
        positions = position_ids(contract, prepared['pixel_values'].device)

        with torch.inference_mode():
            vit_embeds = vision(prepared['pixel_values'], positions)
            input_embeds = prepare_language_inputs(
                eagle,
                vit_embeds,
                prepared['lang_tokens'],
                contract,
            )
            embeddings = language(input_embeds, prepared['attention_mask'])
            backbone_features = backbone_output(embeddings)
            state_features = action_head.state_encoder(
                prepared['state'],
                prepared['embodiment_id'],
            )
            timestep = torch.zeros(
                contract.batch,
                dtype=torch.int64,
                device=prepared['noise'].device,
            )
            actions = prepared['noise'].to(torch.float16)
            action_features = action_head.action_encoder(
                actions,
                timestep,
                prepared['embodiment_id'],
            )
            action_positions = torch.arange(
                contract.action_horizon,
                dtype=torch.int64,
                device=actions.device,
            )
            action_features = action_features + action_head.position_embedding(
                action_positions).unsqueeze(0)
            future = action_head.future_tokens.weight.unsqueeze(0)
            sa_embs = torch.cat((state_features, future, action_features), dim=1)
            model_output = action_head.model(
                sa_embs,
                backbone_features,
                timestep,
            )

        export_options = {'opset': 19, 'dynamo': False}
        build_options = {'precision': 'fp16', 'strongly_typed': False}
        graphs = {
            'vit':
            PreparedGraph(
                name='vit',
                module=vision,
                inputs={
                    'pixel_values': prepared['pixel_values'],
                    'position_ids': positions,
                },
                input_specs=(
                    TensorSpec(
                        'pixel_values',
                        'float16',
                        (
                            contract.image_views,
                            3,
                            contract.image_size,
                            contract.image_size,
                        ),
                    ),
                    TensorSpec(
                        'position_ids',
                        'int64',
                        (contract.image_views, contract.vision_patches),
                    ),
                ),
                output_names=('vit_embeds', ),
                export_options=export_options,
                build_options=build_options,
            ),
            'llm':
            PreparedGraph(
                name='llm',
                module=language,
                inputs={
                    'inputs_embeds': input_embeds,
                    'attention_mask': prepared['attention_mask'],
                },
                input_specs=(
                    TensorSpec(
                        'inputs_embeds',
                        'float16',
                        (
                            contract.batch,
                            contract.sequence_length,
                            contract.backbone_dim,
                        ),
                    ),
                    TensorSpec(
                        'attention_mask',
                        'int64',
                        (contract.batch, contract.sequence_length),
                    ),
                ),
                output_names=('embeddings', ),
                export_options=export_options,
                build_options=build_options,
            ),
            'backbone_output':
            PreparedGraph(
                name='backbone_output',
                module=backbone_output,
                inputs={'backbone_features': embeddings},
                input_specs=(TensorSpec(
                    'backbone_features',
                    'float16',
                    (
                        contract.batch,
                        contract.sequence_length,
                        contract.backbone_dim,
                    ),
                ), ),
                output_names=('output', ),
                export_options=export_options,
                build_options=build_options,
            ),
            'state_encoder':
            PreparedGraph(
                name='state_encoder',
                module=action_head.state_encoder,
                inputs={
                    'state': prepared['state'],
                    'embodiment_id': prepared['embodiment_id'],
                },
                input_specs=(
                    TensorSpec(
                        'state',
                        'float16',
                        (contract.batch, 1, contract.state_dim),
                    ),
                    TensorSpec(
                        'embodiment_id',
                        'int64',
                        (contract.batch, ),
                    ),
                ),
                output_names=('output', ),
                export_options=export_options,
                build_options=build_options,
            ),
            'action_encoder':
            PreparedGraph(
                name='action_encoder',
                module=action_head.action_encoder,
                inputs={
                    'actions': actions,
                    'timesteps_tensor': timestep,
                    'embodiment_id': prepared['embodiment_id'],
                },
                input_specs=(
                    TensorSpec(
                        'actions',
                        'float16',
                        (
                            contract.batch,
                            contract.action_horizon,
                            contract.action_dim,
                        ),
                    ),
                    TensorSpec(
                        'timesteps_tensor',
                        'int64',
                        (contract.batch, ),
                    ),
                    TensorSpec(
                        'embodiment_id',
                        'int64',
                        (contract.batch, ),
                    ),
                ),
                output_names=('output', ),
                export_options=export_options,
                build_options=build_options,
            ),
            'dit':
            PreparedGraph(
                name='dit',
                module=action_head.model,
                inputs={
                    'sa_embs': sa_embs,
                    'vl_embs': backbone_features,
                    'timesteps_tensor': timestep,
                },
                input_specs=(
                    TensorSpec(
                        'sa_embs',
                        'float16',
                        (
                            contract.batch,
                            contract.sa_length,
                            contract.input_embedding_dim,
                        ),
                    ),
                    TensorSpec(
                        'vl_embs',
                        'float16',
                        (
                            contract.batch,
                            contract.sequence_length,
                            contract.backbone_dim,
                        ),
                    ),
                    TensorSpec(
                        'timesteps_tensor',
                        'int64',
                        (contract.batch, ),
                    ),
                ),
                output_names=('output', ),
                export_options=export_options,
                build_options=build_options,
            ),
            'action_decoder':
            PreparedGraph(
                name='action_decoder',
                module=action_head.action_decoder,
                inputs={
                    'model_output': model_output,
                    'embodiment_id': prepared['embodiment_id'],
                },
                input_specs=(
                    TensorSpec(
                        'model_output',
                        'float16',
                        (
                            contract.batch,
                            contract.sa_length,
                            contract.hidden_size,
                        ),
                    ),
                    TensorSpec(
                        'embodiment_id',
                        'int64',
                        (contract.batch, ),
                    ),
                ),
                output_names=('output', ),
                export_options=export_options,
                build_options=build_options,
            ),
        }
        reference_inputs = dict(representative_inputs)
        reference = GR00TReferenceExecutor(model, contract)
        return PreparedDeployment(
            family=self.name,
            graphs=graphs,
            reference_runner=lambda: reference(reference_inputs),
            output_names=('actions', ),
            metadata={'contract': vars(contract).copy()},
            runtime_resources={
                'eagle_backbone': eagle,
                'action_head': action_head,
            },
        )

    def create_executor(
        self,
        engines: Mapping[str, Any],
        deployment: PreparedDeployment,
    ) -> GR00TTensorRTExecutor:
        contract = GR00TN15Contract(**deployment.metadata['contract'])
        resources = deployment.runtime_resources
        return GR00TTensorRTExecutor(
            engines,
            resources['eagle_backbone'],
            resources['action_head'],
            contract,
        )
