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
"""PI0.5 model-family graph preparation and TensorRT orchestration."""

from __future__ import annotations

import inspect
from typing import Any, Mapping

import torch

from ...adapters.base import PreparedDeployment, PreparedGraph
from ...contracts import TensorSpec
from ..base import ModelFamily, config_mappings, config_nodes
from ..registry import MODEL_FAMILIES
from .contract import PI05Contract, cache_names
from .executor import (PI05SourceReferenceExecutor, PI05TensorRTExecutor,
                       prepare_policy_inputs)
from .export import PI05DenoiseExport, PI05PrefixExport, PI05VisionExport


def _image_view_count(images: Any) -> int:
    if isinstance(images, (list, tuple)):
        return len(images)
    if not isinstance(images, torch.Tensor):
        raise TypeError('images must be a tensor or a list of tensors.')
    if images.ndim != 4 or images.shape[1] % 3 != 0:
        raise ValueError('Flattened images must have shape (B, 3V, H, W).')
    return int(images.shape[1] // 3)


def _model_type(model_config: Any) -> str:
    if isinstance(model_config, Mapping):
        return str(model_config['type'])
    return str(model_config.type)


def _language_tokens(training_config: Any) -> int:
    from fluxvla.transforms.transform_prompts import (LiberoPromptFromInputs,
                                                      ProcessPrompts)

    prompt_types = {
        'ProcessPrompts': ProcessPrompts,
        'LiberoPromptFromInputs': LiberoPromptFromInputs,
    }
    lengths = set()
    for name, prompt_type in prompt_types.items():
        default = inspect.signature(
            prompt_type.__init__).parameters['max_len'].default
        lengths.update(
            int(node['max_len']) if 'max_len' in node else int(default)
            for node in config_nodes(training_config, name))
    if len(lengths) != 1:
        raise ValueError(
            'PI0.5 training config must resolve to one ProcessPrompts max_len; '
            f'found {sorted(lengths)}.')
    return lengths.pop()


def _image_views_from_config(training_config: Any) -> int:
    counts = {
        len(node[key])
        for node in config_mappings(training_config)
        for key in ('video_keys', 'img_keys') if key in node
    }
    if len(counts) != 1:
        raise ValueError(
            'PI0.5 training config must resolve to one image-view count; '
            f'found {sorted(counts)}.')
    return counts.pop()


@MODEL_FAMILIES.register_module('pi05')
class PI05ModelFamily(ModelFamily):
    """Three-graph TensorRT deployment for FluxVLA PI0.5."""

    name = 'pi05'
    export_dtype = 'bfloat16'

    @classmethod
    def matches(cls, model_config: Any) -> bool:
        return _model_type(model_config) == 'PI05FlowMatching'

    def make_representative_inputs(
        self,
        model: Any,
        training_config: Any,
        device: Any,
    ) -> Mapping[str, Any]:
        contract = PI05Contract(
            image_views=_image_views_from_config(training_config),
            language_tokens=_language_tokens(training_config),
            action_horizon=int(model.n_action_steps),
            action_dim=int(model.max_action_dim),
            denoise_steps=int(model.num_steps),
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        vocab_size = int(model.llm_backbone.config.vocab_size)
        return {
            'images':
            torch.randn(
                contract.batch,
                contract.image_views * 3,
                contract.image_height,
                contract.image_width,
                dtype=torch.float32,
                device=device,
                generator=generator,
            ),
            'lang_tokens':
            torch.randint(
                0,
                vocab_size,
                (contract.batch, contract.language_tokens),
                dtype=torch.int32,
                device=device,
                generator=generator,
            ),
            'states':
            torch.zeros(
                contract.batch,
                contract.action_dim,
                dtype=torch.float32,
                device=device,
            ),
            'img_masks':
            torch.ones(
                contract.batch,
                contract.image_views,
                dtype=torch.bool,
                device=device,
            ),
            'lang_masks':
            torch.ones(
                contract.batch,
                contract.language_tokens,
                dtype=torch.bool,
                device=device,
            ),
            'noise':
            torch.randn(
                contract.batch,
                contract.action_horizon,
                contract.action_dim,
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
            raise ValueError('PI0.5 requires model type PI05FlowMatching.')
        lang_tokens = representative_inputs['lang_tokens']
        if not isinstance(lang_tokens, torch.Tensor):
            raise TypeError('lang_tokens must be a torch.Tensor.')
        if lang_tokens.ndim != 2:
            raise ValueError('lang_tokens must have shape (B, L).')
        contract = PI05Contract(
            image_views=_image_view_count(representative_inputs['images']),
            language_tokens=int(lang_tokens.shape[1]),
            action_horizon=int(model.n_action_steps),
            action_dim=int(model.max_action_dim),
            denoise_steps=int(model.num_steps),
        )
        vision = PI05VisionExport(model, contract)
        prefix = PI05PrefixExport(model, contract)
        denoise = PI05DenoiseExport(model, contract)
        prepared = prepare_policy_inputs(representative_inputs, contract)
        cache_output_names = cache_names(contract)

        with torch.inference_mode():
            image_prefix = vision(prepared['images'])
            cache_values = prefix(
                image_prefix,
                prepared['lang_tokens'],
                prepared['lang_mask'],
            )
        if len(cache_values) != len(cache_output_names):
            raise ValueError('PI0.5 prefix returned the wrong cache count.')

        export_options = {'opset': 19}
        build_options = {'strongly_typed': True}
        cache_specs = tuple(
            TensorSpec(name, 'bfloat16', contract.cache_shape)
            for name in cache_output_names)
        vision_graph = PreparedGraph(
            name='vision',
            module=vision,
            inputs={'images': prepared['images']},
            input_specs=(TensorSpec(
                'images',
                'bfloat16',
                (
                    contract.image_views,
                    3,
                    contract.image_height,
                    contract.image_width,
                ),
            ), ),
            output_names=('image_prefix', ),
            export_options=export_options,
            build_options=build_options,
        )
        prefix_graph = PreparedGraph(
            name='prefix',
            module=prefix,
            inputs={
                'image_prefix': image_prefix,
                'lang_tokens': prepared['lang_tokens'],
                'lang_mask': prepared['lang_mask'],
            },
            input_specs=(
                TensorSpec(
                    'image_prefix',
                    'bfloat16',
                    (
                        contract.batch,
                        contract.image_tokens,
                        contract.prefix_hidden,
                    ),
                ),
                TensorSpec(
                    'lang_tokens',
                    'int32',
                    (contract.batch, contract.language_tokens),
                ),
                TensorSpec(
                    'lang_mask',
                    'bool',
                    (contract.batch, contract.language_tokens),
                ),
            ),
            output_names=cache_output_names,
            export_options=export_options,
            build_options=build_options,
        )
        denoise_inputs = {
            'x_t': prepared['noise'],
            'timestep': torch.ones(
                contract.batch,
                dtype=torch.float32,
                device=prepared['noise'].device,
            ),
            'lang_mask': prepared['lang_mask'],
            **dict(zip(cache_output_names, cache_values)),
        }
        denoise_graph = PreparedGraph(
            name='denoise',
            module=denoise,
            inputs=denoise_inputs,
            input_specs=(
                TensorSpec(
                    'x_t',
                    'float32',
                    (
                        contract.batch,
                        contract.action_horizon,
                        contract.action_dim,
                    ),
                ),
                TensorSpec(
                    'timestep',
                    'float32',
                    (contract.batch, ),
                ),
                TensorSpec(
                    'lang_mask',
                    'bool',
                    (contract.batch, contract.language_tokens),
                ),
                *cache_specs,
            ),
            output_names=('x_next', ),
            export_options=export_options,
            build_options=build_options,
        )
        graphs = {
            'vision': vision_graph,
            'prefix': prefix_graph,
            'denoise': denoise_graph,
        }
        policy_inputs = dict(representative_inputs)
        reference = PI05SourceReferenceExecutor(model, contract)
        return PreparedDeployment(
            family=self.name,
            graphs=graphs,
            reference_runner=lambda: reference(policy_inputs),
            output_names=('actions', ),
            metadata={'contract': vars(contract).copy()},
        )

    def create_executor(
        self,
        engines: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> PI05TensorRTExecutor:
        contract = PI05Contract(**metadata['contract'])
        return PI05TensorRTExecutor(engines, contract)
