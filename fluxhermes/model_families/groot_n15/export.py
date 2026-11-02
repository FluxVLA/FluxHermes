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
"""ONNX-exportable learned partitions for GR00T-N1.5."""

from __future__ import annotations

import torch
from torch import nn

from .contract import GR00TN15Contract


class GR00TVisionExport(nn.Module):
    """Export the Eagle vision transformer as one graph."""

    def __init__(self, eagle_backbone: nn.Module) -> None:
        super().__init__()
        self.vision = eagle_backbone.vlm.vision_model.vision_model
        self.eval()

    def forward(self, pixel_values: torch.Tensor,
                position_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.vision.embeddings
        patch = embeddings.patch_embedding(
            pixel_values.to(embeddings.patch_embedding.weight.dtype))
        hidden = patch.flatten(2).transpose(1, 2)
        hidden = hidden + embeddings.position_embedding(position_ids)
        output = self.vision.encoder(
            inputs_embeds=hidden,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return self.vision.post_layernorm(output.last_hidden_state)


class GR00TLanguageExport(nn.Module):
    """Export image projection, token embedding and Eagle language layers."""

    def __init__(self, eagle_backbone: nn.Module,
                 contract: GR00TN15Contract) -> None:
        super().__init__()
        self.language_model = eagle_backbone.vlm.language_model
        self.connector = eagle_backbone.vlm.mlp1
        self.eagle_linear = eagle_backbone.eagle_linear
        self.select_layer = int(eagle_backbone.select_layer)
        self.contract = contract
        self.image_token_index = int(eagle_backbone.vlm.image_token_index)
        self.use_pixel_shuffle = bool(eagle_backbone.vlm.use_pixel_shuffle)
        self.downsample_ratio = float(eagle_backbone.vlm.downsample_ratio)
        if self.use_pixel_shuffle and self.downsample_ratio != 0.5:
            raise ValueError(
                'GR00T-N1.5 TensorRT supports Eagle pixel shuffle ratio 0.5.')
        self.eval()

    @staticmethod
    def _pixel_shuffle(value: torch.Tensor) -> torch.Tensor:
        batch, width, height, channels = value.shape
        value = value.reshape(batch, width, height // 2, channels * 2)
        value = value.permute(0, 2, 1, 3).contiguous()
        value = value.reshape(batch, height // 2, width // 2, channels * 4)
        return value.permute(0, 2, 1, 3).contiguous()

    def forward(self, vit_embeds: torch.Tensor, lang_tokens: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        if self.use_pixel_shuffle:
            side = int(vit_embeds.shape[1]**0.5)
            vit_embeds = vit_embeds.reshape(
                self.contract.image_views,
                side,
                side,
                self.contract.vision_hidden,
            )
            vit_embeds = self._pixel_shuffle(vit_embeds)
            vit_embeds = vit_embeds.reshape(
                self.contract.image_views,
                -1,
                vit_embeds.shape[-1],
            )
        image_features = self.connector(vit_embeds).reshape(
            self.contract.batch,
            self.contract.image_tokens,
            self.contract.backbone_dim,
        )
        inputs_embeds = self.language_model.get_input_embeddings()(
            lang_tokens)
        selected = lang_tokens == self.image_token_index
        image_indices = torch.cumsum(selected.to(torch.int64), dim=1) - 1
        image_indices = image_indices.clamp(0, self.contract.image_tokens - 1)
        selected_features = torch.gather(
            image_features,
            1,
            image_indices.unsqueeze(-1).expand(
                -1, -1, self.contract.backbone_dim),
        )
        inputs_embeds = torch.where(
            selected.unsqueeze(-1), selected_features, inputs_embeds)
        output = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return self.eagle_linear(output.hidden_states[self.select_layer])


class GR00TActionEncoderExport(nn.Module):
    """Export the complete state/action token assembly graph."""

    def __init__(self, action_head: nn.Module,
                 contract: GR00TN15Contract) -> None:
        super().__init__()
        self.action_encoder = action_head.action_encoder
        self.position_embedding = action_head.position_embedding
        self.future_tokens = action_head.future_tokens
        self.contract = contract
        self.register_buffer(
            'action_positions',
            torch.arange(contract.action_horizon, dtype=torch.int64),
            persistent=False,
        )
        self.eval()

    def forward(
        self,
        actions: torch.Tensor,
        timesteps_tensor: torch.Tensor,
        embodiment_id: torch.Tensor,
        state_features: torch.Tensor,
    ) -> torch.Tensor:
        action_features = self.action_encoder(
            actions,
            timesteps_tensor,
            embodiment_id,
        )
        action_features = action_features + self.position_embedding(
            self.action_positions).unsqueeze(0)
        future = self.future_tokens.weight.unsqueeze(0).expand(
            actions.shape[0], -1, -1)
        return torch.cat((state_features, future, action_features), dim=1)


class GR00TBackboneOutputExport(nn.Module):
    """Export the GR00T visual-language feature adapter."""

    def __init__(self, action_head: nn.Module) -> None:
        super().__init__()
        self.vlln = action_head.vlln
        self.vl_self_attention = action_head.vl_self_attention
        self.eval()

    def forward(self, backbone_features: torch.Tensor) -> torch.Tensor:
        return self.vl_self_attention(self.vlln(backbone_features))
