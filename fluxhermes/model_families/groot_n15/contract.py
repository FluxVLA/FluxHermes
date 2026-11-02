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
"""Static deployment contract for the GR00T-N1.5 TensorRT graph set."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GR00TN15Contract:
    batch: int = 1
    image_views: int = 2
    image_size: int = 224
    vision_patches: int = 256
    projected_image_tokens_per_view: int = 256
    sequence_length: int = 600
    state_dim: int = 64
    action_horizon: int = 10
    action_dim: int = 32
    future_tokens: int = 32
    backbone_dim: int = 2048
    vision_hidden: int = 1152
    input_embedding_dim: int = 1536
    hidden_size: int = 1024
    denoise_steps: int = 4
    timestep_buckets: int = 1000
    output_action_dim: int = 32

    @property
    def image_tokens(self) -> int:
        return self.image_views * self.projected_image_tokens_per_view

    @property
    def sa_length(self) -> int:
        return 1 + self.future_tokens + self.action_horizon

    def __post_init__(self) -> None:
        invalid = {
            name: value
            for name, value in vars(self).items()
            if not isinstance(value, int) or value <= 0
        }
        if invalid:
            raise ValueError(f'GR00T-N1.5 contract values must be positive: {invalid}')
        if self.batch != 1:
            raise ValueError('GR00T-N1.5 TensorRT supports batch size 1.')
        if self.sequence_length < self.image_tokens:
            raise ValueError('GR00T-N1.5 sequence cannot hold all image tokens.')
        if self.output_action_dim > self.action_dim:
            raise ValueError(
                'GR00T-N1.5 output action dimension cannot exceed action_dim.')
