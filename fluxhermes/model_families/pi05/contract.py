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
"""Static deployment contract for the PI0.5 TensorRT partition set."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PI05Contract:
    batch: int = 1
    image_views: int = 2
    image_height: int = 224
    image_width: int = 224
    image_tokens_per_view: int = 256
    language_tokens: int = 180
    kv_layers: int = 18
    kv_heads: int = 1
    head_dim: int = 256
    action_horizon: int = 10
    action_dim: int = 32
    denoise_steps: int = 10
    prefix_hidden: int = 2048
    expert_hidden: int = 1024

    @property
    def image_tokens(self) -> int:
        return self.image_views * self.image_tokens_per_view

    @property
    def prefix_tokens(self) -> int:
        return self.image_tokens + self.language_tokens

    @property
    def cache_shape(self) -> tuple[int, ...]:
        return (self.batch, self.kv_heads, self.prefix_tokens, self.head_dim)

    def __post_init__(self) -> None:
        invalid = {
            name: value
            for name, value in vars(self).items()
            if not isinstance(value, int) or value <= 0
        }
        if invalid:
            raise ValueError(f'PI0.5 contract values must be positive: {invalid}')
        if self.batch != 1:
            raise ValueError('PI0.5 TensorRT supports batch size 1.')
        if self.image_views != 2:
            raise ValueError('PI0.5 TensorRT requires exactly two image views.')
        if self.action_horizon != 10 or self.denoise_steps != 10:
            raise ValueError('PI0.5 TensorRT requires action horizon 10 and exactly '
                             '10 Euler steps.')


def cache_names(contract: PI05Contract) -> tuple[str, ...]:
    return tuple(
        name for layer in range(contract.kv_layers)
        for name in (f'prefix_k_{layer:02d}', f'prefix_v_{layer:02d}'))
