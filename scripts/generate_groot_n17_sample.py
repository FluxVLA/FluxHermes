"""Generate a deterministic GR00T N1.7 TensorRT export sample.

The sample runs a synthetic LIBERO observation through the original FluxVLA
evaluation transforms, then captures the exact boundary consumed by the
FluxHermes N1.7 adapter. It is suitable for export and source/backend parity;
replace the synthetic raw observation with a recorded rollout for calibration.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

# Avoid this repository's ``scripts/inspect.py`` shadowing Python's stdlib
# ``inspect`` module when this file is executed directly.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == _SCRIPT_DIR:
    sys.path.pop(0)

# isort: off
import numpy as np  # noqa: E402
# isort: on


def _gradient_images(size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:size, 0:size]
    x = x * 255 // (size - 1)
    y = y * 255 // (size - 1)
    diagonal = (x + y) // 2
    first = np.stack((x, y, diagonal), axis=-1).astype(np.uint8)
    second = np.stack((diagonal, x, y), axis=-1).astype(np.uint8)
    return first, second


def generate(output: Path, fluxvla_root: Path, config_path: Path) -> Path:
    output = output.expanduser().resolve()
    fluxvla_root = fluxvla_root.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    sys.path.insert(0, str(fluxvla_root))
    os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
    # FluxVLA configs intentionally contain repository-relative component
    # paths, so construct the original preprocessing pipeline from its root.
    os.chdir(fluxvla_root)
    import torch
    from fluxvla.engines import build_dataset_from_cfg
    from fluxvla.models.backbones.vlms.groot_n17_qwen3_backbone import \
        GrootN17Qwen3Backbone
    from fluxvla.models.compat.qwen3vl_457_compat import _get_rope_index_457
    from mmengine import Config
    from mmengine.config import ConfigDict

    cfg = Config.fromfile(str(config_path))
    dataset_cfg = copy.deepcopy(cfg.eval.dataset)
    # The saved checkpoint statistics use pre-transform names. The resolved
    # training config contains the equivalent post-mapping ``proprio`` stats
    # expected by this evaluation pipeline.
    dataset_cfg['norm_stats'] = copy.deepcopy(
        cfg.train_dataloader.dataset.dataset_statistics)
    dataset_cfg['task_suite_name'] = cfg.eval.task_suite_name
    dataset_cfg['norm_stats_key'] = cfg.eval.norm_stats_key
    dataset = build_dataset_from_cfg(dataset_cfg)

    agent_image, wrist_image = _gradient_images()
    raw_observation = dict(
        agentview_image=agent_image,
        robot0_eye_in_hand_image=wrist_image,
        robot0_eef_pos=np.array([-0.04, 0.03, 0.82], dtype=np.float32),
        robot0_eef_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        robot0_gripper_qpos=np.array([0.02, -0.02], dtype=np.float32),
        task_description=(
            'pick up the black bowl and place it on the plate'),
        is_new_episode=True,
    )
    batch, _ = dataset(raw_observation)
    lang_tokens, lang_masks = GrootN17Qwen3Backbone._trim_common_left_padding(
        batch['lang_tokens'], batch['lang_masks'])
    image_grid_thw = batch['image_grid_thw'].flatten(0, 1)

    rope_owner = type('RopeOwner', (), {})()
    rope_owner.config = ConfigDict(cfg.model.vlm_backbone.model_config)
    position_ids, _ = _get_rope_index_457(
        rope_owner,
        input_ids=lang_tokens.cpu(),
        image_grid_thw=image_grid_thw.cpu(),
        attention_mask=lang_masks.cpu(),
    )
    generator = torch.Generator(device='cpu').manual_seed(42)
    noise = torch.randn((1, 40, 132), generator=generator, dtype=torch.float32)

    arrays = dict(
        images=batch['images'].detach().cpu().float().numpy(),
        lang_tokens=lang_tokens.detach().cpu().numpy().astype(np.int64),
        position_ids=position_ids.detach().cpu().numpy().astype(np.int64),
        states=batch['states'].detach().cpu().float().numpy(),
        embodiment_ids=(
            batch['embodiment_ids'].detach().cpu().numpy().astype(np.int32)),
        noise=noise.numpy(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output',
        default=(
            '/root/projects/FluxHermes/work_dirs/gr00t-n17-libero-bf16/'
            'groot_n17_deploy_sample.npz'),
    )
    parser.add_argument('--fluxvla-root', default='/root/projects/FluxVLA')
    parser.add_argument(
        '--config',
        default=(
            '/root/projects/FluxVLA/configs/gr00tn17/'
            'gr00tn17_qwen3vl_2b_libero_10_full_finetune.py'),
    )
    args = parser.parse_args()
    output = generate(
        Path(args.output), Path(args.fluxvla_root), Path(args.config))
    print(output)


if __name__ == '__main__':
    main()
