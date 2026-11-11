#!/usr/bin/env python3
"""Generate 5 synthetic RGB/prompt observations x seeds 7,17,29 on native PI05.

Requires Python 3.10, pinned FluxVLA checkout and local model assets.
Uses original preprocessing and predict_action; does not download assets.
This script is intended for execution on the GPU host, not local validation.
Only a complete run publishes validation.json; partial directories are retained
for diagnosis and must not be reused. Golden actions are normalized model outputs,
not denormalized robot commands or task-success measurements.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

REVISION = '249531b78ee922757a6e4b56d2f5965487f9f0fc'
CHECKPOINT_SHA256 = '4a79542cbb79b75deea3804d91f87f9c2180ff704f0cd1988469ee72655fbe15'
TOKEN_LIMIT = 32
SEEDS = (7, 17, 29)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(root, manifest_path=None):
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text())
        require(manifest['revision'] == REVISION, 'Wrong source manifest revision')
        require(len(manifest['files']) == 715, 'Expected all 715 tracked source files')
        for name, expected in manifest['files'].items():
            path = (root / name).resolve()
            require(path.is_relative_to(root), 'Source manifest path escapes checkout')
            require(sha256_file(path) == expected, f'Source checksum mismatch: {name}')
        return
    revision = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    require(revision == REVISION, 'Wrong FluxVLA revision')
    status = subprocess.check_output(
        ['git', '-C',
         str(root), 'status', '--porcelain', '--untracked-files=no'],
        text=True)
    require(not status.strip(), 'FluxVLA tracked files must be unchanged')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def dump(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def resolve_tokenizer(root, config, checkpoint, source):
    if root is not None:
        path = root.expanduser().resolve()
        if path.is_dir():
            path /= 'tokenizer.model'
        if not path.is_file():
            raise FileNotFoundError(
                f'--tokenizer-root requires existing SentencePiece file: {path}')
        return path
    candidates = [checkpoint.parent.parent / 'tokenizer' / 'tokenizer.model']
    for field in ('model_path', 'download_path'):
        value = config.get(field)
        if value and '://' not in str(value):
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = source / path
            if path.is_dir():
                path /= 'tokenizer.model'
            candidates.append(path)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        'No local Paligemma tokenizer; pass --tokenizer-root pointing to '
        'tokenizer.model or its directory. Checked: ' + ', '.join(map(str, candidates)))


def choose_prompts(tokenizer):
    """Search whole natural-language prompts, never clip token ids or input text."""

    def length(text):
        return len(tokenizer(text)['input_ids'])

    short = [
        'pick up the cup',
        'place the red cup on the table',
        'move the small blue bowl next to the plate on the left side of the table',
        'open the drawer',
        'put the spoon inside the bowl',
        'pick up the black bowl and place it on the wooden tray',
    ]
    chosen, lengths = [], set()
    for text in short:
        n = length(text)
        if n < TOKEN_LIMIT and n not in lengths:
            chosen.append(text)
            lengths.add(n)
        if len(chosen) == 3:
            break
    require(
        len(chosen) == 3,
        'Tokenizer did not yield three distinct padded prompt lengths')
    # O4 tries an exact 32-token sentence using real tokenizer results.
    candidates = []
    for obj in ('cup', 'red cup', 'small blue bowl', 'large black ceramic bowl'):
        for dest in ('wooden tray', 'white plate', 'open drawer',
                     'left side of the table'):
            for suffix in ('', ' carefully', ' without touching the other objects', (
                    ' while leaving the spoon and the plate in their original '
                    'positions'
            ), ' then return the gripper to its starting position above the table'):
                text = (f'pick up the {obj} from the table '
                        f'and place it on the {dest}{suffix}')
                n = length(text)
                if n <= TOKEN_LIMIT:
                    candidates.append((n, text))
    require(
        bool(candidates), 'No O4 prompt fits the explicit deployment token capacity')
    n, longest_text = max(candidates, key=lambda pair: pair[0])
    for text in short:
        if text not in chosen and text != longest_text and length(text) < TOKEN_LIMIT:
            chosen.append(text)
            break
    chosen.append(longest_text)  # Fifth observation is O4 (zero-based IDs).
    require(len(chosen) == 5, 'Expected five distinct task prompts')
    return chosen, {
        'o4_target': TOKEN_LIMIT,
        'o4_actual': n,
        'o4_full_length_achieved': n == TOKEN_LIMIT,
        'o4_gap_tokens': TOKEN_LIMIT - n,
        'search': 'finite whole-sentence candidate search; no token truncation'
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--fluxvla-root', type=Path, required=True)
    parser.add_argument(
        '--source-manifest',
        type=Path,
        help='Explicit SHA256 manifest for a source snapshot without Git')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument(
        '--config', type=Path, help='Default: checkpoint.parent.parent/config.json')
    parser.add_argument(
        '--tokenizer-root',
        type=Path,
        help='Local tokenizer.model file or containing directory')
    args = parser.parse_args()
    source = args.fluxvla_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = (args.config
                   or checkpoint.parent.parent / 'config.json').expanduser().resolve()
    output = args.root.expanduser().resolve() / 'cases' / 'pi05-01'
    if output.exists():
        raise FileExistsError(output)
    for path in (source / 'fluxvla/__init__.py', checkpoint, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    require(sys.version_info[:2] == (3, 10), 'Use existing cp310 fluxvla interpreter')
    verify_source(source, args.source_manifest)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    import numpy as np
    import psutil
    import torch
    from mmengine import Config
    from safetensors.torch import load_file

    require(psutil.virtual_memory().available >= 24 * 1024**3,
            'Requires >=24 GiB available host RAM before model load')
    require(torch.cuda.is_available(), 'A real CUDA GPU is required')
    torch.cuda.set_device(0)
    gpu_free, gpu_total = torch.cuda.mem_get_info()
    require(gpu_free >= 12 * 1024**3,
            'Requires >=12 GiB free GPU memory before model load')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    import fluxvla
    from fluxvla.engines import build_transform_from_cfg, build_vla_from_cfg
    require(
        Path(fluxvla.__file__).resolve().parent == source / 'fluxvla',
        'Wrong FluxVLA import origin')
    require(
        not any(x == 'fluxhermes' or x.startswith(('fluxhermes.', 'fluxvla_export'))
                for x in sys.modules),
        'Native oracle must not import Hermes or export SDK')
    cfg = Config.fromfile(str(config_path))
    original_model_config = copy.deepcopy(dict(cfg.model))
    require(
        cfg.model.type == 'PI05FlowMatching',
        'Only ordinary PI05FlowMatching is supported; not PI0 or inference variants')
    require(cfg.model.n_action_steps == 10 and cfg.model.max_action_dim == 32,
            'Unexpected action profile')
    require(
        cfg.runner.get('mixed_precision_dtype') == 'bf16'
        and cfg.runner.get('enable_mixed_precision_training'),
        'Saved runner must select BF16 mixed precision')
    transforms_cfg = copy.deepcopy(list(cfg.eval.dataset.transforms))
    require([t['type'] for t in transforms_cfg[:3]] == [
        'ProcessLiberoEvalInputs', 'TransformImage', 'LiberoPromptFromInputs'
    ], 'Unexpected saved eval transform order')
    prompt_cfg = transforms_cfg[2]
    require('max_len' not in prompt_cfg,
            ('Expected saved config without max_len; review changed '
             'deployment capacity explicitly'))
    require(
        prompt_cfg.get('use_conversation') is False
        and not prompt_cfg.get('prompt_template') and not prompt_cfg.get('add_new_line')
        and not prompt_cfg.get('negative_prompt'),
        'Prompt formatting changed; raw-token length preflight needs review')
    tokenizer_path = resolve_tokenizer(args.tokenizer_root, prompt_cfg['tokenizer'],
                                       checkpoint, source)
    prompt_cfg['tokenizer']['model_path'] = str(tokenizer_path)
    # Deployment-only choice; preserve the saved config and checkpoint.
    prompt_cfg['max_len'] = TOKEN_LIMIT
    processors = [build_transform_from_cfg(t) for t in transforms_cfg[:3]]
    prompts, coverage = choose_prompts(processors[2].tokenizer)
    observations = []
    for obs_index, text in enumerate(prompts):
        rng = np.random.default_rng(1000 + obs_index)
        # Two distinct RGB uint8 views, including spatial gradients and texture.
        y, x = np.indices((256, 256))
        rgb = np.stack([(x + obs_index * 31) % 256, (y * obs_index) % 256,
                        ((x + y) // 2 + obs_index * 17) % 256],
                       axis=-1).astype(np.uint8)
        wrist = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        raw = {
            'agentview_image': rgb,
            'robot0_eye_in_hand_image': wrist,
            'task_description': text
        }
        token_ids = processors[2].tokenizer(text)['input_ids']
        require(
            len(token_ids) <= TOKEN_LIMIT,
            f'O{obs_index} would truncate {len(token_ids)} tokens')
        processed = raw.copy()
        for transform in processors:
            processed = transform(processed)
        require(processed['lang_tokens'][:len(token_ids)].tolist() == token_ids,
                'Transform altered raw token ids')
        require(
            int(processed['lang_masks'].sum()) == len(token_ids),
            'Token mask length mismatch')
        inputs = {
            'images': processed['pixel_values'].unsqueeze(0).numpy(),
            'img_masks': np.asarray(processed['img_masks'], dtype=np.bool_)[None],
            'lang_tokens': processed['lang_tokens'][None],
            'lang_masks': processed['lang_masks'][None]
        }
        states = rng.uniform(-1, 1, (1, 32)).astype(np.float32)
        raw_info = {
            k: {
                'shape': list(v.shape),
                'dtype': str(v.dtype),
                'sha256': hashlib.sha256(v.tobytes()).hexdigest()
            }
            for k, v in raw.items() if k != 'task_description'
        }
        observations.append((text, inputs, states, raw_info, len(token_ids)))
    require(
        len({o[4]
             for o in observations}) >= 3, 'Require >=3 effective token lengths')
    require(
        len({32 - o[4]
             for o in observations
             if o[4] < 32}) >= 2, 'Require multiple padding lengths')
    checkpoint_hash = sha256_file(checkpoint)
    require(checkpoint_hash == CHECKPOINT_SHA256,
            'Checkpoint hash differs from verified original asset')
    host_available = psutil.virtual_memory().available
    require(host_available >= 24 * 1024**3,
            'Requires >=24 GiB available host RAM immediately before model load')
    require(torch.cuda.mem_get_info()[0] >= 12 * 1024**3,
            'Requires >=12 GiB free GPU memory immediately before model load')
    print('Preflight passed; loading original model serially', flush=True)
    model = build_vla_from_cfg(copy.deepcopy(cfg.model))
    require(
        type(model).__name__ == 'PI05FlowMatching', 'Builder returned unexpected model')
    stored = load_file(str(checkpoint), device='cpu')
    require(
        all(v.dtype == torch.bfloat16 for v in stored.values()),
        'Checkpoint storage is not uniformly BF16')
    model.load_state_dict(stored, strict=True)
    del stored
    model.eval().to(device='cuda:0', dtype=torch.bfloat16)
    gc.collect()
    require(
        all(p.dtype == torch.bfloat16 for p in model.parameters()),
        'Model parameters must be BF16')
    require(
        dict(cfg.model) == original_model_config,
        'Saved model configuration was modified')
    output.mkdir(parents=True, exist_ok=False)
    rows, records = [], []
    shape_dtype = {
        'images': ((1, 6, 224, 224), 'float32'),
        'img_masks': ((1, 2), 'bool'),
        'lang_tokens': ((1, 32), 'int64'),
        'lang_masks': ((1, 32), 'bool'),
        'noise': ((1, 10, 32), 'float32')
    }
    for obs_index, (text, cpu, states, raw_info,
                    valid_tokens) in enumerate(observations):
        for seed in SEEDS:
            sample_id = f'O{obs_index}-S{seed}'
            native = {
                k: torch.from_numpy(v.copy()).to('cuda:0')
                for k, v in cpu.items()
            }
            native['states'] = torch.from_numpy(states.copy()).to('cuda:0')
            sampler = model.sample_noise
            captured = []

            def observe_noise(shape, device):
                value = sampler(shape, device)
                captured.append(
                    value.detach().clone())  # Preserve BEFORE in-place Euler.
                return value  # Native receives its original unmodified object.

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            require('sample_noise' not in model.__dict__,
                    'Unexpected instance sampler override')
            model.sample_noise = observe_noise
            try:
                with torch.inference_mode(), torch.autocast(
                        'cuda', dtype=torch.bfloat16):
                    golden = model.predict_action(**native).detach().float().cpu()
            finally:
                del model.sample_noise  # Restore original class method even on errors.
            require(len(captured) == 1, 'Native no-noise call must sample exactly once')
            noise = captured[0]
            before = noise.clone()
            for repeat in range(2):
                with torch.inference_mode(), torch.autocast(
                        'cuda', dtype=torch.bfloat16):
                    explicit = model.predict_action(
                        **native, noise=noise.clone()).detach().float().cpu()
                require(
                    torch.equal(explicit, golden),
                    f'{sample_id}: no-noise vs explicit clone repeat {repeat} differs')
            require(
                torch.equal(noise, before),
                f'{sample_id}: immutable captured noise changed')
            require(
                tuple(golden.shape) == (1, 10, 32)
                and torch.isfinite(golden).all().item(), 'Invalid golden output')
            arrays = {**cpu, 'noise': noise.cpu().numpy().copy()}
            for key, (shape, dtype) in shape_dtype.items():
                require(arrays[key].shape == shape and str(arrays[key].dtype) == dtype,
                        f'{sample_id}: invalid {key} profile')
                require(
                    np.isfinite(arrays[key]).all(), f'{sample_id}: non-finite {key}')
            name = sample_id + '.npz'
            golden_name = sample_id + '.golden.npy'
            np.savez(output / name, **arrays)
            np.save(output / golden_name, golden.numpy(), allow_pickle=False)
            rows.append({'id': sample_id, 'inputs': name})
            records.append({
                'id':
                sample_id,
                'observation':
                obs_index,
                'seed':
                seed,
                'prompt':
                text,
                'effective_tokens':
                valid_tokens,
                'padding_tokens':
                32 - valid_tokens,
                'truncated':
                False,
                'inputs':
                name,
                'golden':
                golden_name,
                'noise_sha256':
                hashlib.sha256(arrays['noise'].tobytes()).hexdigest(),
                'inputs_sha256':
                sha256_file(output / name),
                'golden_sha256':
                sha256_file(output / golden_name),
                'input_profile': {
                    k: {
                        'shape': list(v.shape),
                        'dtype': str(v.dtype)
                    }
                    for k, v in arrays.items()
                },
                'native_states':
                states.tolist(),
                'native_states_dtype':
                'float32',
                'raw_rgb':
                raw_info,
                'rgb_generator_seed':
                1000 + obs_index,
                'synthetic':
                True,
                'sampler_calls':
                1,
                'explicit_clone_repeats':
                2,
                'native_bitwise_equal':
                True,
                'input_noise_preserved':
                True
            })
            print(
                f'{sample_id}: tokens={valid_tokens}, '
                f'padding={32-valid_tokens}; native parity PASS',
                flush=True)
    require(len(rows) == 15, 'Expected exactly 15 completed cases')
    require(
        not any(x == 'fluxhermes' or x.startswith(('fluxhermes.', 'fluxvla_export'))
                for x in sys.modules), 'Unexpected SDK import')
    verify_source(source, args.source_manifest)
    source_files = [
        inspect.getfile(type(model)),
        inspect.getfile(model.predict_action),
        inspect.getfile(model.sample_noise)
    ] + [inspect.getfile(type(t)) for t in processors]
    provenance = {
        'schema_version':
        1,
        'synthetic':
        True,
        'purpose':
        'Numerical input coverage only, not real robot data or task success rate',
        'source_root':
        str(source),
        'source_revision':
        REVISION,
        'clean_git_revision_verified':
        True,
        'source_files': {
            p: sha256_file(p)
            for p in sorted(set(source_files))
        },
        'generator_sha256':
        sha256_file(__file__),
        'python':
        sys.version,
        'executable':
        sys.executable,
        'torch':
        torch.__version__,
        'numpy':
        np.__version__,
        'gpu':
        torch.cuda.get_device_name(0),
        'gpu_free_preflight':
        gpu_free,
        'gpu_total':
        gpu_total,
        'host_available_before_load':
        host_available,
        'checkpoint':
        str(checkpoint),
        'checkpoint_sha256':
        checkpoint_hash,
        'config':
        str(config_path),
        'config_sha256':
        sha256_file(config_path),
        'tokenizer':
        str(tokenizer_path),
        'tokenizer_sha256':
        sha256_file(tokenizer_path),
        'saved_eval_transforms':
        list(cfg.eval.dataset.transforms),
        'effective_eval_transforms':
        transforms_cfg[:3],
        'deployment_token_limit':
        TOKEN_LIMIT,
        'saved_config_has_max_len':
        False,
        'original_transform_default_max_len':
        inspect.signature(type(processors[2])).parameters['max_len'].default,
        'capacity_origin':
        'Existing explicit Hermes PI05 deployment profile, NOT training max default',
        'state_handling':
        ('Synthetic already-normalized FP32 [1,32] native states; '
         'five-input deployment binds states to noise metadata (ordinary PI05 only)'),
        'sampler': ('Native predict_action with noise omitted; observer clones '
                    'original sample before Euler and returns original unchanged'),
        'golden': ('Independent original predict_action output; exact equality '
                   'against two explicit noise.clone repeats per case'),
        'parameter_dtype':
        'bfloat16',
        'autocast_dtype':
        'bfloat16',
        'allow_tf32':
        False,
        'coverage':
        coverage,
        'samples':
        records
    }
    dump(output / 'provenance.json', provenance)
    dump(output / 'validation.json',
         {'samples': rows})  # Exact public manifest schema, publish last.
    print(
        json.dumps({
            'output': str(output),
            'samples': len(rows),
            'coverage': coverage
        },
                   indent=2))


if __name__ == '__main__':
    main()
