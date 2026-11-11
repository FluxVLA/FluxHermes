"""Generate synthetic N1.7 cases using original transforms and predict_action."""
import argparse
import copy
import json
import sys
import traceback
from pathlib import Path

from prepare_pi05_cases import verify_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--fluxvla-root', type=Path, required=True)
    parser.add_argument(
        '--source-manifest',
        type=Path,
        help='Explicit SHA256 manifest for a source snapshot without Git')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--tokenizer-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    source = args.fluxvla_root.expanduser().resolve()
    verify_source(source, args.source_manifest)
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from fluxvla.engines import build_vla_from_cfg
    from fluxvla.engines.utils.builder import build_transform_from_cfg
    from mmengine.config import Config
    from PIL import Image
    folder = root / 'cases/gr00t-n17-01'
    folder.mkdir(parents=True, exist_ok=False)
    report_path = folder / 'provenance.json'
    manifest = {'samples': []}
    report = {
        'passed':
        False,
        'cases': [],
        'fixture_kind': ('synthetic RGB and normalized states, original image/token '
                         'transforms; not task success')
    }
    try:
        import psutil
        assert psutil.virtual_memory().available >= 24 * 1024**3
        assert torch.cuda.mem_get_info()[0] >= 12 * 1024**3
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        cfg = Config.fromfile(
            str(source /
                'configs/gr00tn17/gr00tn17_qwen3vl_2b_libero_10_full_finetune.py'))
        mc = copy.deepcopy(cfg.model)
        mc['model_path'] = str(args.checkpoint.expanduser().resolve())
        model = build_vla_from_cfg(mc)
        model.from_pretrained()
        model.eval().to('cuda', dtype=torch.bfloat16)
        backbone = model.vlm_backbone
        vlm = backbone.vlm
        head = model.vla_head
        report['rope_implementation'] = vlm.model.get_rope_index.__func__.__module__
        prompts = [
            'Move.', 'Pick up the black bowl.',
            'Place the black bowl beside the white plate on the table.',
            ('Open the drawer and put the small bowl inside it before '
             'closing the drawer.'),
            ('Pick up the black bowl from the left side of the table, '
             'move it carefully past the plate, and place it inside the drawer.')
        ]
        for index, prompt in enumerate(prompts):
            rng = np.random.default_rng(100 + index)
            data = dict(
                pixel_values=[
                    Image.fromarray(
                        rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
                    for _ in range(2)
                ],
                img_masks=[True, True],
                task_description=prompt,
                states=rng.uniform(-0.1, 0.1, (1, 132)).astype(np.float32),
                embodiment_ids=np.asarray(
                    model.embodiment_id_map[model.active_embodiment_key],
                    dtype=np.int64))
            for tc in cfg.eval.dataset.transforms:
                if tc['type'] not in ('GrootN17ImageAugmentation',
                                      'QWen2VLImageTransform',
                                      'ProcessPromptsWithImage'):
                    continue
                tc = copy.deepcopy(tc)
                if tc['type'] == 'ProcessPromptsWithImage':
                    tc['tokenizer']['model_path'] = str(
                        args.tokenizer_root.expanduser().resolve())
                transform = build_transform_from_cfg(tc)
                if index == 4 and tc['type'] == 'ProcessPromptsWithImage':
                    for repetitions in range(50):
                        boundary_prompt = ('Pick up the black bowl' +
                                           ' carefully' * repetitions)
                        candidate_data = transform(
                            dict(data, task_description=boundary_prompt))
                        if int(np.asarray(candidate_data['lang_masks']).sum()) == 180:
                            data = candidate_data
                            prompt = boundary_prompt
                            break
                    else:
                        raise AssertionError(
                            'Could not construct real-tokenizer zero-pad boundary')
                else:
                    data = transform(data)
            inputs = dict(
                images=torch.as_tensor(data['pixel_values']).unsqueeze(0).cuda(),
                image_grid_thw=torch.as_tensor(
                    data['image_grid_thw']).unsqueeze(0).cuda(),
                lang_tokens=torch.as_tensor(data['lang_tokens']).unsqueeze(0).cuda(),
                lang_masks=torch.as_tensor(data['lang_masks']).unsqueeze(0).cuda(),
                states=torch.as_tensor(data['states']).unsqueeze(0).bfloat16().cuda(),
                embodiment_ids=torch.as_tensor(
                    data['embodiment_ids']).reshape(1).int().cuda())
            trimmed, mask = backbone._trim_common_left_padding(
                inputs['lang_tokens'], inputs['lang_masks'])
            assert bool(mask.bool().all())
            positions, deltas = vlm.model.get_rope_index(
                trimmed,
                image_grid_thw=inputs['image_grid_thw'].flatten(0, 1),
                attention_mask=mask)
            for seed in (7, 17, 29):
                with torch.inference_mode(), torch.autocast(
                        'cuda', dtype=torch.bfloat16):
                    reference = model.predict_action(**inputs, seed=seed)
                    sampled = []
                    original = head._sample_initial_actions
                    assert '_sample_initial_actions' not in head.__dict__

                    def observe(*args, **kwargs):
                        noise = original(*args, **kwargs)
                        sampled.append(noise.detach().clone())
                        return noise

                    head._sample_initial_actions = observe
                    try:
                        observed = model.predict_action(**inputs, seed=seed)
                    finally:
                        del head._sample_initial_actions
                    assert len(sampled) == 1 and torch.equal(reference, observed)
                    noise = sampled[0]
                    calls = []

                    def supply(*args, **kwargs):
                        calls.append(1)
                        return noise.clone()

                    def positions_hook(module, args, kwargs):
                        assert kwargs.get('position_ids') is None
                        return (args, dict(kwargs, position_ids=positions))

                    handle = vlm.register_forward_pre_hook(
                        positions_hook, with_kwargs=True)
                    head._sample_initial_actions = supply
                    try:
                        projected = dict(
                            inputs,
                            lang_tokens=trimmed,
                            lang_masks=torch.ones_like(trimmed))
                        candidate = model.predict_action(**projected)
                    finally:
                        handle.remove()
                        del head._sample_initial_actions
                    equal = torch.equal(reference, candidate)
                    item = dict(
                        observation=index,
                        seed=seed,
                        prompt=prompt,
                        effective_length=int(trimmed.shape[1]),
                        padding=int(inputs['lang_tokens'].shape[1] - trimmed.shape[1]),
                        noise_dtype=str(noise.dtype),
                        sampler_calls=len(calls),
                        bitwise_equal=equal,
                        max_abs=float((reference - candidate).abs().max()))
                    report['cases'].append(item)
                    print(json.dumps(item), flush=True)
                    assert equal and len(calls) == 1
                    case_id = f'o{index}-s{seed}'
                    deployed = dict(
                        images=inputs['images'],
                        lang_tokens=trimmed,
                        position_ids=positions,
                        states=inputs['states'].float(),
                        embodiment_ids=inputs['embodiment_ids'],
                        noise=noise)
                    np.savez(
                        folder / f'{case_id}.npz', **{
                            k: v.detach().cpu().numpy()
                            for k, v in deployed.items()
                        })
                    np.save(folder / f'{case_id}.golden.npy',
                            reference.detach().cpu().numpy())
                    manifest['samples'].append(
                        dict(id=case_id, inputs=f'{case_id}.npz'))
        assert len({x['effective_length'] for x in report['cases']}) >= 3
        assert any((x['padding'] == 0 for x in report['cases']))
        (folder / 'validation.json').write_text(json.dumps(manifest, indent=2) + '\n')
        verify_source(source, args.source_manifest)
        report['passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
