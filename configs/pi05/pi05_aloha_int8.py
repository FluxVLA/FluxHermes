"""PI0.5 ALOHA INT8 Q/DQ deployment template."""

from pathlib import Path


_base = Path(__file__).with_name('pi05_aloha_fp16.py')
namespace = {}
exec(compile(_base.read_text(encoding='utf-8'), str(_base), 'exec'), namespace)
deployment = namespace['deployment']
deployment = {
    **deployment,
    'name': 'pi05-aloha-int8',
    'work_dir': '../../work_dirs/pi05-aloha-int8',
    'codebase_config': {
        **deployment['codebase_config'],
        'source': {
            **deployment['codebase_config']['source'],
            'calibration_inputs': '$PI05_CALIBRATION_NPZ_DIR',
        },
    },
    'quantization': {
        'mode': 'int8',
        'calibration_method': 'percentile',
        'per_channel': True,
        'use_external_data_format': True,
        'activation_symmetric': True,
        'weight_symmetric': True,
        # Exclude numerically sensitive or unsupported nodes here.
        'nodes_to_exclude': [],
    },
    'backend_config': {
        **deployment['backend_config'],
        'precision': 'int8',
        'timing_cache': '../../work_dirs/timing/pi05-aloha-int8.timing',
    },
    'validation': {
        **deployment['validation'],
        'atol': 8e-2,
        'rtol': 8e-2,
        'min_cosine_similarity': 0.98,
    },
    'package': {
        **deployment['package'],
        'output_dir': '../../artifacts/pi05-aloha-int8',
    },
}
