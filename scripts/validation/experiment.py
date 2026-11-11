"""Diagnostic packaging of the shipped recipe; select MODEL=pi05 or n17."""
import os
import runpy
from pathlib import Path

root = Path(__file__).resolve().parents[2]
recipes = {
    'pi05': ('configs/pi05/pi05_libero_bf16.py', 'PI05_VALIDATION_MANIFEST'),
    'n17': ('configs/gr00t/gr00t_n17_libero_bf16.py', 'GROOT_N17_VALIDATION_MANIFEST'),
}
recipe, manifest = recipes[os.environ['MODEL']]
deployment = runpy.run_path(str(root / recipe))['deployment']
deployment['validation']['samples_manifest'] = os.environ[manifest]
deployment['validation']['fail_on_error'] = False
deployment['codebase_config']['isolate_export'] = True
deployment['backend_config']['tactic_dram_size_gib'] = 8
deployment['backend_config']['builder_optimization_level'] = 0
