"""Internal opt-in export worker; the public entry remains `compile --execute`."""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from .config import DeploymentConfig
from .pipeline import CompilePipeline
from .utils import write_json


def main():
    config_path, run_id = sys.argv[1:]
    data = json.loads(Path(config_path).read_text(encoding='utf-8'))
    original_path = Path(data['config_path']) if data.get('config_path') else None
    config = DeploymentConfig.from_dict(data, config_path=original_path)
    pipeline = CompilePipeline(config)
    state = dict(run_id=run_id, pid=os.getpid(), passed=False)
    try:
        device = str(
            config.codebase.get('device', config.source.get('device', 'cuda:0')))
        processor, prepared, graph_path = pipeline._prepare_ir(device)
        import numpy as np
        arrays = {}
        import torch
        state['input_dtypes'] = {}
        for name, value in prepared.inputs.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    'Isolated export requires tensor representative inputs.')
            state['input_dtypes'][name] = str(value.dtype)
            # Every BF16 value is exactly representable in FP32; NumPy lacks BF16.
            value = value.detach().cpu()
            if value.dtype == torch.bfloat16:
                value = value.float()
            arrays[name] = value.numpy()
        np.savez(config.work_dir / 'export-inputs.npz', **arrays)
        metadata = dict(prepared.metadata)
        metadata.update(processor.runtime_metadata())
        state.update(passed=True, graph_path=str(graph_path), source_metadata=metadata)
    except Exception as error:
        state['error'] = dict(
            type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc())
        raise
    finally:
        state['stages'] = pipeline.pipeline_manager.report()
        write_json(config.work_dir / 'export-stage.json', state)


if __name__ == '__main__':
    main()
