"""Execute a config-defined isolated export chain and snapshot its inputs."""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from .contracts import DeploymentConfig
from .pipeline import CompilePipeline
from .utils import write_json


def main():
    config_path, run_id = sys.argv[1:]
    document = json.loads(Path(config_path).read_text(encoding='utf-8'))
    original = Path(document['config_path']) if document.get('config_path') else None
    config = DeploymentConfig.from_dict(document, config_path=original)
    state = dict(run_id=run_id, pid=os.getpid(), passed=False, stages=[])
    pipeline = None
    try:
        pipeline = CompilePipeline(config)
        data = pipeline.transforms(pipeline.new_context(run_id))
        prepared, processor = data['prepared'], data['task_processor']
        import numpy as np
        import torch

        arrays = {}
        state['input_dtypes'] = {}
        for name, value in prepared.inputs.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    'Isolated export requires tensor representative inputs.')
            state['input_dtypes'][name] = str(value.dtype)
            value = value.detach().cpu()
            if value.dtype == torch.bfloat16:
                value = value.float()
            arrays[name] = value.numpy()
        np.savez(config.work_dir / 'export-inputs.npz', **arrays)
        state.update(
            passed=True,
            graph_path=str(data['graph_path']),
            source_metadata={
                **prepared.metadata,
                **processor.runtime_metadata()
            },
            precision_ready=data.get('precision_ready', False),
            quantized=data.get('quantized', False))
    except Exception as error:
        state['error'] = dict(
            type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc())
        raise
    finally:
        if pipeline is not None:
            state['stages'] = pipeline.pipeline_manager.report()
        write_json(config.work_dir / 'export-stage.json', state)


if __name__ == '__main__':
    main()
