#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${FLUXVLA_CONDA_ENV:-fluxvla}"
FLUXVLA_ROOT="${FLUXVLA_ROOT:?Set FLUXVLA_ROOT to the FluxVLA source tree}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${FLUXHERMES_STATE_DIR:-${HOME}/.local/state/fluxhermes}"

CONDA_PREFIX="$(conda run -n "${CONDA_ENV}" python -c 'import sys; print(sys.prefix)')"
PYTHON="${CONDA_PREFIX}/bin/python"

if [[ ! -f "${FLUXVLA_ROOT}/setup.py" ]]; then
  echo "FluxVLA setup.py not found: ${FLUXVLA_ROOT}/setup.py" >&2
  exit 2
fi
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA compiler not found: ${CUDA_HOME}/bin/nvcc" >&2
  exit 2
fi

"${PYTHON}" - <<'PY'
from packaging.version import Version
import torch

version = Version(torch.__version__.split('+', 1)[0])
if version.release[:2] != (2, 8):
    raise SystemExit(f'Expected PyTorch 2.8.x, found {torch.__version__}')
if torch.version.cuda != '12.8':
    raise SystemExit(f'Expected PyTorch CUDA 12.8, found {torch.version.cuda}')
if not torch.cuda.is_available():
    raise SystemExit('CUDA is not available to PyTorch')
if not torch._C._GLIBCXX_USE_CXX11_ABI:
    raise SystemExit('PyTorch must use the CXX11 ABI for this environment')
print('PyTorch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(0))
PY

export FLUXVLA_CONDA_ENV="${CONDA_ENV}"
bash "${PROJECT_ROOT}/scripts/setup_conda_env.sh"

if ! "${PYTHON}" - <<'PY' >/dev/null 2>&1
import torch
import flash_attn
from flash_attn import flash_attn_func

q = torch.randn(1, 16, 2, 64, device='cuda', dtype=torch.bfloat16)
y = flash_attn_func(q, q, q)
torch.cuda.synchronize()
assert torch.isfinite(y).all()
PY
then
  "${PYTHON}" -m pip uninstall -y flash-attn || true
  export CUDA_HOME
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export FLASH_ATTENTION_FORCE_BUILD=TRUE
  export FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE
  export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80}"
  export MAX_JOBS="${MAX_JOBS:-2}"
  export NVCC_THREADS="${NVCC_THREADS:-2}"
  "${PYTHON}" -m pip install --no-cache-dir --no-build-isolation \
    'flash-attn==2.8.3'
fi

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-2}"
(
  cd "${FLUXVLA_ROOT}"
  "${PYTHON}" setup.py build_ext --inplace --force
)

export FLUXHERMES_STATE_DIR_RESOLVED="${STATE_DIR}"
PYTHONPATH="${FLUXVLA_ROOT}:${PYTHONPATH:-}" "${PYTHON}" - <<'PY'
import json
import platform
from pathlib import Path

import flash_attn
import ml_dtypes
from google.protobuf import __version__ as protobuf_version
import numpy
import onnx
import onnxruntime
import scipy
import tensorrt
import torch
from fluxvla.engines import build_vla_from_cfg  # noqa: F401
from fluxvla.ops.cuda.gemma_rotary_embedding import gemma_rotary_embedding_ext
from fluxvla.ops.cuda.matmul_bias import matmul_bias_ext
from fluxvla.ops.cuda.rotary_pos_embedding import rotary_pos_embedding_ext

report = {
    'python': platform.python_version(),
    'glibc': platform.libc_ver()[1],
    'torch': torch.__version__,
    'torch_cuda': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0),
    'numpy': numpy.__version__,
    'scipy': scipy.__version__,
    'protobuf': protobuf_version,
    'ml_dtypes': ml_dtypes.__version__,
    'onnx': onnx.__version__,
    'onnxruntime': onnxruntime.__version__,
    'tensorrt': tensorrt.__version__,
    'flash_attn': flash_attn.__version__,
    'native_extensions': [
        gemma_rotary_embedding_ext.__file__,
        rotary_pos_embedding_ext.__file__,
        matmul_bias_ext.__file__,
    ],
}
state_dir = Path(__import__('os').environ['FLUXHERMES_STATE_DIR_RESOLVED'])
state_dir.mkdir(parents=True, exist_ok=True)
path = state_dir / 'groot-n17-torch28-environment.json'
path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2, sort_keys=True))
print('Environment report:', path)
PY
