#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${FLUXVLA_CONDA_ENV:-fluxvla}"
TRT_TARGET="${FLUXHERMES_TRT_TARGET:-/mnt/workspace/fluxhermes-trt}"
TMP_ROOT="${FLUXHERMES_TMPDIR:-/mnt/workspace/fluxhermes-tmp}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_PREFIX="$(conda run -n "${CONDA_ENV}" python -c 'import sys; print(sys.prefix)')"
PYTHON="${CONDA_PREFIX}/bin/python"
SITE_PACKAGES="$(${PYTHON} -c 'import site; print(site.getsitepackages()[0])')"

mkdir -p "${TMP_ROOT}" "${TRT_TARGET}"
export TMPDIR="${TMP_ROOT}"

# FluxHermes reuses the full FluxVLA conda environment so both model families
# keep their original Python/CUDA dependencies.
"${PYTHON}" -m pip install --no-cache-dir \
  --constraint "${PROJECT_ROOT}/requirements-conda.txt" \
  -e "${PROJECT_ROOT}[export,quantization,dev]"

# Reinstall the numerical ABI set together. This prevents stale NumPy 2.x
# extension files from being mixed with the pinned NumPy 1.26 runtime.
"${PYTHON}" -m pip install --no-cache-dir --force-reinstall \
  'numpy==1.26.4' \
  'scipy==1.15.3' \
  'protobuf==5.29.6' \
  'ml-dtypes==0.2.0'

# Keep the large TensorRT libraries on the shared data disk instead of the
# small container root filesystem. Skip the download when the exact target is
# already healthy.
if ! PYTHONPATH="${TRT_TARGET}" "${PYTHON}" -c \
  'import tensorrt; assert tensorrt.__version__ == "11.4.0.106"' \
  >/dev/null 2>&1; then
  "${PYTHON}" -m pip install --no-cache-dir --upgrade \
    --constraint "${PROJECT_ROOT}/requirements-conda.txt" \
    --target "${TRT_TARGET}" tensorrt-cu12
fi

# A plain path line is appended after site-packages and can leave an older
# TensorRT binding active. Insert the isolated target at the front instead.
printf '%s\n' \
  "import sys; sys.path.insert(0, r'${TRT_TARGET}')" \
  > "${SITE_PACKAGES}/fluxhermes_tensorrt.pth"

"${PYTHON}" - <<'PY'
import numpy
import scipy
import tensorrt

assert numpy.__version__ == '1.26.4', numpy.__version__
assert scipy.__version__ == '1.15.3', scipy.__version__
assert tensorrt.__version__ == '11.4.0.106', tensorrt.__version__
PY
"${PYTHON}" -m pip check
"${PYTHON}" -m fluxhermes.cli.doctor

cat <<MSG

FluxHermes is ready in the conda environment:
  conda activate ${CONDA_ENV}
MSG
