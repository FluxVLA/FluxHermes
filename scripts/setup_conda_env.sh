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

# Keep the large TensorRT libraries on the shared data disk instead of the
# small container root filesystem, then expose them to the conda interpreter.
"${PYTHON}" -m pip install --no-cache-dir --upgrade \
  --constraint "${PROJECT_ROOT}/requirements-conda.txt" \
  --target "${TRT_TARGET}" tensorrt-cu12
printf '%s\n' "${TRT_TARGET}" > "${SITE_PACKAGES}/fluxhermes_tensorrt.pth"

"${PYTHON}" -m fluxhermes.cli.doctor

cat <<MSG

FluxHermes is ready in the conda environment:
  conda activate ${CONDA_ENV}
MSG
