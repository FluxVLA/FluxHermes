# GR00T N1.7 export environment (PyTorch 2.8)

This workflow is validated on an RTX 3090 host with glibc 2.31.

## Validated versions

- Python 3.10.12
- PyTorch 2.8.0+cu128
- FlashAttention 2.8.3 (locally compiled)
- NumPy 1.26.4
- SciPy 1.15.3
- ONNX 1.18.0
- ONNX Runtime 1.23.2
- TensorRT 11.4.0.106
- CUDA driver runtime reported by PyTorch: 12.8
- Local CUDA compiler: 12.2

The CUDA compiler/PyTorch CUDA minor-version warning is expected on this host.
FlashAttention and the FluxVLA native extensions were compiled and executed on
the RTX 3090 after installation.

## Prepare the environment

```bash
export CONDA_ROOT=/path/to/miniconda3
export FLUXHERMES_ROOT=/path/to/FluxHermes
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate fluxvla
cd "${FLUXHERMES_ROOT}"

export FLUXVLA_ROOT=/path/to/FluxVLA
export FLUXVLA_CONDA_ENV=fluxvla
export FLUXHERMES_TRT_TARGET="${HOME}/.local/fluxhermes-trt-11.4"
export FLUXHERMES_TMPDIR="${HOME}/.cache/fluxhermes-tmp"
export FLUXHERMES_STATE_DIR="${HOME}/.local/state/fluxhermes"

bash scripts/setup_groot_n17_torch28_env.sh
```

The setup script verifies the exact PyTorch/CUDA ABI, normalizes the numerical
Python packages, installs the isolated TensorRT runtime, validates or builds
FlashAttention, rebuilds all three FluxVLA C++/CUDA extensions, imports
FluxVLA, and writes an environment report.

## Compile the N1.7 checkpoint

```bash
export FLUXHERMES_ROOT=/path/to/FluxHermes
export FLUXVLA_ROOT=/path/to/FluxVLA
export GROOT_N17_CONFIG=/path/to/config.json
export GROOT_N17_CHECKPOINT=/path/to/checkpoint.safetensors
export GROOT_N17_DEPLOY_SAMPLE=/path/to/groot_n17_deploy_sample.npz
export GROOT_N17_WORK_DIR=/path/to/work
export GROOT_N17_BUNDLE_DIR=/path/to/bundle
export GROOT_N17_ISOLATE_EXPORT=true
export PYTHONPATH="$FLUXHERMES_ROOT:$FLUXVLA_ROOT:${PYTHONPATH:-}"

# Use this only when work and bundle directories are on the same filesystem
# and disk capacity cannot hold a second 7 GB engine copy.
export FLUXHERMES_BUNDLE_FILE_MODE=hardlink

python scripts/compile.py \
  configs/gr00t/gr00t_n17_libero_bf16.py \
  --execute 2>&1 | tee "${GROOT_N17_WORK_DIR}/compile.log"
```

The hardlink mode is opt-in. Do not modify or overwrite
`work/model.plan` after packaging because it shares its inode with the bundle
engine. Use the default `copy` mode when an independent bundle copy is needed.

## Verify and run

```bash
python scripts/inspect_bundle.py "${GROOT_N17_BUNDLE_DIR}"

python scripts/run_bundle.py \
  "${GROOT_N17_BUNDLE_DIR}" \
  --inputs "${GROOT_N17_DEPLOY_SAMPLE}" \
  --output "${GROOT_N17_WORK_DIR}/actions.npz"
```

The validated output contract is `actions: float32[1, 40, 132]`.
