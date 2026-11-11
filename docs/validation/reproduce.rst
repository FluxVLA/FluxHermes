PR8 full-model reproduction
==========================

Scope and recorded environment
------------------------------

The recorded runs completed export, TensorRT build, packaging, fresh-process
inference and timing for ordinary PI05 and GR00T N1.7. Both full-model numerical
matrices failed the requested tolerances. Workflow completion does not establish numerical parity. See ``full_model_results.json`` beside this
page for the separately collected results.

The measured host used Linux x86-64, an RTX 3090, Python 3.10.12,
Torch 2.6.0+cu124, torchvision 0.21.0+cu124, TensorRT 10.13.3.9,
Transformers 5.3.0 and FluxVLA commit
``249531b78ee922757a6e4b56d2f5965487f9f0fc``. FluxVLA's bundled
Transformers compatibility implementation remains part of that source revision.
The original environment used existing Python 3.10 packages plus isolated
package overlays. These instructions have not been tested in a fresh environment;
the parameterized fixture generators were rerun on the recorded GPU host using
an explicitly checksum-verified source snapshot. Both sets of 15 inputs and native
golden outputs matched the original fixtures exactly. See
``reproduction_check.json`` for counts and helper source fingerprints.

Setup and external assets
-------------------------

Use a dedicated Python 3.10 environment and a CUDA 12.4-compatible compiler/toolkit
for FluxVLA's native extensions. The Hermes export extra is not a complete
FluxVLA environment. FluxVLA's setup script declares no Python dependencies and
builds three CUDA extensions; its original dependency instructions and Flash
Attention requirements still apply. Do not put Python 3.12 wheels on a Python
3.10 import path. Check actual imports, not only pip distribution metadata: the
recorded host also had a different TensorRT installation outside the selected
10.13 overlay.

Set paths to local checkouts and externally obtained assets. Generated cases,
weights, engines and reports stay outside the repository::

    export HERMES_ROOT="$PWD"
    export FLUXVLA_ROOT="/path/to/FluxVLA"
    export VALIDATION_ROOT="/path/to/new-validation-directory"
    export PYTHON="/path/to/python3.10"
    export PI05_CONFIG="/path/to/pi05/config.json"
    export PI05_CHECKPOINT="/path/to/pi05/checkpoints/step-038064-epoch-24-loss=0.0170.safetensors"
    export PI05_TOKENIZER="/path/to/pi05/tokenizer/tokenizer.model"
    export GROOT_N17_CHECKPOINT="/path/to/n17/libero_10"
    export N17_TOKENIZER="/path/to/n17-tokenizer"
    export GROOT_N17_CONFIG="$FLUXVLA_ROOT/configs/gr00tn17/gr00tn17_qwen3vl_2b_libero_10_full_finetune.py"
    mkdir -p "$VALIDATION_ROOT"
    git -C "$FLUXVLA_ROOT" checkout 249531b78ee922757a6e4b56d2f5965487f9f0fc
    "$PYTHON" -m pip install -e "$HERMES_ROOT[export]" psutil

Preserve the Torch/TensorRT versions above when resolving dependencies. Install
FluxVLA's dependencies and compile its extensions with this same interpreter
and Torch ABI, following its pinned source instructions. This is a setup
constraint, not a tested lockfile or a claim that the one pip command installs
the whole stack. Once the original native model imports successfully::

    export PYTHONPATH="$FLUXVLA_ROOT:$HERMES_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export JE_ARROW_MALLOC_CONF=background_thread:false
    export PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1
    "$PYTHON" - <<'PY'
    import torch, tensorrt, transformers
    import fluxvla
    from fluxvla.engines import build_vla_from_cfg, build_transform_from_cfg
    assert torch.__version__ == '2.6.0+cu124', torch.__version__
    assert tensorrt.__version__ == '10.13.3.9', tensorrt.__version__
    assert transformers.__version__ == '5.3.0', transformers.__version__
    print(fluxvla.__file__, tensorrt.__file__)
    PY

Set ``JE_ARROW_MALLOC_CONF=background_thread:false`` before starting **each**
compile, generator, run and benchmark process, including subprocesses. On the
recorded host this Arrow 25 allocator setting avoided a native crash when loading
the reference model after TensorRT initialization. It is a process environment
workaround, not an engine change.

The N1.7 weights are from ``nvidia/GR00T-N1.7-LIBERO`` at HF revision
``2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21``, subdirectory ``libero_10``.
The tokenizer files (``merges.txt``, ``tokenizer.json``,
``tokenizer_config.json``, ``vocab.json``) match
``nvidia/Cosmos-Reason2-2B`` revision
``9ce19a195e423419c349abfc86fd07178b230561`` and
``Qwen/Qwen3-VL-2B-Instruct`` revision
``89644892e4d85e24eaac8bacfd4f463576704203``. Those four files had identical
Git blob hashes in the recorded metadata. Download before enabling offline mode::

    hf download nvidia/GR00T-N1.7-LIBERO \
      --revision 2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21 \
      --include 'libero_10/*' --local-dir "$(dirname "$GROOT_N17_CHECKPOINT")"
    hf download Qwen/Qwen3-VL-2B-Instruct \
      --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
      --include merges.txt tokenizer.json tokenizer_config.json vocab.json \
      --local-dir "$N17_TOKENIZER"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

PI05 requires the saved training configuration, original Paligemma tokenizer and
external fine-tuned checkpoint named above. The checkpoint SHA-256 is
``4a79542cbb79b75deea3804d91f87f9c2180ff704f0cd1988469ee72655fbe15``;
the generator checks it. No public HF repository/revision for this fine-tuned
checkpoint was established in the recorded evidence. Supplying that exact asset
is a prerequisite for reproducing the PI05 numbers; substitute weights are a
separate experiment. Neither model's weights or tokenizer are bundled here.

Generate actual fixtures
------------------------

Run serially with at least 24 GiB available host RAM and 12 GiB free GPU memory
before model loading. Keep more than 22 GiB free disk for each build and retain
at least 4 GiB host **available** memory throughout. The guard polls available
memory and process-tree RSS every 0.1 seconds, terminates its owned process group
below the reserve or above 40 GiB RSS, and records the exit status. It is not a
hard allocation limit. Use fresh output directories after an interrupted run::

    "$PYTHON" scripts/validation/guarded_cli.py \
      --report "$VALIDATION_ROOT/prepare-pi05-guard.json" --reserve-gib 4 \
      --command "$PYTHON" scripts/validation/prepare_pi05_cases.py \
      --root "$VALIDATION_ROOT" --fluxvla-root "$FLUXVLA_ROOT" \
      --config "$PI05_CONFIG" --checkpoint "$PI05_CHECKPOINT" \
      --tokenizer-root "$PI05_TOKENIZER"
    "$PYTHON" scripts/validation/guarded_cli.py \
      --report "$VALIDATION_ROOT/prepare-n17-guard.json" --reserve-gib 4 \
      --command "$PYTHON" scripts/validation/prepare_n17_cases.py \
      --root "$VALIDATION_ROOT" --fluxvla-root "$FLUXVLA_ROOT" \
      --checkpoint "$GROOT_N17_CHECKPOINT" --tokenizer-root "$N17_TOKENIZER"

Each generator creates five synthetic RGB/prompt observations times seeds
7, 17 and 29, plus model-ready NPZ inputs, independent native golden actions,
provenance and a public ``validation.json`` manifest. No private input files or
network access are needed once the external model assets are present. The source
checkout must have the pinned HEAD and unchanged tracked files; extension build
outputs may be untracked. For a source snapshot without Git metadata, both generators
accept an explicit ``--source-manifest`` with SHA-256 hashes of all 715 tracked files.
Create that manifest from a trusted checkout of the pinned revision before copying
its source files::

    "$PYTHON" - <<'PY'
    import hashlib, json, os, subprocess
    from pathlib import Path
    source = Path(os.environ['FLUXVLA_ROOT'])
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == '249531b78ee922757a6e4b56d2f5965487f9f0fc'
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    names = subprocess.check_output(['git', '-C', str(source), 'ls-files', '-z']).decode().split('\0')[:-1]
    files = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names}
    assert len(files) == 715
    output = Path(os.environ['VALIDATION_ROOT']) / 'source-manifest.json'
    with output.open('x') as handle:
        json.dump({'revision': revision, 'files': files}, handle, indent=2)
    PY

Pass the manifest explicitly for archive-based runs; a failed Git check does not
select this mode automatically. Do not run Python with ``-O``: N1.7 uses assertions
for its fixture checks.

PI05 uses original saved eval image/token transforms, a deployment capacity of
32 tokens, varied padding and a whole-prompt search for the zero-padding boundary.
It captures original ``sample_noise`` before in-place Euler updates and checks
native no-noise inference against two explicit cloned-noise calls per case.
Its five-input deployment binds states to noise metadata only for ordinary PI05;
this is not a PI0 state-binding recipe.

N1.7 uses original image/token transforms, original common-left-padding trim
and original ``get_rope_index``. It searches for a real-tokenizer length-180
boundary, captures the original action sampler and checks bitwise native parity
with explicit noise and positions. Recorded effective lengths were
138, 142, 148, 151 and 180. The generators contain no neural-network implementation.
Their normalized synthetic states and actions are not robot commands, real
observations, denormalized outputs, or evidence of task success.

Compile through the public CLI
------------------------------

The diagnostic config reuses the shipped recipes, sets
``codebase_config.isolate_export=True``, tactic DRAM 8 GiB and optimization level
0, and preserves BF16, strong typing, no TF32 and the original tolerances
(``atol=rtol=0.02``, cosine at least 0.999). ``fail_on_error=False`` permits
packaging a failed numerical result; it does not mark the bundle validated::

    export PI05_DEPLOY_SAMPLE="$VALIDATION_ROOT/cases/pi05-01/O2-S7.npz"
    export PI05_VALIDATION_MANIFEST="$VALIDATION_ROOT/cases/pi05-01/validation.json"
    export PI05_WORK_DIR="$VALIDATION_ROOT/work/pi05"
    export PI05_BUNDLE_DIR="$VALIDATION_ROOT/bundles/pi05"
    export GROOT_N17_DEPLOY_SAMPLE="$VALIDATION_ROOT/cases/gr00t-n17-01/o2-s7.npz"
    export GROOT_N17_VALIDATION_MANIFEST="$VALIDATION_ROOT/cases/gr00t-n17-01/validation.json"
    export GROOT_N17_WORK_DIR="$VALIDATION_ROOT/work/n17"
    export GROOT_N17_BUNDLE_DIR="$VALIDATION_ROOT/bundles/n17"
    for MODEL in pi05 n17; do
      export MODEL
      "$PYTHON" scripts/validation/guarded_cli.py \
        --report "$VALIDATION_ROOT/compile-$MODEL.json" --reserve-gib 4 \
        --command "$PYTHON" -m fluxhermes.cli.main compile \
        scripts/validation/experiment.py --execute || exit $?
    done

Inspect each bundle's validation result before inference. Default ``run`` and
``benchmark`` reject unvalidated bundles before inference. For example, the
recorded PI05 bundle requires a diagnostic override::

    "$PYTHON" -m fluxhermes.cli.main run "$PI05_BUNDLE_DIR" \
      --inputs "$PI05_DEPLOY_SAMPLE" --output "$VALIDATION_ROOT/default-gate.npz"

For a failed bundle this command must exit nonzero without creating that output.
Do not delete or change validation fields to bypass the gate.

Fresh-process checks and timing
-------------------------------

Run the following for PI05, then set ``CASES`` to
``$VALIDATION_ROOT/cases/gr00t-n17-01``, ``BUNDLE`` to
``$GROOT_N17_BUNDLE_DIR``, and ``REP=o2-s7`` for N1.7. Each public CLI call loads
a fresh engine. The override is explicitly diagnostic::

    export CASES="$VALIDATION_ROOT/cases/pi05-01"
    export BUNDLE="$PI05_BUNDLE_DIR"
    export REP=O2-S7
    export OUTPUT="$VALIDATION_ROOT/checks-$REP"
    mkdir "$OUTPUT"
    for input in "$CASES"/*.npz; do
      id=$(basename "$input" .npz)
      "$PYTHON" -m fluxhermes.cli.main run "$BUNDLE" --inputs "$input" \
        --output "$OUTPUT/$id.npz" --allow-unvalidated || exit $?
    done
    "$PYTHON" - <<'PY'
    import json, os
    from pathlib import Path
    import numpy as np
    from fluxhermes.validation.parity import compare_outputs
    cases, output = Path(os.environ['CASES']), Path(os.environ['OUTPUT'])
    rows = []
    for sample in json.loads((cases / 'validation.json').read_text())['samples']:
        name = sample['id']
        golden = np.load(cases / (name + '.golden.npy'), allow_pickle=False)
        with np.load(output / (name + '.npz'), allow_pickle=False) as actual:
            result = compare_outputs({'actions': golden}, {'actions': actual['actions']},
                                     atol=0.02, rtol=0.02, min_cosine_similarity=0.999)
        rows.append({'id': name, 'comparison': result})
    report = {'workflow_completed': len(rows) == 15 and all(
                  r['comparison']['outputs']['actions'].get('finite', False) for r in rows),
              'numerical_passed': len(rows) == 15 and all(r['comparison']['passed'] for r in rows),
              'samples': rows}
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['numerical_passed'] else 1)
    PY
    "$PYTHON" -m fluxhermes.cli.main benchmark "$BUNDLE" \
      --inputs "$CASES/$REP.npz" --warmup 10 --repeats 100 \
      --allow-unvalidated > "$OUTPUT/benchmark.txt"

Numerical failure is expected for the recorded engines. Inspect the comparison
report separately from command completion. Benchmark stdout can contain TensorRT
log prefixes before its JSON object. Timing covers model-ready NumPy inputs,
device transfer, inference and synchronization; loading, raw preprocessing and
robot control are excluded.

The independently collected original evidence used 45 calls per model in one
fresh public ``RobotPolicy`` process: 15 forward calls with immediate repeats
(30), then reverse inter-case replay (15). It also checked NumPy input
immutability, replay bitwise stability, checksum failures, default validation
gates, a representative fresh CLI run and CLI benchmark. The shorter 15-case
fresh-process loop above reproduces numerical checks, not the original
same-process replay/immutability matrix or its negative tests. It must not be
reported as a rerun of that 45-call evidence.
