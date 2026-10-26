# Finite CPU GoTrack model diagnostic

This is **not** the geometric tracking benchmark, full-video tracking, automatic
segmentation, independent attachment accuracy, live phone performance, or
`workbench_integer_metric_v1` serving admission. It performs inference only;
there is no optimizer, backward pass, training loop, or new model selection.

The adapter uses actual production `quality_gotrack.load_network`,
`TexturedRenderer`, and `GoTrackRefiner`. The public immutable artifact record is
authenticated with `quality_adaptation_export_parity.read_manifest`; its
supported `context_for_export` is passed to the unchanged atomic
`quality_gotrack_inference.install_adapted_head`. These lower-level research APIs
do not grant workbench serving admission. No existing gate is patched or skipped.

## What the two predictions mean

The first selected real HB object supplies authorized onboarding images 8 and 9,
onboarding visible-mask annotations, and known poses from preceding onboarding
images 7 and 8. Each variant gets exactly the same independent inputs/seeds.
This intentionally isolates model/refinement behavior; it is not a causal
tracker replay or evidence of automatic mask/pose initialization. The original
metric model is converted from millimetres to a vertex-coloured metre GLB,
without changing object axes.

Three variants run sequentially:

1. Frozen base with the exact saved `best.pt['head']` reference.
2. Frozen base with the exact exported trained head, installed atomically.
3. Unmodified frozen upstream baseline head.

Each uses the ordinary legacy refiner's five iterations, stopping early if its
normal solver becomes unavailable: at most **30 complete network forwards**.
Forward hooks count the actual backbone, decoder and head, and save every query,
template, rendered mask, flow and confidence tensor. Reference/export inputs
must be byte-identical with matching shape/dtype; output tolerance is fixed
before execution at `atol=1e-6, rtol=0`. Baseline versus adapted comparisons use
identical first-iteration crops, not different model-selected trajectories.
Returned candidates and solver failures are retained separately from numerical
parity. Parity does not imply good poses.

## Isolated restoration, never change the benchmark checkout

From the project checkout containing these new tools:

```bash
NEURAL=.cache/tracking-benchmark/neural-worktree
git worktree add --detach "$NEURAL" f58d532bebd8700c07fc3f529b18bd646bd14a73
cp tools/{tracking_neural_diagnostic,restore_tracking_neural_sources,test_tracking_neural_diagnostic}.py "$NEURAL/tools/"
cp bench/runtime/requirements-tracking-neural-cpu.txt "$NEURAL/bench/runtime/"
cd "$NEURAL"
python tools/restore_tracking_neural_sources.py \
  --head /absolute/path/to/user-provided/head.pt \
  --best /absolute/path/to/user-provided/best.pt \
  --download-public-base
```

Alternatively provide `--base /absolute/path/to/gotrack_checkpoint.pt` instead
of downloading. The helper preserves at least 10 GiB free disk, authenticates
all weights and pinned public source archives, and refuses an unexpected
worktree revision. It restores only three known files to exact hash-verified
CRLF bytes; global line-ending conversion is incorrect:

| File | Restored SHA-256 |
|---|---|
| `bench/runtime/adaptation-sources.json` | `44e4c8a84d602a6fbe623622a3488db014aaddf562b06748e3a13afc88693d33` |
| `bench/quality_gotrack.py` | `0b18516894690de0fbf3565d5d542ced17865c161e646369fa4f7e5412a28c29` |
| `bench/quality_runner.py` | `bca306d1342dce2566c786fed94a7564ae090e11bf42243da2ea7bc9ce841ad1` |

The installer remains unchanged LF bytes, hash
`d1cd703ffe389ea72947d86a37ddbd42a065323af35c9e73547362027d3486b3`.
The legitimate historical source-review diff is reproduced byte-for-byte by
`git diff eddcd4386b1459f1ecdf8713ee8d047055c9cc1a f58d532 -- bench/quality_gotrack.py bench/quality_runner.py`:
77,650 bytes, SHA-256
`30520de45084ce1f948929e38331eb3b4691be9a731d15381d487c54ed262398`.
This is source evidence, not a manufactured execution receipt. The restored
320-file GoTrack closure has no missing or unreviewed replacements.

## Required artifacts and licensing

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Official base | 1,608,850,339 | `f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086` |
| User's exported head | 80,165,526 | `27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311` |
| User's reference best checkpoint | 80,203,390 | `0c211531ffdee325e6504411f9dee07c83cc242cd8d048c115548aed6de2fa3b` |

The base's size/hash are independently declared in the pinned official
[Git LFS pointer](https://raw.githubusercontent.com/facebookresearch/gotrack/68f76055755f2a4a8967e13ece834f975f008bdf/gotrack_checkpoint.pt).
The helper downloads only that exact
[public base](https://media.githubusercontent.com/media/facebookresearch/gotrack/68f76055755f2a4a8967e13ece834f975f008bdf/gotrack_checkpoint.pt).
GoTrack's [official licence](https://github.com/facebookresearch/gotrack/blob/68f76055755f2a4a8967e13ece834f975f008bdf/LICENSE)
is CC BY-NC 4.0: keep this research noncommercial and retain attribution.
Model weights and third-party source archives are **not** committed here.

## CPU runtime

Use a separate virtual environment, not the geometric benchmark's NumPy 2.x
environment. The measured lock is
`bench/runtime/requirements-tracking-neural-cpu.txt`; downloaded/build wheel
hashes are recorded in `bench/runtime/tracking-neural-cpu-wheels.json`.

```bash
python -m venv .venv
.venv/bin/python -m pip install pip==25.0.1 setuptools==75.3.0 wheel==0.44.0
.venv/bin/python -m pip install torch==2.5.1+cpu torchvision==0.20.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu
```

All other packages have prebuilt wheels except exact `PyOpenGL==3.1.0`, required
by pinned `pyrender==0.1.45`. Its official PyPI sdist is pure Python; inspected
`setup.py` declares no native extensions. The source SHA-256 is
`9b47c5c3a094fa518ca88aeed35ae75834d53e4285512c61879f67a48c94ddaf`.
After fetching and checking that exact sdist, build it with no compiler allowed:

```bash
CC=/bin/false CXX=/bin/false timeout 120s .venv/bin/python -m pip wheel \
  --no-deps --no-build-isolation --wheel-dir .cache/cpu-wheels \
  /absolute/path/to/verified/PyOpenGL-3.1.0.tar.gz
.venv/bin/python -m pip install --only-binary=:all: --find-links .cache/cpu-wheels \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r bench/runtime/requirements-tracking-neural-cpu.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest tools.test_tracking_neural_diagnostic -q
```

The measured environment is Python 3.12.14, Torch 2.5.1+cpu,
Torchvision 0.20.1+cpu, NumPy 1.26.4, OpenCV 4.10.0, and software EGL
`llvmpipe (LLVM 19.1.7, 256 bits)`, Mesa 25.0.7. No CUDA/xFormers is needed.
This newly measured runtime is not asserted to be the original training runtime.

## Prepare and execute, serially after speed benchmarks

```bash
.venv/bin/python tools/tracking_neural_diagnostic.py inspect
.venv/bin/python tools/tracking_neural_diagnostic.py prepare \
  --dataset /absolute/path/to/tracking-benchmark/real-v1 \
  --output .cache/neural-diagnostic-inputs-v1
nice -n 10 .venv/bin/python tools/tracking_neural_diagnostic.py supervise \
  --plan .cache/neural-diagnostic-inputs-v1/plan.json \
  --output .cache/neural-diagnostic-run-v1 \
  --max-seconds 900 --max-rss 7516192768
```

Use fresh output directories. The parent supervisor checks aggregate process
RSS every 250 ms, terminates above 7 GiB RSS or 900 seconds, caps output at
512 MiB, and preserves 10 GiB free disk. Thus RSS is sampled, not an exact
continuous peak. Torch/OpenBLAS/OpenCV use one thread; software rendering is
explicit. Full-model inference must not overlap the geometric speed trials.
The production base loader runs inside a scoped security-only wrapper forcing
`torch.load(weights_only=True, map_location='cpu')`; unsafe pickle fallback and
additional safe-global allowlists are forbidden. Head and best are also safely
deserialized only after exact byte hashes match.

Outputs include `supervision.json`, `freeze.json`, per-variant JSON,
captured forward NPZs, `worker.log`, and `results.json` when all variants finish.
Keep raw captures local; publish only reviewed compact summaries. Inspect
`fullNetworkExecuted`, parity, returned-candidate counts, errors, and supervisor
exit separately; a completed invocation is not a quality pass.

The full workbench still requires its exact historical small-records file at
the `verify_primary_record` gate, followed by genuinely earned normal
contract/geometry/full-image parity/admission stages. The missing historical
backup and loader receipts are additional preflight diagnostics, not hard
ready-to-install or admission gates. This diagnostic never fabricates those
artifacts or changes native/live admission.
