# Pinned SAM 2.1 CPU runtime (recorded, offline only)

This extends the isolated GoTrack diagnostic environment; it does not replace the
older measured `requirements-tracking-neural-cpu.txt` lock. The new complete
35-package runtime is `bench/runtime/requirements-tracking-sam-cpu.txt`.
`bench/runtime/tracking-sam-cpu-setup.json` records all 35 wheel hashes, the two
new pure-Python source-build hashes, source/checkpoint hashes and setup scope.
No third-party weights or source archives are committed.

## Public assets and license

Use the isolated worktree described in [the GoTrack setup](tracking-neural-diagnostic.md),
not the geometric benchmark's NumPy 2.x environment. The pinned SAM source is
[`facebookresearch/sam2` at `2b90b9f5ceec907a1c18123530e92e794ad901a4`](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4).
The [official README](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/README.md)
and [license](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/LICENSE)
license SAM code and checkpoints under Apache-2.0. GoTrack's separate
noncommercial restriction still applies to the combined research pipeline.

- Official source archive:
  `https://codeload.github.com/facebookresearch/sam2/zip/2b90b9f5ceec907a1c18123530e92e794ad901a4`,
  55,952,852 bytes, SHA-256
  `fe93082a71a885a427894b1eab76341768781b6b58a298a0717e03862097d137`.
- Official checkpoint:
  `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt`,
  323,606,802 bytes, SHA-256
  `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5`.
- Source destination: `.cache/model-quality/sources/sam2`.
- Checkpoint destination:
  `.cache/model-quality/checkpoints/sam2.1_hiera_base_plus.pt`.

The recorded restoration downloaded the archive with a 64 MiB compressed bound,
validated paths inside the destination, and capped expanded data at 256 MiB
(64,563,748 bytes observed). Every SAM file in the exact historical source lock
matched. The checkpoint was streamed into a new `.part`, capped at its exact
size, SHA-verified and atomically renamed; no rejected private-download endpoint
was reused. Preserve at least 10 GiB free disk.

For a fresh checkout, pinned Git fetch is an alternative to extracting that
verified archive (do not overwrite an existing source directory):

```bash
mkdir -p .cache/model-quality/sources/sam2
cd .cache/model-quality/sources/sam2
git init
git remote add origin https://github.com/facebookresearch/sam2.git
git fetch --depth 1 origin 2b90b9f5ceec907a1c18123530e92e794ad901a4
git checkout --detach FETCH_HEAD
cd ../../../..
```

Download the checkpoint from its official URL above into a new `.part` file;
check both `stat -c %s` and `sha256sum` against the exact values before renaming.
The workbench loads the source directly; a SAM editable install is unnecessary.
Set `SAM2_BUILD_CUDA=0`; do not compile/install its optional CUDA extension.

## CPU dependencies

The tested Python 3.12 environment retains Torch 2.5.1+cpu, torchvision
0.20.1+cpu and NumPy 1.26.4. Added packages are Hydra-core 1.3.2, OmegaConf 2.3.0,
iopath 0.1.10, tqdm 4.66.5, PyYAML 6.0.2, portalocker 2.10.1,
antlr4-python3-runtime 4.9.3 and packaging 24.1. `pip check` passed.

Exact iopath 0.1.10 and ANTLR runtime 4.9.3 have only source distributions.
Their official PyPI `setup.py` files were inspected: no native extensions or
compiler are required. Fetch those exact official source distributions, verify
`purePythonSourceBuilds` in the setup manifest, then build only these bounded
pure-Python wheels (PyOpenGL 3.1.0 was already handled by the prior GoTrack setup):

```bash
CC=/bin/false CXX=/bin/false timeout 120s .venv/bin/python -m pip wheel \
  --no-deps --no-build-isolation --wheel-dir .cache/cpu-wheels \
  /path/to/verified/iopath-0.1.10.tar.gz \
  /path/to/verified/antlr4-python3-runtime-4.9.3.tar.gz
SAM2_BUILD_CUDA=0 .venv/bin/python -m pip install --only-binary=:all: \
  --find-links .cache/cpu-wheels \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r bench/runtime/requirements-tracking-sam-cpu.txt
.venv/bin/python -m pip check
```

Source hashes reproduce the inputs; local pure-Python wheel hashes can vary
with build metadata. The manifest records the exact wheels actually installed
on this VM, not an assertion that arbitrary rebuilt wheels are byte-identical.

## Readiness versus measured inference

Imports, Hydra composition of `configs/sam2.1/sam2.1_hiera_b+.yaml` (image size
1024), and `segmentation_preflight(verify_hash=True)` passed without constructing
a model. CPU is available; CUDA is absent. The pinned SAM loader uses
`torch.load(weights_only=True)`. Its optional connected-component CUDA
postprocessing is disabled by the unchanged production adapter.

Actual recorded segmentation uses the existing native CLI:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 SAM2_BUILD_CUDA=0 \
  .venv/bin/python -m bench.workbench_cli --workspace /path/to/native-workspace \
  segment CLIP_ID --mask-set NEW_IMMUTABLE_MASK_SET --mode regression \
  --device cpu --verify-hash
```

For the VM pilot the bounded supervisor additionally sets Torch threads 8,
interop threads 1 and OpenCV threads 1 explicitly. It enforces 7 GiB sampled
aggregate RSS, 900 seconds wall time, 512 MiB output and 10 GiB free disk. Only
one model stage runs at a time. The native mask manifest preserves initialization
and per-frame time, actual observed masks, checkpoint identity, reviewed point
prompts and causal-release guard. Runtime/source/input hashes, partial masks,
resource samples and supervisor failures remain local.

A prompt-assisted segmentation result is not automatic object discovery, pose
accuracy, end-to-end tracking, or historical serving admission. Only the first
onboarding frame supplies reviewed points; no scored masks, poses or depth may
enter segmentation. Any full trained-pose stage still needs separately earned,
explicitly scoped parity/geometry admission; the old historical small-records
pin must not be manufactured or relaxed.
