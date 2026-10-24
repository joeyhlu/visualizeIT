# Kaggle CUDA activation

This bounded comparison checks that the legacy and amortized candidate preparation
paths produce equivalent retained states on the actual Kaggle CUDA runtime. It
does not train a model, measure independent accuracy, or replace the later
candidate-feature parity check.

Use the private prepared source, the completed pinned runtime receipt, and the
existing full development recipe and exposure audit. The helper selects object 1,
views 0 and 1, from the unchanged prospective split. It never opens the sealed
test object's payload. Each branch computes its own observed masks and retained
states from the common rendered inputs and template banks.

## Launch

The observed Kaggle image uses Ubuntu 24.04. Install its missing GLU library in
the cloud before rendering; the pinned Python packages alone do not supply it:

```bash
apt-get --no-install-recommends -y install libglu1-mesa=9.0.2-1.1build1
export PYOPENGL_PLATFORM=egl VISUALIZEIT_CLOUD=1
```

This installs one 152 kB Ubuntu package. The completed analytical rendering check
observed 0.900003433 m depth for a cube front plane at 0.9 m; this is a software
fixture, not a pose-accuracy measurement or a training example.

Create a `kaggle_activation_request_v1` JSON document containing the actual source
root, fresh output and scratch directories, exact source hashes, model-cache
path, full recipe and exposure-audit file descriptors, immutable source-lock
descriptor, limits and the observed free-resource/CUDA provenance. Use
`quality_kaggle_activation.expected_source_paths()` for the required source
inventory; each hash must describe the reviewed bytes actually installed in the
cloud. A descriptor contains the absolute path, byte length and SHA256.

Run these stages sequentially with the pinned runtime's Python:

```bash
PY=/kaggle/working/visualizeit-pinned-runtime-v2/venv/bin/python
REQUEST=/kaggle/working/visualizeit-activation-request-v1.json
SHA=$(sha256sum "$REQUEST" | cut -d ' ' -f 1)
for STAGE in freeze render banks legacy amortized compare; do
  "$PY" -I -B scripts/kaggle_activation.py "$STAGE" \
    --request "$REQUEST" --request-sha256 "$SHA" --cloud-execution || break
done
```

Run from the prepared source directory. Start one GPU stage at a time. The
default limits are 1,800 cumulative seconds, 8 GiB process-tree RAM, 18 GiB
combined activation/cache storage and at least 1 GiB free space. The model cache
has its separate 8 GiB ceiling. The workflow does not select a provider plan or
authorize charges.

## Inspect failures and outputs

Keep the request, freeze record, individual stage records, logs and independent
branch outputs. A failed or incomplete stage is not a successful parity result.
Do not rerun into its output directory or remove evidence to force admission.
The comparison must include available states on the same actual CUDA device;
synthetic software fixtures do not satisfy this requirement.

After a successful comparison, use its authenticated snapshots for the existing
portable raw-preparation preflight. Continue raw preparation, frozen-feature
extraction and prelabel parity before generating supervised labels. This small
activation workload is a software integration check, not a training dataset or
an untouched evaluation set.

The pinned runtime recovery preserves its failed v1 directory. The v2 installer
uses a fresh UV-only directory, counts live extraction entries without following
temporary links, and performs strict complete link validation at final admission.
Runtime success and GPU availability alone are not evidence of training updates.
