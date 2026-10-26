# Licensed real examples and complete annotations

These are real **HB / HomebrewedDB** RGB frames, not synthetic renders.
License: CC0 1.0 Universal. Credit: Roman Kaskman, Ivan Shugurov, Sergey Zakharov, Slobodan Ilic (HomebrewedDB, ICCVW 2019); BOP conversion by the BOP authors.
Pinned source: https://huggingface.co/datasets/bop-benchmark/hb/tree/985ebaef7b1bd4bffb2039f18dd81faa647fd060
Official license listing: https://bop.felk.cvut.cz/datasets/

The eight PNGs are byte-identical source images: frames 0 and 309 from each of
scenes 1–4. ``index.json`` binds each image to its original member and SHA-256.
Frame 0 is from authorized annotated onboarding; frame 309 is an evaluated frame.
``contact-sheet.jpg`` is a labeled/resized display derivative, never a metric input.

``evaluator.json.gz`` contains the **complete 3,600 evaluated object-frame
annotation records**, with 6D pose, calibration, visibility, and depth/mask
resource references for all 12 objects. It is not just annotation for these
eight examples. Pixel depth/mask files are referenced, not embedded in JSON.
``inference.json.gz`` contains the complete input manifest, including 120
authorized onboarding records. The 3,600 scored input records contain no pose,
depth, mask or visibility labels. Gzip filename and mtime are fixed; index.json
records compressed and expanded hashes. These are original dataset-reference
annotations, **not newly collected independent human labels**.

Extract annotations without changing the raw corpus:
    python -c "import gzip,pathlib; pathlib.Path('/tmp/hb-evaluator.json').write_bytes(gzip.open('datasets/tracking-v1/real-examples/evaluator.json.gz','rb').read())"

Acquire the complete selected media, verify it, or package it from the repo root:
    python -m bench.tracking_real_data acquire --output .cache/tracking-benchmark/real-v1 --recipe-output datasets/tracking-v1
    python -m bench.tracking_real_data verify --output .cache/tracking-benchmark/real-v1
    python -m bench.tracking_real_data package --output .cache/tracking-benchmark/real-v1 --archive artifacts/tracking-benchmark/releases/visualizeit-hb-tracking-v1.tar.gz

Recreate this committed example/annotation subset:
    python -m bench.tracking_real_data examples --output .cache/tracking-benchmark/real-v1 --examples-output datasets/tracking-v1/real-examples

The full 1.4 GiB resource corpus is not embedded as Git blobs. A separate complete
717 MB compressed media archive is now [published as a verified GitHub research prerelease asset](https://github.com/joeyhlu/visualizeIT/releases/download/tracking-dataset-v1/visualizeit-hb-tracking-v1.tar.gz). Uploaded size and GitHub-computed SHA-256 match the source archive. No private/user media is included.
