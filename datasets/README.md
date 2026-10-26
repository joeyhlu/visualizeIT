# VisualizeIT datasets

| Dataset | Input/evaluation coverage | Purpose |
|---|---|---|
| [offline-sam-pose-v1](offline-sam-pose-v1/README.md) | Real HB, 12 objects × 300 scored frames, plus one authorized frame-9 seed per object | Native recorded RGB → observed SAM → metric pose/tracking inputs, metric GLBs, reviewed annotation-assisted onboarding prompts and separate evaluator |
| [tracking-v1](tracking-v1/README.md) | Real HB + explicitly synthetic stress cases | Geometric tracker regression datasets, frozen annotations and recipes |

The [published full real/synthetic media release](https://github.com/joeyhlu/visualizeIT/releases/tag/tracking-dataset-v1)
is reused rather than duplicated as large Git blobs. The offline SAM/pose adapter
adds full native input metadata and reproducible conversion recipes for all 12 metric models; it is **not** cached
neural features or a claim of model quality. Bounded smoke/pilot/long execution
profiles do not shrink the published full dataset.

Keep runtime inputs and evaluator labels separate. Source licensing, annotation
assistance, initialization, coordinate conventions, hashes, reproducible setup,
and exact native-budget handling are documented in each dataset card. Private
model weights/configurations and unapproved new model results are not datasets.
