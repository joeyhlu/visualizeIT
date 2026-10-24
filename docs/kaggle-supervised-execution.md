# Versioned Kaggle supervision

The execution bridge is `bench.quality_adaptation_supervised_v1`. It preserves the
original RAW context, seal, source root, three wrapper descriptors and adapter
descriptor. Add the new bridge and supervisor source files alongside the original
source package without overwriting any file. The externally authenticated request
must add `supervision_sources` with exactly three ordinary file descriptors:

```python
request['supervision_sources'] = {
    'bridge': pin(source_root / 'bench/quality_adaptation_supervised_v1.py'),
    'supervisor': pin(source_root / 'scripts/kaggle_supervision_v1.py'),
    'scanner': pin(source_root / 'bench/quality_adaptation_portable.py'),
}
```

The bridge compares those descriptors with the actual loaded module paths and
bytes. It also verifies fixed hashes for the original orchestration helpers and
the original no-follow scanner. No caller-provided success flag replaces these
checks. The source root must remain the root in the existing RAW context.

Use `prepare_stage(stage, request)` for identity, extraction, feature sealing and
the private label worker; use `run_labels(request, request_path, request_sha256)`
for the supervised label stage. The worker invocation is authenticated by both the
request hash and its private environment marker. The original prelabel verifier
runs before the label plan is read. The numerical label loop is identical to the
original function; the assembler remains the original subprocess producer.

Use `execute(stage, request, stop_after_steps=None, pilot=False)` for preflight,
training, resume and export. Use `pilot_execute(pilot_request)` for the original
one to four genuine update measurement protocol. Both `pilot_request['training']`
and `pilot_request['full_run']` must carry the same `supervision_sources` as
preparation. The pilot still uses original checkpoint inspection, reconciliation,
prepared admission and full-scope bounds. A pilot checkpoint is never promoted.

The equivalent CLI accepts these stages, plus `labels`, `pilot`, `inspect` and
`monitor`, with `--request` and `--request-sha256`. An absolute bridge file path
works when the bridge is installed in the original source package. The `-B`
interpreter flag is required by the original source-only package checks.

Outer operators can use `scripts.kaggle_supervision_v1.Facade(k, raw)` as an
explicit adapter-compatible dependency. This object verifies the fixed original
adapter and scanner source hashes. Validation calls delegate to that adapter;
`supervise` is explicitly the new implementation. Its output counter uses the
original `raw._usage` fd/no-follow scanner, including its three complete scan
attempts. Runtime accounting uses at most three complete original
`k.unique_tree_bytes` calls with the same approved link policy; only
`FileNotFoundError` causes a retry. Other errors fail closed.

Extraction writes supervision provenance into `extraction-binding.json` before
the original feature seal is created. Existing v2 schemas and the original
prepared identity remain authoritative; the binding descriptor makes the added
provenance transitive through that identity. Failure results preserve partial
output and do not attempt successful extraction finalization. The bridge retains
the original model producers, trainer, dataset identity, config identity and
checkpoint resume checks.

Enforced limits retain 8 GiB RSS, 8 GiB feature/cache allowances, 18 GiB combined
storage, at least 1 GiB reserve, 1200 seconds per stage and 900 seconds per trainer
invocation. Fresh observed free Kaggle evidence remains required. Local tests are
model-free software evidence; Linux process groups, symlinks and hardlinks must
also pass in the actual cloud source-test stage before neural execution.
