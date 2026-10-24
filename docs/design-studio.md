# Private phone design studio

The [camera workspace](phone-camera.md) adds on-device object selection and a
mask-clipped frozen-frame preview. Use **Try this artwork with the camera** to
transfer the current design locally. The camera’s **Back to design studio** link returns
the generated artwork as an unsaved design for editing or saving after pairing. Phone live-camera access requires trusted
HTTPS; the original private HTTP URL supports artwork editing but not phone
camera permission.

The studio creates patterns, colors, and text locally in your browser. It uses a
finite recipe grammar, not a language model. The saved PNG is the actual artwork;
recipes preserve pattern choices and seeds, while text pixels can vary with the
platform font. A browser photo is only a flat placement preview and is not part of
the artwork PNG.

Run from the repository using the existing approved Python runtime and packages:

```powershell
& C:/Users/Achita/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -B -S tools/prototype_bootstrap.py tools.serve_design_studio
```

This binds `127.0.0.1:8891`. Open the printed private pairing URL. Its fragment
contains a fresh startup token; the frontend consumes it into memory and clears
the fragment. A restart requires pairing again. API calls and PNG downloads
require `Authorization: Bearer TOKEN`; tokens do not belong in query strings,
image URLs, logs, or browser storage. Static artwork creation is available before
pairing. Do not share the pairing URL.

To open on a phone on the same trusted private network, bind the computer's actual
private IPv4 address explicitly:

```powershell
& C:/Users/Achita/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -B -S tools/prototype_bootstrap.py tools.serve_design_studio --lan-host 192.168.1.20
```

Replace that example with the computer's LAN address. The server accepts only
loopback or explicit RFC1918 IPv4 addresses; wildcard, public, and hostname binds
are rejected. LAN transport is HTTP, so use a trusted private network. No public
tunnel is created. Host, exact Origin, and Fetch Metadata checks reject foreign
browser origins; there is no permissive CORS. The service is a small local tool,
not an internet-facing deployment.

The optional read-only bridge connects directly to numeric loopback
`127.0.0.1:8765`, using only `GET /api/status` and `GET /api/clips`. Set
`--upstream-port PORT` if the existing operator service uses another port. The
bridge does not follow redirects, read proxy environment variables, or expose
upstream paths or private metadata. A stopped or invalid upstream is shown as
disconnected. It does not start the workbench, models, or inference.

Saved designs are immutable bundles under `.cache/design-studio` (override with
`--storage-root`). Limits are 100 designs, 100 handoffs, 32 MiB aggregate storage,
2 MiB per PNG, and 256â€“1024 pixels per edge. Saves require the 1 GiB free disk
reserve plus the complete new output size. PNG chunk CRCs, dimensions, and full
Pillow decode are checked before publication. Metadata and PNG hashes are checked
again when read. Storage ancestors and contents reject symlinks and Windows
reparse points. Keep this directory private to the local account; this is not a
multi-user filesystem security boundary.

Writes use a lock, exclusive temporary files, file flush/fsync, and a single
directory rename to publish a complete bundle. Failed writes remove their
temporary bundle. An abrupt process or machine failure can leave a hidden
`.pending-*` directory; it counts toward the byte cap but is never listed as a
design. Do not delete it while the server runs. Directory rename durability over
sudden power loss is filesystem-dependent. Restart does not overwrite records.
Repeating a POST creates another immutable record; no idempotency key is offered,
so inspect saved designs after an uncertain network result before retrying.

Handoffs are saved manifests for
`mobile/Assets/VisualizeIt/Runtime/DesignImporter.cs`. Download the PNG and select
it with the native image picker. A selected clip must exist in the current actual
workbench catalog; an unbound handoff works while the workbench is disconnected.
Handoffs report `ready_for_manual_import`, `tracking_applied: false`, and
`native_import_verified: false`. They do not queue tracking or prove native import.
Actual phone import and attachment tracking remain separate verification steps.

The API uses the frozen `visualizeit_design_v1`,
`visualizeit_studio_status_v1`, and `visualizeit_artwork_handoff_v1` contracts.
Unknown or duplicate JSON fields, invalid types, nonfinite numbers, oversized
bodies, corrupt images, and unvalidated clips fail explicitly. Errors have shape
`{"error":{"code":"...","message":"..."}}`; authentication failures return
401, foreign origin/host 403, missing records 404, tampered records 409, body size
413, unavailable workbench/decoder 503, and storage/reserve limits 507. Operator
logs contain rejection codes and server failures without request URLs or tokens.

Run the backend checks without installation or model execution:

```powershell
& C:/Users/Achita/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -B -S tools/prototype_bootstrap.py unittest tools.test_design_studio
```

These checks cover normal HTTP save/download/handoff, auth and browser boundary
enforcement, JSON and PNG rejection, persistence and tamper detection, quota and
disk reserve, rollback, path/link guards, and fixed upstream catalog/redirect
handling. A synthetic Windows reparse-attribute check runs on all platforms;
actual symlink creation is skipped if account privileges prohibit it. Physical
phone networking, native picker behavior, and power-loss durability require
separate environment checks.
