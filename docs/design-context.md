# Design Context v1 contract

This is a current-design snapshot, containing all retained submitted text and zero or one selected PNG. It is separate from the strict studio generation-request schema. JSON uses UTF-8, camelCase member names, schemaVersion 1 and documentType `visualizeit-project`. Every field below is required, including nullable fields. Unknown members reject; only the root `extensions` object permits unknown inert data. IDs are 1–128 ASCII letters/digits/`._:-`. JSON order is immaterial; array order is meaningful.

Frozen JSON profile (notation `?` means nullable, not optional):

```
root {schemaVersion:1, documentType:"visualizeit-project",
 projectId,projectRevisionId,parentRevisionId:ID?,title,origin:{projectId,revisionId}?,
 messages:[{id,role:"user"|"assistant"|"system",source,text,artworkIds:[ID]}],
 acceptedBrief:{revisionId,text}?,
 promptDraft:{revisionId,baseBriefRevisionId:ID?,text,capability:string?,settings:object}?,
 chosenSuggestionEffects:[{id,actionId,projectId,contextRevisionId,draftRevisionId:ID?,briefRevisionId:ID?,
                 effect:{type:"draft"|"clarify"|"import",payload:object},resultDraftRevisionId:ID?}],
 generationAttempt:{id,projectId,contextRevisionId,brief:{revisionId,text},prompt,
                    capability,effectiveRequest:object}?,
 selectedArtwork:{id,png:{sha256,byteCount,width,height,base64},
   provenance:{kind:"generated"|"imported"|"legacy",producer:{app,version,capability}?,
               brief:{revisionId,text}?,prompt:string?,effectiveRequest:object?,source:string?}}?,
 extensions:object}
```

Settings, effectiveRequest and effect payload are opaque inert JSON objects. They are never executed by this module. Source is an opaque descriptive string, never a fetchable reference. Producer fields are strings, with version/capability nullable. Unknown historical facts are null. Conversation artworkIds may reference excluded artwork; availability is derived from the local catalog. No geometry or attachment fields exist. A generated record requires producing capability, frozen creating brief, prompt and exact consumed request; imported/legacy records may have unknown provenance.

Canonical size/semantic serialization: Python-equivalent JSON `ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False`, UTF-8; no BOM. Metadata is this serialization with selectedArtwork.png.base64 replaced by `""`. Caps: file 25,165,824 bytes; metadata 1,048,576; extensions 65,536; messages/effects each 1,000; nesting 32; PNG 16,777,216 bytes; each dimension 4,096; pixels 16,777,216. Invalid Unicode, duplicate members, nonfinite values, noncanonical base64, malformed/trailing PNG, wrong dimensions/hash/size reject before persistence. Pillow verifies and decodes the original PNG without re-encoding.

## Python callable API (tools/design_context.py)

`validate_document(dict) -> dict`, `decode_document(bytes) -> dict`, `encode_document(dict) -> bytes` return validated deep copies/canonical bytes and raise `DocumentError`. `new_project(title="Untitled") -> dict` creates context-only state. `make_artwork(png_bytes, provenance, artwork_id=None) -> dict` validates original bytes. `new_id() -> str` makes UUID strings.

`ProjectStore(root, fault=None)` stores app-owned immutable assets/catalog/revisions and an atomic per-project current+previous pointer. fault(stage) may raise at `assetStaging`, `assetCreated`, `manifestStaging`, `beforeActivation`, `afterActivation`. `create(title="Untitled") -> dict`; `open(project_id) -> dict` (sets `recovered` boolean on the store); `list() -> list[dict]`; `update(document, expected_revision) -> dict` creates a new revision from candidate fields; `import_snapshot(bytes, choice=None) -> dict|None`, with choice `update`, `copy`, or `cancel`; `export_snapshot(project_id) -> bytes`; `catalog(project_id) -> dict[artworkId,artwork]`. Conflicting revision/artwork semantics raise `ConflictError`; a different imported revision requires explicit choice. An identical import is idempotent. Copy gets fresh project/revision IDs and origin, retains immutable artwork identity. Cancel performs no writes. Corrupt current falls back to validated previous with `recovered=True`; both invalid raise `RecoveryError` and preserve files. Switching project is `open` and does not mutate state.

Authority helpers take store, project ID, expected revision and return committed state: `append_message(..., text, role="user", artwork_ids=None, source="user")`, `edit_draft(..., text, capability=None, settings=None)`, `accept_brief(..., text=None)` (None accepts displayed draft; another proposal preserves an edited draft as stale), `load_accepted_brief(...)`, `select_artwork(..., artwork)`. `prepare_generation(..., capability, effective_request, supported)` validates support and draft freshness, commits acceptance plus immutable attempt before any renderer executes, and returns `(document, attempt)`. `complete_generation(store, attempt, png_bytes, producer, active_project_id)` saves valid output into its originating project's immutable catalog; selects it only when origin revision still equals attempt.contextRevisionId AND the active project matches. It never invents successful records on invalid output or failed persistence. No renderer is called by core.

`apply_chip(store, project_id, expected_revision, chip)` accepts `{id,actionId,projectId,contextRevisionId,draftRevisionId:ID?,briefRevisionId:ID?,effect:{type,payload}}`. All origin IDs must match; chosen IDs are single-use. Draft payload is a subset of `{text,capability,settings}`; clarify payload `{question}` and import payload `{}` record/navigation only. Chips never accept or generate. Adapters inventory actual capabilities and construct honest chips; the core does not infer arbitrary object support. `artwork_available(store, project_id, artwork_id)` derives local availability without modifying exported history.

## Evidence boundaries

The offline adapter's CLI reports recovery explicitly on stderr, outside the immutable document. The iPhone source currently gates snapshot import/export because general opaque-number and Unicode-key parity is unresolved; local project persistence and Photos PNG handling are separate source paths. See the native README for blocked versus deferred gates. This is not a completed cross-platform handoff.

Local manifests add one envelope object around portable state, so their parser permits one additional container level while portable validation remains capped at 32. The full serialized local manifest, including retained catalog metadata, must fit its reader's 24 MiB bound before any write. At that local limit, saving is refused without advancing the active pointer or deleting original assets. Immutable identity uses type-sensitive canonical JSON comparison: Boolean true cannot equal numeric 1.

Core synthetic tests establish document/storage behavior only. Adapter UI, real offline generation compatibility, native compilation/device interchange, memory suitability and attachment behavior require their separate checks. Android is a separately mandatory next slice. Export includes the complete saved transcript and selected PNG; adapters must disclose that scope. No network, model, camera or automatic export occurs here. Immutable local older artwork survives current-design imports and exports.

`draft_status(document)` returns `absent`, `clean`, `unaccepted`, or `stale`: missing draft is absent; mismatched nullable base/current accepted revision is stale; matching base and accepted text is clean; other text is unaccepted. No imported Boolean asserts acceptance. New projects have null brief/draft; explicit acceptance creates a clean draft. Copies preserve frozen historical attempt/effect origins. Chip brief/draft/result IDs may be null when no such record exists.

Local storage maps every project/commit identifier to its SHA-256 UTF-8 name. Imported IDs (including `.`, `..`, `:` and Windows device names) are semantic strings, never paths. Listing recovers identity from validated manifests rather than folder names. Assets use only validated SHA-256 digest filenames. Local manifests contain `document`, `catalog`, and `parentCommit`; they are distinct from portable snapshots. Only activated manifests and their parent chains establish saved request authority; staging/orphan files are inert.

All store writes and authority helpers hold a per-root reentrant thread lock plus a kernel byte/file lock for the entire read/compare/write transaction, including across separate processes. The persistent `.store.lock` is never removed or stolen; process termination releases its kernel lock. Contention times out visibly after ten seconds. `ProjectStore.transaction()` is an optional reentrant context for trusted host sequences. Imported content never acquires authority to call generation or mutate another project. Historical attempt `projectId` values are retained as origin references across copies; completion requires the actual originating store's committed attempt.

Initial test attempt: 20 errors at setup from atomic rename denied in the OS sandbox temporary directory; no test bodies ran. The authorized correction uses `.cache/design-context-core-tests/` under the existing workspace without changing permissions. This initial blocked trial is retained as evidence, not counted as passing tests.

Corrected core execution (26 October 2026): `python -B -m unittest tools.test_design_context -v` ran 25 tests in 12.421 seconds, all passed, zero skips. Every test body ran. Coverage includes retention, accepted/draft authority, generation request persistence, failure preservation, chip effects/staleness, delayed results, immutable provenance, context-only and selected-image interchange, semantic/artwork conflicts, parser/PNG rejection, exact/over limits, five synthetic copy-on-write interruption points, missing/corrupt state recovery, opaque storage IDs, and stale-writer checks with threads and separate processes. The pixel cap equals the square of the maximum dimension, so exceeding pixels necessarily exceeds a dimension too; that coupled rejection is tested. A 4,096-square compressed PNG is actually decoded. The 24 MiB file boundary uses legal JSON whitespace padding; a canonical valid snapshot cannot reach that envelope with the other caps.

Fixtures `context-only.visualizeit-project.json` and `selected-png.visualizeit-project.json` contain synthetic text only and one tiny generated-in-test-data PNG respectively; their creation descriptions truthfully remain imported/unknown. Core tests do not establish adapter capability inventories/suggestion honesty, strict legacy studio loading, native UI mapping, native storage durability, physical-device interchange, attachment accuracy, or Android completion. Those remain separate integration/review obligations. Fault injection is a software boundary test, not a hardware power-loss claim.
