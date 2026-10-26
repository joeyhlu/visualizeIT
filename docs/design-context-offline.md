# Saved design projects for the offline tool

The local project command workflow saves submitted notes/conversations, explicitly accepted intent, a separate prompt draft and zero or one selected PNG. It does not execute tracking, a model, a camera or a cloud service. The existing browser recipe editor and immutable artwork library keep their existing format.

Run from the repository root with the existing Python environment containing Pillow:

```powershell
python -B -m tools.design_context_offline --help
python -B -m tools.design_context_offline new --title "Bottle label study"
python -B -m tools.design_context_offline list
python -B -m tools.design_context_offline open PROJECT_ID
```

The default private local store is `.cache/design-context/projects`. Use `--root PATH` before the command to select another app-owned store. Project IDs and current revision IDs are returned as JSON. The following commands require the **current** project revision so a stale command cannot overwrite newer work:

```powershell
python -B -m tools.design_context_offline note PROJECT_ID --expected-revision REVISION "A calm green label"
python -B -m tools.design_context_offline draft PROJECT_ID --expected-revision NEW_REVISION "green dots" --capability dots
python -B -m tools.design_context_offline accept PROJECT_ID --expected-revision NEWEST_REVISION
python -B -m tools.design_context_offline suggest PROJECT_ID
python -B -m tools.design_context_offline choose PROJECT_ID blue-pattern --expected-revision CURRENT_REVISION
```

Use the revision from each completed mutation for the next command; the placeholders above are not literal IDs. `note` appends explicitly labelled user text. `accept --text "..."` accepts a particular proposal; omit `--text` to accept the displayed draft. Chat/notes do not automatically accept intent. Choosing a suggestion records its stated effect and can edit the draft; it never generates or accepts a brief automatically. Import/clarification suggestions require a deliberate subsequent action. `load-brief` deliberately loads the accepted brief into the draft; unrelated manual edits are not silently replaced.

## Preserve an existing artwork

Generate/save artwork through the existing studio as before. Then explicitly associate its immutable v1 bundle with a project:

```powershell
python -B -m tools.design_context_offline adopt-studio PROJECT_ID DESIGN_ID --expected-revision REVISION --studio-root PATH_TO_EXISTING_STUDIO_STORE
python -B -m tools.design_context_offline import-png PROJECT_ID SELECTED_IMAGE.png --expected-revision REVISION
```

Studio adoption verifies the old metadata and PNG through its existing reader and preserves the bundle byte-for-byte. Unknown historical creating brief/producer remain unknown. Native or external PNG input has a direct import path; no fake studio recipe is required. App-owned bytes survive removal of the external source image. This association does not select an object, construct a surface or apply tracking; use the existing workbench's deliberate clip/attachment workflow separately.

The adapter exposes `OfflineProjects.generate(..., render, supported=...)` for trusted host integration with the existing finite generator. It commits accepted intent and the exact consumed recipe before invoking the callback, validates the returned original PNG, and binds success to the originating project/request. Imported metadata never supplies an executable callback. This is a tested software integration seam; the browser editor does not yet automatically use project context. It remains a separate deliberate legacy-save/adoption workflow.

## Explicit local interchange

```powershell
python -B -m tools.design_context_offline export PROJECT_ID --output design.visualizeit-project.json
python -B -m tools.design_context_offline export PROJECT_ID --output design.visualizeit-project.json --confirm-full-transcript
python -B -m tools.design_context_offline import design.visualizeit-project.json
python -B -m tools.design_context_offline import design.visualizeit-project.json --choice update
python -B -m tools.design_context_offline import design.visualizeit-project.json --choice copy
```

Export first reports its scope. Writing requires explicit confirmation: **all retained saved messages and the selected PNG**, or no PNG for a context-only project. No other artwork-library images, camera/video pixels, geometry, calibration, model weights or clip state are appended. User-authored messages are preserved as text, including private details they may contain. The command does not overwrite an existing destination. Transfer the file manually; there is no automatic upload or sync.

A different revision of an existing project requires explicit update/copy/cancel. Reimporting the same validated semantics is idempotent. Copies get new project/revision IDs and an origin reference; immutable artwork identity remains. A snapshot with an unsupported source generator still opens from its verified PNG bytes; it is not automatically regenerated or converted to an equivalent recipe.

See [the document contract](design-context.md) for caps, provenance, corruption recovery and strict format rules. Current and previous complete states remain recoverable. Both-corrupt recovery preserves files and reports failure instead of initializing a blank project over them. General project history, cloud/database services and artwork garbage collection are outside this slice.

## Verification limits

Core and offline synthetic checks exercise storage, authority, imports, legacy compatibility and failure handling. They do not establish native compilation, real offline-to-phone interchange, exact rendered phone UI, memory suitability, attachment accuracy or FPS. The iPhone source must be independently reviewed and later built/tested. A real native Android app remains the mandatory separate next slice.
