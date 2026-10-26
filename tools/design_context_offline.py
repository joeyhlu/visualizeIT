"""Local design-project commands and an adapter to the existing artwork studio.

No model, camera, renderer, network client, or provider is constructed here.
Imported recipes are descriptive and are never executed. Run with --help.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

from tools import design_context as dc
from tools.design_studio import StudioStore, StudioError, validate_recipe, validate_png


CAPABILITIES = ("solid", "stripes", "checker", "dots", "waves", "grid")
CONSUMED_FIELDS = ("pattern", "colors", "text", "scale", "rotation_deg", "seed", "width", "height")
DEFAULT_RECIPE = {"pattern": "stripes", "colors": ["#f4ebd8", "#42634b"],
                  "text": "", "scale": 1, "rotation_deg": -25, "seed": 731,
                  "width": 1024, "height": 1024}


class OfflineProjects:
    """Navigation is local bookkeeping, outside immutable portable revisions."""

    def __init__(self, root, *, store=None):
        self.store = store if store is not None else dc.ProjectStore(root)
        self.active_project_id = None
        self.recovery_notice = None

    def create(self, title="Untitled"):
        document = self.store.create(title)
        self.active_project_id = document["projectId"]
        return document

    def open(self, project_id):
        document = self.store.open(project_id)
        self.recovery_notice = ("Recovered the previous complete design; current files were preserved."
                                if self.store.recovered else None)
        self.active_project_id = project_id
        return document

    def import_snapshot(self, data, choice=None):
        document = self.store.import_snapshot(data, choice=choice)
        if document is not None:
            self.active_project_id = document["projectId"]
        return document

    def adopt_studio(self, project_id, expected_revision, studio, design_id):
        """Associate validated v1 artwork; never modify its immutable bundle.

        A saved recipe is historical metadata, not proof of its actual renderer.
        Unknown creating brief and producer remain null, rather than inferred.
        """
        metadata = studio.get_design(design_id)
        png = studio.get_image(design_id)
        provenance = {"kind": "legacy", "producer": None, "brief": None,
                      "prompt": metadata["prompt"], "effectiveRequest": copy.deepcopy(metadata["recipe"]),
                      "source": "Saved studio v1 metadata; historical creation is unverified"}
        artwork = dc.make_artwork(png, provenance, artwork_id="studio-" + metadata["id"])
        return dc.select_artwork(self.store, project_id, expected_revision, artwork)

    def import_png(self, project_id, expected_revision, png):
        provenance = {"kind": "imported", "producer": None, "brief": None,
                      "prompt": None, "effectiveRequest": None, "source": None}
        artwork = dc.make_artwork(png, provenance)
        return dc.select_artwork(self.store, project_id, expected_revision, artwork)

    def generate(self, project_id, expected_revision, recipe, render, *, supported):
        """Explicit host integration seam for the existing finite generator.

        `render` is a trusted local callable supplied by the host workflow, never
        imported data. The request is committed before calling it. A failed
        renderer leaves the accepted intent and the prior selected image intact.
        This module does not implement another renderer or parse arbitrary chat.
        """
        request = copy.deepcopy(validate_recipe(copy.deepcopy(recipe)))
        if request["pattern"] not in CAPABILITIES or type(supported) is not bool:
            raise dc.DocumentError("Unsupported offline generation capability")
        _, attempt = dc.prepare_generation(self.store, project_id, expected_revision,
            capability=request["pattern"], effective_request=request, supported=supported)
        png = render(copy.deepcopy(request))
        validate_png(png, request)
        return dc.complete_generation(self.store, attempt, png,
            {"app": "visualizeit-offline-studio", "version": "design-context-v1",
             "capability": request["pattern"]}, self.active_project_id)


def suggestions(document):
    """Deterministic alternatives; no proposed chip implies object synthesis."""
    document = dc.validate_document(document)
    draft = document["promptDraft"]
    brief = document["acceptedBrief"]
    text = draft["text"] if draft else brief["text"] if brief else ""
    known = re.search(r"\b(solid|stripes|checker|dots|waves|grid)\b", text.lower())
    pattern = known.group(1) if known else "waves"
    proposals = []
    # These are explicit alternatives, not a guess that an object can be made.
    if known:
        proposals.append(("blue-pattern", "Try blue " + pattern, "draft",
            {"text": "blue " + pattern, "capability": pattern,
             "settings": {**DEFAULT_RECIPE, "pattern": pattern, "colors": ["#376aa0", "#ffffff"]}}))
        proposals.append(("cream-pattern", "Try cream " + pattern, "draft",
            {"text": "cream " + pattern, "capability": pattern,
             "settings": {**DEFAULT_RECIPE, "pattern": pattern, "colors": ["#f4ebd8", "#202820"]}}))
    else:
        proposals.append(("upload", "Upload artwork", "import", {}))
        proposals.append(("literal-text", "Use literal text", "draft",
            {"text": text, "capability": "solid", "settings": {**DEFAULT_RECIPE,
              "pattern": "solid", "text": text[:80]}}))
        # Never silently truncate a text request to fit this finite renderer.
        if len(text) > 80:
            proposals.pop()
    proposals.append(("ask-colors", "Choose colors", "clarify", {"question": "Which colors do you want?"}))
    if known:
        proposals.append(("upload", "Upload artwork", "import", {}))
    proposals.append(("waves-alternative", "Try blue waves", "draft",
        {"text": "blue waves", "capability": "waves", "settings": {**DEFAULT_RECIPE,
         "pattern": "waves", "colors": ["#376aa0", "#ffffff"]}}))
    result = []
    for action_id, label, kind, payload in proposals[:5]:
        origin = {"actionId": action_id, "projectId": document["projectId"],
                  "contextRevisionId": document["projectRevisionId"],
                  "draftRevisionId": draft["revisionId"] if draft else None,
                  "briefRevisionId": brief["revisionId"] if brief else None,
                  "effect": {"type": kind, "payload": payload}}
        digest = hashlib.sha256(json.dumps(origin, sort_keys=True).encode()).hexdigest()[:32]
        result.append({"label": label, "chip": {"id": "chip-" + digest, **origin}})
    return result


def read_bounded(path, limit):
    with Path(path).open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise dc.DocumentError("Input file exceeds the supported limit")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local current-design snapshots; no inference or camera access")
    parser.add_argument("--root", type=Path, default=Path(".cache/design-context/projects"))
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("new", help="Create a context-only local project")
    create.add_argument("--title", default="Untitled")
    commands.add_parser("list", help="List local projects without changing revisions")
    for name in ("open", "note", "draft", "accept", "load-brief", "suggest", "choose", "adopt-studio", "import-png", "export"):
        command = commands.add_parser(name)
        command.add_argument("project_id")
        if name not in ("open", "suggest", "export"):
            command.add_argument("--expected-revision", required=True)
        if name in ("note", "draft"):
            command.add_argument("text")
        if name == "draft":
            command.add_argument("--capability", choices=CAPABILITIES)
        if name == "accept":
            command.add_argument("--text", help="Accept this explicit proposal; omitted accepts the displayed draft")
        if name == "choose":
            command.add_argument("action_id")
        if name == "adopt-studio":
            command.add_argument("design_id")
            command.add_argument("--studio-root", type=Path, required=True)
        if name == "import-png":
            command.add_argument("file", type=Path)
        if name == "export":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--confirm-full-transcript", action="store_true")
    importing = commands.add_parser("import", help="Import a deliberate self-contained snapshot")
    importing.add_argument("file", type=Path)
    importing.add_argument("--choice", choices=("update", "copy", "cancel"))
    args = parser.parse_args(argv)
    adapter = OfflineProjects(args.root)
    try:
        if args.command == "new":
            output = adapter.create(args.title)
        elif args.command == "list":
            output = adapter.store.list()
        elif args.command == "import":
            output = adapter.import_snapshot(read_bounded(args.file, 25_165_824), args.choice)
        elif args.command == "open":
            output = adapter.open(args.project_id)
        elif args.command == "suggest":
            output = suggestions(adapter.open(args.project_id))
        elif args.command == "export":
            document = adapter.open(args.project_id)
            scope = f"Export contains all {len(document['messages'])} saved messages and {int(document['selectedArtwork'] is not None)} selected PNG. No camera/video data is added."
            print(scope, file=sys.stderr)
            if not args.confirm_full_transcript:
                raise dc.DocumentError("Review this scope, then explicitly pass --confirm-full-transcript to write")
            data = adapter.store.export_snapshot(args.project_id)
            with args.output.open("xb") as stream:
                stream.write(data)
            output = {"exported": True, "bytes": len(data)}
        else:
            project_id, revision = args.project_id, args.expected_revision
            if args.command == "note":
                output = dc.append_message(adapter.store, project_id, revision, args.text, source="offline-user")
            elif args.command == "draft":
                output = dc.edit_draft(adapter.store, project_id, revision, args.text, capability=args.capability)
            elif args.command == "accept":
                output = dc.accept_brief(adapter.store, project_id, revision, text=args.text)
            elif args.command == "load-brief":
                output = dc.load_accepted_brief(adapter.store, project_id, revision)
            elif args.command == "adopt-studio":
                output = adapter.adopt_studio(project_id, revision, StudioStore(args.studio_root), args.design_id)
            elif args.command == "import-png":
                output = adapter.import_png(project_id, revision, read_bounded(args.file, 16_777_216))
            elif args.command == "choose":
                document = adapter.open(project_id)
                matches = [item["chip"] for item in suggestions(document) if item["chip"]["actionId"] == args.action_id]
                if len(matches) != 1:
                    raise dc.DocumentError("Suggestion is unavailable; request fresh suggestions")
                output = dc.apply_chip(adapter.store, project_id, revision, matches[0])
                if matches[0]["effect"]["type"] != "draft":
                    print("Action requires explicit follow-up: " + matches[0]["effect"]["type"], file=sys.stderr)
        print(json.dumps(output, ensure_ascii=False, allow_nan=False))
        return 0
    except (dc.DocumentError, StudioError, OSError, ValueError) as exc:
        print("Project operation failed: " + str(exc), file=sys.stderr)
        return 2
    finally:
        if adapter.recovery_notice:
            print("Recovery: " + adapter.recovery_notice, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
