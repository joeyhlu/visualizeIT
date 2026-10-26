"""Bounded, inert current-design documents; no generator or network execution."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct
import uuid
import zlib
import os
import threading
import time
from contextlib import contextmanager
from functools import wraps

from PIL import Image

MAX_FILE = 25_165_824
MAX_METADATA = 1_048_576
MAX_EXTENSIONS = 65_536
MAX_ENTRIES = 1000
MAX_PNG = 16_777_216
MAX_DIMENSION = 4096
MAX_PIXELS = 16_777_216
MAX_DEPTH = 32


class DocumentError(ValueError):
    pass


class ConflictError(DocumentError):
    pass


class RecoveryError(DocumentError):
    pass


# Locks live for the entire read/compare/write transaction. The persistent lock
# file is never unlinked: kernel handle locks release on process termination.
_LOCKS = {}
_LOCKS_GUARD = threading.Lock()
_LOCK_LOCAL = threading.local()


def _serialized(method):
    @wraps(method)
    def invoke(self, *args, **kwargs):
        with self.transaction():
            return method(self, *args, **kwargs)
    return invoke


def _transactional(function):
    @wraps(function)
    def invoke(store, *args, **kwargs):
        with store.transaction():
            return function(store, *args, **kwargs)
    return invoke


def _storage_name(identifier):
    _id(identifier)
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()


def new_id():
    return str(uuid.uuid4())


def _json(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise DocumentError("Invalid JSON value") from exc


def _semantic_equal(left, right):
    """Canonical typed JSON identity; Python's True == 1 is not JSON identity."""
    return _json(left) == _json(right)


def _shape(value, names):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise DocumentError("Missing or unknown members: " + names)


def _text(value):
    if not isinstance(value, str):
        raise DocumentError("Expected text")
    try:
        value.encode("utf-8", "strict")
    except UnicodeError as exc:
        raise DocumentError("Invalid Unicode") from exc


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", value):
        raise DocumentError("Invalid identifier")


def _object(value):
    if not isinstance(value, dict):
        raise DocumentError("Expected inert object")


def _tree(value, depth=1):
    if depth > MAX_DEPTH:
        raise DocumentError("JSON nesting limit")
    if isinstance(value, dict):
        for key, item in value.items():
            _text(key)
            _tree(item, depth + (isinstance(item, (dict, list))))
    elif isinstance(value, list):
        for item in value:
            _tree(item, depth + (isinstance(item, (dict, list))))
    elif isinstance(value, str):
        _text(value)
    elif value is None or type(value) in (bool, int):
        pass
    elif isinstance(value, float) and math.isfinite(value):
        pass
    else:
        raise DocumentError("Invalid JSON type/value")


def _brief(value):
    _shape(value, "revisionId text")
    _id(value["revisionId"])
    _text(value["text"])


def _producer(value):
    _shape(value, "app version capability")
    _text(value["app"])
    for key in ("version", "capability"):
        if value[key] is not None:
            _text(value[key])


def validate_png(data):
    """Validate framing/CRC before an established decoder allocates a raster."""
    if not isinstance(data, bytes) or len(data) > MAX_PNG:
        raise DocumentError("PNG byte limit/type")
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise DocumentError("PNG signature")
    offset, chunks, dimensions, idat = 8, 0, None, False
    palette, ended_idat, color_type = False, False, None
    while offset < len(data):
        if offset + 12 > len(data):
            raise DocumentError("Truncated PNG chunk")
        size = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4:offset + 8]
        end = offset + 12 + size
        if end > len(data) or not re.fullmatch(b"[A-Za-z]{4}", kind) or kind[2] & 32:
            raise DocumentError("Invalid PNG chunk")
        payload = data[offset + 8:end - 4]
        crc = struct.unpack_from(">I", data, end - 4)[0]
        if zlib.crc32(kind + payload) & 0xffffffff != crc:
            raise DocumentError("PNG CRC")
        if chunks == 0:
            if kind != b"IHDR" or size != 13:
                raise DocumentError("PNG IHDR")
            width, height = struct.unpack_from(">II", payload)
            if not (1 <= width <= MAX_DIMENSION and 1 <= height <= MAX_DIMENSION
                    and width * height <= MAX_PIXELS):
                raise DocumentError("PNG dimension/pixel limit")
            dimensions = (width, height)
            bit_depth, color_type, compression, filtering, interlace = payload[8:]
            valid_bits = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
            if (color_type not in valid_bits or bit_depth not in valid_bits[color_type]
                    or compression != 0 or filtering != 0 or interlace not in (0, 1)):
                raise DocumentError("Invalid PNG IHDR fields")
        elif kind == b"IHDR":
            raise DocumentError("Duplicate PNG IHDR")
        if kind == b"IDAT":
            if ended_idat or (color_type == 3 and not palette):
                raise DocumentError("Invalid PNG IDAT order/palette")
            idat = True
        elif idat:
            ended_idat = True
        if kind == b"PLTE":
            if palette or idat or not 0 < size <= 768 or size % 3 or color_type in (0, 4):
                raise DocumentError("Invalid PNG palette")
            palette = True
        if kind in (b"acTL", b"fcTL", b"fdAT"):
            raise DocumentError("Animated PNG unsupported")
        if kind == b"IEND":
            if size != 0 or end != len(data) or not idat:
                raise DocumentError("PNG IEND/trailing data")
            break
        if kind[0] & 32 == 0 and kind not in (b"IHDR", b"PLTE", b"IDAT", b"IEND"):
            raise DocumentError("Unknown critical PNG chunk")
        chunks += 1
        offset = end
    else:
        raise DocumentError("Missing PNG IEND")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG" or image.size != dimensions:
                raise DocumentError("PNG decoder dimensions")
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            if getattr(image, "n_frames", 1) != 1:
                raise DocumentError("Animated PNG unsupported")
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise DocumentError("PNG decoder rejected image") from exc
    return dimensions


def _artwork(value):
    _shape(value, "id png provenance")
    _id(value["id"])
    png = value["png"]
    _shape(png, "sha256 byteCount width height base64")
    if not isinstance(png["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", png["sha256"]):
        raise DocumentError("PNG digest")
    for key, cap in (("byteCount", MAX_PNG), ("width", MAX_DIMENSION), ("height", MAX_DIMENSION)):
        if type(png[key]) is not int or not 1 <= png[key] <= cap:
            raise DocumentError("PNG bounds")
    encoded = png["base64"]
    if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_PNG + 2) // 3):
        raise DocumentError("PNG base64 limit/type")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise DocumentError("PNG base64") from exc
    if base64.b64encode(data).decode("ascii") != encoded:
        raise DocumentError("Noncanonical PNG base64")
    if len(data) != png["byteCount"] or hashlib.sha256(data).hexdigest() != png["sha256"]:
        raise DocumentError("PNG byte count/hash mismatch")
    if validate_png(data) != (png["width"], png["height"]):
        raise DocumentError("PNG dimension mismatch")
    provenance = value["provenance"]
    _shape(provenance, "kind producer brief prompt effectiveRequest source")
    if provenance["kind"] not in ("generated", "imported", "legacy"):
        raise DocumentError("Unknown provenance kind")
    if provenance["producer"] is not None:
        _producer(provenance["producer"])
    if provenance["brief"] is not None:
        _brief(provenance["brief"])
    for key in ("prompt", "source"):
        if provenance[key] is not None:
            _text(provenance[key])
    if provenance["effectiveRequest"] is not None:
        _object(provenance["effectiveRequest"])
    if provenance["kind"] == "generated" and (provenance["producer"] is None
            or not provenance["producer"]["capability"] or provenance["brief"] is None
            or provenance["prompt"] is None or provenance["effectiveRequest"] is None):
        raise DocumentError("Generated provenance lacks actual request")


def validate_document(document):
    _tree(document)
    _shape(document, "schemaVersion documentType projectId projectRevisionId parentRevisionId title origin messages acceptedBrief promptDraft chosenSuggestionEffects generationAttempt selectedArtwork extensions")
    if type(document["schemaVersion"]) is not int or document["schemaVersion"] != 1 or document["documentType"] != "visualizeit-project":
        raise DocumentError("Unsupported project schema")
    _id(document["projectId"])
    _id(document["projectRevisionId"])
    _text(document["title"])
    if document["parentRevisionId"] is not None:
        _id(document["parentRevisionId"])
        if document["parentRevisionId"] == document["projectRevisionId"]:
            raise DocumentError("Self-parent revision")
    if document["origin"] is not None:
        _shape(document["origin"], "projectId revisionId")
        for item in document["origin"].values():
            _id(item)
    brief = document["acceptedBrief"]
    if brief is not None:
        _brief(brief)
    draft = document["promptDraft"]
    if draft is not None:
        _shape(draft, "revisionId baseBriefRevisionId text capability settings")
        _id(draft["revisionId"])
        if draft["baseBriefRevisionId"] is not None:
            _id(draft["baseBriefRevisionId"])
        _text(draft["text"])
        if draft["capability"] is not None:
            _text(draft["capability"])
        _object(draft["settings"])
    for key in ("messages", "chosenSuggestionEffects"):
        entries = document[key]
        if not isinstance(entries, list) or len(entries) > MAX_ENTRIES:
            raise DocumentError("Entry count/type")
        seen = set()
        for item in entries:
            if not isinstance(item, dict):
                raise DocumentError("Entry object")
            _id(item.get("id"))
            if item["id"] in seen:
                raise DocumentError("Duplicate entry ID")
            seen.add(item["id"])
            if key == "messages":
                _shape(item, "id role source text artworkIds")
                _text(item["source"])
                if item["role"] not in ("user", "assistant", "system"):
                    raise DocumentError("Message role")
                _text(item["text"])
                if not isinstance(item["artworkIds"], list):
                    raise DocumentError("Message references")
                for identifier in item["artworkIds"]:
                    _id(identifier)
                if len(set(item["artworkIds"])) != len(item["artworkIds"]):
                    raise DocumentError("Duplicate message reference")
            else:
                _shape(item, "id actionId projectId contextRevisionId draftRevisionId briefRevisionId effect resultDraftRevisionId")
                for field in ("actionId", "projectId", "contextRevisionId"):
                    _id(item[field])
                for field in ("draftRevisionId", "briefRevisionId", "resultDraftRevisionId"):
                    if item[field] is not None:
                        _id(item[field])
                _effect(item["effect"])
    attempt = document["generationAttempt"]
    if attempt is not None:
        _shape(attempt, "id projectId contextRevisionId brief prompt capability effectiveRequest")
        for key in ("id", "projectId", "contextRevisionId"):
            _id(attempt[key])
        _brief(attempt["brief"])
        _text(attempt["prompt"])
        _text(attempt["capability"])
        _object(attempt["effectiveRequest"])
    _object(document["extensions"])
    if len(_json(document["extensions"])) > MAX_EXTENSIONS:
        raise DocumentError("Extension byte limit")
    metadata = copy.deepcopy(document)
    if document["selectedArtwork"] is not None:
        # Reject metadata overflow before allocating/decoding the selected PNG.
        try:
            metadata["selectedArtwork"]["png"]["base64"] = ""
        except (KeyError, TypeError) as exc:
            raise DocumentError("Artwork shape") from exc
    if len(_json(metadata)) > MAX_METADATA:
        raise DocumentError("Metadata byte limit")
    if document["selectedArtwork"] is not None:
        _artwork(document["selectedArtwork"])
    return copy.deepcopy(document)


def _effect(effect):
    _shape(effect, "type payload")
    _object(effect["payload"])
    payload = effect["payload"]
    if effect["type"] == "draft":
        if not payload or not set(payload) <= {"text", "capability", "settings"}:
            raise DocumentError("Draft chip effect")
        for key in ("text", "capability"):
            if key in payload and not (key == "capability" and payload[key] is None):
                _text(payload[key])
        if "settings" in payload:
            _object(payload["settings"])
    elif effect["type"] == "clarify":
        _shape(payload, "question")
        _text(payload["question"])
    elif effect["type"] == "import":
        _shape(payload, "")
    else:
        raise DocumentError("Unknown chip type")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DocumentError("Duplicate JSON member")
        result[key] = value
    return result


def _parse(data, *, max_depth=MAX_DEPTH):
    if not isinstance(data, bytes) or len(data) > MAX_FILE:
        raise DocumentError("Document file limit/type")
    try:
        text = data.decode("utf-8", "strict")
        depth, quoted, escaped = 0, False, False
        for char in text:
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "[{":
                depth += 1
                if depth > max_depth:
                    raise DocumentError("JSON nesting limit")
            elif char in "]}":
                depth -= 1
        return json.loads(text, object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(DocumentError("Nonfinite JSON")))
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise DocumentError("Invalid bounded JSON") from exc


def decode_document(data):
    return validate_document(_parse(data))


def encode_document(document):
    data = _json(validate_document(document))
    if len(data) > MAX_FILE:
        raise DocumentError("Document file limit")
    return data


def new_project(title="Untitled"):
    return validate_document({"schemaVersion": 1, "documentType": "visualizeit-project",
        "projectId": new_id(), "title": title, "projectRevisionId": new_id(), "parentRevisionId": None, "origin": None,
        "messages": [], "acceptedBrief": None, "promptDraft": None,
        "chosenSuggestionEffects": [], "generationAttempt": None, "selectedArtwork": None, "extensions": {}})


def make_artwork(png_bytes, provenance, artwork_id=None):
    width, height = validate_png(png_bytes)
    result = {"id": artwork_id or new_id(), "png": {"sha256": hashlib.sha256(png_bytes).hexdigest(),
        "byteCount": len(png_bytes), "width": width, "height": height,
        "base64": base64.b64encode(png_bytes).decode("ascii")}, "provenance": copy.deepcopy(provenance)}
    _tree(result)
    _artwork(result)
    return result


class ProjectStore:
    """Copy-on-write manifests and assets, activated by one atomic pointer pair.

    Store-wide OS locks and reentrant thread locks serialize all writers,
    including separate instances/processes. expected_revision rejects stale work.
    """
    def __init__(self, root, fault=None):
        self.root = Path(root)
        self.fault = fault or (lambda stage: None)
        self.recovered = False

    @contextmanager
    def transaction(self):
        key = os.path.normcase(str(self.root.resolve()))
        with _LOCKS_GUARD:
            mutex = _LOCKS.setdefault(key, threading.RLock())
        with mutex:
            held = getattr(_LOCK_LOCAL, "held", None)
            if held is None:
                held = _LOCK_LOCAL.held = set()
            if key in held:
                yield
                return
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / ".store.lock").open("a+b") as handle:
                if handle.seek(0, 2) == 0:
                    handle.write(b"0")
                    handle.flush()
                deadline = time.monotonic() + 10
                while True:
                    try:
                        handle.seek(0)
                        if os.name == "nt":
                            import msvcrt
                            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        else:
                            import fcntl
                            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise ConflictError("Store busy; no mutation performed")
                        time.sleep(0.01)
                held.add(key)
                try:
                    yield
                finally:
                    held.remove(key)
                    handle.seek(0)
                    if os.name == "nt":
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _directory(self, project_id):
        return self.root / "projects" / _storage_name(project_id)

    def _manifest_path(self, project_id, commit_id):
        return self._directory(project_id) / "revisions" / (_storage_name(commit_id) + ".json")

    def _project_ids(self):
        directory = self.root / "projects"
        if not directory.exists():
            return
        for path in sorted(directory.iterdir()):
            if not path.is_dir() or not (path / "pointer.json").exists():
                continue
            try:
                pointer = _parse(self._read(path / "pointer.json"))
                _shape(pointer, "current previous")
                for commit_id in (pointer["current"], pointer["previous"]):
                    if commit_id is None:
                        continue
                    try:
                        manifest = _parse(self._read(path / "revisions" / (_storage_name(commit_id) + ".json")), max_depth=MAX_DEPTH + 1)
                        project_id = manifest["document"]["projectId"]
                        if self._directory(project_id) != path:
                            raise DocumentError("Storage identity mismatch")
                        # Verify complete state rather than trusting a filename.
                        self._state(project_id)
                        yield project_id
                        break
                    except (OSError, DocumentError, KeyError, TypeError, AttributeError):
                        continue
                else:
                    raise RecoveryError("Project identity/current/previous invalid; preserved")
            except (OSError, DocumentError, KeyError, TypeError, AttributeError) as exc:
                raise RecoveryError("Project listing contains unrecoverable state; preserved") from exc

    def _read(self, path, limit=MAX_FILE):
        if path.stat().st_size > limit:
            raise DocumentError("Local file limit")
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise DocumentError("Local file limit")
        return data

    def _pointer(self, project_id):
        pointer = _parse(self._read(self._directory(project_id) / "pointer.json"))
        _shape(pointer, "current previous")
        _id(pointer["current"])
        if pointer["previous"] is not None:
            _id(pointer["previous"])
        return pointer

    def _manifest(self, project_id, commit_id):
        _id(commit_id)
        raw = _parse(self._read(self._manifest_path(project_id, commit_id)), max_depth=MAX_DEPTH + 1)
        _shape(raw, "document catalog parentCommit")
        if raw["parentCommit"] is not None:
            _id(raw["parentCommit"])
        _object(raw["catalog"])
        for identifier, artwork in raw["catalog"].items():
            _id(identifier)
            if artwork.get("id") != identifier:
                raise DocumentError("Catalog identity")
            self._hydrate(artwork)
            _artwork(artwork)
        document = raw["document"]
        if document["selectedArtwork"] is not None:
            selected_id = document["selectedArtwork"]["id"]
            if selected_id not in raw["catalog"]:
                raise DocumentError("Selected artwork absent from catalog")
            self._hydrate(document["selectedArtwork"])
            if not _semantic_equal(document["selectedArtwork"], raw["catalog"][selected_id]):
                raise DocumentError("Selected catalog mismatch")
        document = validate_document(document)
        if document["projectId"] != project_id:
            raise DocumentError("Project identity mismatch")
        return document, raw["catalog"]

    def _hydrate(self, artwork):
        png = artwork["png"]
        digest = png["sha256"]
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise DocumentError("Unsafe digest")
        data = self._read(self.root / "assets" / (digest + ".png"), MAX_PNG)
        png["base64"] = base64.b64encode(data).decode("ascii")

    def _state(self, project_id):
        self.recovered = False
        try:
            pointer = self._pointer(project_id)
        except (OSError, DocumentError, KeyError, TypeError) as exc:
            raise RecoveryError("Project activation pointer unavailable; files preserved") from exc
        for index, commit_id in enumerate((pointer["current"], pointer["previous"])):
            if commit_id is None:
                continue
            try:
                document, catalog = self._manifest(project_id, commit_id)
                self.recovered = index == 1
                return document, catalog, commit_id
            except (OSError, DocumentError, KeyError, TypeError, AttributeError):
                continue
        raise RecoveryError("Current and previous invalid; files preserved")

    def open(self, project_id):
        return self._state(project_id)[0]

    def catalog(self, project_id):
        return self._state(project_id)[1]

    def list(self):
        return [{key: document[key] for key in ("projectId", "projectRevisionId", "title")}
                for project_id in self._project_ids() for document in [self.open(project_id)]]

    def _write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

    def _commit(self, document, catalog, previous=None):
        document = validate_document(document)
        catalog = copy.deepcopy(catalog)
        selected = document["selectedArtwork"]
        if selected is not None:
            existing = catalog.get(selected["id"])
            if existing is not None and not _semantic_equal(existing, selected):
                raise ConflictError("Immutable artwork conflict")
            catalog[selected["id"]] = copy.deepcopy(selected)
        # Validate every catalog item before any writes.
        for identifier, artwork in catalog.items():
            if artwork["id"] != identifier:
                raise ConflictError("Catalog identity mismatch")
            _artwork(artwork)
        for project_id in self._project_ids():
            _, existing_catalog, _ = self._state(project_id)
            for identifier, artwork in catalog.items():
                if identifier in existing_catalog and not _semantic_equal(existing_catalog[identifier], artwork):
                    raise ConflictError("Immutable artwork identity conflict across projects")
        # Local storage wraps portable state in one additional object. Preflight
        # the complete envelope (including accumulated catalog) against the same
        # reader byte limit before writing an asset, manifest or active pointer.
        local_document = copy.deepcopy(document)
        if selected is not None:
            local_document["selectedArtwork"]["png"]["base64"] = ""
        local_catalog = copy.deepcopy(catalog)
        for artwork in local_catalog.values():
            artwork["png"]["base64"] = ""
        manifest_data = _json({"document": local_document, "catalog": local_catalog, "parentCommit": previous})
        _parse(manifest_data, max_depth=MAX_DEPTH + 1)
        directory = self._directory(document["projectId"])
        for artwork in catalog.values():
            png = artwork["png"]
            asset = self.root / "assets" / (png["sha256"] + ".png")
            data = base64.b64decode(png["base64"], validate=True)
            if asset.exists():
                if self._read(asset, MAX_PNG) != data:
                    raise ConflictError("Immutable asset conflict")
            else:
                staging = asset.with_name(new_id() + ".stage")
                self.fault("assetStaging")
                self._write(staging, data)
                os.replace(staging, asset)
                self.fault("assetCreated")
        commit_id = new_id()
        self.fault("manifestStaging")
        self._write(self._manifest_path(document["projectId"], commit_id), manifest_data)
        staging = directory / (new_id() + ".stage")
        self._write(staging, _json({"current": commit_id, "previous": previous}))
        self.fault("beforeActivation")
        os.replace(staging, directory / "pointer.json")
        self.fault("afterActivation")
        return copy.deepcopy(document)

    @_serialized
    def create(self, title="Untitled"):
        return self._commit(new_project(title), {})

    @_serialized
    def update(self, document, expected_revision):
        document = validate_document(document)
        current, catalog, previous = self._state(document["projectId"])
        if current["projectRevisionId"] != expected_revision or document["projectRevisionId"] != expected_revision:
            raise ConflictError("Stale project revision")
        document["parentRevisionId"] = expected_revision
        document["projectRevisionId"] = new_id()
        return self._commit(document, catalog, previous)

    def _history(self, project_id):
        pointer = self._pointer(project_id)
        pending = [pointer["current"], pointer["previous"]]
        seen = set()
        while pending:
            commit_id = pending.pop()
            if commit_id is None or commit_id in seen:
                continue
            seen.add(commit_id)
            try:
                raw = _parse(self._read(self._manifest_path(project_id, commit_id)), max_depth=MAX_DEPTH + 1)
                _shape(raw, "document catalog parentCommit")
                parent = raw["parentCommit"]
                if parent is not None:
                    _id(parent)
                pending.append(parent)
                yield self._manifest(project_id, commit_id)
            except (OSError, DocumentError, KeyError, TypeError, AttributeError):
                continue

    def _check_known_revision(self, project_id, document):
        for known, catalog in self._history(project_id):
            if known["projectRevisionId"] == document["projectRevisionId"] and _json(known) != _json(document):
                raise ConflictError("Same revision with different semantic content")
            selected = document["selectedArtwork"]
            if selected is not None and selected["id"] in catalog and not _semantic_equal(catalog[selected["id"]], selected):
                raise ConflictError("Immutable artwork conflict")

    def import_snapshot(self, data, choice=None):
        document = decode_document(data)
        if choice not in (None, "update", "copy", "cancel"):
            raise DocumentError("Unknown import choice")
        if choice == "cancel":
            return None
        with self.transaction():
            return self._import_validated(document, choice)

    def _import_validated(self, document, choice):
        project_id = document["projectId"]
        exists = (self._directory(project_id) / "pointer.json").exists()
        catalog, previous = {}, None
        if exists:
            current, catalog, previous = self._state(project_id)
            self._check_known_revision(project_id, document)
            if _semantic_equal(current, document):
                return current
            if choice is None:
                raise ConflictError("Explicit update/copy/cancel required")
        if choice == "copy":
            origin = {"projectId": project_id, "revisionId": document["projectRevisionId"]}
            document.update(projectId=new_id(), projectRevisionId=new_id(), parentRevisionId=None, origin=origin)
            catalog, previous = {}, None
        return self._commit(document, catalog, previous)

    def export_snapshot(self, project_id):
        return encode_document(self.open(project_id))


def _current(store, project_id, expected_revision):
    document = store.open(project_id)
    if document["projectRevisionId"] != expected_revision:
        raise ConflictError("Stale project revision")
    return document


@_transactional
def append_message(store, project_id, expected_revision, text, role="user", artwork_ids=None, source="user"):
    document = _current(store, project_id, expected_revision)
    document["messages"].append({"id": new_id(), "role": role, "source": source, "text": text, "artworkIds": artwork_ids or []})
    return store.update(document, expected_revision)


@_transactional
def edit_draft(store, project_id, expected_revision, text, capability=None, settings=None):
    document = _current(store, project_id, expected_revision)
    draft = document["promptDraft"]
    if draft is None:
        brief = document["acceptedBrief"]
        draft = {"revisionId": new_id(), "baseBriefRevisionId": brief["revisionId"] if brief else None,
                 "text": text, "capability": None, "settings": {}}
        document["promptDraft"] = draft
    draft.update(revisionId=new_id(), text=text)
    if capability is not None:
        draft["capability"] = capability
    if settings is not None:
        draft["settings"] = copy.deepcopy(settings)
    return store.update(document, expected_revision)


def draft_status(document):
    """No imported mutable Boolean can independently assert acceptance."""
    draft, brief = document["promptDraft"], document["acceptedBrief"]
    if draft is None:
        return "absent"
    if draft["baseBriefRevisionId"] != (brief["revisionId"] if brief else None):
        return "stale"
    return "clean" if brief is not None and draft["text"] == brief["text"] else "unaccepted"


def _accept(document, text=None):
    draft, brief = document["promptDraft"], document["acceptedBrief"]
    previous_status = draft_status(document)
    if text is None and draft is None:
        raise DocumentError("No draft to accept")
    accepting_draft = draft is not None and (text is None or text == draft["text"])
    text = draft["text"] if text is None else text
    _text(text)
    if brief is None or text != brief["text"]:
        document["acceptedBrief"] = {"revisionId": new_id(), "text": text}
    brief = document["acceptedBrief"]
    if accepting_draft or previous_status in ("absent", "clean"):
        if draft is None:
            draft = {"capability": None, "settings": {}}
            document["promptDraft"] = draft
        draft.update(revisionId=new_id(), baseBriefRevisionId=brief["revisionId"], text=text)


@_transactional
def accept_brief(store, project_id, expected_revision, text=None):
    document = _current(store, project_id, expected_revision)
    _accept(document, text)
    return store.update(document, expected_revision)


@_transactional
def load_accepted_brief(store, project_id, expected_revision):
    document = _current(store, project_id, expected_revision)
    brief = document["acceptedBrief"]
    if brief is None:
        raise DocumentError("No accepted brief")
    draft = document["promptDraft"] or {"capability": None, "settings": {}}
    draft.update(revisionId=new_id(), baseBriefRevisionId=brief["revisionId"], text=brief["text"])
    document["promptDraft"] = draft
    return store.update(document, expected_revision)


@_transactional
def select_artwork(store, project_id, expected_revision, artwork):
    document = _current(store, project_id, expected_revision)
    document["selectedArtwork"] = copy.deepcopy(artwork)
    return store.update(document, expected_revision)


@_transactional
def prepare_generation(store, project_id, expected_revision, capability, effective_request, supported):
    document = _current(store, project_id, expected_revision)
    if not supported or not capability or document["promptDraft"] is None or document["promptDraft"]["capability"] != capability:
        raise DocumentError("Unsupported generation; save intent separately")
    if draft_status(document) == "stale":
        raise ConflictError("Stale draft: save draft as brief or load accepted brief")
    _object(effective_request)
    _accept(document)
    # Own the post-acceptance revision so the saved request and result guard agree.
    revision_id = new_id()
    attempt = {"id": new_id(), "projectId": project_id, "contextRevisionId": revision_id,
        "brief": copy.deepcopy(document["acceptedBrief"]), "prompt": document["promptDraft"]["text"],
        "capability": capability, "effectiveRequest": copy.deepcopy(effective_request)}
    document["generationAttempt"] = attempt
    document.update(projectRevisionId=revision_id, parentRevisionId=expected_revision)
    current, catalog, previous = store._state(project_id)
    if current["projectRevisionId"] != expected_revision:
        raise ConflictError("Stale generation revision")
    committed = store._commit(document, catalog, previous)
    return committed, copy.deepcopy(attempt)


@_transactional
def complete_generation(store, attempt, png_bytes, producer, active_project_id):
    project_id = attempt["projectId"]
    current, catalog, previous = store._state(project_id)
    # Attempts must be genuine saved request records, including when now stale.
    genuine = False
    for historical, _ in store._history(project_id):
        if _semantic_equal(historical["generationAttempt"], attempt):
            genuine = True
            break
    if not genuine or producer.get("capability") != attempt["capability"]:
        raise ConflictError("Unknown or mismatched generation request")
    artwork = make_artwork(png_bytes, {"kind": "generated", "producer": producer,
        "brief": attempt["brief"], "prompt": attempt["prompt"], "effectiveRequest": attempt["effectiveRequest"], "source": None}, artwork_id=attempt["id"])
    if artwork["id"] in catalog and not _semantic_equal(catalog[artwork["id"]], artwork):
        raise ConflictError("Attempt already completed differently")
    catalog[artwork["id"]] = artwork
    if current["projectRevisionId"] == attempt["contextRevisionId"] and active_project_id == project_id:
        current["selectedArtwork"] = artwork
        current.update(parentRevisionId=current["projectRevisionId"], projectRevisionId=new_id())
    return store._commit(current, catalog, previous)


@_transactional
def apply_chip(store, project_id, expected_revision, chip):
    _shape(chip, "id actionId projectId contextRevisionId draftRevisionId briefRevisionId effect")
    document = _current(store, project_id, expected_revision)
    for key in ("id", "actionId", "projectId", "contextRevisionId"):
        _id(chip[key])
    for key in ("draftRevisionId", "briefRevisionId"):
        if chip[key] is not None:
            _id(chip[key])
    if (chip["projectId"], chip["contextRevisionId"], chip["draftRevisionId"], chip["briefRevisionId"]) != (project_id, expected_revision, (document["promptDraft"]["revisionId"] if document["promptDraft"] else None), (document["acceptedBrief"]["revisionId"] if document["acceptedBrief"] else None)):
        raise ConflictError("Stale chip context")
    if any(effect["id"] == chip["id"] for effect in document["chosenSuggestionEffects"]):
        raise ConflictError("Chip already applied")
    _effect(chip["effect"])
    effect = copy.deepcopy(chip)
    if effect["effect"]["type"] == "draft":
        if document["promptDraft"] is None:
            brief = document["acceptedBrief"]
            document["promptDraft"] = {"revisionId": new_id(), "baseBriefRevisionId": brief["revisionId"] if brief else None,
                "text": brief["text"] if brief else "", "capability": None, "settings": {}}
        document["promptDraft"].update(effect["effect"]["payload"])
        document["promptDraft"].update(revisionId=new_id())
    effect["resultDraftRevisionId"] = document["promptDraft"]["revisionId"] if document["promptDraft"] else None
    document["chosenSuggestionEffects"].append(effect)
    return store.update(document, expected_revision)


def artwork_available(store, project_id, artwork_id):
    return artwork_id in store.catalog(project_id)
