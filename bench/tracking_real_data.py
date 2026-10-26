"""Acquire a pinned, bounded real HB tracking corpus, never a full ZIP archive.

This is a dataset-reference initialized rigid-tracking benchmark, not an official
BOP challenge score or independent human annotation. Only the ten onboarding
frames expose object pose, depth and visible mask to inference. Evaluation RGB
is label-free in inference.json; evaluator.json contains the reference labels.

Run from the repository root::

    python -m bench.tracking_real_data acquire --output .cache/tracking-benchmark/real-v1
    python -m bench.tracking_real_data verify --output .cache/tracking-benchmark/real-v1

The frozen recipe is emitted under ``--recipe-output`` (default: the dataset
directory); commit that recipe, not a multi-gigabyte duplicate of upstream data.
"""
from __future__ import annotations

import argparse
import binascii
from concurrent.futures import ThreadPoolExecutor
import hashlib
import gzip
import io
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import struct
import statistics as statistics_lib
import tarfile
import time
import urllib.error
import zlib

from .remote_zip import RemoteZip

REVISION = "985ebaef7b1bd4bffb2039f18dd81faa647fd060"
BASE_URL = f"https://huggingface.co/datasets/bop-benchmark/hb/resolve/{REVISION}/"
RGB_ARCHIVE = "hb_val_primesense.zip"
MODEL_ARCHIVE = "hb_models.zip"
LICENSE = "CC0-1.0"
DATASET_PAGE = "https://bop.felk.cvut.cz/datasets/#HB"
AUTHOR = "Roman Kaskman, Ivan Shugurov, Sergey Zakharov, Slobodan Ilic (HomebrewedDB, ICCVW 2019); BOP conversion by the BOP authors"
SCENES = ("000001", "000002", "000003", "000004")
OBJECTS_PER_SCENE = 3
FRAME_COUNT = 310
ONBOARDING_COUNT = 10
MAX_DATASET_BYTES = 3 * 1024**3
MIN_FREE_BYTES = 1024**3
MAX_RANGE_BYTES = 16 * 1024**2
MAX_COALESCE_GAP = 256 * 1024
MAX_MODEL_EXPANDED_BYTES = 64 * 1024**2
ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def safe_member(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Unsafe member path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError("Unsafe member path")
    return path


def atomic_write(path, raw):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Frozen output differs; use a fresh directory: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    if temporary.exists():
        raise ValueError(f"Unfinished output needs inspection: {temporary}")
    temporary.write_bytes(raw)
    temporary.replace(path)


def retry(call):
    for attempt in range(3):
        try:
            return call()
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def ranges_for_members(entries, members, *, range_limit=MAX_RANGE_BYTES, gap_limit=MAX_COALESCE_GAP, archive_size=None):
    """Coalesce local ZIP records; no member or range exceeds the hard bound.

    The next central-directory local offset bounds the complete local record,
    including variable extra fields and optional data descriptors. Selected
    records are never expanded before the response has passed range validation.
    """
    ordered = sorted((row["offset"], name) for name, row in entries.items())
    next_offsets = {name: ordered[i + 1][0] for i, (_, name) in enumerate(ordered[:-1])}
    spans = []
    for name in sorted(set(members), key=lambda n: entries[n]["offset"]):
        safe_member(name)
        row = entries[name]
        if row["flags"] & 1 or row["method"] not in (0, 8):
            raise ValueError("Encrypted/unsupported ZIP member")
        start = row["offset"]
        # The final record is rare but bounded using ZIP's maximum extra length.
        stop = next_offsets.get(name, start + 30 + len(name.encode()) + 65535 + row["compressed"])
        if archive_size is not None:
            stop = min(stop, archive_size)
        expanded_limit = MAX_MODEL_EXPANDED_BYTES if name.startswith("models/") and name.endswith(".ply") else 32 * 1024**2
        if row["size"] > expanded_limit or not 0 < stop - start <= range_limit:
            raise ValueError(f"Selected ZIP member exceeds bounded range: {name}")
        if spans and start - spans[-1][1] <= gap_limit and stop - spans[-1][0] <= range_limit:
            spans[-1][1] = max(stop, spans[-1][1])
            spans[-1][2].append(name)
        else:
            spans.append([start, stop, [name]])
    return spans


def extract_record(payload, offset, name, row):
    expanded_limit = MAX_MODEL_EXPANDED_BYTES if name.startswith("models/") and name.endswith(".ply") else 32 * 1024**2
    if not 0 <= row["size"] <= expanded_limit:
        raise ValueError("ZIP expansion exceeds member-type limit")
    if offset < 0 or len(payload) - offset < 30:
        raise ValueError("Truncated ZIP local header")
    fields = struct.unpack_from("<4s5H3L2H", payload, offset)
    if fields[0] != b"PK\x03\x04" or fields[2] & 1 or fields[3] != row["method"]:
        raise ValueError("ZIP header identity/method mismatch")
    name_length, extra_length = fields[-2:]
    raw_name = payload[offset + 30: offset + 30 + name_length]
    decoded_name = raw_name.decode("utf-8" if fields[2] & 2048 else "cp437")
    if decoded_name != name:
        raise ValueError("ZIP local member name differs from central directory")
    start = offset + 30 + name_length + extra_length
    raw = payload[start:start + row["compressed"]]
    if len(raw) != row["compressed"]:
        raise ValueError("Truncated ZIP member")
    if row["method"] == 0:
        result = raw
    else:
        decoder = zlib.decompressobj(-15)
        result = decoder.decompress(raw, row["size"] + 1)
        if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
            raise ValueError("Invalid ZIP expansion")
    if len(result) != row["size"] or binascii.crc32(result) & 0xffffffff != row["crc"]:
        raise ValueError("ZIP member size/CRC differs")
    return result


def cached_member(output, archive, member):
    target = output / "metadata" / member
    if target.exists():
        raw = target.read_bytes()
        row = archive.entries[member]
        if len(raw) != row["size"] or binascii.crc32(raw) & 0xffffffff != row["crc"]:
            raise ValueError("Cached metadata size/CRC differs")
        return raw
    raw = retry(lambda: archive.read(member))
    atomic_write(target, raw)
    return raw


def selection_for(metadata):
    """Freeze scene coverage by identity/order, never by tracker predictions.

    For each of four scene IDs take the first three globally unused object IDs
    appearing exactly once in every frame 0..309. All 300 evaluation frames stay
    included regardless of visibility, motion, difficulty or eventual failure.
    """
    selected, used = [], set()
    for scene in SCENES:
        gt = metadata[scene]["scene_gt.json"]
        candidates = sorted({row["obj_id"] for rows in gt.values() for row in rows} - used)
        accepted = []
        for object_id in candidates:
            instances = []
            for frame_id in range(FRAME_COUNT):
                found = [i for i, row in enumerate(gt.get(str(frame_id), [])) if row["obj_id"] == object_id]
                if len(found) != 1:
                    break
                instances.append(found[0])
            if len(instances) == FRAME_COUNT:
                accepted.append(dict(sceneId=scene, objectId=object_id, frameIds=list(range(FRAME_COUNT)), instanceIds=instances))
            if len(accepted) == OBJECTS_PER_SCENE:
                break
        if len(accepted) != OBJECTS_PER_SCENE:
            raise ValueError(f"Scene {scene} does not satisfy the fixed selection policy")
        selected.extend(accepted)
        used.update(choice["objectId"] for choice in accepted)
    return selected


def member_plan(selected, test_entries, model_entries):
    members = {RGB_ARCHIVE: set(), MODEL_ARCHIVE: set()}
    for choice in selected:
        members[MODEL_ARCHIVE].add(f'models/obj_{choice["objectId"]:06d}.ply')
        for frame, instance in zip(choice["frameIds"], choice["instanceIds"]):
            prefix = f'val_primesense/{choice["sceneId"]}'
            rgb = f"{prefix}/rgb/{frame:06d}.png"
            if rgb not in test_entries:
                rgb = f"{prefix}/rgb/{frame:06d}.jpg"
            members[RGB_ARCHIVE].update((rgb, f"{prefix}/depth/{frame:06d}.png", f"{prefix}/mask_visib/{frame:06d}_{instance:06d}.png"))
    members[MODEL_ARCHIVE].add("models/models_info.json")
    for archive, names in members.items():
        entries = test_entries if archive == RGB_ARCHIVE else model_entries
        if any(name not in entries for name in names):
            raise ValueError("Pinned source is missing required member")
    return members


def acquire_members(output, archive_name, archive, names, *, workers):
    pending, hashes = [], {}
    for name in sorted(names):
        target = output / safe_member(name)
        row = archive.entries[name]
        if target.exists():
            raw = target.read_bytes()
            if len(raw) != row["size"] or binascii.crc32(raw) & 0xffffffff != row["crc"]:
                raise ValueError(f"Cached member differs: {name}")
            hashes[name] = sha(raw)
        else:
            pending.append(name)
    ranges = ranges_for_members(archive.entries, pending, archive_size=archive.total)
    print(f"{archive_name}: {len(pending)} pending members in {len(ranges)} bounded ranges", flush=True)

    def fetch(span):
        start, stop, members = span
        raw, total = retry(lambda: archive.request(f"{start}-{stop - 1}"))
        if total != archive.total:
            raise ValueError("Pinned archive size changed")
        result = {}
        for name in members:
            data = extract_record(raw, archive.entries[name]["offset"] - start, name, archive.entries[name])
            if shutil.disk_usage(output).free - len(data) < MIN_FREE_BYTES:
                raise ValueError("Acquisition would cross 1 GiB free-space reserve")
            atomic_write(output / name, data)
            result[name] = sha(data)
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(fetch, ranges)):
            hashes.update(result)
            if index % 5 == 0 or index + 1 == len(ranges):
                print(f"{archive_name}: {index + 1}/{len(ranges)} ranges complete", flush=True)
    return hashes, sum(stop - start for start, stop, _ in ranges)


def manifests(selected, metadata, hashes, selection_digest):
    common = dict(schemaVersion=1, dataset="BOP HB val_primesense", datasetRevision=REVISION,
                  source=DATASET_PAGE, license=LICENSE, authors=AUTHOR,
                  evidenceKind="real_dataset_reference", benchmarkId="visualizeit-hb-tracking-v1",
                  selectionDigest=selection_digest, labelKind="Dataset-reference 6D poses/depth/masks, not independent human annotation",
                  sourceFrameRate=None, timestampKind="frame_index_no_fps")
    inference = dict(**common, objects=[])
    evaluator = dict(**common, objects=[])
    def ref(name):
        return dict(path=name, sha256=hashes[name])
    for choice in selected:
        object_id, scene = choice["objectId"], choice["sceneId"]
        obj_common = dict(objectId=object_id, name=f"HB object {object_id}", sceneId=scene,
                          referenceId=f"hb-{REVISION[:12]}-{scene}-{object_id:06d}-000000",
                          model=ref(f"models/obj_{object_id:06d}.ply"))
        inf_obj = dict(**obj_common, onboarding=[], frames=[])
        eval_obj = dict(**obj_common, frames=[])
        for logical_index, (frame, instance) in enumerate(zip(choice["frameIds"], choice["instanceIds"])):
            prefix = f"val_primesense/{scene}"
            rgb = f"{prefix}/rgb/{frame:06d}.png"
            if rgb not in hashes:
                rgb = f"{prefix}/rgb/{frame:06d}.jpg"
            record = dict(frameId=frame, logicalIndex=logical_index, timestampKind="frame_index_no_fps",
                          timestampSeconds=None, calibration=metadata[scene]["scene_camera.json"][str(frame)], rgb=ref(rgb))
            depth = ref(f"{prefix}/depth/{frame:06d}.png")
            mask = ref(f"{prefix}/mask_visib/{frame:06d}_{instance:06d}.png")
            pose = metadata[scene]["scene_gt.json"][str(frame)][instance]
            if logical_index < ONBOARDING_COUNT:
                inf_obj["onboarding"].append(dict(**record, depth=depth, onboardingMask=mask, evaluationPose=pose))
            else:
                inf_obj["frames"].append(record)
                eval_obj["frames"].append(dict(**record, depth=depth, evaluationMask=mask, evaluationPose=pose,
                                               evaluationInfo=metadata[scene]["scene_gt_info.json"][str(frame)][instance]))
        inference["objects"].append(inf_obj)
        evaluator["objects"].append(eval_obj)
    inference["evaluatorManifestSHA256"] = sha(canonical(evaluator))
    return inference, evaluator


def corpus_statistics(inference, evaluator):
    """Describe coverage from source annotations without looking at predictions."""
    sequences = []
    for obj, reference in zip(inference["objects"], evaluator["objects"]):
        if obj["objectId"] != reference["objectId"]:
            raise ValueError("Manifest object ordering differs")
        rows = reference["frames"]
        visibility = [row["evaluationInfo"]["visib_fract"] for row in rows]
        directions = []
        for row in rows:
            pose = row["evaluationPose"]
            r, t = pose["cam_R_m2c"], pose["cam_t_m2c"]
            direction = [-sum(r[3*j+k]*t[j] for j in range(3)) for k in range(3)]
            norm = math.sqrt(sum(value*value for value in direction))
            directions.append([value/norm for value in direction])
        minimum_dot = min(sum(a*b for a, b in zip(left, right))
                          for i, left in enumerate(directions) for right in directions[i:])
        sequences.append(dict(objectId=obj["objectId"], sceneId=obj["sceneId"], scoredFrames=len(rows),
                              minimumVisibleFraction=min(visibility), medianVisibleFraction=statistics_lib.median(visibility),
                              visibleFractionAtLeastHalfFrames=sum(value >= .5 for value in visibility),
                              viewpointSpanDegrees=math.degrees(math.acos(max(-1., min(1., minimum_dot)))),
                              modelSHA256=obj["model"]["sha256"]))
    return dict(schemaVersion=1, evidenceKind="original_dataset_reference_metadata",
                inferenceSHA256=sha(canonical(inference)), realObjects=len(sequences),
                scenes=len({row["sceneId"] for row in sequences}),
                uniqueRGBFrames=len({row["rgb"]["path"] for obj in inference["objects"]
                                     for row in obj["onboarding"]+obj["frames"]}),
                scoredObjectFrames=sum(row["scoredFrames"] for row in sequences),
                sourceFPS=None, sequences=sequences)


def acquire(output, recipe_output=None, *, workers=4):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < MIN_FREE_BYTES:
        raise ValueError("Need at least 1 GiB free before acquisition")
    archives = {name: retry(lambda name=name: RemoteZip(BASE_URL + name)) for name in (RGB_ARCHIVE, MODEL_ARCHIVE)}
    metadata, metadata_hashes = {scene: {} for scene in SCENES}, {}
    metadata_members = [(scene, name, f"val_primesense/{scene}/{name}") for scene in SCENES
                        for name in ("scene_gt.json", "scene_gt_info.json", "scene_camera.json")]
    def fetch_metadata(item):
        scene, name, member = item
        raw = cached_member(output, archives[RGB_ARCHIVE], member)
        return scene, name, member, raw
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for scene, name, member, raw in pool.map(fetch_metadata, metadata_members):
            metadata[scene][name] = json.loads(raw)
            metadata_hashes[member] = sha(raw)
    selected = selection_for(metadata)
    plan = member_plan(selected, archives[RGB_ARCHIVE].entries, archives[MODEL_ARCHIVE].entries)
    planned_bytes = sum(archives[name].entries[member]["size"] for name, members in plan.items() for member in members)
    if planned_bytes > MAX_DATASET_BYTES:
        raise ValueError("Frozen corpus would exceed the 3 GiB data budget")
    selection = dict(schemaVersion=1, dataset="BOP HB val_primesense", datasetRevision=REVISION,
                     recipeVersion="visualizeit-hb-tracking-v1", license=LICENSE, authors=AUTHOR,
                     licenseEvidence=["https://bop.felk.cvut.cz/datasets/", f"https://huggingface.co/datasets/bop-benchmark/hb/blob/{REVISION}/README.md"],
                     selectionPolicy="First three globally unused IDs in each of scenes 1,2,3,4, exactly one instance in every frame 0..309; no tracking-result or visibility filtering",
                     selectedObjects=selected, onboardingFramesPerObject=ONBOARDING_COUNT,
                     evaluationFramesPerObject=FRAME_COUNT-ONBOARDING_COUNT, uniqueRGBFrames=len({n for n in plan[RGB_ARCHIVE] if "/rgb/" in n}),
                     evaluationObjectFrames=len(selected)*(FRAME_COUNT-ONBOARDING_COUNT),
                     sourceFrameRate=None, temporalUnits="frame index only; no physical FPS inferred",
                     realFootage=True, previouslyUsedProjectDataset=True, pretrainedExposure="unknown",
                     metadataSHA256=metadata_hashes, plannedExpandedBytes=planned_bytes,
                     archiveSizes={name: archive.total for name, archive in archives.items()},
                     storagePolicy="Pinned download recipe + hashes in Git; full real RGB/depth/masks cached locally, not duplicated as Git blobs")
    selection_digest = sha(canonical(selection))
    atomic_write(output/"selection.json", canonical(dict(**selection, selectionDigest=selection_digest)))
    hashes, network_ranges = {}, 0
    for name, archive in archives.items():
        current, transferred = acquire_members(output, name, archive, plan[name], workers=workers)
        hashes.update(current)
        network_ranges += transferred
    inference, evaluator = manifests(selected, metadata, hashes, selection_digest)
    atomic_write(output/"inference.json", canonical(inference))
    atomic_write(output/"evaluator.json", canonical(evaluator))
    lock = dict(schemaVersion=1, selectionSHA256=sha((output/"selection.json").read_bytes()),
                inferenceSHA256=sha(canonical(inference)), evaluatorSHA256=sha(canonical(evaluator)),
                datasetRevision=REVISION, expandedResourceBytes=planned_bytes, resources=[
                    dict(archive=name, member=member, bytes=archives[name].entries[member]["size"], sha256=hashes[member])
                    for name in sorted(plan) for member in sorted(plan[name])])
    atomic_write(output/"sources.lock.json", canonical(lock))
    if recipe_output:
        recipe_output = Path(recipe_output)
        atomic_write(recipe_output/"sources.hb-selection.json", (output/"selection.json").read_bytes())
        atomic_write(recipe_output/"sources.hb-lock.json", canonical(lock))
        atomic_write(recipe_output/"sources.hb-statistics.json", canonical(corpus_statistics(inference, evaluator)))
    print(json.dumps(dict(status="complete", objects=len(selected), realEvaluationObjectFrames=len(selected)*300,
                          uniqueRGBFrames=selection["uniqueRGBFrames"], expandedBytes=planned_bytes,
                          currentRunRangeBytes=network_ranges, inferenceSHA256=lock["inferenceSHA256"])), flush=True)
    return lock


def verify(output):
    output = Path(output)
    lock = json.loads((output/"sources.lock.json").read_text())
    for name, key in (("selection.json", "selectionSHA256"), ("inference.json", "inferenceSHA256"), ("evaluator.json", "evaluatorSHA256")):
        if sha((output/name).read_bytes()) != lock[key]:
            raise ValueError(f"Manifest hash differs: {name}")
    for item in lock["resources"]:
        path = output/safe_member(item["member"])
        if path.stat().st_size != item["bytes"] or sha(path.read_bytes()) != item["sha256"]:
            raise ValueError(f"Resource differs: {item['member']}")
    print(f"Verified {len(lock['resources'])} real resources and all sealed manifests", flush=True)
    return lock


def package(output, archive_path):
    """Build a deterministic distributable; never upload or mutate GitHub."""
    output, archive_path = Path(output), Path(archive_path)
    lock = verify(output)
    if archive_path.exists() or archive_path.with_name(archive_path.name+".part").exists():
        raise ValueError("Release archive already exists; retain it or choose a fresh path")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    README = (f"""# VisualizeIT HB tracking corpus v1

Real RGB-D footage from HB / HomebrewedDB (BOP Primesense validation).
Source revision: {REVISION}
Source: {DATASET_PAGE}
License: CC0 1.0 Universal (https://creativecommons.org/publicdomain/zero/1.0/).
Upstream license evidence: https://huggingface.co/datasets/bop-benchmark/hb/blob/{REVISION}/README.md
Credit: {AUTHOR}.

This archive includes the exact 12-object/4-scene frozen selection, 1,240 real
RGB frames plus depth/visible masks/models, ten annotated onboarding frames per
object and 300 scored frames per object (3,600 scored object-frames total).
RGB files are shared between multiple object sequences in each scene; do not
describe 3,600 object-frames as 3,600 unique images. Labels are original dataset
reference poses, not newly collected independent human annotations. The
project previously used HB and pretrained exposure is unknown. No source FPS
is asserted. No private/user media or pretrained model weights are included.

inference.json contains only RGB/calibration/identity for scored frames.
Only the first ten annotated onboarding records supply depth, visible mask
and pose. evaluator.json contains scored reference pose/depth/mask/visibility.
Use separate inference and evaluation stages; do not load evaluator labels in
the tracker. This is initialized tracking, not automatic object detection.

Integrity: sources.lock.json lists every resource's SHA-256 and byte count,
plus manifest seals. Verify from the VisualizeIT checkout:
    python -m bench.tracking_real_data verify --output <extracted>/real-v1

Code and reproducible acquisition: bench/tracking_real_data.py in VisualizeIT.
Inference SHA-256: {lock['inferenceSHA256']}
Evaluator SHA-256: {lock['evaluatorSHA256']}
""").encode()
    paths = sorted({row["member"] for row in lock["resources"]} |
                   {"selection.json", "inference.json", "evaluator.json", "sources.lock.json"})
    temporary = archive_path.with_name(archive_path.name+".part")
    with temporary.open("wb") as target, gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0, compresslevel=1) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
            for relative in paths:
                path = output/safe_member(relative)
                info = tarfile.TarInfo("real-v1/"+relative)
                info.size, info.mode, info.mtime = path.stat().st_size, 0o644, 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with path.open("rb") as source:
                    archive.addfile(info, source)
            info = tarfile.TarInfo("real-v1/README.md")
            info.size, info.mode, info.mtime = len(README), 0o644, 0
            archive.addfile(info, io.BytesIO(README))
    temporary.replace(archive_path)
    receipt = dict(schemaVersion=1, filename=archive_path.name, bytes=archive_path.stat().st_size,
                   sha256=sha_file(archive_path), resources=len(lock["resources"]),
                   datasetRevision=REVISION, inferenceSHA256=lock["inferenceSHA256"],
                   deterministic=True, compression="gzip level 1; mtime=0; sorted USTAR paths with fixed ownership/mode/time",
                   publicationStatus="prepared_locally_not_uploaded")
    atomic_write(archive_path.with_suffix(archive_path.suffix+".json"), canonical(receipt))
    print(json.dumps(receipt), flush=True)
    return receipt


def export_examples(output, destination):
    """Commit-sized CC0 real images plus complete compressed annotation packs.

    This never changes the frozen corpus, and never converts dataset-reference
    labels into supposed new human ground truth.
    """
    from PIL import Image, ImageDraw
    output, destination = Path(output), Path(destination)
    lock = json.loads((output/"sources.lock.json").read_text())
    indexed = {row["member"]: row for row in lock["resources"]}
    destination.mkdir(parents=True, exist_ok=True)
    examples = []
    sheet = Image.new("RGB", (4*320, 2*265), "#111111")
    drawing = ImageDraw.Draw(sheet)
    for scene_index, scene in enumerate(SCENES):
        for row_index, frame in enumerate((0, FRAME_COUNT-1)):
            member = f"val_primesense/{scene}/rgb/{frame:06d}.png"
            raw = (output/member).read_bytes()
            if sha(raw) != indexed[member]["sha256"]:
                raise ValueError("Example source differs from sealed corpus")
            filename = f"scene-{scene}-frame-{frame:06d}.png"
            atomic_write(destination/filename, raw)
            with Image.open(io.BytesIO(raw)) as original:
                thumbnail = original.convert("RGB")
                thumbnail.thumbnail((320, 240), Image.Resampling.LANCZOS)
                sheet.paste(thumbnail, (scene_index*320, row_index*265))
            drawing.text((scene_index*320+4, row_index*265+243), f"HB scene {scene} frame {frame}", fill="white")
            examples.append(dict(filename=filename, member=member, bytes=len(raw), sha256=sha(raw),
                                 role="annotated_onboarding_example" if frame == 0 else "scored_evaluation_example"))
    rendered = io.BytesIO()
    sheet.save(rendered, "JPEG", quality=90, optimize=False)
    atomic_write(destination/"contact-sheet.jpg", rendered.getvalue())
    annotations = []
    for name, expected in (("inference.json", lock["inferenceSHA256"]), ("evaluator.json", lock["evaluatorSHA256"])):
        raw = (output/name).read_bytes()
        if sha(raw) != expected:
            raise ValueError("Annotation pack source differs from sealed manifest")
        compressed = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0, compresslevel=9) as stream:
            stream.write(raw)
        filename = name+".gz"
        atomic_write(destination/filename, compressed.getvalue())
        annotations.append(dict(filename=filename, bytes=len(compressed.getvalue()), sha256=sha(compressed.getvalue()),
                                expandedBytes=len(raw), expandedSHA256=expected))
    index = dict(schemaVersion=1, datasetRevision=REVISION, license=LICENSE, authors=AUTHOR,
                 source=DATASET_PAGE, images=examples, annotationPacks=annotations,
                 contactSheet=dict(filename="contact-sheet.jpg", sha256=sha(rendered.getvalue()),
                                   transformation="RGB start/end examples resized to 320x240 thumbnails, labeled, JPEG quality 90; not used for metrics"),
                 disclosure="Eight actual RGB example images and complete compressed inference/evaluator annotation manifests are in Git. Full RGB/depth/mask/model media remain in the pinned acquisition corpus and separate full-media package.")
    atomic_write(destination/"index.json", canonical(index))
    README = f"""# Licensed real examples and complete annotations

These are real **HB / HomebrewedDB** RGB frames, not synthetic renders.
License: CC0 1.0 Universal. Credit: {AUTHOR}.
Pinned source: https://huggingface.co/datasets/bop-benchmark/hb/tree/{REVISION}
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
717 MB compressed media archive was prepared locally; publishing it requires a
separate verified GitHub Release upload. No private/user media is included.
"""
    atomic_write(destination/"README.md", README.encode())
    print(json.dumps(dict(status="complete", destination=str(destination), actualRealExampleImages=len(examples),
                          completeEvaluationAnnotations=3600, bytes=sum(path.stat().st_size for path in destination.iterdir() if path.is_file()))), flush=True)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("acquire", "verify", "package", "examples"))
    parser.add_argument("--output", type=Path, default=ROOT/".cache/tracking-benchmark/real-v1")
    parser.add_argument("--recipe-output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--archive", type=Path, default=ROOT/"artifacts/tracking-benchmark/releases/visualizeit-hb-tracking-v1.tar.gz")
    parser.add_argument("--examples-output", type=Path, default=ROOT/"datasets/tracking-v1/real-examples")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("Use 1..8 network workers")
    if args.action == "acquire":
        acquire(args.output, args.recipe_output, workers=args.workers)
    elif args.action == "package":
        package(args.output, args.archive)
    elif args.action == "examples":
        export_examples(args.output, args.examples_output)
    else:
        verify(args.output)


if __name__ == "__main__":
    main()
