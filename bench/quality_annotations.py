"""Source-only annotation editor and explicit reviewed-label import."""
import argparse
import json
import math
from pathlib import Path
from types import MappingProxyType

from .quality_assets import ROOT
from .storage import write_artifact


# Frozen independently of incoming editor JSON. Keep in sync with the literal
# selection in quality_mask_audit.py; the tests compare both and the three
# checked-in canonical pending files.
FROZEN_ANNOTATION_SELECTIONS = MappingProxyType({
    "keyboard": (
        (1149, "uniform"), (1150, "hard_candidate"), (1154, "hard_candidate"), (1157, "uniform"),
        (1165, "uniform"), (1168, "hard_candidate"), (1174, "uniform"), (1182, "uniform"),
        (1187, "hard_candidate"), (1190, "uniform"), (1198, "uniform"), (1207, "uniform"),
        (1208, "hard_candidate"), (1215, "uniform"), (1223, "uniform"), (1231, "uniform"),
        (1233, "hard_candidate"), (1240, "uniform"), (1248, "uniform"), (1256, "uniform"),
        (1257, "hard_candidate"), (1264, "uniform"), (1273, "uniform"), (1281, "uniform"),
        (1286, "hard_candidate"), (1289, "uniform"), (1297, "uniform"), (1306, "uniform"),
        (1314, "uniform"), (1315, "hard_candidate"), (1322, "uniform"), (1330, "uniform"),
        (1339, "uniform"), (1347, "uniform"), (1348, "hard_candidate"), (1355, "uniform"),
        (1363, "uniform"), (1372, "uniform"), (1380, "uniform"), (1388, "uniform"),
    ),
    "mug": (
        (827, "uniform"), (828, "hard_candidate"), (832, "hard_candidate"), (835, "uniform"),
        (843, "uniform"), (846, "hard_candidate"), (852, "uniform"), (860, "uniform"),
        (865, "hard_candidate"), (868, "uniform"), (876, "uniform"), (885, "uniform"),
        (886, "hard_candidate"), (893, "uniform"), (901, "uniform"), (909, "uniform"),
        (911, "hard_candidate"), (918, "uniform"), (926, "uniform"), (934, "uniform"),
        (935, "hard_candidate"), (942, "uniform"), (951, "uniform"), (959, "uniform"),
        (964, "hard_candidate"), (967, "uniform"), (975, "uniform"), (984, "uniform"),
        (992, "uniform"), (993, "hard_candidate"), (1000, "uniform"), (1008, "uniform"),
        (1017, "uniform"), (1025, "uniform"), (1026, "hard_candidate"), (1033, "uniform"),
        (1041, "uniform"), (1050, "uniform"), (1058, "uniform"), (1066, "uniform"),
    ),
    "ranch": (
        (10, "uniform"), (11, "hard_candidate"), (15, "hard_candidate"), (18, "uniform"),
        (26, "uniform"), (29, "hard_candidate"), (35, "uniform"), (43, "uniform"),
        (48, "hard_candidate"), (51, "uniform"), (59, "uniform"), (68, "uniform"),
        (69, "hard_candidate"), (76, "uniform"), (84, "uniform"), (92, "uniform"),
        (94, "hard_candidate"), (101, "uniform"), (109, "uniform"), (117, "uniform"),
        (118, "hard_candidate"), (125, "uniform"), (134, "uniform"), (142, "uniform"),
        (147, "hard_candidate"), (150, "uniform"), (158, "uniform"), (167, "uniform"),
        (175, "uniform"), (176, "hard_candidate"), (183, "uniform"), (191, "uniform"),
        (200, "uniform"), (208, "uniform"), (209, "hard_candidate"), (216, "uniform"),
        (224, "uniform"), (233, "uniform"), (241, "uniform"), (249, "uniform"),
    ),
})
FROZEN_RESOLUTION = (1024, 1280)
FROZEN_SCORED_FRAME_IDS = MappingProxyType({
    "keyboard": tuple(range(1149, 1389)),
    "mug": tuple(range(827, 1067)),
    "ranch": tuple(range(10, 250)),
})
FROZEN_SOURCE_SHA256 = MappingProxyType({
    "keyboard": "dd638f8f8de4413bfbbd92e8e21e0451c13efd7fb29b1f5d5e89f12040546a02",
    "mug": "ad9e6c4d70adfa62889b08a9f31e0d98a781619f86c5efdce48caef482be3844",
    "ranch": "6dcd8c1dda7381d69db178e2b619447506cfcce6cfe4afce8666a43c6363283e",
})
PENDING_SELECTION_STATUS = "hard candidates need image-only difficulty review"
REVIEWED_SELECTION_STATUS = "reviewed_image_only"
_HASH_CHARS = frozenset("0123456789abcdef")
_GEOMETRY_EPSILON = 1e-9


def _point(value, width, height, label):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must be an [x, y] point")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in value):
        raise ValueError(f"{label} coordinates must be finite numbers")
    x, y = value
    if not 0 <= x < width or not 0 <= y < height:
        raise ValueError(f"{label} must lie inside the original image")
    return float(x), float(y)


def _orientation(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, point):
    return (min(a[0], b[0]) - _GEOMETRY_EPSILON <= point[0] <= max(a[0], b[0]) + _GEOMETRY_EPSILON
            and min(a[1], b[1]) - _GEOMETRY_EPSILON <= point[1] <= max(a[1], b[1]) + _GEOMETRY_EPSILON)


def _segments_intersect(a, b, c, d):
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if ((o1 > _GEOMETRY_EPSILON and o2 < -_GEOMETRY_EPSILON)
            or (o1 < -_GEOMETRY_EPSILON and o2 > _GEOMETRY_EPSILON)):
        if ((o3 > _GEOMETRY_EPSILON and o4 < -_GEOMETRY_EPSILON)
                or (o3 < -_GEOMETRY_EPSILON and o4 > _GEOMETRY_EPSILON)):
            return True
    return ((abs(o1) <= _GEOMETRY_EPSILON and _on_segment(a, b, c))
            or (abs(o2) <= _GEOMETRY_EPSILON and _on_segment(a, b, d))
            or (abs(o3) <= _GEOMETRY_EPSILON and _on_segment(c, d, a))
            or (abs(o4) <= _GEOMETRY_EPSILON and _on_segment(c, d, b)))


def _validate_polygon(paths, width, height, label):
    if not isinstance(paths, list):
        raise ValueError(f"{label} must be a list of closed polygons")
    for polygon_index, polygon in enumerate(paths):
        polygon_label = f"{label} polygon {polygon_index}"
        if isinstance(polygon, dict):
            if "hole" in polygon and not isinstance(polygon["hole"], bool):
                raise ValueError(f"{polygon_label} hole flag must be boolean")
            points = polygon.get("points")
        else:
            points = polygon
        if not isinstance(points, list):
            raise ValueError(f"{polygon_label} points must be a list")
        if len(points) > 2048:
            raise ValueError(f"{polygon_label} has too many vertices")
        coords = [_point(point, width, height, polygon_label) for point in points]
        if len(coords) > 3 and coords[0] == coords[-1]:
            coords.pop()
        if len(set(coords)) < 3:
            raise ValueError(f"{polygon_label} must have at least three distinct points")
        area_twice = sum(
            coords[i][0] * coords[(i + 1) % len(coords)][1]
            - coords[(i + 1) % len(coords)][0] * coords[i][1]
            for i in range(len(coords))
        )
        if abs(area_twice) <= _GEOMETRY_EPSILON:
            raise ValueError(f"{polygon_label} is degenerate")
        edge_count = len(coords)
        for first in range(edge_count):
            a, b = coords[first], coords[(first + 1) % edge_count]
            for second in range(first + 1, edge_count):
                if second == first or second == first + 1 or (first == 0 and second == edge_count - 1):
                    continue
                c, d = coords[second], coords[(second + 1) % edge_count]
                if _segments_intersect(a, b, c, d):
                    raise ValueError(f"{polygon_label} self-intersects")


def _validate_landmarks(landmarks, width, height, label):
    if not isinstance(landmarks, list):
        raise ValueError(f"{label} landmarks must be a list")
    seen_ids = set()
    for index, landmark in enumerate(landmarks):
        prefix = f"{label} landmark {index}"
        if not isinstance(landmark, dict):
            raise ValueError(f"{prefix} must be an object")
        landmark_id = landmark.get("landmark_id")
        if not isinstance(landmark_id, str) or not landmark_id.strip() or landmark_id in seen_ids:
            raise ValueError(f"{prefix} needs a unique stable ID")
        seen_ids.add(landmark_id)
        if not isinstance(landmark.get("visible", True), bool):
            raise ValueError(f"{prefix} visibility must be boolean")
        if not isinstance(landmark.get("reviewed"), bool):
            raise ValueError(f"{prefix} correspondence review status must be boolean")
        xyz = landmark.get("object_point_m")
        if not isinstance(xyz, list) or len(xyz) != 3:
            raise ValueError(f"{prefix} needs a reviewed 3D model point")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in xyz):
            raise ValueError(f"{prefix} model coordinates must be finite numbers")
        _point(landmark.get("pixel"), width, height, f"{prefix} image point")


def _frozen_rows(alias):
    if alias not in FROZEN_ANNOTATION_SELECTIONS:
        raise ValueError("Unknown object")
    return FROZEN_ANNOTATION_SELECTIONS[alias]


def _validate_annotation_document_v1(document, expected_object=None):
    """Validate editor/import/evaluator labels against the immutable RGB selection."""
    if not isinstance(document, dict):
        raise ValueError("Annotation document must be an object")
    alias = document.get("object")
    selection = _frozen_rows(alias)
    if expected_object is not None and alias != expected_object:
        raise ValueError("Annotation object does not match the selected object")
    version = document.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("Unsupported annotation schema version")
    if document.get("resolution") != list(FROZEN_RESOLUTION):
        raise ValueError("Annotation resolution must match the frozen native RGB resolution")
    source_sha256 = document.get("source_sha256")
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(char not in _HASH_CHARS for char in source_sha256)
            or source_sha256 != FROZEN_SOURCE_SHA256[alias]):
        raise ValueError("Annotation source hash does not match the frozen original RGB video")
    selection_status = document.get("selection_status")
    if selection_status not in (PENDING_SELECTION_STATUS, REVIEWED_SELECTION_STATUS):
        raise ValueError("Invalid image-only hard-selection review status")
    operator = document.get("operator")
    if operator is not None and (not isinstance(operator, str) or not operator.strip()):
        raise ValueError("Reviewer name must be a non-empty string when supplied")
    frames = document.get("frames")
    if not isinstance(frames, list) or len(frames) != len(selection):
        raise ValueError("Exactly forty canonical annotation frames are required")
    if selection_status == REVIEWED_SELECTION_STATUS and not operator:
        raise ValueError("Image-only selection review requires a reviewer name")

    expected_ids = [frame_id for frame_id, _ in selection]
    actual_ids = []
    for row in frames:
        if not isinstance(row, dict):
            raise ValueError("Every annotation frame must be an object")
        frame_id = row.get("frame_id")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int):
            raise ValueError("Annotation frame IDs must be integers")
        actual_ids.append(frame_id)
    if actual_ids != expected_ids:
        raise ValueError("Annotation frame IDs must match the immutable original candidate selection")

    width, height = FROZEN_RESOLUTION
    for row, (frame_id, group) in zip(frames, selection):
        if row.get("group") != group or row.get("image") != f"{frame_id}.jpg":
            raise ValueError(f"Annotation group or original RGB image changed at frame {frame_id}")
        status = row.get("status")
        if status not in ("pending", "reviewed"):
            raise ValueError(f"Frame {frame_id} status must be explicitly pending or reviewed")
        visibility = row.get("visibility")
        if visibility not in (None, "visible", "fully_hidden"):
            raise ValueError(f"Frame {frame_id} visibility must be visible or fully_hidden")
        _validate_polygon(row.get("visible_object"), width, height, f"Frame {frame_id} visible object")
        _validate_polygon(row.get("overlapping_hands"), width, height, f"Frame {frame_id} overlapping hands")
        _validate_landmarks(row.get("landmarks"), width, height, f"Frame {frame_id}")
        if visibility == "fully_hidden" and (row["visible_object"] or row["landmarks"]):
            raise ValueError(f"Fully hidden frame {frame_id} cannot contain target polygons or landmarks")
        if status == "reviewed":
            if not operator:
                raise ValueError(f"Reviewed frame {frame_id} requires a reviewer name")
            if visibility == "fully_hidden":
                pass
            elif visibility == "visible":
                if not row["visible_object"]:
                    raise ValueError(f"Visible reviewed frame {frame_id} needs a target polygon")
                if not row["landmarks"] or not all(
                    landmark.get("visible", True) is True and landmark.get("reviewed") is True
                    for landmark in row["landmarks"]
                ):
                    raise ValueError(f"Visible reviewed frame {frame_id} needs reviewed visible landmarks")
            else:
                raise ValueError(f"Reviewed frame {frame_id} needs an explicit visibility choice")
        if row.get("review_notes") is not None and not isinstance(row.get("review_notes"), str):
            raise ValueError(f"Frame {frame_id} review notes must be text")

    # Reject visible-object / hand contradictions before labels can be imported,
    # scored or exported as diagnostic oracle masks. Empty pending rows require
    # no image-sized buffers.
    from .quality_evaluate import polygon_mask
    for row in frames:
        target = polygon_mask(row["visible_object"], (height, width)) if row["visible_object"] else None
        hands = polygon_mask(row["overlapping_hands"], (height, width)) if row["overlapping_hands"] else None
        if target is not None and hands is not None and (target & hands).any():
            raise ValueError(f"Visible object annotations overlap a hand at frame {row['frame_id']}")
        if (row["status"] == "reviewed" and row["visibility"] == "visible"
                and (target is None or not target.any())):
            raise ValueError(f"Visible reviewed frame {row['frame_id']} has an empty target mask")
    return document


def normalize_annotation_document(document, expected_object=None):
    """Validate legacy/v2 labels and return a v2 in-memory scoring/editor view."""
    if not isinstance(document, dict):
        raise ValueError("Annotation document must be an object")
    version = document.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("Unsupported annotation schema version")
    if version == 1:
        _validate_annotation_document_v1(document, expected_object)
        normalized = json.loads(json.dumps(document, allow_nan=False))
        normalized["schema_version"] = 2
        for row in normalized["frames"]:
            if row["status"] == "pending":
                row["landmark_status"] = "pending"
            elif row["visibility"] == "fully_hidden":
                row["landmark_status"] = "not_applicable_fully_hidden"
            else:
                row["landmark_status"] = "reviewed"
        return normalized
    if version != 2:
        raise ValueError("Unsupported annotation schema version")
    return validate_annotation_document(document, expected_object)


def validate_annotation_document(document, expected_object=None):
    """Validate schema v1/v2 while keeping v1 acceptance rules unchanged."""
    if not isinstance(document, dict):
        raise ValueError("Annotation document must be an object")
    version = document.get("schema_version")
    if version == 1 and not isinstance(version, bool):
        return _validate_annotation_document_v1(document, expected_object)
    if isinstance(version, bool) or not isinstance(version, int) or version != 2:
        raise ValueError("Unsupported annotation schema version")
    alias = document.get("object")
    selection = _frozen_rows(alias)
    if expected_object is not None and alias != expected_object:
        raise ValueError("Annotation object does not match the selected object")
    if document.get("resolution") != list(FROZEN_RESOLUTION):
        raise ValueError("Annotation resolution must match the frozen native RGB resolution")
    source_sha256 = document.get("source_sha256")
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(char not in _HASH_CHARS for char in source_sha256)
            or source_sha256 != FROZEN_SOURCE_SHA256[alias]):
        raise ValueError("Annotation source hash does not match the frozen original RGB video")
    selection_status = document.get("selection_status")
    if selection_status not in (PENDING_SELECTION_STATUS, REVIEWED_SELECTION_STATUS):
        raise ValueError("Invalid image-only hard-selection review status")
    operator = document.get("operator")
    if operator is not None and (not isinstance(operator, str) or not operator.strip()):
        raise ValueError("Reviewer name must be a non-empty string when supplied")
    frames = document.get("frames")
    if not isinstance(frames, list) or len(frames) != len(selection):
        raise ValueError("Exactly forty canonical annotation frames are required")
    if selection_status == REVIEWED_SELECTION_STATUS and not operator:
        raise ValueError("Image-only selection review requires a reviewer name")
    expected_ids = [frame_id for frame_id, _ in selection]
    actual_ids = []
    for row in frames:
        if not isinstance(row, dict):
            raise ValueError("Every annotation frame must be an object")
        frame_id = row.get("frame_id")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int):
            raise ValueError("Annotation frame IDs must be integers")
        actual_ids.append(frame_id)
    if actual_ids != expected_ids:
        raise ValueError("Annotation frame IDs must match the immutable original candidate selection")

    width, height = FROZEN_RESOLUTION
    allowed_landmark_statuses = {
        "pending", "reviewed", "unobservable", "not_applicable_fully_hidden",
    }
    for row, (frame_id, group) in zip(frames, selection):
        if row.get("group") != group or row.get("image") != f"{frame_id}.jpg":
            raise ValueError(f"Annotation group or original RGB image changed at frame {frame_id}")
        status = row.get("status")
        if status not in ("pending", "reviewed"):
            raise ValueError(f"Frame {frame_id} status must be explicitly pending or reviewed")
        visibility = row.get("visibility")
        if visibility not in (None, "visible", "fully_hidden"):
            raise ValueError(f"Frame {frame_id} visibility must be visible or fully_hidden")
        landmark_status = row.get("landmark_status")
        if landmark_status not in allowed_landmark_statuses:
            raise ValueError(f"Frame {frame_id} needs an explicit landmark review status")
        if landmark_status in ("reviewed", "unobservable") and not operator:
            raise ValueError(f"Frame {frame_id} landmark review requires a reviewer name")
        _validate_polygon(row.get("visible_object"), width, height, f"Frame {frame_id} visible object")
        _validate_polygon(row.get("overlapping_hands"), width, height, f"Frame {frame_id} overlapping hands")
        _validate_landmarks(row.get("landmarks"), width, height, f"Frame {frame_id}")
        if visibility == "fully_hidden" and (row["visible_object"] or row["landmarks"]):
            raise ValueError(f"Fully hidden frame {frame_id} cannot contain target polygons or landmarks")
        if status == "reviewed":
            if not operator:
                raise ValueError(f"Reviewed frame {frame_id} requires a reviewer name")
            if visibility is None:
                raise ValueError(f"Reviewed frame {frame_id} needs an explicit visibility choice")
            if visibility == "visible" and not row["visible_object"]:
                raise ValueError(f"Visible reviewed frame {frame_id} needs a target polygon")
            if visibility == "fully_hidden" and landmark_status != "not_applicable_fully_hidden":
                raise ValueError(f"Fully hidden reviewed frame {frame_id} has an invalid landmark status")
        if visibility == "fully_hidden" and landmark_status in ("reviewed", "unobservable"):
            raise ValueError(f"Fully hidden frame {frame_id} cannot have reviewed or unobservable landmarks")
        reason = row.get("landmark_review_reason")
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError(f"Frame {frame_id} landmark review reason must be non-empty text")
        visible_landmarks = [landmark for landmark in row["landmarks"]
                             if landmark.get("visible", True) is True]
        if landmark_status == "reviewed":
            if visibility != "visible" or not visible_landmarks or not all(
                landmark.get("reviewed") is True for landmark in visible_landmarks
            ):
                raise ValueError(f"Frame {frame_id} reviewed landmark status needs independently reviewed visible points")
            if reason is not None:
                raise ValueError(f"Frame {frame_id} reviewed landmarks cannot include an unobservable reason")
        elif landmark_status == "unobservable":
            if visibility != "visible" or row["landmarks"] or not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"Frame {frame_id} unobservable landmarks need visible status, an explicit reason, and no landmarks")
        elif landmark_status == "not_applicable_fully_hidden":
            if status != "reviewed" or visibility != "fully_hidden" or row["visible_object"] or row["landmarks"]:
                raise ValueError(f"Frame {frame_id} not-applicable landmarks require a reviewed fully hidden frame with empty target and landmarks")
            if reason is not None:
                raise ValueError(f"Frame {frame_id} fully hidden landmarks cannot have an unobservable reason")
        elif reason is not None:
            raise ValueError(f"Frame {frame_id} pending landmarks cannot include a review reason")
        if row.get("review_notes") is not None and not isinstance(row.get("review_notes"), str):
            raise ValueError(f"Frame {frame_id} review notes must be text")

    from .quality_evaluate import polygon_mask
    for row in frames:
        target = polygon_mask(row["visible_object"], (height, width)) if row["visible_object"] else None
        hands = polygon_mask(row["overlapping_hands"], (height, width)) if row["overlapping_hands"] else None
        if target is not None and hands is not None and (target & hands).any():
            raise ValueError(f"Visible object annotations overlap a hand at frame {row['frame_id']}")
        if (row["status"] == "reviewed" and row["visibility"] == "visible"
                and (target is None or not target.any())):
            raise ValueError(f"Visible reviewed frame {row['frame_id']} has an empty target mask")
    return document


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VisualizeIt · independent annotations</title>
<style>body{background:#101b21;color:#e6eff4;font:16px system-ui;margin:24px}main{max-width:1280px;margin:auto}p{color:#bdd0da;line-height:1.5}button,select,input{background:#203c47;color:white;padding:9px;border:1px solid #52818d;border-radius:6px}button{cursor:pointer}.controls{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}canvas{height:65vh;max-width:100%;object-fit:contain;background:#080f14;cursor:crosshair}a{color:#70ebd2}#status{color:#70ebd2}label{display:inline-flex;align-items:center;gap:6px}</style>
<main><h1>Independent image annotations</h1><p>Review original pixels only. Mask/visibility review and landmark review are independent. Visible targets need a target polygon for mask review; landmarks can be reviewed separately or marked unobservable with a reason. Fully hidden targets have no target polygon or landmarks. No predicted masks or dataset poses appear here.</p>
<div class="controls"><select id="object"><option>keyboard</option><option>mug</option><option>ranch</option></select><button id="previous">Previous frame</button><select id="frame"></select><button id="next">Next frame</button><span id="status"></span></div>
<div class="controls"><label>Visibility <select id="visibility"><option value="">Choose visibility</option><option value="visible">Visible</option><option value="fully_hidden">Fully hidden</option></select></label><select id="tool"><option value="visible_object">Visible object polygon · green</option><option value="overlapping_hands">Overlapping hand polygon · orange</option><option value="landmark">Surface landmark · yellow</option></select><label><input id="hole" type="checkbox">Polygon is a hole</label><button id="close">Close polygon</button><button id="undo">Undo last point / region</button><button id="clear">Clear this frame</button></div>
<canvas id="canvas" width="1024" height="1280"></canvas>
<div class="controls"><input id="landmarkId" placeholder="Stable landmark ID"><input id="xyz" placeholder="Model point x,y,z in metres"><label><input id="correspondence" type="checkbox">3D correspondence reviewed</label></div>
<p>Click to add polygon points, then close it. Regions may have separate islands and holes. In landmark mode, enter the independently reviewed model coordinate and click its image location. Object polygons must exclude hands. Polygon edits invalidate mask review; landmark edits invalidate landmark review only; visibility edits invalidate both reviews.</p>
<div class="controls"><input id="landmarkReason" placeholder="Reason landmarks are unobservable"><input id="operator" placeholder="Reviewer name"><label><input id="hardReview" type="checkbox">Hard-frame selection reviewed using images only</label><button id="reviewMask">Review mask and visibility</button><button id="reviewLandmarks">Review landmarks</button><button id="markUnobservable">Mark landmarks unobservable</button><button id="download">Download annotations JSON</button><button id="reload">Load edited JSON</button><input id="file" type="file" accept="application/json" hidden><a href="index.html">Comparison viewer</a></div>
<p id="notes">All 120 labels start pending. A returned mask or a visible overlay is not successful attachment. An export with pending frames cannot pass the independent gate.</p>
<script>
const FROZEN_CASES=__FROZEN_CASES__;
const $=id=>document.getElementById(id),canvas=$('canvas'),ctx=canvas.getContext('2d');
let data,image=new Image(),draft=[],savedTool=null,ticket=0;
const PENDING_SELECTION='hard candidates need image-only difficulty review';
const REVIEWED_SELECTION='reviewed_image_only';
function current(){return data.frames[Number($('frame').value)];}
function message(text){$('status').textContent=text;}
function setMaskPending(){current().status='pending';}
function setLandmarkPending(){current().landmark_status='pending';delete current().landmark_review_reason;}
function setBothPending(){setMaskPending();setLandmarkPending();}
function finitePoint(p,w,h){return Array.isArray(p)&&p.length===2&&p.every(Number.isFinite)&&p[0]>=0&&p[0]<w&&p[1]>=0&&p[1]<h;}
function cross(u,v,z){return(v[0]-u[0])*(z[1]-u[1])-(v[1]-u[1])*(z[0]-u[0]);}
function onSegment(a,b,p){return p[0]>=Math.min(a[0],b[0])-1e-9&&p[0]<=Math.max(a[0],b[0])+1e-9&&p[1]>=Math.min(a[1],b[1])-1e-9&&p[1]<=Math.max(a[1],b[1])+1e-9;}
function intersects(a,b,c,d){const o1=cross(a,b,c),o2=cross(a,b,d),o3=cross(c,d,a),o4=cross(c,d,b),proper=((o1>1e-9&&o2< -1e-9)||(o1< -1e-9&&o2>1e-9))&&((o3>1e-9&&o4< -1e-9)||(o3< -1e-9&&o4>1e-9));return proper||(Math.abs(o1)<=1e-9&&onSegment(a,b,c))||(Math.abs(o2)<=1e-9&&onSegment(a,b,d))||(Math.abs(o3)<=1e-9&&onSegment(c,d,a))||(Math.abs(o4)<=1e-9&&onSegment(c,d,b));}
function validatePath(path,w,h,label){const points=Array.isArray(path)?path:path&&path.points;if(!Array.isArray(points))throw Error(label+' points must be a list');if(path&&!Array.isArray(path)&&path.hole!==undefined&&typeof path.hole!=='boolean')throw Error(label+' hole flag must be boolean');if(points.length>2048)throw Error(label+' has too many vertices');let p=points.map((q)=>{if(!finitePoint(q,w,h))throw Error(label+' has invalid coordinates');return [Number(q[0]),Number(q[1])];});if(p.length>3&&p[0][0]===p[p.length-1][0]&&p[0][1]===p[p.length-1][1])p.pop();if(new Set(p.map(q=>q.join(','))).size<3)throw Error(label+' is degenerate');let area=0;for(let i=0;i<p.length;i++)area+=p[i][0]*p[(i+1)%p.length][1]-p[(i+1)%p.length][0]*p[i][1];if(Math.abs(area)<=1e-9)throw Error(label+' is degenerate');for(let i=0;i<p.length;i++){for(let j=i+1;j<p.length;j++){if(j===i+1||(i===0&&j===p.length-1))continue;if(intersects(p[i],p[(i+1)%p.length],p[j],p[(j+1)%p.length]))throw Error(label+' self-intersects');}}}
function validateLandmarks(f,w,h){if(!Array.isArray(f.landmarks))throw Error('Frame landmarks must be a list');const ids=new Set();for(const [i,l] of f.landmarks.entries()){if(!l||typeof l!=='object'||typeof l.landmark_id!=='string'||!l.landmark_id.trim()||ids.has(l.landmark_id)||typeof l.reviewed!=='boolean'||typeof (l.visible===undefined?true:l.visible)!=='boolean'||!Array.isArray(l.object_point_m)||l.object_point_m.length!==3||!l.object_point_m.every(Number.isFinite)||!finitePoint(l.pixel,w,h))throw Error('Landmark '+i+' is malformed');ids.add(l.landmark_id);}}
function validateGeometry(f,w,h){if(!Array.isArray(f.visible_object)||!Array.isArray(f.overlapping_hands))throw Error('Frame geometry must use polygon lists');f.visible_object.forEach((p,i)=>validatePath(p,w,h,'Visible polygon '+i));f.overlapping_hands.forEach((p,i)=>validatePath(p,w,h,'Hand polygon '+i));validateLandmarks(f,w,h);}
function validateLegacyFrame(f,w,h){validateGeometry(f,w,h);if(f.visibility!==undefined&&f.visibility!==null&&!['visible','fully_hidden'].includes(f.visibility))throw Error('Choose visible or fully hidden');if(f.visibility==='fully_hidden'&&(f.visible_object.length||f.landmarks.length))throw Error('Clear target polygons and landmarks on a fully hidden frame');if(!['pending','reviewed'].includes(f.status))throw Error('Frame status must be pending or reviewed');if(f.status==='reviewed'&&(f.visibility===undefined||f.visibility===null))throw Error('Reviewed frame needs an explicit visibility choice');if(f.status==='reviewed'&&f.visibility==='visible'&&(!f.visible_object.some(p=>Array.isArray(p)||p.hole!==true)||!f.landmarks.length||!f.landmarks.every(l=>l.visible!==false&&l.reviewed===true)))throw Error('Visible legacy review needs target polygon and reviewed landmarks');}
function validateFrameV2(f,w,h){validateGeometry(f,w,h);if(!['pending','reviewed'].includes(f.status))throw Error('Mask review status must be pending or reviewed');if(f.visibility!==undefined&&f.visibility!==null&&!['visible','fully_hidden'].includes(f.visibility))throw Error('Choose visible or fully hidden');if(!['pending','reviewed','unobservable','not_applicable_fully_hidden'].includes(f.landmark_status))throw Error('Choose an explicit landmark review status');if(f.visibility==='fully_hidden'&&(f.visible_object.length||f.landmarks.length))throw Error('Fully hidden targets cannot contain target polygons or landmarks');if(f.status==='reviewed'&&(f.visibility===undefined||f.visibility===null))throw Error('Reviewed mask frame needs an explicit visibility choice');if(f.status==='reviewed'&&f.visibility==='visible'&&!f.visible_object.some(p=>Array.isArray(p)||p.hole!==true))throw Error('Visible mask review needs a target polygon');if(f.visibility==='fully_hidden'&&['reviewed','unobservable'].includes(f.landmark_status))throw Error('Fully hidden frame cannot have visible landmark review');if(f.landmark_status==='reviewed'&&(f.visibility!=='visible'||!f.landmarks.some(l=>l.visible!==false)||!f.landmarks.filter(l=>l.visible!==false).every(l=>l.reviewed===true)))throw Error('Reviewed landmarks need visible reviewed points');if(f.landmark_status==='unobservable'&&(f.visibility!=='visible'||f.landmarks.length||typeof f.landmark_review_reason!=='string'||!f.landmark_review_reason.trim()))throw Error('Unobservable landmarks need a visible frame, reason, and no landmarks');if(f.landmark_status==='not_applicable_fully_hidden'&&(f.status!=='reviewed'||f.visibility!=='fully_hidden'||f.visible_object.length||f.landmarks.length))throw Error('Not-applicable landmarks require a reviewed fully hidden frame');if(f.landmark_status==='pending'&&f.landmark_review_reason!==undefined)throw Error('Pending landmarks cannot include a review reason');if(f.landmark_status==='reviewed'&&f.landmark_review_reason!==undefined)throw Error('Reviewed landmarks cannot include an unobservable reason');}
function validateDocument(d,alias){
 const frozen=FROZEN_CASES[alias];
 if(!frozen||!d||typeof d!=='object')throw Error('Unknown annotation case');
 if(d.schema_version!==1&&d.schema_version!==2)throw Error('Unsupported annotation schema');
 if(d.object!==alias||JSON.stringify(d.resolution)!==JSON.stringify(frozen.resolution)||d.source_sha256!==frozen.source_sha256)throw Error('Annotation provenance changed');
 if(![frozen.pending_selection_status,frozen.reviewed_selection_status].includes(d.selection_status))throw Error('Invalid image-only selection status');
 if(d.operator!==null&&d.operator!==undefined&&(typeof d.operator!=='string'||!d.operator.trim()))throw Error('Reviewer name must be text');
 if(!Array.isArray(d.frames)||d.frames.length!==frozen.frames.length)throw Error('Expected 40 annotation frames');
 if(d.selection_status===REVIEWED_SELECTION&&!d.operator)throw Error('Image-only selection review needs a reviewer name');
 for(let i=0;i<d.frames.length;i++){
  const f=d.frames[i],expected=frozen.frames[i];
  if(!f||f.frame_id!==expected[0]||f.group!==expected[1]||f.image!==expected[2])throw Error('Frozen frame ID, group, or original RGB image changed at row '+i);
  if(d.schema_version===1)validateLegacyFrame(f,frozen.resolution[0],frozen.resolution[1]);
  else {
   validateFrameV2(f,frozen.resolution[0],frozen.resolution[1]);
   if(['reviewed','unobservable'].includes(f.landmark_status)&&!d.operator)throw Error('Landmark review needs a reviewer name');
  }
  if(f.status==='reviewed'&&!d.operator)throw Error('Reviewed frame needs a reviewer name');
 }
 return d;
}
function normalizeDocument(d,alias){validateDocument(d,alias);const normalized=JSON.parse(JSON.stringify(d));if(normalized.schema_version===1){normalized.schema_version=2;for(const f of normalized.frames){if(f.status==='pending')f.landmark_status='pending';else if(f.visibility==='fully_hidden')f.landmark_status='not_applicable_fully_hidden';else f.landmark_status='reviewed';}}return normalized;}
function paint(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0);let f=current();$('visibility').value=f.visibility||'';for(const [key,color] of [['visible_object','#28e7ad'],['overlapping_hands','#ffa340']]){for(const p of f[key]){let pts=p.points||p;if(!pts.length)continue;ctx.beginPath();ctx.moveTo(...pts[0]);for(const q of pts.slice(1))ctx.lineTo(...q);ctx.closePath();ctx.fillStyle=color+'55';if(!p.hole)ctx.fill();ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();}}for(const p of f.landmarks){ctx.fillStyle='#ffdc42';ctx.beginPath();ctx.arc(...p.pixel,5,0,Math.PI*2);ctx.fill();ctx.font='20px system-ui';ctx.fillText(p.landmark_id,p.pixel[0]+8,p.pixel[1]);}if(draft.length){ctx.beginPath();ctx.moveTo(...draft[0]);for(const p of draft.slice(1))ctx.lineTo(...p);ctx.strokeStyle='#fff';ctx.lineWidth=3;ctx.stroke();}message('Frame '+f.frame_id+' · '+f.group+' · mask '+f.status+' · landmarks '+f.landmark_status+' · '+data.frames.filter(x=>x.status==='reviewed').length+'/40 mask reviews');}
async function loadFrame(){draft=[];savedTool=null;const mine=++ticket;image=new Image();image.src='annotations/'+data.object+'/'+current().image;await image.decode();if(mine!==ticket)return;canvas.width=data.resolution[0];canvas.height=data.resolution[1];paint();}
function install(d,alias){data=normalizeDocument(d,alias);$('frame').replaceChildren(...data.frames.map((f,i)=>Object.assign(document.createElement('option'),{value:i,textContent:f.frame_id+' · '+f.group})));$('operator').value=data.operator||'';$('hardReview').checked=data.selection_status===REVIEWED_SELECTION;loadFrame();}
async function loadCase(){const alias=$('object').value,r=await fetch('annotations/'+alias+'/annotations.json');if(!r.ok)throw Error('Missing annotations');install(await r.json(),alias);}
function updateReviewerAndSelection(){data.operator=$('operator').value.trim()||null;data.selection_status=$('hardReview').checked?REVIEWED_SELECTION:PENDING_SELECTION;}
function validateCurrent(){updateReviewerAndSelection();validateDocument(data,data.object);}
function pointFromEvent(e){const rect=canvas.getBoundingClientRect(),scale=Math.min(rect.width/canvas.width,rect.height/canvas.height),left=rect.left+(rect.width-canvas.width*scale)/2,top=rect.top+(rect.height-canvas.height*scale)/2;return [Math.round((e.clientX-left)/scale),Math.round((e.clientY-top)/scale)];}
function reviewMask(){try{validateCurrent();if(draft.length)throw Error('Close the current polygon first');const f=current(),visibility=$('visibility').value||null;if(!visibility)throw Error('Choose visible or fully hidden');if(visibility==='fully_hidden'&&(f.visible_object.length||f.landmarks.length))throw Error('Clear target polygons and landmarks before reviewing fully hidden');f.visibility=visibility;if(visibility==='fully_hidden'){f.landmark_status='not_applicable_fully_hidden';delete f.landmark_review_reason;}else if(!f.visible_object.some(p=>Array.isArray(p)||p.hole!==true))throw Error('Visible mask review needs a target polygon');f.status='reviewed';validateDocument(data,data.object);paint();message('Mask and visibility review saved independently');}catch(e){message(e.message);}}
function reviewLandmarks(){try{validateCurrent();const f=current();if(!data.operator)throw Error('Landmark review needs a reviewer name');if(f.visibility!=='visible')throw Error('Landmark review requires a visible object');if(!f.landmarks.length||!f.landmarks.some(l=>l.visible!==false)||!f.landmarks.filter(l=>l.visible!==false).every(l=>l.reviewed===true))throw Error('Add independently reviewed visible landmarks first');f.landmark_status='reviewed';delete f.landmark_review_reason;validateDocument(data,data.object);paint();message('Landmark review saved independently');}catch(e){message(e.message);}}
function markLandmarksUnobservable(){try{validateCurrent();const f=current();if(!data.operator)throw Error('Landmark review needs a reviewer name');if(f.visibility!=='visible')throw Error('Unobservable landmarks require a visible object');const reason=$('landmarkReason').value.trim();if(!reason)throw Error('Enter why no landmark is measurable');f.landmarks=[];f.landmark_status='unobservable';f.landmark_review_reason=reason;validateDocument(data,data.object);paint();message('Landmarks marked unobservable; mask review unchanged');}catch(e){message(e.message);}}
canvas.onclick=e=>{const p=pointFromEvent(e);if(p[0]<0||p[0]>=canvas.width||p[1]<0||p[1]>=canvas.height)return;if($('tool').value==='landmark'){const xyz=$('xyz').value.split(',').map(Number),id=$('landmarkId').value.trim();if(!id||xyz.length!==3||!xyz.every(Number.isFinite)||!$('correspondence').checked){message('Enter ID and independently reviewed model coordinate first');return;}if(current().landmarks.some(l=>l.landmark_id===id)){message('Use a unique stable landmark ID');return;}setLandmarkPending();current().landmarks.push({landmark_id:id,object_point_m:xyz,pixel:p,visible:true,reviewed:true});}else{if(savedTool&&savedTool!==$('tool').value){message('Close the current polygon before switching regions');return;}setMaskPending();savedTool=$('tool').value;draft.push(p);}paint();};
$('visibility').onchange=()=>{const value=$('visibility').value||null,f=current();if(f.visibility===value)return;if(value==='fully_hidden'&&(f.visible_object.length||f.landmarks.length)){$('visibility').value=f.visibility||'';message('Clear target polygons and landmarks before choosing fully hidden');return;}f.visibility=value;setBothPending();delete f.landmark_review_reason;paint();};
$('close').onclick=()=>{if(!savedTool||draft.length<3){message('At least three points required');return;}try{validatePath({points:draft,hole:$('hole').checked},data.resolution[0],data.resolution[1],'Polygon');}catch(e){message(e.message);return;}current()[savedTool].push({points:draft,hole:$('hole').checked});setMaskPending();draft=[];savedTool=null;paint();};
$('undo').onclick=()=>{let changed=false;if(draft.length){draft.pop();changed=true;}else if($('tool').value==='landmark'&&current().landmarks.length){current().landmarks.pop();setLandmarkPending();changed=true;}else if($('tool').value!=='landmark'&&current()[$('tool').value].length){current()[$('tool').value].pop();setMaskPending();changed=true;}if(!changed)message('Nothing to undo');paint();};
$('clear').onclick=()=>{Object.assign(current(),{visible_object:[],overlapping_hands:[],landmarks:[],status:'pending',landmark_status:'pending'});delete current().landmark_review_reason;draft=[];savedTool=null;paint();};
$('reviewMask').onclick=reviewMask;$('reviewLandmarks').onclick=reviewLandmarks;$('markUnobservable').onclick=markLandmarksUnobservable;
$('download').onclick=()=>{try{validateCurrent();const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=data.object+'-annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);message('Validated annotations downloaded');}catch(e){message(e.message);}};
$('reload').onclick=()=>$('file').click();$('file').onchange=async()=>{try{const file=$('file').files[0];if(!file)return;const d=JSON.parse(await file.text());install(d,$('object').value);message('Validated annotations loaded');}catch(e){message(e.message);}finally{$('file').value='';}};
$('object').onchange=()=>loadCase().catch(e=>message(e.message));$('frame').onchange=loadFrame;$('previous').onclick=()=>{$('frame').value=Math.max(0,Number($('frame').value)-1);loadFrame();};$('next').onclick=()=>{$('frame').value=Math.min(39,Number($('frame').value)+1);loadFrame();};loadCase().catch(e=>message(e.message));
</script></main></html>'''


def _frozen_cases_for_javascript():
    return {
        alias: {
            "resolution": list(FROZEN_RESOLUTION),
            "source_sha256": FROZEN_SOURCE_SHA256[alias],
            "pending_selection_status": PENDING_SELECTION_STATUS,
            "reviewed_selection_status": REVIEWED_SELECTION_STATUS,
            "frames": [[frame_id, group, f"{frame_id}.jpg"] for frame_id, group in rows],
        }
        for alias, rows in FROZEN_ANNOTATION_SELECTIONS.items()
    }


def render_html():
    return HTML.replace("__FROZEN_CASES__", json.dumps(_frozen_cases_for_javascript(), separators=(",", ":")))


def build():
    write_artifact(ROOT / "artifacts/model-quality/annotate.html", render_html())


def import_labels(path):
    path = Path(path)
    try:
        incoming = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid annotation JSON: {path}") from exc
    if not isinstance(incoming, dict):
        raise ValueError("Annotation document must be an object")
    alias = incoming.get("object")
    validate_annotation_document(incoming, alias)
    target = ROOT / "artifacts/model-quality/annotations" / alias / "annotations.json"
    try:
        prior = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot read canonical annotations: {target}") from exc
    validate_annotation_document(prior, alias)
    for field in ("object", "resolution", "source_sha256"):
        if incoming.get(field) != prior.get(field):
            raise ValueError("Annotation provenance changed")
    for edited, canonical in zip(incoming["frames"], prior["frames"]):
        for field in ("frame_id", "group", "image"):
            if edited.get(field) != canonical.get(field):
                raise ValueError(f"Frozen annotation {field} changed")
    write_artifact(target, json.dumps(incoming, indent=2, allow_nan=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--import-labels", type=Path)
    args = p.parse_args()
    if args.import_labels:
        import_labels(args.import_labels)
    else:
        build()
