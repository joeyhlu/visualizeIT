"""Freeze and acquire a bounded, label-isolated YCB-Video prototype set.

The preflight reads only pinned ZIP metadata and scene annotations. Acquisition
is a separate explicit phase and never falls back to downloading an archive.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil

from .remote_zip import RemoteZip
from .ycb_assets import ROOT

REVISION = '5c2c4aa229800355648cd268040aa814f8dc94f0'
BASE = f'https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/{REVISION}/'
DATASET_URL = 'https://bop.felk.cvut.cz/datasets/'
OUTPUT = ROOT/'.cache'/'prototype-sprint'/'dataset'
MAX_CACHE_BYTES = 384*1024*1024
DEFAULT_MIN_FREE_BYTES = 1024*1024*1024
MAX_METADATA_TRANSFER_BYTES = 512*1024*1024
MAX_TOTAL_NETWORK_BYTES = 1024*1024*1024
MAX_MEMBER_BYTES = 32*1024*1024
FRAME_COUNT = 120
ONBOARDING_COUNT = 10
EVALUATION_COUNT = 110
MIN_VISIBLE_EVALUATION = 55
MIN_VIEW_CHANGE_DEGREES = 15.0
PILOT_OBJECT_IDS = (2, 4, 6, 9, 18)
PILOT_OUTPUT = ROOT/'.cache'/'prototype-sprint'/'dataset-pilot-v1'
PILOT_SOURCE_SELECTION = ROOT/'.cache'/'prototype-sprint'/'dataset-v1'/'selection.json'

# BOP YCB-Video object IDs and their canonical original YCB scan identities.
OBJECTS = {
    1: ('002_master_chef_can', 'Master Chef can'),
    2: ('003_cracker_box', 'Cracker box'),
    3: ('004_sugar_box', 'Sugar box'),
    4: ('005_tomato_soup_can', 'Tomato soup can'),
    5: ('006_mustard_bottle', 'Mustard bottle'),
    6: ('007_tuna_fish_can', 'Tuna fish can'),
    7: ('008_pudding_box', 'Pudding box'),
    8: ('009_gelatin_box', 'Gelatin box'),
    9: ('010_potted_meat_can', 'Potted meat can'),
    10: ('011_banana', 'Banana'),
    11: ('019_pitcher_base', 'Pitcher base'),
    12: ('021_bleach_cleanser', 'Bleach cleanser'),
    13: ('024_bowl', 'Bowl'),
    14: ('025_mug', 'Mug'),
    15: ('035_power_drill', 'Power drill'),
    16: ('036_wood_block', 'Wood block'),
    17: ('037_scissors', 'Scissors'),
    18: ('040_large_marker', 'Large marker'),
    19: ('051_large_clamp', 'Large clamp'),
    20: ('052_extra_large_clamp', 'Extra large clamp'),
    21: ('061_foam_brick', 'Foam brick'),
}

# Exclude physical YCB objects already used through BOP YCB-Video or the
# original YCB scans. Reasons and source evidence are serialized in selection.
EXCLUSIONS = {
    5: ['BOP YCB-Video object-000005 manifest and bench/bop_assets.py TARGETS',
        'original YCB 006_mustard_bottle in bench/ycb_assets.py'],
    10: ['original YCB 011_banana in bench/ycb_assets.py'],
    13: ['original YCB 024_bowl in bench/ycb_assets.py'],
    14: ['BOP YCB-Video object-000014 manifest and bench/bop_assets.py TARGETS',
         'original YCB 025_mug in bench/ycb_assets.py'],
    15: ['BOP YCB-Video object-000015 manifest and bench/bop_assets.py TARGETS',
         'original YCB 035_power_drill in bench/ycb_assets.py'],
}

PROJECT_IDENTITY_INVENTORY = [
    {'source': 'bench/hd_experiment.py; .cache/hope-hd/object-000005.json',
     'identity': 'HOPE toy Chocolate Pudding box',
     'crossDatasetAlias': None,
     'reason': 'HOPE dataset metadata describes separately scanned toy grocery objects; no exact YCB physical-object alias is established.'},
    {'source': 'bench/hd_experiment.py; .cache/hope-hd/object-000007.json',
     'identity': 'HOPE toy Corn can',
     'crossDatasetAlias': None,
     'reason': 'Different HOPE toy grocery object; a shared package category is not an exact YCB physical-object identity.'},
    {'source': 'bench/show3d_experiment.py CASES; bench/OFFLINE_PIPELINE.md',
     'identity': 'HOT3D keyboard id 28, mug_white id 9, bottle_ranch id 15',
     'crossDatasetAlias': None,
     'reason': 'HOT3D IDs are dataset-specific; no exact YCBV physical-object correspondence is established. Generic mug/bottle labels are not treated as identity evidence.'},
]


class PreflightBlocked(RuntimeError):
    """The metadata-only preflight cannot safely proceed."""


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True)+'\n').encode('utf-8')


def _selection_digest(selection):
    core = {key: value for key, value in selection.items()
            if key not in ('selectionDigest', 'benchmarkId', 'plannedDatasetBytes')}
    return hashlib.sha256(_json_bytes(core)).hexdigest()


def _disk_free(path):
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free


def _safe_root(output):
    output = Path(output).resolve()
    protected = (ROOT/'.cache'/'bop').resolve()
    if output == protected or protected in output.parents or output in protected.parents:
        raise ValueError('Prototype output may not overlap the original .cache/bop tree.')
    return output


def _rotation(value):
    r = [float(x) for x in value]
    if len(r) != 9 or not all(math.isfinite(x) for x in r):
        raise ValueError('Invalid BOP rotation.')
    matrix = [r[0:3], r[3:6], r[6:9]]
    det = (matrix[0][0]*(matrix[1][1]*matrix[2][2]-matrix[1][2]*matrix[2][1])
           - matrix[0][1]*(matrix[1][0]*matrix[2][2]-matrix[1][2]*matrix[2][0])
           + matrix[0][2]*(matrix[1][0]*matrix[2][1]-matrix[1][1]*matrix[2][0]))
    error = math.sqrt(sum((sum(matrix[k][i]*matrix[k][j] for k in range(3))-
                           (1.0 if i == j else 0.0))**2
                          for i in range(3) for j in range(3)))
    if abs(det-1.0) > 1e-3 or error > 1e-3:
        raise ValueError('BOP rotation is not a proper near-SO(3) matrix.')
    return matrix


def _view_direction(pose):
    r = _rotation(pose['cam_R_m2c'])
    t = [float(x) for x in pose['cam_t_m2c']]
    if len(t) != 3 or not all(math.isfinite(x) for x in t):
        raise ValueError('Invalid BOP translation.')
    # Camera center in object coordinates: -R^T t.
    d = [-sum(r[row][col]*t[row] for row in range(3)) for col in range(3)]
    norm = math.sqrt(sum(x*x for x in d))
    if norm <= 1e-12:
        raise ValueError('Degenerate object-to-camera direction.')
    return [x/norm for x in d]


def viewpoint_span_degrees(poses):
    directions = [_view_direction(pose) for pose in poses]
    if len(directions) < 2:
        return 0.0
    minimum_dot = min(max(-1.0, min(1.0,
        sum(a*b for a, b in zip(left, right))))
        for i, left in enumerate(directions) for right in directions[i+1:])
    return math.degrees(math.acos(minimum_dot))


def rotation_span_degrees(poses):
    """Report SO(3) span as secondary metadata; it is not the selection gate."""
    rotations = [_rotation(pose['cam_R_m2c']) for pose in poses]
    maximum = 0.0
    for i, a in enumerate(rotations):
        for b in rotations[i+1:]:
            trace = sum(sum(a[row][k]*b[col][k] for k in range(3))
                        for row, col in ((0, 0), (1, 1), (2, 2)))
            angle = math.degrees(math.acos(max(-1.0, min(1.0, (trace-1.0)/2.0))))
            maximum = max(maximum, angle)
    return maximum


def find_earliest_window(scene_id, object_id, gt, info, camera):
    """Return the earliest qualifying contiguous 120-frame window in a scene."""
    frame_ids = sorted(int(key) for key in gt)
    if not frame_ids:
        return None
    keys = set(frame_ids)
    for start in range(frame_ids[0], frame_ids[-1]-FRAME_COUNT+2):
        ids = list(range(start, start+FRAME_COUNT))
        if any(frame not in keys or str(frame) not in info or str(frame) not in camera for frame in ids):
            continue
        instances = []
        poses = []
        visibility = []
        valid = True
        for frame in ids:
            matches = [i for i, pose in enumerate(gt[str(frame)]) if pose.get('obj_id') == object_id]
            if len(matches) != 1:
                valid = False
                break
            instance = matches[0]
            try:
                pose = gt[str(frame)][instance]
                vis = float(info[str(frame)][instance]['visib_fract'])
                _view_direction(pose)
                if not math.isfinite(vis) or vis < 0 or vis > 1:
                    valid = False
                    break
            except (KeyError, IndexError, TypeError, ValueError):
                valid = False
                break
            instances.append(instance)
            poses.append(pose)
            visibility.append(vis)
        if not valid:
            continue
        eval_poses = poses[ONBOARDING_COUNT:]
        eval_visibility = visibility[ONBOARDING_COUNT:]
        visible_count = sum(vis >= 0.5 for vis in eval_visibility)
        span = viewpoint_span_degrees(eval_poses)
        if visible_count >= MIN_VISIBLE_EVALUATION and span >= MIN_VIEW_CHANGE_DEGREES:
            return {
                'sceneId': str(scene_id), 'frameIds': ids, 'instanceIds': instances,
                'viewpointSpanDegrees': span,
                'rotationSpanDegrees': rotation_span_degrees(eval_poses),
                'visibleEvaluationFrames': visible_count,
                'metadataMembers': {
                    'scene_gt.json': f'test/{scene_id}/scene_gt.json',
                    'scene_gt_info.json': f'test/{scene_id}/scene_gt_info.json',
                    'scene_camera.json': f'test/{scene_id}/scene_camera.json',
                },
            }
    return None


def _asset_ref(member, entries, missing_reason='source_member_missing'):
    if member not in entries:
        return {'path': None, 'sha256': None, 'available': False, 'failureReason': missing_reason}
    size = int(entries[member]['size'])
    if size > MAX_MEMBER_BYTES:
        raise PreflightBlocked(f'Archive member exceeds {MAX_MEMBER_BYTES} byte bound: {member}')
    return {'path': member, 'sha256': '0'*64, 'available': True}


def _rgb_member(entries, scene_id, frame_id):
    stem = f'test/{scene_id}/rgb/{frame_id:06d}'
    for suffix in ('.png', '.jpg', '.jpeg'):
        if stem+suffix in entries:
            return stem+suffix
    return stem+'.png'


def _mask_member(scene_id, frame_id, instance_id):
    return f'test/{scene_id}/mask_visib/{frame_id:06d}_{instance_id:06d}.png'


def _depth_member(scene_id, frame_id):
    return f'test/{scene_id}/depth/{frame_id:06d}.png'


def _model_member(object_id):
    return f'models/obj_{object_id:06d}.ply'


def _build_manifests(selected, metadata_by_scene, test_entries, model_entries,
                     benchmark_id, selection_digest, *, evidence_kind=None,
                     benchmark_mode=None, real_unseen_objects=None,
                     ten_objects_evaluated=None, planned_objects=None):
    inference = {'schemaVersion': 1, 'dataset': 'BOP YCB-Video', 'datasetRevision': REVISION,
                 'source': DATASET_URL, 'onboardingFrames': ONBOARDING_COUNT,
                 'evaluationFrames': EVALUATION_COUNT, 'benchmarkId': benchmark_id,
                 'selectionDigest': selection_digest,
                 'evaluatorManifestSHA256': '0'*64, 'objects': []}
    evaluator = {'schemaVersion': 1, 'dataset': 'BOP YCB-Video', 'datasetRevision': REVISION,
                 'source': DATASET_URL, 'labelKind': 'dataset-reference; not independent human truth',
                 'benchmarkId': benchmark_id, 'selectionDigest': selection_digest,
                 'objects': []}
    if evidence_kind is not None:
        mode = {'evidenceKind': evidence_kind, 'benchmarkMode': benchmark_mode,
                'realUnseenObjects': real_unseen_objects,
                'tenObjectsEvaluated': ten_objects_evaluated,
                'plannedObjects': planned_objects}
        inference.update(mode)
        evaluator.update(mode)
        evaluator['labelKind'] = ('dataset-reference pilot; not independent human truth'
                                  if benchmark_mode == 'pilot'
                                  else 'dataset-reference; not independent human truth')
    for choice in selected:
        object_id = choice['objectId']
        canonical, name = OBJECTS[object_id]
        scene_id = choice['sceneId']
        gt, info, camera = metadata_by_scene[scene_id]
        model_member = _model_member(object_id)
        model = _asset_ref(model_member, model_entries)
        inf_obj = {'objectId': object_id, 'canonicalYcbId': canonical, 'name': name,
                   'sceneId': scene_id,
                   'referenceId': f'ycbv-{REVISION[:12]}-{scene_id}-{object_id:06d}-{choice["frameIds"][0]:06d}',
                   'model': model, 'onboarding': [], 'frames': []}
        eval_obj = {'objectId': object_id, 'canonicalYcbId': canonical, 'name': name,
                    'sceneId': scene_id,
                    'referenceId': inf_obj['referenceId'], 'model': model, 'frames': []}
        for logical_index, (frame_id, instance_id) in enumerate(zip(choice['frameIds'], choice['instanceIds'])):
            rgb = _asset_ref(_rgb_member(test_entries, scene_id, frame_id), test_entries)
            depth = _asset_ref(_depth_member(scene_id, frame_id), test_entries)
            common = {'frameId': frame_id, 'logicalIndex': logical_index,
                      'timestampKind': 'frame_index_no_fps', 'timestampSeconds': None,
                      'calibration': camera[str(frame_id)], 'rgb': rgb}
            if logical_index < ONBOARDING_COUNT:
                inf_obj['onboarding'].append({**common, 'depth': depth,
                    'evaluationPose': gt[str(frame_id)][instance_id]})
            else:
                inf_obj['frames'].append(common)
                mask = _asset_ref(_mask_member(scene_id, frame_id, instance_id), test_entries)
                eval_obj['frames'].append({
                    'frameId': frame_id, 'logicalIndex': logical_index,
                    'timestampKind': 'frame_index_no_fps', 'timestampSeconds': None,
                    'rgb': rgb,
                    'calibration': camera[str(frame_id)],
                    'evaluationPose': gt[str(frame_id)][instance_id],
                    'evaluationInfo': info[str(frame_id)][instance_id],
                    'evaluationMask': mask, 'depth': depth,
                })
        inference['objects'].append(inf_obj)
        evaluator['objects'].append(eval_obj)
    return inference, evaluator


def _members_for(selected, test_entries, model_entries):
    members = {}
    for choice in selected:
        object_id = choice['objectId']
        model = _model_member(object_id)
        if model in model_entries:
            members[model] = int(model_entries[model]['size'])
        for frame_id, instance_id in zip(choice['frameIds'], choice['instanceIds']):
            for member in (_rgb_member(test_entries, choice['sceneId'], frame_id),
                           _depth_member(choice['sceneId'], frame_id)):
                if member in test_entries:
                    members[member] = int(test_entries[member]['size'])
            if frame_id >= choice['frameIds'][ONBOARDING_COUNT]:
                member = _mask_member(choice['sceneId'], frame_id, instance_id)
                if member in test_entries:
                    members[member] = int(test_entries[member]['size'])
    return members


def _selection_core(selected, candidates, exclusions, member_sizes, manifest_sizes,
                    metadata_transfer_bytes, free_bytes, minimum_free_bytes, status):
    return {
        'schemaVersion': 1, 'status': status, 'dataset': 'BOP YCB-Video',
        'datasetRevision': REVISION, 'source': DATASET_URL,
        'identityMapEvidence': [
            'BOP YCB-Video canonical object ID table, pinned revision '+REVISION,
            'bench/bop_assets.py TARGETS and .cache/bop/object-000005.json, object-000014.json, object-000015.json',
            'bench/ycb_assets.py OBJECTS; .cache/ycb/011_banana/source.json and .cache/ycb/024_bowl/source.json',
            '.cache/generalization_ready_input_audit_v1.md records the five prior YCB scan identities and BOP model inventory',
        ],
        'selectionOrder': 'ascending BOP objectId; earliest lexicographic sceneId then frameId',
        'thresholds': {'consecutiveFrames': FRAME_COUNT, 'onboardingFrames': ONBOARDING_COUNT,
                       'evaluationFrames': EVALUATION_COUNT,
                       'minimumViewpointSpanDegrees': MIN_VIEW_CHANGE_DEGREES,
                       'minimumVisibleEvaluationFrames': MIN_VISIBLE_EVALUATION,
                       'visibilityFraction': 0.5},
        'excludedObjectIds': {str(k): {'canonicalYcbId': OBJECTS[k][0], 'reasons': v}
                              for k, v in sorted(exclusions.items())},
        'otherProjectIdentityInventory': PROJECT_IDENTITY_INVENTORY,
        'selectedObjects': selected, 'eligibleCandidates': candidates,
        'members': [{'path': path, 'bytes': size} for path, size in sorted(member_sizes.items())],
        'memberUnionBytes': sum(member_sizes.values()),
        'manifestByteSizes': manifest_sizes,
        'metadataPreflightTransferredBytes': metadata_transfer_bytes,
        'storage': {'cacheLimitBytes': MAX_CACHE_BYTES, 'freeBytesAtPreflight': free_bytes,
                    'minimumFreeBytes': minimum_free_bytes},
    }


def _write_selection(output, selection, dataset_bytes=None):
    output.mkdir(parents=True, exist_ok=True)
    if dataset_bytes is not None:
        selection['plannedDatasetBytes'] = dataset_bytes
    raw = _json_bytes(selection)
    if dataset_bytes is not None:
        total = selection['memberUnionBytes'] + sum(selection['manifestByteSizes'].values()) + len(raw)
        if total != dataset_bytes:
            selection['plannedDatasetBytes'] = total
            raw = _json_bytes(selection)
            total = selection['memberUnionBytes'] + sum(selection['manifestByteSizes'].values()) + len(raw)
            selection['plannedDatasetBytes'] = total
            raw = _json_bytes(selection)
    target = output/'selection.json'
    if target.exists():
        existing = target.read_bytes()
        if existing != raw:
            raise ValueError('A frozen prototype selection already exists; refusing to replace it.')
        return target
    _atomic_write(output, target, raw, reserve=0, count_limit=MAX_CACHE_BYTES)
    return target


def _atomic_write(root, target, data, reserve, count_limit):
    root = Path(root).resolve()
    target = Path(target).resolve()
    if target != root and root not in target.parents:
        raise ValueError('Write escaped the prototype dataset cache.')
    if target.exists():
        if target.read_bytes() == data:
            return target
        raise ValueError(f'Retained output differs; refusing overwrite: {target.name}')
    current = sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) if root.exists() else 0
    if current + len(data) > count_limit:
        raise PreflightBlocked('Prototype dataset cache would exceed 384 MiB.')
    if _disk_free(root)-len(data) < reserve:
        raise PreflightBlocked('Disk reserve would be crossed by the next write.')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+'.part')
    if temporary.exists():
        raise ValueError(f'Retained partial file exists; refusing overwrite: {temporary.name}')
    try:
        temporary.write_bytes(data)
        temporary.replace(target)
    except Exception:
        # Keep a failed .part for inspection; never erase acquisition evidence.
        raise


def preflight(output=OUTPUT, *, minimum_free_bytes=DEFAULT_MIN_FREE_BYTES,
              remote_zip_factory=RemoteZip):
    """Freeze a metadata-only selection and exact deduplicated acquisition byte plan."""
    output = _safe_root(output)
    if minimum_free_bytes < 1024*1024*1024:
        raise ValueError('The authorized prototype free-space reserve is at least 1 GiB.')
    free_bytes = _disk_free(output)
    if free_bytes < minimum_free_bytes:
        return {'schemaVersion': 1, 'status': 'blocked_low_disk', 'freeBytes': free_bytes,
                'minimumFreeBytes': minimum_free_bytes, 'networkBytes': 0,
                'message': 'No metadata or dataset requests were made because the disk reserve is not met.'}

    test_archive = remote_zip_factory(BASE+'ycbv_test_all.zip')
    model_archive = remote_zip_factory(BASE+'ycbv_models.zip')
    transfer = test_archive.transferred + model_archive.transferred
    if transfer > MAX_METADATA_TRANSFER_BYTES:
        raise PreflightBlocked('ZIP metadata directories exceeded the preflight network budget.')
    scenes = sorted({PurePosixPath(path).parts[1] for path in test_archive.entries
                     if path.startswith('test/') and path.endswith('/scene_gt.json')})
    candidates = [object_id for object_id in sorted(OBJECTS) if object_id not in EXCLUSIONS]
    earliest = {}
    metadata_hashes = {}
    for scene_id in scenes:
        names = {key: f'test/{scene_id}/{key}' for key in
                 ('scene_gt.json', 'scene_gt_info.json', 'scene_camera.json')}
        if any(name not in test_archive.entries for name in names.values()):
            continue
        metadata = {}
        raw_metadata = {}
        try:
            for key, member in names.items():
                if int(test_archive.entries[member]['size']) > MAX_MEMBER_BYTES:
                    raise PreflightBlocked(f'Scene metadata exceeds member bound: {member}')
                raw = test_archive.read(member)
                raw_metadata[key] = raw
                metadata[key] = json.loads(raw)
                transfer = test_archive.transferred + model_archive.transferred
                if transfer > MAX_METADATA_TRANSFER_BYTES:
                    raise PreflightBlocked('Scene metadata exceeded the bounded preflight transfer budget.')
        except (ValueError, KeyError, json.JSONDecodeError):
            # Retain this scene as unavailable metadata; do not invent annotations.
            continue
        gt, info, camera = (metadata[key] for key in
                            ('scene_gt.json', 'scene_gt_info.json', 'scene_camera.json'))
        for object_id in candidates:
            if object_id in earliest:
                continue
            found = find_earliest_window(scene_id, object_id, gt, info, camera)
            if found is not None:
                earliest[object_id] = found
                metadata_hashes[(scene_id, 'scene_gt.json')] = hashlib.sha256(raw_metadata['scene_gt.json']).hexdigest()
                metadata_hashes[(scene_id, 'scene_gt_info.json')] = hashlib.sha256(raw_metadata['scene_gt_info.json']).hexdigest()
                metadata_hashes[(scene_id, 'scene_camera.json')] = hashlib.sha256(raw_metadata['scene_camera.json']).hexdigest()
        if len(earliest) == len(candidates):
            break

    eligible = []
    for object_id in sorted(earliest):
        choice = {'objectId': object_id, 'canonicalYcbId': OBJECTS[object_id][0],
                  'name': OBJECTS[object_id][1], **earliest[object_id]}
        choice['metadataSha256'] = {
            member: metadata_hashes[(choice['sceneId'], name)]
            for name, member in choice['metadataMembers'].items()
        }
        eligible.append(choice)
    selected = eligible[:10]
    if len(selected) < 10:
        status = 'blocked_fewer_than_ten_eligible_identities'
        selection = _selection_core(selected, eligible, EXCLUSIONS, {}, {}, transfer,
                                    free_bytes, minimum_free_bytes, status)
        _write_selection(output, selection)
        return selection

    selected_scene_ids = {row['sceneId'] for row in selected}
    metadata_by_scene = {}
    for scene_id in selected_scene_ids:
        def check_preflight_transfer():
            if test_archive.transferred + model_archive.transferred > MAX_METADATA_TRANSFER_BYTES:
                raise PreflightBlocked('Selected-scene metadata exceeded the preflight transfer budget.')
        metadata_by_scene[scene_id] = _read_scene_metadata(
            test_archive, scene_id, check_preflight_transfer)
    transfer = test_archive.transferred + model_archive.transferred
    if transfer > MAX_METADATA_TRANSFER_BYTES:
        raise PreflightBlocked('Selected-scene metadata exceeded the bounded preflight transfer budget.')
    # Verify metadata bytes still match the hashes frozen while selecting.
    for row in selected:
        current = metadata_by_scene[row['sceneId']][3]
        if any(hashlib.sha256(current[key]).hexdigest() != row['metadataSha256'][member]
               for key, member in row['metadataMembers'].items()):
            raise PreflightBlocked('Selected scene metadata changed during preflight.')
    decoded = {scene: values[:3] for scene, values in metadata_by_scene.items()}
    member_sizes = _members_for(selected, test_archive.entries, model_archive.entries)
    placeholder_digest = '0'*64
    placeholder_id = 'ycbv-prototype-'+'0'*16
    inference, evaluator = _build_manifests(selected, decoded, test_archive.entries,
                                             model_archive.entries, placeholder_id, placeholder_digest)
    manifest_sizes = {'inference.json': len(_json_bytes(inference)),
                      'evaluator.json': len(_json_bytes(evaluator))}
    selection = _selection_core(selected, eligible, EXCLUSIONS, member_sizes, manifest_sizes,
                                transfer, free_bytes, minimum_free_bytes, 'ready')
    selection_digest = _selection_digest(selection)
    benchmark_id = 'ycbv-prototype-'+selection_digest[:16]
    selection['selectionDigest'] = selection_digest
    selection['benchmarkId'] = benchmark_id
    # Fixed-point because selection.json reports its exact own byte contribution.
    total = sum(member_sizes.values()) + sum(manifest_sizes.values())
    for _ in range(5):
        raw = _json_bytes({**selection, 'plannedDatasetBytes': total})
        new_total = sum(member_sizes.values()) + sum(manifest_sizes.values()) + len(raw)
        if new_total == total:
            break
        total = new_total
    selection['plannedDatasetBytes'] = total
    if total > MAX_CACHE_BYTES:
        selection['status'] = 'blocked_cache_limit'
    _write_selection(output, selection, total)
    return selection


def _validate_pilot_source(source, source_raw):
    if not isinstance(source, dict) or source.get('status') != 'blocked_fewer_than_ten_eligible_identities':
        raise PreflightBlocked('Pilot source must be the retained blocked full-benchmark selection.')
    if source.get('datasetRevision') != REVISION:
        raise ValueError('Pilot source selection is not pinned to the expected dataset revision.')
    if source.get('selectionDigest') and source['selectionDigest'] != _selection_digest(source):
        raise ValueError('Pilot source selection digest mismatch.')
    selected = source.get('selectedObjects')
    eligible = source.get('eligibleCandidates')
    if (not isinstance(selected, list) or not isinstance(eligible, list)
            or [row.get('objectId') for row in selected] != list(PILOT_OBJECT_IDS)
            or [row.get('objectId') for row in eligible] != list(PILOT_OBJECT_IDS)
            or selected != eligible):
        raise PreflightBlocked('Pilot requires exactly the five identities frozen by the blocked selection.')
    if source.get('thresholds') != {
            'consecutiveFrames': FRAME_COUNT, 'onboardingFrames': ONBOARDING_COUNT,
            'evaluationFrames': EVALUATION_COUNT,
            'minimumViewpointSpanDegrees': MIN_VIEW_CHANGE_DEGREES,
            'minimumVisibleEvaluationFrames': MIN_VISIBLE_EVALUATION,
            'visibilityFraction': 0.5}:
        raise PreflightBlocked('Pilot source thresholds differ from the approved frozen gate.')
    for row in selected:
        if (row.get('canonicalYcbId') != OBJECTS[row['objectId']][0]
                or row.get('name') != OBJECTS[row['objectId']][1]
                or len(row.get('frameIds', [])) != FRAME_COUNT
                or len(row.get('instanceIds', [])) != FRAME_COUNT
                or set(row.get('metadataSha256', {})) != set(row.get('metadataMembers', {}).values())):
            raise PreflightBlocked('Pilot source row is incomplete or does not preserve its frozen identity/window.')
        if row['frameIds'] != list(range(row['frameIds'][0], row['frameIds'][0] + FRAME_COUNT)):
            raise PreflightBlocked('Pilot source contains a nonconsecutive frozen window.')
    # Kept as a separate argument to make the byte-for-byte source pin explicit.
    return hashlib.sha256(source_raw).hexdigest()


def _pilot_manifest_options(selection):
    if selection.get('benchmarkMode') != 'pilot':
        return {}
    return {'evidence_kind': 'dataset_reference', 'benchmark_mode': 'pilot',
            'real_unseen_objects': len(selection['selectedObjects']),
            'ten_objects_evaluated': False,
            'planned_objects': len(selection['selectedObjects'])}


def pilot_preflight(source_selection=PILOT_SOURCE_SELECTION, output=PILOT_OUTPUT, *,
                    minimum_free_bytes=DEFAULT_MIN_FREE_BYTES,
                    remote_zip_factory=RemoteZip):
    """Derive an exact five-object pilot from the retained blocked selection."""
    output = _safe_root(output)
    source_path = Path(source_selection).resolve()
    sprint_root = (ROOT/'.cache'/'prototype-sprint').resolve()
    if source_path == sprint_root or sprint_root not in source_path.parents:
        raise ValueError('Pilot source selection must remain inside .cache/prototype-sprint.')
    if (output == source_path.parent or output in source_path.parent.parents
            or source_path.parent in output.parents):
        raise ValueError('Pilot output must be separate from the retained source selection.')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Pilot output already contains retained files; refusing to replace or mix selections.')
    if minimum_free_bytes < 1024*1024*1024:
        raise ValueError('The authorized prototype free-space reserve is at least 1 GiB.')
    if not source_path.is_file():
        raise ValueError('The retained blocked source selection does not exist.')
    source_raw = source_path.read_bytes()
    source = json.loads(source_raw.decode('utf-8'))
    source_sha256 = _validate_pilot_source(source, source_raw)
    free_bytes = _disk_free(output)
    if free_bytes < minimum_free_bytes:
        return {'schemaVersion': 1, 'status': 'blocked_low_disk', 'freeBytes': free_bytes,
                'minimumFreeBytes': minimum_free_bytes, 'networkBytes': 0,
                'sourceSelectionSHA256': source_sha256,
                'message': 'No metadata or dataset requests were made because the disk reserve is not met.'}

    test_archive = remote_zip_factory(BASE+'ycbv_test_all.zip')
    model_archive = remote_zip_factory(BASE+'ycbv_models.zip')
    transfer = test_archive.transferred + model_archive.transferred
    if transfer > MAX_METADATA_TRANSFER_BYTES:
        raise PreflightBlocked('ZIP metadata directories exceeded the preflight network budget.')
    metadata = _load_selected_metadata(test_archive, model_archive, source)
    transfer = test_archive.transferred + model_archive.transferred
    cumulative_transfer = int(source.get('metadataPreflightTransferredBytes', 0)) + transfer
    if transfer > MAX_METADATA_TRANSFER_BYTES or cumulative_transfer > MAX_TOTAL_NETWORK_BYTES:
        raise PreflightBlocked('Pilot metadata recheck exceeded the bounded network budget.')

    decoded = {scene: values[:3] for scene, values in metadata.items()}
    selected = source['selectedObjects']
    for choice in selected:
        gt, info, camera = decoded[choice['sceneId']]
        current = find_earliest_window(choice['sceneId'], choice['objectId'], gt, info, camera)
        if current is None:
            raise PreflightBlocked(f'Frozen pilot window is no longer eligible: object {choice["objectId"]}.')
        for key in ('sceneId', 'frameIds', 'instanceIds', 'visibleEvaluationFrames', 'metadataMembers'):
            if current[key] != choice[key]:
                raise PreflightBlocked(f'Frozen pilot {key} changed for object {choice["objectId"]}.')
        if (not math.isclose(current['viewpointSpanDegrees'], choice['viewpointSpanDegrees'], abs_tol=1e-9)
                or not math.isclose(current['rotationSpanDegrees'], choice['rotationSpanDegrees'], abs_tol=1e-9)):
            raise PreflightBlocked(f'Frozen pilot geometry gate changed for object {choice["objectId"]}.')

    member_sizes = _members_for(selected, test_archive.entries, model_archive.entries)
    placeholder_digest = '0'*64
    placeholder_id = 'ycbv-pilot-'+'0'*16
    pilot_options = {'evidence_kind': 'dataset_reference', 'benchmark_mode': 'pilot',
                     'real_unseen_objects': len(selected), 'ten_objects_evaluated': False,
                     'planned_objects': len(selected)}
    inference, evaluator = _build_manifests(
        selected, decoded, test_archive.entries, model_archive.entries,
        placeholder_id, placeholder_digest, **pilot_options)
    manifest_sizes = {'inference.json': len(_json_bytes(inference)),
                      'evaluator.json': len(_json_bytes(evaluator))}

    # Preserve the original frozen rows and gates verbatim, while deriving a new
    # selection identity and an independent exact deduplicated byte union.
    pilot = dict(source)
    pilot.update({
        'status': 'ready_pilot', 'evidenceKind': 'dataset_reference',
        'benchmarkMode': 'pilot', 'realUnseenObjects': len(selected),
        'plannedObjects': len(selected), 'tenObjectsEvaluated': False,
        'sourceSelection': {
            'path': source_path.relative_to(ROOT).as_posix(),
            'sha256': source_sha256,
            'selectionDigest': source.get('selectionDigest'),
            'status': source['status'],
        },
        'members': [{'path': path, 'bytes': size} for path, size in sorted(member_sizes.items())],
        'memberUnionBytes': sum(member_sizes.values()),
        'manifestByteSizes': manifest_sizes,
        'metadataPreflightTransferredBytes': cumulative_transfer,
        'storage': {'cacheLimitBytes': MAX_CACHE_BYTES, 'freeBytesAtPreflight': free_bytes,
                    'minimumFreeBytes': minimum_free_bytes},
    })
    pilot['selectionDigest'] = _selection_digest(pilot)
    pilot['benchmarkId'] = 'ycbv-pilot-'+pilot['selectionDigest'][:16]
    total = sum(member_sizes.values()) + sum(manifest_sizes.values())
    for _ in range(5):
        raw = _json_bytes({**pilot, 'plannedDatasetBytes': total})
        new_total = sum(member_sizes.values()) + sum(manifest_sizes.values()) + len(raw)
        if new_total == total:
            break
        total = new_total
    if total > MAX_CACHE_BYTES:
        pilot['status'] = 'blocked_cache_limit'
        pilot['selectionDigest'] = _selection_digest(pilot)
        pilot['benchmarkId'] = 'ycbv-pilot-'+pilot['selectionDigest'][:16]
        for _ in range(5):
            raw = _json_bytes({**pilot, 'plannedDatasetBytes': total})
            new_total = sum(member_sizes.values()) + sum(manifest_sizes.values()) + len(raw)
            if new_total == total:
                break
            total = new_total
    pilot['plannedDatasetBytes'] = total
    _write_selection(output, pilot, total)
    return pilot


def _read_scene_metadata(archive, scene_id, transfer_guard=None):
    names = {'scene_gt.json': f'test/{scene_id}/scene_gt.json',
             'scene_gt_info.json': f'test/{scene_id}/scene_gt_info.json',
             'scene_camera.json': f'test/{scene_id}/scene_camera.json'}
    raw = {}
    for key, member in names.items():
        raw[key] = archive.read(member)
        if transfer_guard is not None:
            transfer_guard()
    parsed = {key: json.loads(value) for key, value in raw.items()}
    return parsed['scene_gt.json'], parsed['scene_gt_info.json'], parsed['scene_camera.json'], raw


def _load_selected_metadata(archive, model_archive, selection):
    result = {}
    for scene_id in sorted({row['sceneId'] for row in selection['selectedObjects']}):
        def check_total_network():
            total = (selection['metadataPreflightTransferredBytes'] +
                     archive.transferred + model_archive.transferred)
            if total > MAX_TOTAL_NETWORK_BYTES:
                raise PreflightBlocked('Metadata recheck exceeded the 1 GiB total network budget.')
        gt, info, camera, raw = _read_scene_metadata(archive, scene_id, check_total_network)
        result[scene_id] = (gt, info, camera, raw)
    for row in selection['selectedObjects']:
        raw = result[row['sceneId']][3]
        for label, member in row['metadataMembers'].items():
            name = PurePosixPath(member).name
            if hashlib.sha256(raw[name]).hexdigest() != row['metadataSha256'][member]:
                raise PreflightBlocked(f'Frozen scene metadata hash changed: {member}')
    return result


def acquire(output=OUTPUT, *, minimum_free_bytes=DEFAULT_MIN_FREE_BYTES,
            remote_zip_factory=RemoteZip):
    """Acquire only the frozen union; caller must explicitly invoke this phase."""
    output = _safe_root(output)
    if minimum_free_bytes < 1024*1024*1024:
        raise ValueError('The authorized prototype free-space reserve is at least 1 GiB.')
    selection_path = output/'selection.json'
    if not selection_path.is_file():
        raise ValueError('Run metadata-only preflight first.')
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if selection.get('status') not in ('ready', 'ready_pilot'):
        raise PreflightBlocked(f'Frozen selection is not acquisition-ready: {selection.get("status")}')
    if selection.get('datasetRevision') != REVISION:
        raise ValueError('Frozen dataset revision differs from the pinned revision.')
    if selection['plannedDatasetBytes'] > MAX_CACHE_BYTES:
        raise PreflightBlocked('Frozen exact union exceeds the 384 MiB dataset cache cap.')
    free_bytes = _disk_free(output)
    if free_bytes < minimum_free_bytes:
        raise PreflightBlocked('Current disk free space is below the configured reserve; no assets requested.')
    test_archive = remote_zip_factory(BASE+'ycbv_test_all.zip')
    model_archive = remote_zip_factory(BASE+'ycbv_models.zip')
    metadata = _load_selected_metadata(test_archive, model_archive, selection)
    if test_archive.transferred + model_archive.transferred > MAX_METADATA_TRANSFER_BYTES:
        raise PreflightBlocked('Frozen-scene metadata exceeded the bounded preflight transfer budget.')
    decoded = {scene: values[:3] for scene, values in metadata.items()}
    if selection.get('selectionDigest') != _selection_digest(selection):
        raise ValueError('Frozen selection digest mismatch.')
    benchmark_id = selection['benchmarkId']
    selection_digest = selection['selectionDigest']
    inference, evaluator = _build_manifests(selection['selectedObjects'], decoded,
                                             test_archive.entries, model_archive.entries,
                                             benchmark_id, selection_digest,
                                             **_pilot_manifest_options(selection))
    expected_sizes = selection['manifestByteSizes']
    if len(_json_bytes(inference)) != expected_sizes['inference.json'] or len(_json_bytes(evaluator)) != expected_sizes['evaluator.json']:
        raise PreflightBlocked('Frozen manifest byte estimate changed; refusing asset acquisition.')
    members = {row['path']: row['bytes'] for row in selection['members']}
    if members != _members_for(selection['selectedObjects'], test_archive.entries, model_archive.entries):
        raise PreflightBlocked('Frozen member union changed; refusing asset acquisition.')
    current_files = {p.relative_to(output).as_posix(): p.stat().st_size
                     for p in output.rglob('*') if p.is_file() and p.name != 'selection.json'}
    planned_paths = set(members) | {'inference.json', 'evaluator.json'}
    if set(current_files)-planned_paths:
        raise ValueError('Unexpected files in prototype dataset cache; refusing to overwrite or delete.')
    missing_bytes = sum(size for path, size in members.items() if path not in current_files)
    manifest_missing = sum(expected_sizes[name] for name in ('inference.json', 'evaluator.json')
                           if name not in current_files)
    projected = selection_path.stat().st_size + sum(current_files.values()) + missing_bytes + manifest_missing
    if projected > MAX_CACHE_BYTES:
        raise PreflightBlocked('Current cache plus remaining frozen members exceeds 384 MiB.')
    if free_bytes - missing_bytes - manifest_missing < minimum_free_bytes:
        raise PreflightBlocked('Exact remaining acquisition bytes would cross the free-space reserve.')

    failed = {}
    hashes = {}
    for member, expected_size in sorted(members.items()):
        target = output.joinpath(*PurePosixPath(member).parts)
        archive = model_archive if member.startswith('models/') else test_archive
        entry = archive.entries[member]
        projected_network = (selection['metadataPreflightTransferredBytes'] +
                             test_archive.transferred + model_archive.transferred +
                             int(entry['compressed']) + 30)
        if projected_network > MAX_TOTAL_NETWORK_BYTES:
            failed[member] = {'errorType': 'TotalNetworkBudgetReached'}
            continue
        try:
            raw = archive.read(member)
            if len(raw) != expected_size:
                raise ValueError('Expanded member size differs from frozen plan.')
            member_digest = hashlib.sha256(raw).hexdigest()
            if target.exists():
                cached = target.read_bytes()
                if hashlib.sha256(cached).hexdigest() != member_digest:
                    raise ValueError(f'Retained cache hash mismatch; preserving existing file: {member}')
            else:
                _atomic_write(output, target, raw, reserve=minimum_free_bytes, count_limit=MAX_CACHE_BYTES)
            hashes[member] = member_digest
        except Exception as exc:
            failed[member] = {'errorType': type(exc).__name__}

    inference, evaluator = _build_manifests(selection['selectedObjects'], decoded,
                                             test_archive.entries, model_archive.entries,
                                             benchmark_id, selection_digest,
                                             **_pilot_manifest_options(selection))
    def replace_refs(value):
        if isinstance(value, dict):
            if value.get('available') and value.get('path') in members:
                path = value['path']
                if path in hashes:
                    value['sha256'] = hashes[path]
                elif path in failed:
                    value.update(path=None, sha256=None, available=False,
                                 failureReason='acquisition_failed')
            for child in value.values():
                replace_refs(child)
        elif isinstance(value, list):
            for child in value:
                replace_refs(child)
    replace_refs(inference)
    replace_refs(evaluator)
    evaluator_bytes = _json_bytes(evaluator)
    inference['evaluatorManifestSHA256'] = hashlib.sha256(evaluator_bytes).hexdigest()
    _atomic_write(output, output/'inference.json', _json_bytes(inference),
                  reserve=minimum_free_bytes, count_limit=MAX_CACHE_BYTES)
    _atomic_write(output, output/'evaluator.json', evaluator_bytes,
                  reserve=minimum_free_bytes, count_limit=MAX_CACHE_BYTES)
    return {'status': 'complete' if not failed else 'completed_with_retained_failures',
            'datasetRevision': REVISION, 'selectedObjects': len(selection['selectedObjects']),
            'evidenceKind': selection.get('evidenceKind', 'dataset_reference'),
            'benchmarkMode': selection.get('benchmarkMode', 'full'),
            'realUnseenObjects': selection.get('realUnseenObjects', len(selection['selectedObjects'])),
            'plannedObjects': selection.get('plannedObjects', len(selection['selectedObjects'])),
            'tenObjectsEvaluated': selection.get('tenObjectsEvaluated', len(selection['selectedObjects']) == 10),
            'inferenceManifest': 'inference.json', 'evaluatorManifest': 'evaluator.json',
            'failedMembers': failed, 'cachedBytes': sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
            'cacheLimitBytes': MAX_CACHE_BYTES, 'freeBytesAfterAcquisition': _disk_free(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'pilot-preflight', 'acquire'), default='preflight')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source-selection', type=Path, default=PILOT_SOURCE_SELECTION)
    parser.add_argument('--minimum-free-gib', type=float, default=1.0)
    args = parser.parse_args(argv)
    minimum = int(args.minimum_free_gib*1024**3)
    if args.phase == 'pilot-preflight':
        result = pilot_preflight(args.source_selection, args.output or PILOT_OUTPUT,
                                 minimum_free_bytes=minimum)
    else:
        output = args.output or OUTPUT
        result = (preflight if args.phase == 'preflight' else acquire)(
            output, minimum_free_bytes=minimum)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
