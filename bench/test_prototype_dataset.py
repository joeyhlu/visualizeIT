"""Synthetic contracts for the frozen YCB-Video prototype data boundary."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
import uuid
from unittest import mock

from . import prototype_dataset as dataset


def pose_at(degrees):
    angle = math.radians(degrees)
    return {'obj_id': 2, 'cam_R_m2c': [1, 0, 0, 0, 1, 0, 0, 0, 1],
            'cam_t_m2c': [-math.sin(angle), 0, -math.cos(angle)]}


def scene_metadata(start=0, visible=110):
    gt, info, camera = {}, {}, {}
    for logical in range(dataset.FRAME_COUNT):
        frame = start+logical
        gt[str(frame)] = [pose_at(logical if logical >= 10 else 0)]
        info[str(frame)] = [{'visib_fract': 0.5 if logical-10 < visible else 0.49}]
        camera[str(frame)] = {'cam_K': [1, 0, 0, 0, 1, 0, 0, 0, 1]}
    return gt, info, camera


class PrototypeDatasetTests(unittest.TestCase):
    def test_exclusions_cover_prior_canonical_ycb_ids_with_reasons(self):
        self.assertEqual(set(dataset.EXCLUSIONS), {5, 10, 13, 14, 15})
        self.assertTrue(all(dataset.EXCLUSIONS[key] for key in dataset.EXCLUSIONS))
        self.assertEqual(dataset.OBJECTS[10][0], '011_banana')
        self.assertEqual(dataset.OBJECTS[11][0], '019_pitcher_base')
        self.assertEqual(dataset.OBJECTS[13][0], '024_bowl')

    def test_earliest_window_uses_eval_visibility_and_viewpoint(self):
        gt, info, camera = scene_metadata(visible=55)
        chosen = dataset.find_earliest_window('000001', 2, gt, info, camera)
        self.assertEqual(chosen['frameIds'], list(range(120)))
        self.assertEqual(chosen['visibleEvaluationFrames'], 55)
        self.assertGreaterEqual(chosen['viewpointSpanDegrees'], 15)
        gt, info, camera = scene_metadata(visible=54)
        self.assertIsNone(dataset.find_earliest_window('000001', 2, gt, info, camera))

    def test_window_must_be_consecutive_and_one_target_instance_per_frame(self):
        gt, info, camera = scene_metadata()
        del gt['44']
        self.assertIsNone(dataset.find_earliest_window('000001', 2, gt, info, camera))
        gt, info, camera = scene_metadata()
        gt['75'].append(dict(gt['75'][0]))
        self.assertIsNone(dataset.find_earliest_window('000001', 2, gt, info, camera))

    def test_manifest_split_hides_every_eval_label_and_onboarding_masks(self):
        gt, info, camera = scene_metadata()
        selected = {'objectId': 2, 'canonicalYcbId': dataset.OBJECTS[2][0],
                    'name': dataset.OBJECTS[2][1], **dataset.find_earliest_window('000001', 2, gt, info, camera)}
        test_entries = {}
        model_entries = {'models/obj_000002.ply': {'size': 4}}
        for frame in range(120):
            test_entries[f'test/000001/rgb/{frame:06d}.png'] = {'size': 3}
            test_entries[f'test/000001/depth/{frame:06d}.png'] = {'size': 3}
            if frame >= 10:
                test_entries[f'test/000001/mask_visib/{frame:06d}_000000.png'] = {'size': 1}
        inference, evaluator = dataset._build_manifests(
            [selected], {'000001': (gt, info, camera)}, test_entries, model_entries,
            'ycbv-prototype-0123456789abcdef', 'a'*64)
        obj = inference['objects'][0]
        scored = obj['frames']
        self.assertEqual(len(obj['onboarding']), 10)
        self.assertEqual(len(scored), 110)
        self.assertTrue(all('evaluationPose' in row for row in obj['onboarding']))
        self.assertTrue(all('depth' in row for row in obj['onboarding']))
        self.assertTrue(all(not ({'evaluationPose', 'evaluationInfo', 'evaluationMask', 'depth'} & set(row))
                            for row in scored))
        eval_obj = evaluator['objects'][0]
        self.assertEqual(len(eval_obj['frames']), 110)
        self.assertTrue(all({'rgb', 'depth', 'evaluationPose', 'evaluationInfo', 'evaluationMask'} <= set(row)
                            for row in eval_obj['frames']))
        self.assertEqual(obj['referenceId'], eval_obj['referenceId'])
        self.assertEqual(inference['selectionDigest'], evaluator['selectionDigest'])

    def test_member_union_deduplicates_rgb_depth_but_keeps_instance_masks(self):
        selected = []
        for object_id, instance in ((2, 0), (3, 1)):
            selected.append({'objectId': object_id, 'sceneId': '000001',
                             'frameIds': list(range(120)), 'instanceIds': [instance]*120})
        test_entries = {}
        model_entries = {}
        for object_id in (2, 3):
            model_entries[f'models/obj_{object_id:06d}.ply'] = {'size': 5}
        for frame in range(120):
            for kind in ('rgb', 'depth'):
                test_entries[f'test/000001/{kind}/{frame:06d}.png'] = {'size': 7}
            for instance in (0, 1):
                test_entries[f'test/000001/mask_visib/{frame:06d}_{instance:06d}.png'] = {'size': 2}
        union = dataset._members_for(selected, test_entries, model_entries)
        self.assertEqual(len(union), 2 + 120*2 + 110*2)
        self.assertEqual(sum(union.values()), 10 + 120*14 + 220*2)

    def test_missing_assets_remain_explicit_and_no_failed_frame_is_dropped(self):
        missing = dataset._asset_ref('absent.png', {})
        self.assertIsNone(missing['path'])
        self.assertFalse(missing['available'])
        self.assertEqual(missing['failureReason'], 'source_member_missing')

    def test_hashes_are_frozen_by_full_metadata_member_path(self):
        raw = {f'test/000001/{name}': b'{}' for name in
               ('scene_gt.json', 'scene_gt_info.json', 'scene_camera.json')}
        archive = _MemoryArchive(raw)
        selection = {'metadataPreflightTransferredBytes': 0,
            'selectedObjects': [{'sceneId': '000001',
            'metadataMembers': {Path(member).name: member for member in raw},
            'metadataSha256': {member: hashlib.sha256(data).hexdigest()
                               for member, data in raw.items()}}]}
        parsed = dataset._load_selected_metadata(archive, _MemoryArchive({}), selection)
        self.assertIn('000001', parsed)

    def test_preflight_low_disk_returns_before_remotezip_or_download(self):
        factory = mock.Mock(side_effect=AssertionError('RemoteZip must not be constructed'))
        path = dataset.OUTPUT.parent/'low-disk-preflight-must-not-write'
        with mock.patch.object(dataset, '_disk_free', return_value=0):
            result = dataset.preflight(path, remote_zip_factory=factory)
        self.assertEqual(result['status'], 'blocked_low_disk')
        self.assertEqual(result['networkBytes'], 0)
        factory.assert_not_called()

    def test_fake_full_preflight_acquire_keeps_digest_and_split_boundaries(self):
        test_members, model_members = fake_ycbv_archive_members()
        def factory(url):
            members = model_members if url.endswith('ycbv_models.zip') else test_members
            return _MemoryArchive(members)
        with tempfile.TemporaryDirectory(dir=dataset.OUTPUT.parent,
                                         ignore_cleanup_errors=True) as temp:
            output = Path(temp)/'dataset'
            def quick_write(root, target, data, reserve, count_limit):
                target = Path(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                return target
            with mock.patch.object(dataset, '_disk_free', return_value=8*1024**3), \
                 mock.patch.object(dataset, '_atomic_write', side_effect=quick_write):
                frozen = dataset.preflight(output, remote_zip_factory=factory)
                self.assertEqual(frozen['status'], 'ready')
                self.assertLessEqual(frozen['plannedDatasetBytes'], dataset.MAX_CACHE_BYTES)
                result = dataset.acquire(output, remote_zip_factory=factory)
            self.assertEqual(result['status'], 'complete')
            inference_raw = (output/'inference.json').read_bytes()
            evaluator_raw = (output/'evaluator.json').read_bytes()
            inference = json.loads(inference_raw)
            evaluator = json.loads(evaluator_raw)
            self.assertEqual(inference['selectionDigest'], frozen['selectionDigest'])
            self.assertEqual(evaluator['selectionDigest'], frozen['selectionDigest'])
            self.assertEqual(inference['benchmarkId'], evaluator['benchmarkId'])
            self.assertEqual(inference['evaluatorManifestSHA256'], hashlib.sha256(evaluator_raw).hexdigest())
            self.assertEqual(len(inference['objects']), 10)
            for object_row, eval_row in zip(inference['objects'], evaluator['objects']):
                self.assertEqual(len(object_row['onboarding']), 10)
                self.assertEqual(len(object_row['frames']), 110)
                self.assertEqual(len(eval_row['frames']), 110)
                self.assertTrue(all('evaluationPose' in row for row in object_row['onboarding']))
                self.assertTrue(all('evaluationPose' not in row and 'depth' not in row
                                    for row in object_row['frames']))
            self.assertEqual(result['cachedBytes'], frozen['plannedDatasetBytes'])

    def test_pilot_revalidates_and_acquires_only_the_five_frozen_choices(self):
        test_members, model_members = pilot_ycbv_archive_members()

        def factory(url):
            members = model_members if url.endswith('ycbv_models.zip') else test_members
            return _MemoryArchive(members)

        temporary_root = dataset.OUTPUT.parent / f'test-pilot-{uuid.uuid4().hex}'
        source_path = temporary_root / 'dataset-v1' / 'selection.json'
        output = temporary_root / 'dataset-pilot-v1'
        temporary_root.mkdir(parents=True)
        source = pilot_blocked_selection(test_members)
        source_path.parent.mkdir(parents=True)
        source_path.write_bytes(dataset._json_bytes(source))

        def quick_write(root, target, data, reserve, count_limit):
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return target

        try:
            with mock.patch.object(dataset, '_disk_free', return_value=8*1024**3), \
                 mock.patch.object(dataset, '_atomic_write', side_effect=quick_write):
                frozen = dataset.pilot_preflight(source_path, output, remote_zip_factory=factory)
                self.assertEqual(frozen['status'], 'ready_pilot')
                self.assertEqual([row['objectId'] for row in frozen['selectedObjects']], [2, 4, 6, 9, 18])
                self.assertEqual(frozen['selectedObjects'], source['selectedObjects'])
                self.assertEqual(frozen['thresholds'], source['thresholds'])
                self.assertEqual(frozen['evidenceKind'], 'dataset_reference')
                self.assertEqual(frozen['benchmarkMode'], 'pilot')
                self.assertEqual(frozen['plannedObjects'], 5)
                self.assertEqual(frozen['tenObjectsEvaluated'], False)
                self.assertEqual(frozen['sourceSelection']['status'], source['status'])
                self.assertLessEqual(frozen['plannedDatasetBytes'], dataset.MAX_CACHE_BYTES)
                acquired = dataset.acquire(output, remote_zip_factory=factory)

            inference = json.loads((output/'inference.json').read_bytes())
            evaluator = json.loads((output/'evaluator.json').read_bytes())
            self.assertEqual(acquired['status'], 'complete')
            self.assertEqual(acquired['selectedObjects'], 5)
            self.assertEqual(inference['selectionDigest'], frozen['selectionDigest'])
            self.assertEqual(evaluator['selectionDigest'], frozen['selectionDigest'])
            self.assertEqual(inference['benchmarkMode'], 'pilot')
            self.assertEqual(inference['evidenceKind'], 'dataset_reference')
            self.assertEqual(inference['realUnseenObjects'], 5)
            self.assertFalse(inference['tenObjectsEvaluated'])
            self.assertEqual(len(inference['objects']), 5)
            for inf_obj, eval_obj in zip(inference['objects'], evaluator['objects']):
                self.assertEqual(len(inf_obj['onboarding']), 10)
                self.assertEqual(len(inf_obj['frames']), 110)
                self.assertEqual(len(eval_obj['frames']), 110)
                self.assertTrue(all('evaluationPose' not in row and 'depth' not in row
                                    for row in inf_obj['frames']))
            self.assertEqual(acquired['cachedBytes'], frozen['plannedDatasetBytes'])
        finally:
            if temporary_root.exists() and temporary_root.resolve().parent == dataset.OUTPUT.parent.resolve():
                shutil.rmtree(temporary_root, ignore_errors=True)

    def test_pilot_source_fails_closed_on_changed_gate_or_identity_order(self):
        test_members, _ = pilot_ycbv_archive_members()
        source = pilot_blocked_selection(test_members)
        source['thresholds']['minimumVisibleEvaluationFrames'] -= 1
        with self.assertRaises(dataset.PreflightBlocked):
            dataset._validate_pilot_source(source, dataset._json_bytes(source))
        source = pilot_blocked_selection(test_members)
        source['selectedObjects'].reverse()
        with self.assertRaises(dataset.PreflightBlocked):
            dataset._validate_pilot_source(source, dataset._json_bytes(source))

    def test_selection_digest_ignores_only_derived_size_and_identity_fields(self):
        value = {'selectedObjects': [1], 'plannedDatasetBytes': 10,
                 'selectionDigest': 'a', 'benchmarkId': 'b'}
        digest = dataset._selection_digest(value)
        value['plannedDatasetBytes'] = 20
        self.assertEqual(dataset._selection_digest(value), digest)
        value['selectedObjects'] = [2]
        self.assertNotEqual(dataset._selection_digest(value), digest)


class _MemoryArchive:
    def __init__(self, members):
        self.members = members
        self.entries = {name: {'size': len(data), 'compressed': len(data)}
                        for name, data in members.items()}
        self.transferred = 0

    def read(self, name):
        self.transferred += len(self.members[name])+30
        return self.members[name]


def fake_ycbv_archive_members():
    test, models = {}, {}
    gt, info, camera = {}, {}, {}
    for logical in range(dataset.FRAME_COUNT):
        frame = logical
        annotations, infos = [], []
        theta = math.radians(logical*0.3)
        for object_id in range(1, 22):
            annotations.append({'obj_id': object_id,
                'cam_R_m2c': [1, 0, 0, 0, 1, 0, 0, 0, 1],
                'cam_t_m2c': [-math.sin(theta), 0, -math.cos(theta)]})
            infos.append({'visib_fract': 1.0})
        gt[str(frame)] = annotations
        info[str(frame)] = infos
        camera[str(frame)] = {'cam_K': [500, 0, 320, 0, 500, 240, 0, 0, 1]}
    for name, value in (('scene_gt.json', gt), ('scene_gt_info.json', info),
                        ('scene_camera.json', camera)):
        member = f'test/000001/{name}'
        test[member] = json.dumps(value).encode()
    selected_ids = [object_id for object_id in sorted(dataset.OBJECTS)
                    if object_id not in dataset.EXCLUSIONS][:10]
    for frame in range(dataset.FRAME_COUNT):
        for kind, payload in (('rgb', b'rgb'), ('depth', b'dep')):
            test[f'test/000001/{kind}/{frame:06d}.png'] = payload+bytes([frame % 256])
        if frame >= dataset.ONBOARDING_COUNT:
            for object_id in selected_ids:
                instance = object_id-1
                test[f'test/000001/mask_visib/{frame:06d}_{instance:06d}.png'] = b'mask'
    for object_id in selected_ids:
        models[f'models/obj_{object_id:06d}.ply'] = f'ply-{object_id}'.encode()
    return test, models


def pilot_ycbv_archive_members():
    test, models = fake_ycbv_archive_members()
    for object_id in dataset.PILOT_OBJECT_IDS:
        models[f'models/obj_{object_id:06d}.ply'] = f'pilot-ply-{object_id}'.encode()
        for frame in range(dataset.ONBOARDING_COUNT, dataset.FRAME_COUNT):
            test[f'test/000001/mask_visib/{frame:06d}_{object_id-1:06d}.png'] = b'pilot-mask'
    return test, models


def pilot_blocked_selection(test_members):
    scene_id = '000001'
    parsed = {name: json.loads(test_members[f'test/{scene_id}/{name}'])
              for name in ('scene_gt.json', 'scene_gt_info.json', 'scene_camera.json')}
    names = {key: f'test/{scene_id}/{key}' for key in parsed}
    choices = []
    for object_id in dataset.PILOT_OBJECT_IDS:
        window = dataset.find_earliest_window(scene_id, object_id,
            parsed['scene_gt.json'], parsed['scene_gt_info.json'], parsed['scene_camera.json'])
        choice = {'objectId': object_id, 'canonicalYcbId': dataset.OBJECTS[object_id][0],
                  'name': dataset.OBJECTS[object_id][1], **window}
        choice['metadataSha256'] = {member: hashlib.sha256(test_members[member]).hexdigest()
                                    for member in names.values()}
        choices.append(choice)
    return dataset._selection_core(choices, choices, dataset.EXCLUSIONS, {}, {}, 0,
        8*1024**3, dataset.DEFAULT_MIN_FREE_BYTES, 'blocked_fewer_than_ten_eligible_identities')


if __name__ == '__main__':
    unittest.main()
