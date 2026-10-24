"""Codex-owned media integration tests; all pixels, CAD and metadata are manufactured."""
from pathlib import Path
import binascii
import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from unittest import mock
import zlib

import numpy as np

from tools import workbench_hb_scene3_profile as hp
from tools import workbench_hb_prefix_scout as sc
from tools.test_workbench_hb_prefix_scout import scene_three_catalog
from tools.test_workbench_hb_metric_inputs import (
    SCRATCH, REVISION, digest, json_bytes, manufactured_ply, payload_row,
    EXPECTED_VERTICES_M, EXPECTED_FACES, EXPECTED_COLOURS_U8,
    glb_document_and_binary, glb_accessor,
)
from tools.test_workbench_hb_camera_k import K


def manufactured_scene_three_inputs(root, workspace):
    """Build independently sealed 120-frame RGB, asymmetric CAD and K fixtures.

    The annotation sentinel has no bytes. No real images, model, camera or GT
    are read. R/t dummy values in camera metadata must remain unconverted.
    """
    catalog = scene_three_catalog(root)
    index = catalog['documents']['resources_index'][1]
    models = {}
    for object_id in (4, 5):
        raw = manufactured_ply(first_position='0 0 1000' if object_id == 4 else '0 0 2000')
        member = f'models/obj_{object_id:06d}.ply'
        row = payload_row(catalog['dataset'], hp.mi.MODEL_ARCHIVE, member, 'model', raw)
        index['resources'][member] = row
        models[object_id] = row
    index['selectedMeshBindings'] = [{'objectId': oid, 'meshMember': row['sourceMember'],
        'meshSHA256': row['uncompressedSHA256'], 'meshExpandedBytes': row['expandedBytes'],
        'meshCRC32': f"{row['crc32']:08x}"} for oid, row in models.items()]
    reuse_root = 'manufactured-metadata'
    camera = root / reuse_root / hp.CAMERA_MEMBER
    camera.parent.mkdir(parents=True)
    camera_raw = json_bytes({str(i): {'cam_K': K, 'cam_R_w2c': [9, 8, 7],
                                     'cam_t_w2c': [6, 5, 4]} for i in range(120)})
    camera.write_bytes(camera_raw)
    index['resources'][hp.CAMERA_MEMBER] = {
        'datasetRevision': REVISION, 'archive': hp.mi.RGB_ARCHIVE,
        'sourceMember': hp.CAMERA_MEMBER, 'allowedRoles': ['metadata'],
        'sourceFlags': 0, 'sourceOffset': 0, 'method': 0, 'sourceMethod': 8, 'flags': 0,
        'compressedBytes': len(camera_raw), 'expandedBytes': len(camera_raw),
        'sourceCompressedBytes': len(zlib.compress(camera_raw)),
        'crc32': binascii.crc32(camera_raw) & 0xffffffff,
        'compressedSHA256': digest(camera_raw), 'uncompressedSHA256': digest(camera_raw),
        'location': {'kind': 'reuse', 'path': camera.relative_to(root).as_posix()}}
    members = len(index['resources'])
    raw_index = json_bytes(index)
    catalog['documents']['acquisition_receipt'][1]['indexSHA256'] = digest(raw_index)
    catalog['documents']['integrity_record'][1]['indexSHA256'] = digest(raw_index)
    catalog['documents']['integrity_record'][1]['verifiedMembers'] = members
    for key, (_, document) in catalog['documents'].items():
        catalog['paths'][key].write_bytes(json_bytes(document))
        catalog['anchors'][key] = digest(catalog['paths'][key].read_bytes())
    scout = sc.ScoutContext(root, workspace, workspace / 'derived/scout',
        sc.SyntheticPins(catalog['anchors'], members, catalog['totals']['encoded_bytes'],
                         catalog['totals']['expanded_bytes']))
    plan = sc.plan_stage(scout,
        receipt_path=catalog['paths']['acquisition_receipt'], expected_receipt_sha256=catalog['anchors']['acquisition_receipt'],
        index_path=catalog['paths']['resources_index'], expected_index_sha256=catalog['anchors']['resources_index'],
        integrity_path=catalog['paths']['integrity_record'], expected_integrity_sha256=catalog['anchors']['integrity_record'],
        existing_input_bytes=0, out=scout.mi().scope / 'plan')
    plan_path = root / plan['plan']
    plan_doc = json.loads(plan_path.read_bytes())
    sc.preview_stage(scout, plan=plan_path, expected_plan_sha256=plan['plan_sha256'],
                     out=scout.mi().scope / 'preview')
    preview_path = scout.mi().scope / 'preview' / sc.RECEIPT_NAME
    pins = hp.SyntheticPins(catalog['anchors'], members, catalog['totals']['encoded_bytes'],
        catalog['totals']['expanded_bytes'], digest(plan_path.read_bytes()), digest(preview_path.read_bytes()),
        digest(camera_raw), plan_doc['input_budget']['total'], reuse_root)
    scope = workspace / 'derived/preparation'
    ctx = hp.ProfileContext(root, workspace, scope, pins)
    profile = hp.build_profile(ctx, scout_plan=plan_path, scout_receipt=preview_path,
        accepted_scout_hashes={'plan': pins.scout_plan_sha256, 'preview_receipt': pins.scout_receipt_sha256},
        historical_anchor_hashes=catalog['anchors'], out=scope / 'profile')
    profile_path = root / profile['profile']
    k_result = hp.extract_camera_k(ctx, profile=profile_path,
        expected_profile_sha256=profile['profile_sha256'], out=scope / 'camera-k')
    profile_doc = json.loads(profile_path.read_bytes())
    source = index['resources']['val_primesense/000003/rgb/000009.png']
    association = {'schema': 'workbench_hb_scene3_opening_association_v1',
        'timestamp_utc': '2000-01-01T00:00:00+00:00', 'reviewer': 'Codex manufactured test fixture',
        'scene': '000003', 'verdict': 'positive_visual_match',
        'profile': {'path': profile['profile'], 'sha256': profile['profile_sha256']},
        'selection': {'frame_id': 's000009', 'source_frame_index': 9,
             'source_png_sha256': source['uncompressedSHA256'], 'native_image_size': [640, 480],
             'positive_point_xy': [320, 340], 'roi_xyxy': [285, 270, 370, 440]},
        'cad': {'object_id': 4, 'member': models[4]['sourceMember'],
             'compressed_sha256': models[4]['compressedSHA256'],
             'uncompressed_sha256': models[4]['uncompressedSHA256']},
        'instance_ordinal': None,
        'evidence': {'opening_source_indices_inspected': [0, 9],
            'opening_preview': profile_doc['scout']['preview_receipt'],
            'cad_orientation_receipt': {'path': 'synthetic-only.json', 'sha256': 'a' * 64},
            'orientation_script_sha256': 'b' * 64, 'matching_features': ['manufactured only'],
            'rejected_candidate_5': 'Manufactured distinct geometry.',
            'profile_candidate_hint': 'Unbound candidate is not selected automatically.',
            'orientation_images_persisted': False},
        'boundaries': {'scored_images_used': False, 'gt_used': False, 'poses_used': False,
            'evaluator_masks_or_depth_used': False, 'neural_inference_used': False},
        'limits': 'Synthetic manufactured review, not an actual instance association.', 'quality_claim': False}
    association_path = root / 'association.json'
    association_path.write_bytes(json_bytes(association))
    return {'catalog': catalog, 'profile': profile_path, 'profile_sha256': profile['profile_sha256'],
        'camera_k': root / k_result['camera_k'], 'camera_k_sha256': k_result['camera_k_sha256'],
        'association': association_path, 'association_sha256': digest(association_path.read_bytes()),
        'association_doc': association, 'profile_doc': profile_doc, 'scope': scope,
        'profile_pins': pins, 'models': models, 'camera': camera}


class SceneThreeMediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools import workbench_hb_scene3_media as media
        cls.media = media
        SCRATCH.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(prefix='hb-media-unit-', dir=SCRATCH)
        cls.root = Path(cls.temp.name).resolve()
        assert cls.root.is_relative_to(SCRATCH.resolve())
        # These are small manufactured scratch stages, not actual preparation.
        # Keep their budget inputs deterministic while concurrent host workloads
        # change free space. Real workspace/admission/inference budgets are never
        # patched. Require the real reserve plus scratch room before starting.
        real_disk_usage = shutil.disk_usage
        measured = real_disk_usage(SCRATCH)
        if measured.free < hp.mi.FREE_RESERVE + (8 << 20):
            cls.temp.cleanup()
            raise unittest.SkipTest('real disk reserve unavailable for temporary manufactured fixture')
        def manufactured_disk_usage(path):
            if Path(path).resolve().is_relative_to(cls.root):
                return type(measured)(measured.total, measured.used, 2 << 30)
            return real_disk_usage(path)
        cls.disk_fixture = mock.patch.object(hp.mi.shutil, 'disk_usage', side_effect=manufactured_disk_usage)
        cls.disk_fixture.start()
        cls.addClassCleanup(cls.disk_fixture.stop)
        cls.addClassCleanup(cls.temp.cleanup)
        cls.workspace = cls.root / 'workspace'
        cls.workspace.mkdir()
        cls.fixture = manufactured_scene_three_inputs(cls.root, cls.workspace)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        f = self.fixture
        self.pins = self.media.SyntheticMediaPins(f['profile_pins'], f['profile_sha256'],
            f['camera_k_sha256'], f['association_sha256'], 4)
        self.ctx = self.media.MediaContext(self.root, self.workspace, f['scope'], self.pins)
        # Keep owned temporary leaf names below Windows path limits even for
        # the longest descriptive test and atomic receipt staging suffix.
        self.leaf = f['scope'] / digest(self._testMethodName.encode())[:10]

    def tearDown(self):
        if self.leaf.exists():
            assert self.leaf.resolve().is_relative_to(self.fixture['scope'].resolve()) and not self.leaf.is_symlink()
            shutil.rmtree(self.leaf)

    def build(self, **changes):
        f = self.fixture
        args = dict(profile=f['profile'], expected_profile_sha256=f['profile_sha256'],
            camera_k=f['camera_k'], expected_camera_k_sha256=f['camera_k_sha256'],
            association_review=f['association'], expected_association_review_sha256=f['association_sha256'],
            out=self.leaf / 'allowlist')
        args.update(changes)
        return self.media.build_allowlist(self.ctx, **args)

    def model(self, allow, **changes):
        args = dict(allowlist=self.root / allow['allowlist'],
            expected_allowlist_sha256=allow['allowlist_sha256'], out=self.leaf / 'model')
        args.update(changes)
        return self.media.prepare_model(self.ctx, **args)

    def video(self, allow, **changes):
        args = dict(allowlist=self.root / allow['allowlist'],
            expected_allowlist_sha256=allow['allowlist_sha256'], out=self.leaf / 'video')
        args.update(changes)
        return self.media.prepare_video(self.ctx, **args)

    def test_metadata_gate_opens_no_payload_or_raw_camera(self):
        with mock.patch.object(hp.mi, 'verified_payload_stream', side_effect=AssertionError('payload opened')), \
                mock.patch.object(hp.ck, 'hash_bounded', side_effect=AssertionError('raw camera read')):
            result = self.build()
        self.assertEqual(result['payload_opens'], 0)
        doc = json.loads((self.root / result['allowlist']).read_bytes())
        self.assertEqual(doc['label'], 'TEST_ONLY')
        self.assertEqual(doc['model']['object_id'], 4)
        self.assertEqual(len(doc['rgb']), 120)
        self.assertFalse(doc['trust']['upstream_authenticated'])
        self.assertNotIn('scene_gt', json.dumps(doc))
        self.assertEqual(self.fixture['profile_doc']['object']['cad_candidate']['object_id'], 5)

    def test_bad_expected_hashes_fail_before_any_payload(self):
        for field in ('expected_profile_sha256', 'expected_camera_k_sha256', 'expected_association_review_sha256'):
            with self.subTest(field=field), mock.patch.object(hp.mi, 'verified_payload_stream',
                    side_effect=AssertionError('payload opened')), self.assertRaises(hp.mi.InputsError):
                self.build(**{field: '0' * 64})
        self.assertFalse((self.leaf / 'allowlist').exists())

    def test_absent_review_cannot_fallback_to_profile_candidate(self):
        with self.assertRaises((hp.mi.InputsError, FileNotFoundError)):
            self.build(association_review=self.root / 'missing-review.json')
        self.assertFalse((self.leaf / 'allowlist').exists())

    def test_production_context_refuses_manufactured_pins(self):
        self.ctx = self.media.MediaContext(self.root, self.workspace, self.fixture['scope'])
        with self.assertRaises(hp.mi.InputsError):
            self.build()

    def test_association_semantics_reject_future_ambiguous_or_numeric_bools(self):
        mutations = [
            lambda d: d.__setitem__('verdict', 'ambiguous'),
            lambda d: d.__setitem__('instance_ordinal', 0),
            lambda d: d.__setitem__('quality_claim', True),
            lambda d: d['selection'].__setitem__('source_frame_index', True),
            lambda d: d['selection'].__setitem__('source_frame_index', 10),
            lambda d: d['selection'].__setitem__('frame_id', 's000010'),
            lambda d: d['selection'].__setitem__('source_png_sha256', '0' * 64),
            lambda d: d['selection'].__setitem__('positive_point_xy', [True, 340]),
            lambda d: d['selection'].__setitem__('positive_point_xy', [0, 0]),
            lambda d: d['selection'].__setitem__('roi_xyxy', [370, 440, 285, 270]),
            lambda d: d['cad'].__setitem__('object_id', True),
            lambda d: d['cad'].__setitem__('object_id', 5),
            lambda d: d['boundaries'].__setitem__('gt_used', True),
            lambda d: d['evidence'].__setitem__('opening_source_indices_inspected', [0, 10]),
            lambda d: d['evidence'].__setitem__('opening_source_indices_inspected', [0, True]),
        ]
        for i, change in enumerate(mutations):
            document = copy.deepcopy(self.fixture['association_doc']); change(document)
            with self.subTest(case=i), self.assertRaises(hp.mi.InputsError):
                self.media.validate_association(document, profile_doc=self.fixture['profile_doc'],
                    profile_rel=self.fixture['profile'].relative_to(self.root).as_posix(), p=self.media.pins(self.ctx))

    def test_association_raw_boundaries_require_exact_false(self):
        for value in (0, None, '', []):
            document = copy.deepcopy(self.fixture['association_doc'])
            document['boundaries']['poses_used'] = value
            with self.subTest(value=value), self.assertRaises(hp.mi.InputsError):
                self.media.validate_association(document, profile_doc=self.fixture['profile_doc'],
                    profile_rel=self.fixture['profile'].relative_to(self.root).as_posix(), p=self.media.pins(self.ctx))

    def test_allowlist_cannot_add_cad_or_change_scoring_split(self):
        result = self.build()
        original = json.loads((self.root / result['allowlist']).read_bytes())
        for key, change in [('model', lambda d: d['model'].__setitem__('object_id', 5)),
                ('split', lambda d: d['frozen_split']['scoring'].append(120)),
                ('scene', lambda d: d.__setitem__('scene', '000002')),
                ('clock', lambda d: d['clock'].__setitem__('physical_fps', 30)),
                ('unknown', lambda d: d.__setitem__('models', [d['model'], d['model']]))]:
            document = copy.deepcopy(original); change(document)
            with self.subTest(case=key), self.assertRaises(hp.mi.InputsError):
                self.media.validate_allowlist_doc(document, self.media.pins(self.ctx))

    def test_model_output_preserves_asymmetric_units_faces_colours_and_normals(self):
        allow = self.build()
        result = self.model(allow)
        receipt = json.loads((self.root / result['model_receipt']).read_bytes())
        document, binary = glb_document_and_binary((self.root / receipt['glb']['path']).read_bytes())
        primitive = document['meshes'][0]['primitives'][0]
        positions = glb_accessor(document, binary, primitive['attributes']['POSITION'])
        faces = glb_accessor(document, binary, primitive['indices']).reshape(-1, 3)
        colours = glb_accessor(document, binary, primitive['attributes']['COLOR_0'])
        normals = glb_accessor(document, binary, primitive['attributes']['NORMAL'])
        np.testing.assert_array_equal(positions, EXPECTED_VERTICES_M)
        np.testing.assert_array_equal(faces, EXPECTED_FACES)
        np.testing.assert_array_equal(colours, EXPECTED_COLOURS_U8)
        np.testing.assert_array_equal(normals, np.tile([0, 0, 1], (3, 1)))
        self.assertEqual(receipt['units']['factor'], .001)
        self.assertFalse(receipt['geometry']['decimated'])
        self.assertIsNone(receipt['geometry']['seed_pose'])

    def test_mutated_compressed_cad_is_refused_and_original_restored(self):
        allow = self.build()
        row = self.fixture['models'][4]
        path = self.fixture['catalog']['dataset'] / row['location']['path']
        original = path.read_bytes()
        try:
            path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaises(hp.mi.InputsError): self.model(allow)
            self.assertFalse((self.leaf / 'model' / self.media.MODEL_RECEIPT_NAME).exists())
        finally:
            path.write_bytes(original)

    def test_foreign_output_directory_remains_untouched(self):
        allow = self.build()
        output = self.leaf / 'foreign'; output.mkdir()
        sentinel = output / 'keep.bin'; sentinel.write_bytes(b'foreign')
        with self.assertRaises(hp.mi.InputsError): self.model(allow, out=output)
        self.assertEqual(sentinel.read_bytes(), b'foreign')
        self.assertEqual(sorted(p.name for p in output.iterdir()), ['keep.bin'])

    def test_encoder_early_record_never_owns_final_video(self):
        self.leaf.mkdir(parents=True)
        ledger = self.media.MediaLedger(self.ctx.mi(), self.leaf, 'prepare-video')
        adapter = self.media._EncoderLedger(ledger)
        adapter.record('video.mp4', 3, metadata=False)
        self.assertEqual(ledger.owned, {})
        self.assertFalse(ledger.owned_file('video.mp4'))

    def test_short_free_space_blocks_before_payload(self):
        usage = hp.mi.shutil.disk_usage(self.workspace)
        low = type(usage)(usage.total, usage.used, hp.mi.FREE_RESERVE)
        with mock.patch.object(hp.mi.shutil, 'disk_usage', return_value=low), \
                mock.patch.object(hp.mi, 'verified_payload_stream', side_effect=AssertionError('payload opened')), \
                self.assertRaises(hp.mi.InputsError):
            self.build()

    def test_sibling_files_count_towards_preparation_budget(self):
        self.leaf.mkdir(parents=True)
        ledger = self.media.MediaLedger(self.ctx.mi(), self.leaf, 'prepare-model')
        real_usage = hp.mi.scope_usage
        def usage(path):
            value = real_usage(path)
            if Path(path) == self.fixture['scope']:
                return {**value, 'total': hp.mi.FINAL_OUTPUTS_CAP - 10}
            return value
        with mock.patch.object(hp.mi, 'scope_usage', side_effect=usage), self.assertRaises(hp.mi.InputsError):
            ledger.check(100, metadata=False)

    def test_bounded_hash_refuses_growth_beyond_cap(self):
        self.leaf.mkdir(parents=True)
        path = self.leaf / 'growing.bin'; path.write_bytes(b'a' * 4)
        original_open = Path.open
        class Growing:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self, n=-1):
                self.n = getattr(self, 'n', 0) + 1
                return b'x' * min(n, 8) if self.n <= 4 else b''
        def opening(p, *args, **kwargs):
            return Growing() if p == path and args and args[0] == 'rb' else original_open(p, *args, **kwargs)
        with mock.patch.object(Path, 'open', opening), self.assertRaises(hp.mi.InputsError):
            self.media.sealed_sha256(path, 5)

    def test_encoder_append_rechecks_global_derived_limit(self):
        self.leaf.mkdir(parents=True)
        ledger = self.media.MediaLedger(self.ctx.mi(), self.leaf, 'prepare-video')
        with mock.patch.object(hp.mi, 'workspace_budget', side_effect=hp.mi.Blocked('global derived grew')), \
                self.assertRaises(hp.mi.InputsError):
            self.media._EncoderLedger(ledger).check(1, metadata=False)

    def test_successful_commit_retains_ownership_if_post_rename_lookup_fails(self):
        self.leaf.mkdir(parents=True)
        ledger = self.media.MediaLedger(self.ctx.mi(), self.leaf, 'prepare-video')
        staging = self.leaf / '.owned-staging'; staging.write_bytes(b'owned')
        with mock.patch.object(ledger, 'own', side_effect=OSError('post-rename lookup failed')):
            try:
                self.media._commit(ledger, staging, 'video.mp4')
            except OSError as exc:
                self.media._fail_stage(ledger, 'prepare-video', exc)
                self.assertFalse((self.leaf / 'video.mp4').exists())
                self.assertEqual((self.leaf / 'invalidated-video.mp4').read_bytes(), b'owned')
            else:
                self.assertTrue(ledger.owned_file('video.mp4'))

    def test_retained_json_staging_counts_towards_metadata_limit(self):
        retained = self.leaf / 'retained'; retained.mkdir(parents=True)
        partial = retained / '.receipt.json.staging-deadbeef'
        with partial.open('wb') as stream: stream.truncate(hp.mi.METADATA_AGGREGATE_CAP)
        target = self.leaf / 'fresh'
        ledger = self.media.MediaLedger(self.ctx.mi(), target, 'media-allowlist')
        with self.assertRaises(hp.mi.InputsError): ledger.check(1, metadata=True)

    def test_rehashed_allowlist_rows_cannot_grant_unapproved_payloads(self):
        allow = self.build()
        path = self.root / allow['allowlist']; original = path.read_bytes()
        for kind in ('model', 'video'):
            document = json.loads(original)
            row = document['model'] if kind == 'model' else document['rgb'][17]
            row['uncompressed_sha256'] = '0' * 64
            path.write_bytes(json_bytes(document))
            changed = {**allow, 'allowlist_sha256': digest(path.read_bytes())}
            try:
                with self.subTest(stage=kind), mock.patch.object(hp.mi, 'verified_payload_stream',
                        side_effect=AssertionError('unapproved payload permission reached')), \
                        self.assertRaises(hp.mi.InputsError):
                    (self.model if kind == 'model' else self.video)(changed)
            finally: path.write_bytes(original)

    def test_video_reseals_every_payload_after_receipt_publication(self):
        allow = self.build()
        doc = json.loads((self.root / allow['allowlist']).read_bytes())
        payload_root = self.fixture['catalog']['dataset']
        expected = {str((payload_root / row['payload_path']).resolve()) for row in doc['rgb']}
        events = []
        original_open, original_publish = Path.open, self.media.publish
        def opening(path, *args, **kwargs):
            if str(path.resolve()) in expected and args and args[0] == 'rb':
                events.append(('payload_read', str(path.resolve())))
            return original_open(path, *args, **kwargs)
        def publish(ledger, name, data, **kwargs):
            result = original_publish(ledger, name, data, **kwargs)
            if name == self.media.VIDEO_RECEIPT_NAME: events.append(('receipt_published', None))
            return result
        with mock.patch.object(Path, 'open', opening), mock.patch.object(self.media, 'publish', publish):
            self.video(allow)
        cutoff = next(i for i, event in enumerate(events) if event[0] == 'receipt_published')
        after = {path for kind, path in events[cutoff+1:] if kind == 'payload_read'}
        self.assertEqual(after, expected, 'stat identity cannot authenticate bytes changed with restored mtime')

    def test_final_prerequisites_reseal_all_source_payloads_after_publication(self):
        allow=self.build();model=self.model(allow);video=self.video(allow)
        doc=json.loads((self.root/allow['allowlist']).read_bytes())
        payload_root=self.fixture['catalog']['dataset']
        expected={str((payload_root/row['payload_path']).resolve()) for row in [*doc['rgb'],doc['model']]}
        events=[]
        original_open,original_publish=Path.open,self.media.publish
        def opening(path,*args,**kwargs):
            if str(path.resolve()) in expected and args and args[0]=='rb':events.append(('read',str(path.resolve())))
            return original_open(path,*args,**kwargs)
        def publish(ledger,name,data,**kwargs):
            value=original_publish(ledger,name,data,**kwargs)
            if name==self.media.PREREQ_NAME:events.append(('published',None))
            return value
        with mock.patch.object(Path,'open',opening),mock.patch.object(self.media,'publish',publish):
            self.media.check_prerequisites(self.ctx,allowlist=self.root/allow['allowlist'],
                expected_allowlist_sha256=allow['allowlist_sha256'],
                model_receipt=self.root/model['model_receipt'],expected_model_receipt_sha256=model['model_receipt_sha256'],
                video_receipt=self.root/video['video_receipt'],expected_video_receipt_sha256=video['video_receipt_sha256'],
                out=self.leaf/'prerequisites')
        cutoff=next(i for i,event in enumerate(events) if event[0]=='published')
        self.assertEqual({path for kind,path in events[cutoff+1:] if kind=='read'},expected)

    def test_actual_synthetic_codec_and_prerequisites_reject_rehashed_contradictions(self):
        from bench.vision import cv2
        allow = self.build(); model = self.model(allow); video = self.video(allow)
        model_path = self.root / model['model_receipt']; video_path = self.root / video['video_receipt']
        model_doc = json.loads(model_path.read_bytes()); video_doc = json.loads(video_path.read_bytes())
        self.assertEqual(len(video_doc['frames']), 120)
        self.assertIsNone(video_doc['clock']['physical_fps'])
        cap = cv2.VideoCapture(str(self.root / video_doc['video']['path']))
        count = 0
        try:
            self.assertTrue(cap.isOpened())
            while True:
                ok, bgr = cap.read()
                if not ok: break
                self.assertEqual(bgr.shape, (480, 640, 3))
                np.testing.assert_allclose(bgr[300, 300, ::-1], [count * 2, 37, 255-count], atol=8)
                np.testing.assert_allclose(bgr[20, 20, ::-1], [11, 177, 53], atol=8)
                count += 1
        finally: cap.release()
        self.assertEqual(count, 120)
        args = dict(allowlist=self.root / allow['allowlist'], expected_allowlist_sha256=allow['allowlist_sha256'],
            model_receipt=model_path, expected_model_receipt_sha256=model['model_receipt_sha256'],
            video_receipt=video_path, expected_video_receipt_sha256=video['video_receipt_sha256'])
        result = self.media.check_prerequisites(self.ctx, **args, out=self.leaf / 'prerequisites')
        self.assertEqual(result['status'], 'prepared_media_only')
        self.assertTrue(result['unresolved'])
        mutations = [('video frame', 'video', lambda d: d['frames'][17].__setitem__('png_sha256', '0'*64)),
            ('video count', 'video', lambda d: d['decode'].__setitem__('sequential_frames', 119)),
            ('physical FPS', 'video', lambda d: d['clock'].__setitem__('physical_fps', 30)),
            ('model units', 'model', lambda d: d['units'].__setitem__('factor', 1)),
            ('model geometry', 'model', lambda d: d['geometry'].__setitem__('triangles', 2)),
            ('process exit', 'video', lambda d: d['process'].__setitem__('exit_code', 7)),
            ('process stdout', 'video', lambda d: d['process'].__setitem__('stdout_bytes', 1)),
            ('video EOF', 'video', lambda d: d['decode'].__setitem__('end_of_video_after', 119)),
            ('video seek', 'video', lambda d: d['decode']['frame_reader_seek_check'].__setitem__('exact', False)),
            ('physical timestamps', 'video', lambda d: d['clock'].__setitem__('physical_timestamps', [0, 1])),
            ('crop declaration', 'video', lambda d: d['image'].__setitem__('crop', [0, 0, 100, 100])),
            ('model decimation', 'model', lambda d: d['geometry'].__setitem__('decimated', True)),
            ('model seed', 'model', lambda d: d['geometry'].__setitem__('seed_pose', [[1, 0], [0, 1]])),
            ('model bounds', 'model', lambda d: d['geometry'].__setitem__('bounds_m', [[0, 0, 0], [99, 99, 99]])),
            ('missing audited CAD reads','model',lambda d:d['reads'].__setitem__('audited_opens',{})),
            ('missing audited RGB reads','video',lambda d:d['reads'].__setitem__('audited_opens',{})),
            ('false planned bytes','model',lambda d:d['budget'].__setitem__('planned_bytes',0)),
            ('invented budget method','video',lambda d:d['budget'].__setitem__('method','arbitrary unverified method')),
            ('impossible ledger','model',lambda d:d['ledger'].__setitem__('scope_usage_now',{'total':0,'metadata':100,'files':0}))]
        for i, (name, kind, change) in enumerate(mutations):
            path = video_path if kind == 'video' else model_path
            original = path.read_bytes()
            document = copy.deepcopy(video_doc if kind == 'video' else model_doc); change(document)
            path.write_bytes(json_bytes(document))
            updated = {**args, f'expected_{kind}_receipt_sha256': digest(path.read_bytes())}
            try:
                with self.subTest(case=name), self.assertRaises(hp.mi.InputsError):
                    self.media.check_prerequisites(self.ctx, **updated, out=self.leaf / f'bad-prereq-{i}')
            finally: path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
