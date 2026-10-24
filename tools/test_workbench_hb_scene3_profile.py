"""Codex-owned scene-3 integration with synthetic RGB, camera and metadata only."""
from pathlib import Path
import copy
import hashlib
import json
import tempfile
import unittest
from unittest import mock
import binascii
import zlib

from tools import workbench_hb_scene3_profile as hp
from tools import workbench_hb_prefix_scout as sc
from tools.test_workbench_hb_prefix_scout import scene_three_catalog
from tools.test_workbench_hb_metric_inputs import SCRATCH, REVISION, digest, json_bytes
from tools.test_workbench_hb_camera_k import K


class SceneThreeProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(prefix='hb-profile-unit-', dir=SCRATCH)
        cls.root = Path(cls.temp.name)
        cls.workspace = cls.root / 'workspace'
        cls.workspace.mkdir()
        cls.catalog = scene_three_catalog(cls.root)
        catalog = cls.catalog
        cls.reuse_root = 'manufactured-metadata'
        cls.camera = cls.root / cls.reuse_root / hp.CAMERA_MEMBER
        cls.camera.parent.mkdir(parents=True)
        cls.camera_raw = json_bytes({str(i): {'cam_K': K, 'cam_R_w2c': [9, 8, 7],
            'cam_t_w2c': [6, 5, 4]} for i in range(120)})
        cls.camera.write_bytes(cls.camera_raw)
        row = {'datasetRevision': REVISION, 'archive': hp.mi.RGB_ARCHIVE,
            'sourceMember': hp.CAMERA_MEMBER, 'allowedRoles': ['metadata'],
            'sourceFlags': 0, 'sourceOffset': 0, 'method': 0, 'sourceMethod': 8, 'flags': 0,
            'compressedBytes': len(cls.camera_raw), 'expandedBytes': len(cls.camera_raw),
            'sourceCompressedBytes': len(zlib.compress(cls.camera_raw)),
            'crc32': binascii.crc32(cls.camera_raw) & 0xffffffff,
            'compressedSHA256': digest(cls.camera_raw), 'uncompressedSHA256': digest(cls.camera_raw),
            'location': {'kind': 'reuse', 'path': cls.camera.relative_to(cls.root).as_posix()}}
        cls.camera_row = row
        index = catalog['documents']['resources_index'][1]
        index['resources'][hp.CAMERA_MEMBER] = row
        raw_index = json_bytes(index)
        catalog['documents']['acquisition_receipt'][1]['indexSHA256'] = digest(raw_index)
        catalog['documents']['integrity_record'][1]['indexSHA256'] = digest(raw_index)
        for key, (_, doc) in catalog['documents'].items():
            catalog['paths'][key].write_bytes(json_bytes(doc))
            catalog['anchors'][key] = digest(catalog['paths'][key].read_bytes())
        scout = sc.ScoutContext(cls.root, cls.workspace, cls.workspace / 'derived/scout',
            sc.SyntheticPins(catalog['anchors'], 122, catalog['totals']['encoded_bytes'], catalog['totals']['expanded_bytes']))
        cls.plan_result = sc.plan_stage(scout,
            receipt_path=catalog['paths']['acquisition_receipt'], expected_receipt_sha256=catalog['anchors']['acquisition_receipt'],
            index_path=catalog['paths']['resources_index'], expected_index_sha256=catalog['anchors']['resources_index'],
            integrity_path=catalog['paths']['integrity_record'], expected_integrity_sha256=catalog['anchors']['integrity_record'],
            existing_input_bytes=0, out=scout.mi().scope / 'plan')
        cls.plan_path = cls.root / cls.plan_result['plan']
        cls.plan = json.loads(cls.plan_path.read_bytes())
        cls.preview_result = sc.preview_stage(scout, plan=cls.plan_path,
            expected_plan_sha256=cls.plan_result['plan_sha256'], out=scout.mi().scope / 'preview')
        cls.preview_path = scout.mi().scope / 'preview' / sc.RECEIPT_NAME
        cls.synthetic = hp.SyntheticPins(catalog['anchors'], 122, catalog['totals']['encoded_bytes'],
            catalog['totals']['expanded_bytes'], digest(cls.plan_path.read_bytes()), digest(cls.preview_path.read_bytes()),
            digest(cls.camera_raw), cls.plan['input_budget']['total'], cls.reuse_root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.scope = self.workspace / 'derived' / self._testMethodName
        self.ctx = hp.ProfileContext(self.root, self.workspace, self.scope, self.synthetic)

    def build(self, **changes):
        args = dict(scout_plan=self.plan_path, scout_receipt=self.preview_path,
            accepted_scout_hashes={'plan': self.synthetic.scout_plan_sha256,
                'preview_receipt': self.synthetic.scout_receipt_sha256},
            historical_anchor_hashes=self.catalog['anchors'], out=self.scope / 'profile')
        args.update(changes)
        return hp.build_profile(self.ctx, **args)

    def extract(self, profile, **changes):
        args = dict(profile=self.root / profile['profile'], expected_profile_sha256=profile['profile_sha256'],
            out=self.scope / 'camera-k')
        args.update(changes)
        return hp.extract_camera_k(self.ctx, **args)

    def test_metadata_profile_never_opens_camera_or_payload(self):
        with mock.patch.object(hp.ck, 'hash_bounded', side_effect=AssertionError('camera must remain unopened')), \
                mock.patch.object(hp.mi, 'verified_payload_stream', side_effect=AssertionError('no RGB/model opens')):
            result = self.build()
        doc = json.loads((self.root / result['profile']).read_bytes())
        self.assertEqual(doc['scene'], '000003')
        self.assertEqual(doc['pins_kind'], 'synthetic_test_only')
        self.assertEqual(doc['camera']['K'], None)
        self.assertEqual(len(doc['frames']['rows']), 120)
        self.assertEqual(doc['frozen_split']['scoring'], list(range(10, 120)))
        self.assertFalse(doc['trust']['upstream_authenticated'])

    def test_complete_two_stages_preserve_raw_k_without_any_pose_output(self):
        profile = self.build()
        with mock.patch.object(hp.mi, 'verified_payload_stream', side_effect=AssertionError('no RGB/model opens')):
            result = self.extract(profile)
        self.assertEqual(result['K'], K)
        doc = json.loads((self.root / result['camera_k']).read_bytes())
        raw = json.dumps(doc)
        self.assertNotIn('cam_R_w2c', raw)
        self.assertNotIn('cam_t_w2c', raw)
        self.assertNotIn('scene_gt', raw)
        self.assertEqual(doc['pins_kind'], 'synthetic_test_only')
        self.assertFalse(doc['trust']['upstream_authenticated'])

    def test_wrong_anchor_argument_refused_before_parse(self):
        bad = dict(self.catalog['anchors']); bad['resources_index'] = '0' * 64
        with mock.patch.object(hp.mi, 'strict_json', side_effect=AssertionError('no parse before pins')), \
                self.assertRaises(hp.mi.Blocked):
            self.build(historical_anchor_hashes=bad)

    def test_wrong_accepted_scout_argument_refused_before_reads(self):
        with mock.patch.object(hp.mi, 'read_bounded', side_effect=AssertionError('no read before pins')), \
                self.assertRaises(hp.mi.Blocked):
            self.build(accepted_scout_hashes={'plan': '0' * 64, 'preview_receipt': self.synthetic.scout_receipt_sha256})

    def test_camera_hash_mismatch_stops_before_scanner(self):
        profile = self.build()
        with mock.patch.object(hp.ck, 'hash_bounded', return_value={'sha256': '0' * 64, 'crc32': self.camera_row['crc32']}), \
                mock.patch.object(hp.ck, 'scan_camera_file', side_effect=AssertionError('never scan unverified bytes')), \
                self.assertRaises(hp.mi.InputsError):
            self.extract(profile)

    def test_camera_crc_mismatch_stops_before_scanner(self):
        profile = self.build()
        with mock.patch.object(hp.ck, 'hash_bounded', return_value={'sha256': self.synthetic.camera_expanded_sha256, 'crc32': 0}), \
                mock.patch.object(hp.ck, 'scan_camera_file', side_effect=AssertionError('never scan unverified CRC')), \
                self.assertRaises(hp.mi.InputsError):
            self.extract(profile)

    def test_camera_closure_never_uses_unbounded_generic_hash(self):
        profile = self.build()
        original = hp.mi.sha256_file
        def checked(path):
            if Path(path) == self.camera:
                raise AssertionError('camera closure must retain expected-size and 1 MiB cap')
            return original(path)
        with mock.patch.object(hp.mi, 'sha256_file', side_effect=checked):
            result = self.extract(profile)
        self.assertEqual(result['K'], K)

    def test_wrong_profile_hash_stops_before_camera(self):
        profile = self.build()
        with mock.patch.object(hp.ck, 'hash_bounded', side_effect=AssertionError('no camera open')), \
                self.assertRaises(hp.mi.InputsError):
            self.extract(profile, expected_profile_sha256='0' * 64)

    def test_free_space_guard_precedes_source_reads(self):
        from types import SimpleNamespace
        with mock.patch.object(hp.mi.shutil, 'disk_usage', return_value=SimpleNamespace(free=0)), \
                mock.patch.object(hp.mi, 'read_bounded', side_effect=AssertionError('no reads on failed budget')), \
                self.assertRaises(hp.mi.Blocked):
            self.build()

    def test_retained_failed_scope_bytes_are_counted(self):
        self.scope.mkdir(parents=True)
        with (self.scope / 'retained-failure.bin').open('wb') as stream:
            stream.truncate(hp.mi.FINAL_OUTPUTS_CAP)
        with mock.patch.object(hp.mi, 'read_bounded', side_effect=AssertionError('no reads on exceeded scope')), \
                self.assertRaises(hp.mi.Blocked):
            self.build()

    def test_existing_output_collision_is_preserved(self):
        out = self.scope / 'profile'
        out.mkdir(parents=True)
        foreign = out / 'foreign.txt'; foreign.write_bytes(b'preserve me')
        with self.assertRaises(hp.mi.InputsError):
            self.build()
        self.assertEqual(foreign.read_bytes(), b'preserve me')

    def test_camera_reuse_row_refuses_other_scene_role_path_and_hash(self):
        args = dict(member=hp.CAMERA_MEMBER, archive=hp.mi.RGB_ARCHIVE, revision=REVISION,
            expected_sha256=self.synthetic.camera_expanded_sha256, max_bytes=hp.CAMERA_CAP,
            root=self.root, canonical=self.camera)
        for changes in ({'sourceMember': 'val_primesense/000002/scene_camera.json'},
                {'allowedRoles': ['evaluation']}, {'method': 8}, {'compressedSHA256': '0' * 64},
                {'location': {'kind': 'reuse', 'path': 'elsewhere/scene_camera.json'}}):
            row = copy.deepcopy(self.camera_row); row.update(changes)
            with self.subTest(changes=changes), self.assertRaises(hp.ck.CameraError):
                hp.ck.validate_camera_row(row, **args)


if __name__ == '__main__':
    unittest.main()
