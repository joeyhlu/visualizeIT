"""Independent object-identity boundary controls using tiny real file assets."""
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bench import workbench_contract as wc
from bench import workbench_neural as neural


class IntSubclass(int):
    pass


class IndependentObjectIdentityChecks(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / '.cache'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.glb = self.root / 'plane.glb'
        self.glb_sha = neural.write_plane_glb(self.glb)
        self.bundle = dict(schema='workbench_metric_bundle_v1', family='manufactured',
            glb_path='plane.glb', glb_sha256=self.glb_sha, obj_id=0,
            principal_point_origin='integer_centers', seed_pose_m=None,
            seed_label=None, seed_frame_id=None, metric_provenance='measured_metric_cad')

    def load_bundle(self, value):
        path = self.root / 'bundle.json'
        path.write_text(json.dumps(dict(self.bundle, obj_id=value)), encoding='utf-8')
        return neural.MetricBundle.from_file(path, root=self.root)

    def case(self, value):
        rgb, mask = self.root / 'rgb.bin', self.root / 'mask.bin'
        # Contract preparation authenticates file bytes; image decoding/parity is
        # a separate gate and is deliberately not claimed by these fixtures.
        rgb.write_bytes(b'manufactured RGB byte fixture')
        mask.write_bytes(b'manufactured mask byte fixture')
        return dict(case_id='manufactured', family='manufactured', obj_id=value,
            rgb_path=str(rgb), rgb_sha256=hashlib.sha256(rgb.read_bytes()).hexdigest(),
            mask_path=str(mask), mask_sha256=hashlib.sha256(mask.read_bytes()).hexdigest(),
            segmentation=dict(producer='manufactured-only', manifest_sha256='a' * 64),
            K=np.eye(3).ravel().tolist(), principal_point_origin='integer_centers',
            initial_pose_m=np.eye(4).tolist(), initialization_source='onboarding:manufactured',
            glb_path=str(self.glb), glb_sha256=self.glb_sha, metric_provenance='measured_metric_cad')

    @contextmanager
    def unrelated_model_prerequisites(self, during_authentication=None):
        # No genuine checkpoint is created, read or loaded. All real file/hash
        # checks below this fixture are limited to tiny inputs in this temp root.
        base = self.root / neural.BASE_CHECKPOINT_RELATIVE
        base.parent.mkdir(parents=True)
        base.write_bytes(b'fixture-only, not a checkpoint')
        pins = dict(neural.PRIMARY, base_checkpoint_bytes=base.stat().st_size,
                    base_checkpoint_sha256=hashlib.sha256(base.read_bytes()).hexdigest())
        with ExitStack() as stack:
            stack.enter_context(patch.object(neural, 'PRIMARY', pins))
            stack.enter_context(patch.object(neural, 'verify_primary_record', return_value={'record_sha256': 'a' * 64}))
            stack.enter_context(patch.object(neural, '_authenticated_bytes', side_effect=during_authentication))
            stack.enter_context(patch.object(neural, 'source_closure', return_value={'fixture_only': True}))
            stack.enter_context(patch.object(neural, '_render_factory', return_value={'fixture_only': True}))
            stack.enter_context(patch.object(neural, 'runtime_packages', return_value={'fixture_only': True}))
            yield

    def prepare(self, cases):
        return neural.prepare_workbench_contract(cases, root=self.root,
            head_path=self.root / 'never-read-head.pt', reference_path=self.root / 'never-read-reference.pt')

    def test_tiny_real_glb_bundle_retains_valid_integer_boundaries(self):
        for value in (0, 9, 18, 2**31 - 1):
            with self.subTest(value=value):
                bundle = self.load_bundle(value)
                self.assertIs(type(bundle.obj_id), int)
                self.assertEqual(bundle.obj_id, value)
                self.assertEqual(bundle.glb_sha256, self.glb_sha)
                self.assertEqual(bundle.glb_path, self.glb)

    def test_malformed_json_identity_never_reaches_geometry_reader(self):
        for value in (False, True, 0., 9.8, '9', -1, 2**31, None, [], {}):
            with self.subTest(value=value), patch.object(neural, 'glb_inventory') as inventory:
                with self.assertRaises(wc.ContractError):
                    self.load_bundle(value)
                inventory.assert_not_called()

    def test_bad_case_identity_refuses_before_model_or_asset_reads(self):
        for value in (False, True, 0., '1', -1, 2**31, None, np.int64(1), IntSubclass(1)):
            case = self.case(value)
            with self.subTest(value=repr(value)), \
                    patch.object(neural, 'verify_primary_record') as record, \
                    patch.object(neural, '_authenticated_bytes') as weights, \
                    patch.object(neural, 'glb_inventory') as mesh:
                with self.assertRaises(wc.ContractError):
                    self.prepare([self.case(0), case])
                record.assert_not_called()
                weights.assert_not_called()
                mesh.assert_not_called()

    def test_contract_keeps_valid_identity_snapshot_during_prerequisite_callbacks(self):
        case = self.case(18)
        def mutate_original(*args, **kwargs):
            case['obj_id'] = True
        with self.unrelated_model_prerequisites(mutate_original):
            contract = self.prepare([case])
        identity = contract.body['cases'][0]['obj_id']
        self.assertIs(type(identity), int)
        self.assertEqual(identity, 18)
        self.assertIs(case['obj_id'], True)
        self.assertEqual(contract.sha256, wc.canonical_sha256(contract.body))
        self.assertFalse(contract.body['quality_claim'])


if __name__ == '__main__':
    unittest.main()
