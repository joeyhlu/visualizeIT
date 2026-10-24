"""Codex-owned scene-000003 scout controls; manufactured bytes only.

The asymmetric RGB fixture identifies each frame independently of the scout's
implementation. It contains no downloaded pixels, calibration or evaluator data.
"""
from pathlib import Path
import copy
import json
import os
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from tools.test_workbench_hb_metric_inputs import (
    ROOT, SCRATCH, REVISION, SELECTION, digest, json_bytes, payload_row,
)


def scene_three_catalog(root):
    from bench.vision import cv2
    dataset = root / 'dataset'
    dataset.mkdir()
    resources = {}
    for index in range(120):
        # Analytical RGB identity: red identifies the frame; unequal channels
        # expose RGB/BGR reversal, while the corner patch exposes a mirror.
        rgb = np.full((480, 640, 3), [index * 2, 37, 255 - index], np.uint8)
        rgb[:40, :60] = [11, 177, 53]
        ok, encoded = cv2.imencode('.png', np.ascontiguousarray(rgb[:, :, ::-1]))
        if not ok:
            raise AssertionError('Manufactured PNG encoding failed')
        member = f'val_primesense/000003/rgb/{index:06d}.png'
        resources[member] = payload_row(dataset, 'hb_val_primesense.zip', member, 'rgb', encoded.tobytes())
    # A path-only sentinel: this fixture must never materialize annotations.
    resources['val_primesense/000003/scene_gt.json'] = {
        'location': {'kind': 'reuse', 'path': 'never-opened/scene_gt.json'}, 'allowedRoles': ['evaluation']}
    resources['val_primesense/000003/scene_camera.json'] = {
        'location': {'kind': 'reuse', 'path': 'never-opened/scene_camera.json'}, 'allowedRoles': ['calibration']}
    source_hashes = {'synthetic-acquisition.py': 'd' * 64}
    index = {'schemaVersion': 1, 'corpus': 'HB', 'datasetRevision': REVISION,
        'selectionDigest': SELECTION, 'storageKind': 'original_zip_members',
        'resources': resources, 'acquisitionSourceHashes': source_hashes}
    index_raw = json_bytes(index)
    receipt = {'status': 'complete', 'datasetRevision': REVISION, 'selectionDigest': SELECTION,
        'indexSHA256': digest(index_raw), 'sourceHashes': source_hashes}
    integrity = {'status': 'verified', 'datasetRevision': REVISION, 'selectionDigest': SELECTION,
        'indexSHA256': digest(index_raw), 'verifiedMembers': len(resources)}
    documents = {'acquisition_receipt': ('acquisition-receipt.json', receipt),
        'resources_index': ('resources.json', index),
        'integrity_record': ('hb-integrity-v1.json', integrity)}
    paths, anchors = {}, {}
    for key, (name, document) in documents.items():
        path = dataset / name
        path.write_bytes(json_bytes(document))
        paths[key], anchors[key] = path, digest(path.read_bytes())
    rows = [resources[f'val_primesense/000003/rgb/{i:06d}.png'] for i in range(120)]
    return {'dataset': dataset, 'documents': documents, 'paths': paths, 'anchors': anchors,
        'totals': {'encoded_bytes': sum(r['compressedBytes'] for r in rows),
                   'expanded_bytes': sum(r['expandedBytes'] for r in rows)}}


class PrefixScoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools import workbench_hb_prefix_scout as scout
        cls.scout = scout
        SCRATCH.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(prefix='hb-scout-unit-', dir=SCRATCH)
        cls.root = Path(cls.temp.name)
        cls.catalog = scene_three_catalog(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.workspace = self.root / self._testMethodName
        self.workspace.mkdir()
        c = self.catalog
        self.ctx = self.scout.ScoutContext(self.root, self.workspace,
            self.workspace / 'derived/scout', self.scout.SyntheticPins(c['anchors'], 122,
                c['totals']['encoded_bytes'], c['totals']['expanded_bytes']))

    def plan(self, **overrides):
        c = self.catalog; a = c['anchors']; p = c['paths']
        args = dict(receipt_path=p['acquisition_receipt'], expected_receipt_sha256=a['acquisition_receipt'],
            index_path=p['resources_index'], expected_index_sha256=a['resources_index'],
            integrity_path=p['integrity_record'], expected_integrity_sha256=a['integrity_record'],
            existing_input_bytes=0, out=self.ctx.mi().scope / 'plan')
        args.update(overrides)
        result = self.scout.plan_stage(self.ctx, **args)
        path = self.root / result['plan']
        return result, path, json.loads(path.read_bytes())

    def preview(self, result, **overrides):
        args = dict(plan=self.root / result['plan'], expected_plan_sha256=result['plan_sha256'],
            out=self.ctx.mi().scope / 'preview')
        args.update(overrides)
        return self.scout.preview_stage(self.ctx, **args)

    def test_plan_is_metadata_only_with_frozen_split_and_pending_calibration(self):
        mi = self.scout.mi
        with mock.patch.object(mi, 'verified_payload_stream', side_effect=AssertionError('no payload reads')):
            result, path, doc = self.plan()
        self.assertEqual(result['payload_opens'], 0)
        self.assertEqual(doc['scene'], '000003')
        self.assertEqual(doc['frozen_split']['opening'], list(range(10)))
        self.assertEqual(doc['frozen_split']['scoring_future'], list(range(10, 120)))
        self.assertEqual(doc['calibration']['status'], self.scout.CALIBRATION_STATUS)
        self.assertIsNone(doc['clock']['physical_fps'])
        self.assertNotIn(b'scene_gt', path.read_bytes())
        self.assertEqual(doc['pins']['kind'], 'synthetic_test_only')

    def test_mismatched_anchor_hash_refused_before_parse_or_payload(self):
        with mock.patch.object(self.scout.mi, 'strict_json', side_effect=AssertionError('must not parse')), \
             self.assertRaises(self.scout.mi.Blocked):
            self.plan(expected_index_sha256='0' * 64)

    def test_future_or_wrong_frame_refused_before_paths_are_touched(self):
        for frames in ([10], [0, 10], [-1], [True], [9, 9]):
            with self.subTest(frames=frames), \
                 mock.patch.object(self.scout.mi, 'read_pinned', side_effect=AssertionError('no reads')), \
                 self.assertRaises(self.scout.mi.UsageError):
                self.scout.preview_stage(self.ctx, plan='absent', expected_plan_sha256='a' * 64,
                    frames=frames, out='absent-output')

    def test_preview_streams_exactly_two_frames_with_independent_rgb_identities(self):
        from bench import workbench_geometry as wg
        from bench.vision import cv2
        result, _, _ = self.plan()
        with mock.patch.object(self.scout.mi, 'verified_payload_stream', wraps=self.scout.mi.verified_payload_stream) as stream:
            preview = self.preview(result)
        self.assertEqual(stream.call_count, 2)
        self.assertEqual([call.args[1]['member'] for call in stream.call_args_list],
            [f'val_primesense/000003/rgb/{i:06d}.png' for i in (0, 9)])
        receipt = json.loads((self.root / preview['receipt']).read_bytes())
        self.assertEqual(receipt['payload_opens'], 2)
        for index, row in zip((0, 9), receipt['frames']):
            expected = np.full((480, 640, 3), [index * 2, 37, 255 - index], np.uint8)
            expected[:40, :60] = [11, 177, 53]
            self.assertEqual(row['decoded_rgb_semantic_sha256'], wg.array_sha256(expected))
            self.assertEqual(row['frame_id'], f's{index:06d}')
        sheet_path = self.root / preview['receipt']; sheet_path = sheet_path.parent / self.scout.SHEET_NAME
        sheet = cv2.imdecode(np.frombuffer(sheet_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        self.assertLessEqual(sheet.shape[1], 1024)
        self.assertEqual(digest(sheet_path.read_bytes()), preview['contact_sheet_sha256'])
        for index, mapping in zip((0, 9), receipt['mapping']):
            self.assertEqual(mapping['native_size'], [640, 480])
            for point, rgb in (([20, 20], [11, 177, 53]), ([120, 120], [index * 2, 37, 255 - index])):
                # Independently apply the documented integer-centre mapping.
                x, y = np.rint(np.array(mapping['offset_xy']) +
                    (np.array(point) + .5) * mapping['scale'] - .5).astype(int)
                np.testing.assert_array_equal(sheet[y, x], rgb[::-1])
        for line in receipt['banner_lines']:
            width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)[0][0]
            self.assertLessEqual(width, sheet.shape[1] - 32)

    def test_plan_tampering_is_refused_even_when_caller_supplies_new_hash(self):
        result, path, doc = self.plan()
        for field, value in [('scene', '000002'), ('flags', {}), ('payload_opens', True),
                             ('calibration', {'status': 'verified'}), ('rgb_totals', {})]:
            with self.subTest(field=field):
                changed = copy.deepcopy(doc); changed[field] = value
                path.write_bytes(json_bytes(changed))
                with mock.patch.object(self.scout.mi, 'verified_payload_stream', side_effect=AssertionError('no payloads')), \
                     self.assertRaises(self.scout.mi.InputsError):
                    self.preview(result, expected_plan_sha256=digest(path.read_bytes()))

    def test_plan_payload_path_escape_is_refused_before_grant(self):
        result, path, doc = self.plan()
        doc['rgb'][0]['payload_path'] = '../escape.bin'
        path.write_bytes(json_bytes(doc))
        with mock.patch.object(self.scout.mi, 'verified_payload_stream', side_effect=AssertionError('no payloads')), \
             self.assertRaises(self.scout.mi.InputsError):
            self.preview(result, expected_plan_sha256=digest(path.read_bytes()))

    def test_excess_input_budget_refuses_before_payloads(self):
        with mock.patch.object(self.scout.mi, 'verified_payload_stream', side_effect=AssertionError('no payloads')), \
             self.assertRaises(self.scout.mi.Blocked):
            self.plan(existing_input_bytes=250_000_000)

    def test_retained_sibling_files_count_toward_scout_budget(self):
        scope = self.ctx.mi().scope; scope.mkdir(parents=True)
        (scope / 'prior-failed.bin').write_bytes(b'x' * self.scout.SCOUT_TOTAL_CAP)
        with mock.patch.object(self.scout.mi, 'verified_payload_stream', side_effect=AssertionError('no payloads')), \
             self.assertRaises(self.scout.mi.Blocked):
            self.plan()

    def test_invalid_existing_input_type_never_reaches_metadata_reads(self):
        for value in (-1, True, 1.5, '0'):
            with self.subTest(value=value), \
                 mock.patch.object(self.scout, '_read_anchors', side_effect=AssertionError('no reads')), \
                 self.assertRaises(self.scout.mi.UsageError):
                self.plan(existing_input_bytes=value)

    def test_actual_free_reserve_refusal_occurs_before_payload_streams(self):
        from types import SimpleNamespace
        with mock.patch.object(self.scout.mi.shutil, 'disk_usage', return_value=SimpleNamespace(free=1024)), \
             mock.patch.object(self.scout.mi, 'verified_payload_stream', side_effect=AssertionError('no payloads')), \
             self.assertRaises(self.scout.mi.Blocked):
            self.plan()

    def test_global_derived_budget_refusal_does_not_create_large_fixtures(self):
        mi = self.scout.mi
        with mock.patch.object(mi, '_tree_bytes', side_effect=lambda p: mi.DERIVED_LIMIT if p.name == 'derived' else 0), \
             mock.patch.object(mi, 'verified_payload_stream', side_effect=AssertionError('no payloads')), \
             self.assertRaises(mi.Blocked):
            self.plan()

    def test_changed_source_identity_refuses_before_plan_publication(self):
        initial = self.scout._sources()
        changed = dict(initial, manufactured_drift='a' * 64)
        with mock.patch.object(self.scout, '_sources', side_effect=[initial, changed]), \
             self.assertRaisesRegex(self.scout.mi.InputsError, 'changed'):
            self.plan()
        self.assertFalse((self.ctx.mi().scope / 'plan' / self.scout.PLAN_NAME).exists())

    def test_failed_preview_receipt_retains_sheet_and_explicit_failure_record(self):
        result, _, _ = self.plan()
        original = self.scout.mi.publish_bytes
        def fail_receipt(ledger, name, data, **kwargs):
            if name == self.scout.RECEIPT_NAME:
                raise OSError('manufactured preview receipt failure')
            return original(ledger, name, data, **kwargs)
        with mock.patch.object(self.scout.mi, 'publish_bytes', side_effect=fail_receipt), \
             self.assertRaisesRegex(OSError, 'manufactured preview receipt failure'):
            self.preview(result)
        folder = self.ctx.mi().scope / 'preview'
        self.assertTrue((folder / self.scout.SHEET_NAME).is_file())
        self.assertFalse((folder / self.scout.RECEIPT_NAME).exists())
        failure = json.loads((folder / 'failure.json').read_bytes())
        self.assertEqual(failure['status'], 'failed')
        self.assertFalse(failure['completed_receipt_published'])
        self.assertIn(self.scout.SHEET_NAME, failure['retained_files'])

    def test_wrong_scene_member_or_flag_in_plan_row_is_refused_before_stream(self):
        result, path, doc = self.plan()
        for field, value in [('member', 'val_primesense/000002/rgb/000000.png'),
                             ('flags', True), ('roles', ['evaluation']), ('crc32', -1)]:
            with self.subTest(field=field):
                changed = copy.deepcopy(doc); changed['rgb'][0][field] = value
                path.write_bytes(json_bytes(changed))
                with mock.patch.object(self.scout.mi, 'verified_payload_stream', side_effect=AssertionError('no payloads')), \
                     self.assertRaises(self.scout.mi.InputsError):
                    self.preview(result, expected_plan_sha256=digest(path.read_bytes()))

    @unittest.skipUnless(os.name == 'nt', 'Native Windows junction control')
    def test_output_junction_refused_without_writing_external_target(self):
        scope = self.ctx.mi().scope; scope.mkdir(parents=True)
        link = scope / 'alias'
        with tempfile.TemporaryDirectory(prefix='scout-outside-', dir=SCRATCH) as directory:
            target = Path(directory)
            proc = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)],
                capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            try:
                with self.assertRaises(self.scout.mi.InputsError):
                    self.plan(out=link / 'new-plan')
                self.assertEqual(list(target.iterdir()), [])
            finally:
                # Nonrecursive removal removes this owned junction, never its target.
                os.rmdir(link)

    def test_payload_cannot_be_read_as_a_plan_before_whole_stage_guard(self):
        mi = self.scout.mi
        row = self.catalog['documents']['resources_index'][1]['resources'][
            'val_primesense/000003/rgb/000000.png']
        payload = self.catalog['dataset'] / row['location']['path']
        opened = []
        original = mi.read_bounded
        def observe(path, *args, **kwargs):
            if Path(path) == payload:
                opened.append(mi._GUARD['active'])
            return original(path, *args, **kwargs)
        with mock.patch.object(mi, 'read_bounded', side_effect=observe), \
             self.assertRaises((mi.InputsError, mi.GuardError, mi.UsageError)):
            self.scout.preview_stage(self.ctx, plan=payload,
                expected_plan_sha256=row['compressedSHA256'], out=self.ctx.mi().scope / 'preview')
        self.assertEqual(opened, [], 'A payload must be refused as a plan before any byte read')

    def test_single_frame_preview_never_claims_that_both_frames_are_shown(self):
        rgb = np.zeros((480, 640, 3), np.uint8)
        try:
            sheet, layout = self.scout.build_contact_sheet([(0, rgb)])
        except self.scout.mi.InputsError:
            return  # Requiring exactly the prospectively authorized pair is valid.
        shown = ' '.join(layout['banner_lines'])
        self.assertNotIn('s000009', shown)

    def test_source_drift_during_plan_publication_is_a_retained_failure(self):
        initial = self.scout._sources()
        state = {'drift': False}
        publish = self.scout.mi.publish_bytes
        def changing_sources():
            return dict(initial, manufactured_drift='a' * 64) if state['drift'] else initial
        def publish_then_drift(ledger, name, data, **kwargs):
            result = publish(ledger, name, data, **kwargs)
            if name == self.scout.PLAN_NAME:
                state['drift'] = True
            return result
        with mock.patch.object(self.scout, '_sources', side_effect=changing_sources), \
             mock.patch.object(self.scout.mi, 'publish_bytes', side_effect=publish_then_drift), \
             self.assertRaisesRegex(self.scout.mi.InputsError, 'changed|closure|drift'):
            self.plan()
        failure = self.ctx.mi().scope / 'plan/failure.json'
        self.assertTrue(failure.is_file())
        self.assertFalse(json.loads(failure.read_bytes())['completed_receipt_published'])

    def test_scope_cap_includes_temporary_and_published_hardlink_names(self):
        mi = self.scout.mi
        for is_failure in (False, True):
            with self.subTest(failure=is_failure):
                scope = self.workspace / 'derived' / ('failure-scope' if is_failure else 'normal-scope')
                scope.mkdir(parents=True)
                (scope / 'retained.bin').write_bytes(b'r' * 400)
                mctx = mi.Context(self.root, self.workspace, scope)
                ledger = self.scout.ScoutLedger(mctx, scope, 'manufactured-publication')
                peak = [400]; actual_link = os.link
                def observed_link(*args, **kwargs):
                    result = actual_link(*args, **kwargs)
                    peak.append(mi.scope_usage(scope)['total'])
                    return result
                blocked = False
                with mock.patch.object(self.scout, 'SCOUT_TOTAL_CAP', 1000), \
                     mock.patch.object(self.scout, 'FAILURE_RESERVE', 0), \
                     mock.patch.object(mi.os, 'link', side_effect=observed_link):
                    try:
                        mi.publish_bytes(ledger, 'output.bin', b'p' * 400, metadata=False, failure=is_failure)
                    except mi.Blocked:
                        blocked = True
                self.assertTrue(blocked, f'Publication exceeded transient logical cap: peak {max(peak)} > 1000')
                self.assertLessEqual(max(peak), 1000)

    def test_source_drift_during_preview_receipt_invalidates_success_and_keeps_failure(self):
        result, _, _ = self.plan()
        initial = self.scout._sources(); state = {'drift': False}
        original = self.scout.mi.publish_bytes
        def changing_sources():
            return dict(initial, manufactured_drift='a' * 64) if state['drift'] else initial
        def publish_then_drift(ledger, name, data, **kwargs):
            published = original(ledger, name, data, **kwargs)
            if name == self.scout.RECEIPT_NAME:
                state['drift'] = True
            return published
        with mock.patch.object(self.scout, '_sources', side_effect=changing_sources), \
             mock.patch.object(self.scout.mi, 'publish_bytes', side_effect=publish_then_drift), \
             self.assertRaisesRegex(self.scout.mi.InputsError, 'changed|closure|drift'):
            self.preview(result)
        folder = self.ctx.mi().scope / 'preview'
        self.assertFalse((folder / self.scout.RECEIPT_NAME).exists())
        self.assertTrue((folder / (self.scout.INVALIDATED_PREFIX + self.scout.RECEIPT_NAME)).exists())
        failure = json.loads((folder / 'failure.json').read_bytes())
        self.assertFalse(failure['completed_receipt_published'])
        self.assertTrue((folder / self.scout.SHEET_NAME).is_file())

    def test_final_plan_hash_failure_invalidates_owned_completed_name(self):
        original = self.scout.mi.sha256_file
        def fail_final(path):
            if Path(path).name == self.scout.PLAN_NAME and Path(path).is_file():
                raise OSError('manufactured final plan hash read failure')
            return original(path)
        with mock.patch.object(self.scout.mi, 'sha256_file', side_effect=fail_final), \
             self.assertRaises(OSError):
            self.plan()
        folder = self.ctx.mi().scope / 'plan'
        self.assertFalse((folder / self.scout.PLAN_NAME).exists())
        self.assertTrue((folder / 'failure.json').exists())

    def test_final_receipt_hash_failure_invalidates_owned_completed_name(self):
        result, _, _ = self.plan()
        original = self.scout.mi.sha256_file
        def fail_final(path):
            if Path(path).name == self.scout.RECEIPT_NAME and Path(path).is_file():
                raise OSError('manufactured final receipt hash read failure')
            return original(path)
        with mock.patch.object(self.scout.mi, 'sha256_file', side_effect=fail_final), \
             self.assertRaises(OSError):
            self.preview(result)
        folder = self.ctx.mi().scope / 'preview'
        self.assertFalse((folder / self.scout.RECEIPT_NAME).exists())
        self.assertTrue((folder / 'failure.json').exists())

    def test_publication_collision_does_not_invalidate_a_foreign_completed_name(self):
        original_scope = self.ctx.mi().scope
        for phase in ('plan', 'preview'):
            with self.subTest(phase=phase):
                self.ctx = self.scout.ScoutContext(self.root, self.workspace,
                    original_scope / phase, self.ctx.synthetic)
                result = self.plan()[0] if phase == 'preview' else None
                name = self.scout.PLAN_NAME if phase == 'plan' else self.scout.RECEIPT_NAME
                publish = self.scout.mi.publish_bytes
                def collision(ledger, filename, data, **kwargs):
                    if filename == name:
                        (ledger.dir / filename).write_bytes(b'FOREIGN OUTPUT: preserve exact bytes')
                    return publish(ledger, filename, data, **kwargs)
                with mock.patch.object(self.scout.mi, 'publish_bytes', side_effect=collision), \
                     self.assertRaises(self.scout.mi.InputsError):
                    self.plan() if phase == 'plan' else self.preview(result)
                folder = self.ctx.mi().scope / phase
                self.assertEqual((folder / name).read_bytes(), b'FOREIGN OUTPUT: preserve exact bytes')
                self.assertFalse((folder / (self.scout.INVALIDATED_PREFIX + name)).exists())

    def test_failure_publication_respects_current_global_derived_cap(self):
        import io
        mi = self.scout.mi; folder = self.ctx.mi().scope / 'failure'
        folder.mkdir(parents=True)
        ledger = self.scout.ScoutLedger(self.ctx.mi(), folder, 'manufactured-failure')
        err = io.StringIO()
        with mock.patch.object(mi, '_tree_bytes', side_effect=lambda p: mi.DERIVED_LIMIT if p.name == 'derived' else 0), \
             mock.patch.object(self.scout.sys, 'stderr', err):
            self.scout._scout_failure(ledger, 'manufactured-failure', OSError('manufactured error'))
        self.assertFalse((folder / 'failure.json').exists())
        self.assertIn('NOT published', err.getvalue())


