"""Fail-closed presentation tests for the optional R8 CPU mesh panel."""
import copy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


def _import_player_without_optional_packages():
    """Load presentation helpers under -S without importing vision libraries."""
    cached = sys.modules.get('bench.quality_player')
    if cached is not None:
        return cached
    names = (
        'numpy', 'cv2', 'bench.vision', 'bench.quality_contract',
        'bench.quality_evaluate', 'bench.quality_player',
    )
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in names}
    numpy_stub = types.ModuleType('numpy')
    numpy_stub.ndarray = object
    cv2_stub = types.ModuleType('cv2')
    sys.modules['numpy'] = numpy_stub
    sys.modules['cv2'] = cv2_stub
    for name in names[2:]:
        sys.modules.pop(name, None)
    try:
        return importlib.import_module('bench.quality_player')
    finally:
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


player = _import_player_without_optional_packages()


class R8MeshParityPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        summary_raw = player.BOTTLE_R8_MESH_PARITY_SUMMARY.read_bytes()
        svg_raw = player.BOTTLE_R8_MESH_PARITY_SVG.read_bytes()
        if hashlib.sha256(summary_raw).hexdigest().upper() != player.BOTTLE_R8_MESH_PARITY_SUMMARY_SHA256:
            raise AssertionError('pinned R8 mesh summary hash changed')
        if hashlib.sha256(svg_raw).hexdigest().upper() != player.BOTTLE_R8_MESH_PARITY_SVG_SHA256:
            raise AssertionError('pinned R8 mesh SVG hash changed')
        cls.summary = json.loads(summary_raw.decode('utf-8'))
        cls.svg_raw = svg_raw

    def test_pinned_loader_and_panel_show_only_bounded_offline_geometry_evidence(self):
        evidence = player.load_r8_mesh_parity_evidence()
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence['expected_samples'], 96)
        self.assertEqual(evidence['measured_samples'], 96)
        self.assertEqual(evidence['vertex_count'], 5003)
        self.assertEqual(evidence['triangle_count'], 8316)
        self.assertAlmostEqual(evidence['elapsed_seconds'], 99.078, places=3)
        self.assertAlmostEqual(evidence['p95_min_mm'], 0.997, places=3)
        self.assertAlmostEqual(evidence['p95_max_mm'], 2.394, places=3)
        self.assertAlmostEqual(evidence['max_residual_mm'], 4.31, places=2)
        self.assertLessEqual(evidence['max_ccw_stat_difference_m'], 5e-16)

        panel = player.render_r8_mesh_parity_panel()
        self.assertIn('id="r8MeshParityDiagnostic"', panel)
        self.assertIn('96/96 fixed SOURCE rays', panel)
        self.assertIn('5e-16 m', panel)
        self.assertIn('0.997 to 2.394 mm', panel)
        self.assertIn('4.31 mm', panel)
        self.assertIn('99.078-second total', panel)
        self.assertIn('synthetic GPU-rendered model camera-z depth', panel)
        self.assertIn('not camera-sensor depth', panel)
        self.assertIn('not live FPS', panel)
        self.assertIn('hardware-GPU depth equivalence', panel)
        self.assertIn('0/12 rows possible', panel)
        self.assertIn('all 28 full-screen cases unmeasured after timeout', panel)
        self.assertIn('<svg ', panel)
        self.assertIn('min-height:44px', panel)
        self.assertIn('overflow-x:auto', panel)
        self.assertIn('min-width:720px', panel)
        self.assertIn('max-width:100%;overflow:hidden', panel)
        self.assertNotIn('href=', panel)
        self.assertNotIn('src=', panel)
        self.assertNotIn('file://', panel)
        self.assertNotIn(str(player.CACHE), panel)
        self.assertNotIn('/diagnostics/', panel)

    def test_summary_semantics_and_svg_structure_are_required(self):
        self.assertIsNotNone(player._r8_mesh_parity_evidence_from_records(
            copy.deepcopy(self.summary), self.svg_raw))

        invalid_cases = []
        changed_scope = copy.deepcopy(self.summary)
        changed_scope['scope'] = 'camera sensor depth and tracking accuracy'
        invalid_cases.append(changed_scope)
        missing_samples = copy.deepcopy(self.summary)
        missing_samples['measured_samples'] = 95
        invalid_cases.append(missing_samples)
        reordered = copy.deepcopy(self.summary)
        reordered['views'].reverse()
        invalid_cases.append(reordered)
        wrong_denominator = copy.deepcopy(self.summary)
        wrong_denominator['views'][0]['denominator'] = 15
        invalid_cases.append(wrong_denominator)
        claim_accuracy = copy.deepcopy(self.summary)
        claim_accuracy['tracking_accuracy_measured'] = True
        invalid_cases.append(claim_accuracy)
        changed_mesh = copy.deepcopy(self.summary)
        changed_mesh['triangle_count'] = 8315
        invalid_cases.append(changed_mesh)
        for record in invalid_cases:
            with self.subTest(record=record.get('scope', record.get('measured_samples'))):
                self.assertIsNone(player._r8_mesh_parity_evidence_from_records(record, self.svg_raw))

        self.assertIsNone(player._r8_mesh_parity_evidence_from_records(
            copy.deepcopy(self.summary), b'<svg xmlns="http://www.w3.org/2000/svg"><script/></svg>'))
        external_reference = self.svg_raw.replace(
            b'<rect width="100%" height="100%"',
            b'<image href="https://example.invalid/" width="1" height="1"/><rect width="100%" height="100%"',
            1,
        )
        self.assertIsNone(player._r8_mesh_parity_evidence_from_records(
            copy.deepcopy(self.summary), external_reference))
        external_stylesheet = self.svg_raw.replace(
            b'<style>', b'<style>@import url("https://example.invalid/style.css");', 1)
        self.assertIsNone(player._r8_mesh_parity_evidence_from_records(
            copy.deepcopy(self.summary), external_stylesheet))

    def test_pinned_loader_hides_missing_mutated_oversized_and_semantically_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / 'missing.json'
            with mock.patch.object(player, 'BOTTLE_R8_MESH_PARITY_SUMMARY', missing):
                self.assertIsNone(player.load_r8_mesh_parity_evidence())

            bad_summary = root / 'summary.json'
            original_summary = player.BOTTLE_R8_MESH_PARITY_SUMMARY.read_bytes()
            bad_summary.write_bytes(original_summary + b' ')
            with mock.patch.object(player, 'BOTTLE_R8_MESH_PARITY_SUMMARY', bad_summary):
                self.assertIsNone(player.load_r8_mesh_parity_evidence())

            bad_svg = root / 'mesh-parity.svg'
            original_svg = player.BOTTLE_R8_MESH_PARITY_SVG.read_bytes()
            bad_svg.write_bytes(original_svg + b' ')
            with mock.patch.object(player, 'BOTTLE_R8_MESH_PARITY_SVG', bad_svg):
                self.assertIsNone(player.load_r8_mesh_parity_evidence())

            oversized = root / 'oversized.json'
            oversized.write_bytes(b'x' * (player.BOTTLE_R8_MESH_PARITY_SUMMARY_MAX_BYTES + 1))
            with mock.patch.object(player, 'BOTTLE_R8_MESH_PARITY_SUMMARY', oversized):
                self.assertIsNone(player.load_r8_mesh_parity_evidence())

            invalid_summary = copy.deepcopy(self.summary)
            invalid_summary['views'].pop()
            invalid_raw = json.dumps(invalid_summary, separators=(',', ':')).encode('utf-8')
            invalid_path = root / 'invalid-semantics.json'
            invalid_path.write_bytes(invalid_raw)
            with mock.patch.multiple(
                player,
                BOTTLE_R8_MESH_PARITY_SUMMARY=invalid_path,
                BOTTLE_R8_MESH_PARITY_SUMMARY_BYTES=len(invalid_raw),
                BOTTLE_R8_MESH_PARITY_SUMMARY_SHA256=hashlib.sha256(invalid_raw).hexdigest().upper(),
            ):
                self.assertIsNone(player.load_r8_mesh_parity_evidence())

    def test_insertion_replaces_interim_card_before_evidence_status_and_removes_stale_card(self):
        summary = copy.deepcopy(self.summary)
        evidence = player._r8_mesh_parity_evidence_from_records(summary, self.svg_raw)
        self.assertIsNotNone(evidence)
        existing = (
            '<html><body>' + player.BOTTLE_R8_MESH_PARITY_START
            + '<details id="r8MeshParityDiagnostic">interim card</details>'
            + player.BOTTLE_R8_MESH_PARITY_END
            + '<h2>Evidence status</h2><p>Original capacity history.</p></body></html>'
        )
        with mock.patch.object(player, 'load_r8_mesh_parity_evidence', return_value=evidence):
            updated = player.insert_r8_mesh_parity_panel(existing)
            self.assertLess(updated.index(player.BOTTLE_R8_MESH_PARITY_START),
                            updated.index('<h2>Evidence status</h2>'))
            self.assertIn('96/96 fixed SOURCE rays', updated)
            self.assertNotIn('interim card', updated)
            self.assertEqual(player.insert_r8_mesh_parity_panel(updated), updated)

        with mock.patch.object(player, 'load_r8_mesh_parity_evidence', return_value=None):
            cleaned = player.insert_r8_mesh_parity_panel(updated)
        self.assertNotIn(player.BOTTLE_R8_MESH_PARITY_START, cleaned)
        self.assertNotIn('r8MeshParityDiagnostic', cleaned)
        self.assertIn('Original capacity history.', cleaned)

    def test_build_integration_is_opt_in(self):
        self.assertIs(
            inspect.signature(player.build).parameters['include_r8_mesh_parity'].default,
            False,
        )


if __name__ == '__main__':
    unittest.main()
