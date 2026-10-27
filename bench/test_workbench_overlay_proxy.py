"""Manufactured render-only proxy checks; no models, inference or GPU."""
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

from bench import workbench_contract as wc, workbench_neural as wn


def write_glb(path, vertices, faces):
    """Tiny independent embedded triangle fixture in metric object coordinates."""
    v = np.asarray(vertices, dtype='<f4').tobytes()
    f = np.asarray(faces, dtype='<u4').tobytes()
    doc = {'asset': {'version': '2.0'}, 'buffers': [{'byteLength': len(v) + len(f)}],
        'bufferViews': [{'buffer': 0, 'byteOffset': 0, 'byteLength': len(v)},
                        {'buffer': 0, 'byteOffset': len(v), 'byteLength': len(f)}],
        'accessors': [{'bufferView': 0, 'componentType': 5126, 'count': len(vertices), 'type': 'VEC3'},
                      {'bufferView': 1, 'componentType': 5125, 'count': len(faces) * 3, 'type': 'SCALAR'}],
        'meshes': [{'primitives': [{'attributes': {'POSITION': 0}, 'indices': 1}]}],
        'nodes': [{'mesh': 0}], 'scenes': [{'nodes': [0]}], 'scene': 0}
    js = json.dumps(doc).encode()
    js += b' ' * (-len(js) % 4)
    binary = v + f
    binary += b'\0' * (-len(binary) % 4)
    raw = (struct.pack('<4sII', b'glTF', 2, 28 + len(js) + len(binary))
           + struct.pack('<I4s', len(js), b'JSON') + js
           + struct.pack('<I4s', len(binary), b'BIN\0') + binary)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


class OverlayProxyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Exact dyadic metre coordinates: one triangle subdivided into four.
        self.v = [[-.5, -.5, 0], [.5, -.5, 0], [0, .5, 0],
                  [0, -.5, 0], [.25, 0, 0], [-.25, 0, 0]]
        self.f = [[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]]
        canonical = self.root / 'canonical.glb'
        sha = write_glb(canonical, self.v, self.f)
        proxy = self.root / 'proxy.glb'
        proxy_sha = write_glb(proxy, self.v[:3], [[0, 1, 2]])
        pose = np.eye(4)
        pose[2, 3] = 2
        self.bundle = wn.MetricBundle('synthetic', canonical, sha, 4, 'integer_centers',
            pose, 'controlled_init:synthetic', 'seed', 'dataset_metric_cad')
        identity = {'obj_id': 4, 'units': 'metres', 'coordinate_frame': 'canonical_object_frame',
                    'canonical_glb_sha256': sha, 'overlay_glb_sha256': proxy_sha}
        vertex_bary = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, .5, .5], [.5, 0, .5]]
        self.receipt = {'schema': 'workbench_overlay_correspondence_v1', **identity,
            'max_deviation_m': 2., 'canonical_to_overlay': {'target_faces': [0] * 4,
                'barycentrics': [[vertex_bary[i] for i in face] for face in self.f]},
            'overlay_to_canonical': {'target_faces': [0], 'barycentrics': [np.eye(3).tolist()]}}
        self.manifest = {'schema': 'workbench_overlay_proxy_v1', **identity,
            'overlay_glb_path': 'proxy.glb', 'receipt_path': 'receipt.json', 'receipt_sha256': ''}
        self.selection = wc.Selection('seed', ((32., 32.),), (), True, 'synthetic reviewer')
        self.kw = dict(clip_K=np.array([[64., 0, 32], [0, 64., 32], [0, 0, 1]]),
                       width=64, height=64, selection_mask=np.ones((64, 64), bool))
        self.pin()

    def pin(self):
        raw = wc.canonical_json(self.receipt)
        (self.root / 'receipt.json').write_bytes(raw)
        self.manifest['receipt_sha256'] = hashlib.sha256(raw).hexdigest()
        raw = wc.canonical_json(self.manifest)
        (self.root / 'manifest.json').write_bytes(raw)
        self.manifest_sha = hashlib.sha256(raw).hexdigest()

    def options(self):
        return dict(overlay_proxy_manifest=self.root / 'manifest.json',
                    overlay_proxy_manifest_sha256=self.manifest_sha, reviewed_proxy_deviation_m=2.)

    def artwork(self, **overrides):
        return wn.NeuralArtwork(self.bundle, self.selection, 'directional', **(self.kw | overrides))

    def verify(self, **overrides):
        opts = self.options() | overrides
        return wn._overlay_proxy_mesh(self.bundle, opts['overlay_proxy_manifest'],
            opts['overlay_proxy_manifest_sha256'], opts['reviewed_proxy_deviation_m'], 200_000)

    def test_no_proxy_preserves_original_record_hash_and_mesh(self):
        before = self.bundle.glb_path.read_bytes()
        # Compare to original implementation from Git, independently loaded into
        # a private module namespace. No proxy kwargs/default record additions.
        import subprocess
        parent = subprocess.check_output(['git', 'show', 'HEAD:bench/workbench_neural.py']).decode()
        namespace = {'__name__': 'bench._overlay_parent', '__package__': 'bench', '__file__': wn.__file__}
        # Only the class is needed; use current unchanged helper bindings.
        namespace.update(vars(wn))
        exec(parent[parent.index('class NeuralArtwork:'):], namespace)
        original = namespace['NeuralArtwork'](self.bundle, self.selection, 'directional', **self.kw)
        actual = self.artwork()
        self.assertEqual(actual.record, original.record)
        self.assertEqual(actual.sha256, original.sha256)
        np.testing.assert_array_equal(actual.vertices, original.vertices)
        np.testing.assert_array_equal(actual.faces, original.faces)
        self.assertEqual(self.bundle.glb_path.read_bytes(), before)

    def test_reduced_proxy_accepts_known_geometry_and_records_both_identities(self):
        before = copy.deepcopy(self.bundle)
        actual = self.artwork(**self.options())
        self.assertEqual(len(actual.faces), 1)
        binding = actual.record['overlay_proxy']
        # Both surfaces are the identical planar triangle. Forward corner
        # interpolation is exact; the reverse correspondence is conservative.
        self.assertEqual(binding['canonical_to_overlay_bound_m'], 0)
        self.assertLess(binding['overlay_to_canonical_bound_m'], 2)
        self.assertEqual(binding['anchor_comparison']['displacement_m'], 0)
        self.assertEqual(actual.record['glb_sha256'], before.glb_sha256)
        self.assertEqual(binding['overlay_glb_sha256'], self.manifest['overlay_glb_sha256'])
        self.assertFalse(binding['quality_claim'])
        self.assertEqual(self.bundle.glb_path, before.glb_path)
        np.testing.assert_array_equal(self.bundle.seed_pose_m, before.seed_pose_m)

    def test_proxy_extents_do_not_change_canonical_artwork_physical_scale(self):
        sha = write_glb(self.root / 'proxy.glb', (np.array(self.v[:3]) * .5).tolist(), [[0, 1, 2]])
        self.manifest['overlay_glb_sha256'] = self.receipt['overlay_glb_sha256'] = sha
        self.pin()
        actual = self.artwork(**self.options())
        # Canonical longest span is exactly one metre; default fraction is 1/4.
        self.assertEqual(actual.marker['size_object_units'], .25)

    def test_canonical_degenerate_faces_filtered_only_for_explicit_overlay_correspondence(self):
        canonical = self.bundle.glb_path
        # Original flattened source indices: repeated=0, coordinate-zero-area=2;
        # the four effective faces remain in source order [1,3,4,5].
        source_faces = [[0, 0, 1], self.f[0], [0, 3, 1], *self.f[1:]]
        sha = write_glb(canonical, self.v, source_faces)
        before = canonical.read_bytes()
        self.bundle = replace(self.bundle, glb_sha256=sha)
        self.manifest['canonical_glb_sha256'] = self.receipt['canonical_glb_sha256'] = sha
        self.pin()  # receipt still covers exactly the four effective faces
        _, public_faces = wn.glb_mesh(canonical)
        np.testing.assert_array_equal(public_faces, source_faces[1:])
        default = self.artwork()
        np.testing.assert_array_equal(default.faces, source_faces[1:])
        self.assertNotIn('overlay_proxy', default.record)

        _, _, binding, geometry = self.verify()
        np.testing.assert_array_equal(geometry[1], self.f)
        self.assertEqual(binding['canonical_raw_faces'], 6)
        self.assertEqual(binding['canonical_effective_faces'], 4)
        self.assertEqual(binding['canonical_dropped_repeated_index_faces'], 1)
        self.assertEqual(binding['canonical_dropped_zero_area_faces'], 1)
        digest = hashlib.sha256(np.array([1, 3, 4, 5], dtype='<i8').tobytes()).hexdigest()
        self.assertEqual(binding['canonical_source_index_digest']['sha256'], digest)
        actual = self.artwork(**self.options())
        self.assertEqual(actual.record['overlay_proxy']['anchor_comparison']['displacement_m'], 0)
        self.assertEqual(canonical.read_bytes(), before)
        # A receipt covering unfiltered faces must fail closed.
        self.receipt['canonical_to_overlay']['target_faces'].insert(0, 0)
        self.pin()
        with self.assertRaisesRegex(ValueError, 'every source triangle'):
            self.verify()

    def test_proxy_coordinate_zero_area_faces_are_still_rejected(self):
        sha = write_glb(self.root / 'proxy.glb', self.v, [[0, 1, 2], [0, 3, 1]])
        self.manifest['overlay_glb_sha256'] = self.receipt['overlay_glb_sha256'] = sha
        self.pin()
        with self.assertRaisesRegex(ValueError, 'degenerate proxy'):
            self.verify()

    def test_authenticated_proxy_cannot_hide_parser_discarded_repeated_index_face(self):
        # Repair 3 reproduction before patch: this exact authenticated raw mesh
        # was accepted as one triangle (manifest 517772d7...4ead46,
        # proxy f5ecffaf...538562). A discarded proxy face must be fatal.
        sha = write_glb(self.root / 'proxy.glb', self.v[:3], [[0, 1, 2], [0, 0, 1]])
        self.manifest['overlay_glb_sha256'] = self.receipt['overlay_glb_sha256'] = sha
        self.pin()
        self.assertEqual(self.manifest_sha, '517772d7ac0012753df0d2d8aef59908f1413d6581ec43393e7ec191cb4ead46')
        self.assertEqual(sha, 'f5ecffaffe73ca28794280a9a8aa0900d0801da82aaa9a93c1bd214951538562')
        # The inherited public/default reader still has its original filtering.
        _, faces = wn.glb_mesh(self.root / 'proxy.glb')
        np.testing.assert_array_equal(faces, [[0, 1, 2]])
        with self.assertRaisesRegex(ValueError, 'parser-discarded repeated-index'):
            self.verify()
        with self.assertRaisesRegex(ValueError, 'parser-discarded repeated-index'):
            self.artwork(**self.options())

        # Removing the invalid face produces a valid authenticated control;
        # receipt mappings, caller limit and canonical geometry are unchanged.
        sha = write_glb(self.root / 'proxy.glb', self.v[:3], [[0, 1, 2]])
        self.manifest['overlay_glb_sha256'] = self.receipt['overlay_glb_sha256'] = sha
        self.pin()
        actual = self.artwork(**self.options())
        self.assertEqual(actual.record['overlay_proxy']['overlay_faces'], 1)

    def test_opt_in_requires_all_explicit_inputs(self):
        for change in ({'overlay_proxy_manifest_sha256': None}, {'reviewed_proxy_deviation_m': None},
                       {'reviewed_proxy_deviation_m': float('inf')}, {'reviewed_proxy_deviation_m': True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.artwork(**(self.options() | change))
        with self.assertRaises(ValueError):
            self.artwork(overlay_proxy_manifest_sha256=self.manifest_sha)
        with self.assertRaises(ValueError):
            self.artwork(**self.options(), selection_mask=None)

    def test_identity_schema_units_frame_and_unknown_keys_fail_closed(self):
        for key, value in (('schema', 'other'), ('obj_id', 5), ('obj_id', True), ('units', 'millimetres'),
                           ('coordinate_frame', 'normalized'), ('canonical_glb_sha256', 'a' * 64), ('extra', 0)):
            with self.subTest(key=key):
                saved = self.manifest.copy()
                self.manifest[key] = value
                self.pin()
                with self.assertRaises(ValueError):
                    self.verify()
                self.manifest = saved
        self.pin()

    def test_exact_bytes_authenticate_every_file(self):
        for name in ('manifest.json', 'receipt.json', 'proxy.glb', 'canonical.glb'):
            path = self.root / name
            original = path.read_bytes()
            with self.subTest(name=name):
                path.write_bytes(original + b' ')
                with self.assertRaises(ValueError):
                    self.verify()
                path.write_bytes(original)

    def test_transient_path_substitution_cannot_change_authenticated_parsed_geometry(self):
        canonical = self.bundle.glb_path
        proxy = self.root / 'proxy.glb'
        original_canonical, original_proxy = canonical.read_bytes(), proxy.read_bytes()
        forged = self.root / 'forged.glb'
        write_glb(forged, (np.array(self.v) * .5).tolist(), self.f)
        forged_raw = forged.read_bytes()
        parse = wn._glb_mesh_bytes
        observed = []

        def substitute_during_parse(raw, **kwargs):
            # An adversarial writer swaps BOTH paths after authentication, then
            # restores them before any post-parse path hashes could observe it.
            canonical.write_bytes(forged_raw)
            proxy.write_bytes(forged_raw)
            try:
                observed.append(hashlib.sha256(raw).hexdigest())
                return parse(raw, **kwargs)
            finally:
                canonical.write_bytes(original_canonical)
                proxy.write_bytes(original_proxy)

        with mock.patch.object(wn, '_glb_mesh_bytes', side_effect=substitute_during_parse), \
                mock.patch.object(wn, 'glb_mesh', side_effect=AssertionError('proxy must never reopen GLBs')):
            v, f, binding, canonical_geometry = self.verify()
        # Parsed geometry comes from exactly the SHA-authenticated byte buffers,
        # even while the files contain different, otherwise valid geometry.
        self.assertEqual(observed, [self.bundle.glb_sha256, self.manifest['overlay_glb_sha256']])
        np.testing.assert_array_equal(canonical_geometry[0], np.asarray(self.v))
        np.testing.assert_array_equal(canonical_geometry[1], np.asarray(self.f))
        np.testing.assert_array_equal(v, np.asarray(self.v[:3]))
        np.testing.assert_array_equal(f, [[0, 1, 2]])
        self.assertEqual(binding['canonical_glb_sha256'], hashlib.sha256(original_canonical).hexdigest())

    def test_substituted_bytes_before_authentication_are_rejected(self):
        canonical = self.bundle.glb_path
        forged = self.root / 'forged.glb'
        write_glb(forged, (np.array(self.v) * .5).tolist(), self.f)
        forged_raw = forged.read_bytes()
        read = Path.read_bytes

        def substituted(path):
            return forged_raw if path == canonical else read(path)

        with mock.patch.object(Path, 'read_bytes', substituted):
            with self.assertRaisesRegex(ValueError, 'canonical GLB bytes differ'):
                self.verify()

    def test_missing_incomplete_invalid_or_flipped_mapping_fails_closed(self):
        original = copy.deepcopy(self.receipt)
        mutations = [lambda d: d.pop('overlay_to_canonical'),
            lambda d: d['canonical_to_overlay']['target_faces'].pop(),
            lambda d: d['canonical_to_overlay']['target_faces'].__setitem__(0, 5),
            lambda d: d['canonical_to_overlay']['target_faces'].__setitem__(0, True),
            lambda d: d['canonical_to_overlay']['barycentrics'][0][0].__setitem__(0, -.1),
            lambda d: d['canonical_to_overlay']['barycentrics'][0][0].__setitem__(0, True),
            lambda d: d['canonical_to_overlay']['barycentrics'][0][0].__setitem__(0, .9),
            lambda d: d.__setitem__('max_deviation_m', .01),
            lambda d: d.__setitem__('units', 'millimetres')]
        for mutation in mutations:
            self.receipt = copy.deepcopy(original)
            mutation(self.receipt)
            self.pin()
            with self.subTest(receipt=self.receipt), self.assertRaises(ValueError):
                self.verify()

    def test_reviewed_limit_and_existing_face_cap_cannot_be_bypassed(self):
        with self.assertRaises(ValueError):
            self.verify(reviewed_proxy_deviation_m=.5)
        with self.assertRaises(ValueError):
            self.artwork(**self.options(), max_faces=200_001)
        sha = write_glb(self.root / 'proxy.glb', self.v, self.f[:2])
        self.manifest['overlay_glb_sha256'] = self.receipt['overlay_glb_sha256'] = sha
        self.pin()
        with self.assertRaisesRegex(ValueError, 'too large'):
            self.artwork(**self.options(), max_faces=1)

    def test_nonmanifold_or_inconsistently_oriented_proxy_refused(self):
        # Three triangles sharing the same directed edge are not a manifold.
        sha = write_glb(self.root / 'proxy.glb', self.v, [[0, 1, 2], [0, 1, 4], [0, 1, 5]])
        self.manifest['overlay_glb_sha256'] = self.receipt['overlay_glb_sha256'] = sha
        self.pin()
        with self.assertRaisesRegex(ValueError, 'edge-manifold'):
            self.verify()

    def test_closed_canonical_cannot_gain_proxy_boundary(self):
        canonical = self.bundle.glb_path
        sha = write_glb(canonical, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
        self.bundle = replace(self.bundle, glb_sha256=sha)
        self.manifest['canonical_glb_sha256'] = self.receipt['canonical_glb_sha256'] = sha
        self.pin()
        with self.assertRaisesRegex(ValueError, 'boundary'):
            self.verify()

    def test_anchor_target_must_agree_with_actual_seed_ray_hit(self):
        sha = write_glb(self.root / 'proxy.glb', self.v, self.f)
        self.manifest['overlay_glb_sha256'] = self.receipt['overlay_glb_sha256'] = sha
        self.receipt['canonical_to_overlay'] = {'target_faces': [0] * 4,
            'barycentrics': [np.eye(3).tolist()] * 4}
        self.receipt['overlay_to_canonical'] = {'target_faces': list(range(4)),
            'barycentrics': [np.eye(3).tolist()] * 4}
        self.pin()
        # Conservative distance/orientation checks pass, but mapping face zero
        # is inconsistent with the actual center-ray hit on another face.
        self.verify()
        with self.assertRaisesRegex(ValueError, 'seed anchor'):
            self.artwork(**self.options())

    def test_flipped_proxy_refused_even_with_large_deviation_limit(self):
        sha = write_glb(self.root / 'proxy.glb', self.v[:3], [[0, 2, 1]])
        self.manifest['overlay_glb_sha256'] = self.receipt['overlay_glb_sha256'] = sha
        self.pin()
        with self.assertRaisesRegex(ValueError, 'flipped'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
