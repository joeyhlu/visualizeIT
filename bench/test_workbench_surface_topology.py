"""Independent manufactured geometry/prefix controls; no files or neural execution."""
from dataclasses import replace
import copy
import json
import unittest
from unittest import mock

import numpy as np

from bench import workbench_surface_topology as st
from bench import workbench_contract as wc
from bench.test_workbench_contract import clip_document, run_document


def quad(split=False):
    vertices = np.array([[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0]], np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    if split:
        vertices = vertices[faces].reshape(-1, 3)
        faces = np.arange(6, dtype=np.int64).reshape(2, 3)
    return vertices, faces


def marker(size=1.0):
    return {'face_id': 0, 'barycentric': [.5, 0, .5], 'tangent': [1., 0, 0],
        'bitangent': [0., 1, 0], 'normal': [0., 0, 1],
        'size_object_units': size, 'artwork_sha256': 'a' * 64}


class SurfaceTopologyTests(unittest.TestCase):
    def setUp(self):
        self.limits = st.portable_20k_limits()
        self.policy = st.default_policy()

    def build(self, vertices, faces, **changes):
        args = dict(units='metres', policy=self.policy, limits=self.limits)
        args.update(changes)
        return st.build_topology(vertices, faces, **args)

    def diagnose(self, vertices, faces, top, **changes):
        args = dict(units=top.units, min_support=3, limits=self.limits)
        args.update(changes)
        return st.diagnose_patch(vertices, faces, marker(), np.full(len(faces), 3, np.int64), top, **args).to_dict()

    def test_split_quad_preserves_original_bytes_and_separates_three_relations(self):
        vertices, faces = quad(True); before = vertices.tobytes(), faces.tobytes()
        top = self.build(vertices, faces)
        self.assertEqual(top.neighbors('index', 0), ())
        self.assertEqual(top.neighbors('geometric_candidate', 0), (1,))
        self.assertEqual(top.neighbors('approved', 0), ())
        self.assertIn('UNVERIFIED', top.attribute_status)
        self.assertEqual((vertices.tobytes(), faces.tobytes()), before)

    def test_diagnosis_enforces_tighter_mesh_limits_before_any_hashing(self):
        vertices, faces = quad(); top = self.build(vertices, faces)
        for limits in (replace(self.limits,max_faces=1),replace(self.limits,max_half_edges=3)):
            with self.subTest(limits=limits.name), mock.patch.object(st,'array_digest',side_effect=AssertionError('hash reached')):
                with self.assertRaises(ValueError):self.diagnose(vertices,faces,top,limits=limits)
        huge=np.broadcast_to(np.zeros((1,3)),(60001,3))
        with mock.patch.object(st,'array_digest',side_effect=AssertionError('oversized array hash reached')):
            with self.assertRaises(ValueError):self.diagnose(huge,faces,top)

    def test_validity_tolerances_cannot_admit_singular_rotations_or_far_outside_anchors(self):
        for field in ('orthonormal_tol','barycentric_tol'):
            with self.subTest(field=field),self.assertRaises(ValueError):replace(self.limits,**{field:2.0})
        vertices,faces=quad();top=self.build(vertices,faces)
        bad=marker();bad['barycentric']=[2.,0.,-1.]
        with self.assertRaises(ValueError):
            st.diagnose_patch(vertices,faces,bad,np.full(2,3),top,units='metres',limits=self.limits)

    def test_explicit_continuous_attributes_approve_equivalent_split_quad(self):
        values = []
        for split in (False, True):
            vertices, faces = quad(split)
            uv = vertices[faces, :2] + .5
            top = self.build(vertices, faces, face_region_ids=np.zeros(2, np.int64), corner_uv=uv)
            self.assertEqual(top.neighbors('approved', 0), (1,))
            result = self.diagnose(vertices, faces, top)
            coverage = result['traversals']['approved']['coverage']
            self.assertAlmostEqual(coverage['patch_surface_area'], 1)
            self.assertAlmostEqual(coverage['surface_area_inside_nominal_footprint'], 1)
            self.assertAlmostEqual(coverage['union_coverage_fraction'], 1)
            self.assertAlmostEqual(coverage['summed_clipped_chart_area_ratio'], 1)
            values.append(coverage)
        self.assertEqual(values[0], values[1])

    def test_uv_seam_and_region_barrier_remain_barriers(self):
        vertices, faces = quad(True); uv = vertices[faces, :2] + .5
        for regions, coords in [(np.array([0, 1]), uv), (np.zeros(2, np.int64), uv + np.array([[[0, 0]], [[1, 0]]]))]:
            top = self.build(vertices, faces, face_region_ids=regions, corner_uv=coords)
            self.assertEqual(top.neighbors('approved', 0), ())
            self.assertEqual(top.neighbors('geometric_candidate', 0), (1,))

    def test_missing_attributes_cannot_be_promoted_by_policy_switches(self):
        vertices, faces = quad(True)
        uv = vertices[faces, :2] + .5
        for regions, coords, a, b in [(None, uv, False, True),
                (np.zeros(2,np.int64), None, True, False), (None,None,False,False)]:
            self.policy = replace(self.policy, require_region_labels=a, require_uv_continuity=b)
            with self.subTest(region_required=a, uv_required=b):
                top = self.build(vertices, faces, face_region_ids=regions, corner_uv=coords)
                self.assertEqual(top.neighbors('approved', 0), (), 'unknown boundaries cannot establish proven continuity')

    def test_point_contact_and_nonzero_offset_do_not_connect(self):
        vertices, faces = quad(True)
        vertices[3:] += [0, 0, 1e-15]
        self.assertEqual(self.build(vertices, faces).neighbors('geometric_candidate', 0), ())
        vertices = np.array([[0,0,0], [1,0,0], [0,1,0], [-1,0,0], [0,-1,0]], float)
        faces = np.array([[0,1,2], [0,3,4]], np.int64)
        self.assertEqual(self.build(vertices, faces).neighbors('geometric_candidate', 0), ())

    def test_signed_zero_matches_without_changing_original_digest_bytes(self):
        vertices, faces = quad(True)
        vertices[0, 2] = -0.0
        top = self.build(vertices, faces)
        self.assertEqual(top.neighbors('geometric_candidate', 0), (1,))
        self.assertTrue(np.signbit(vertices[0, 2]))

    def test_duplicate_layers_and_reversed_neighbors_are_blocked(self):
        vertices, faces = quad(True)
        reversed_faces = faces.copy(); reversed_faces[1] = reversed_faces[1, ::-1]
        self.assertEqual(self.build(vertices, reversed_faces).neighbors('geometric_candidate', 0), ())
        triangle = vertices[:3]
        layers = np.concatenate([triangle, triangle])
        for second in ([3,4,5], [5,4,3]):
            duplicate_faces = np.array([[0,1,2], second], np.int64)
            top = self.build(layers, duplicate_faces)
            self.assertEqual(top.neighbors('geometric_candidate', 0), ())
            self.assertTrue(any(top.blocked_faces.values()))

    def test_nonmanifold_edge_and_hard_crease_are_not_candidate_edges(self):
        vertices = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [0,-1,0]], float)
        faces = np.array([[0,1,2], [1,0,3], [0,1,4]], np.int64)
        top = self.build(vertices, faces)
        self.assertEqual(top.neighbors('geometric_candidate', 0), ())
        self.assertGreater(top.counts['nonmanifold_edges'], 0)
        crease = self.build(vertices, faces[:2])
        self.assertEqual(crease.neighbors('geometric_candidate', 0), ())

    def test_declared_mm_scaling_and_rigid_transform_preserve_fraction(self):
        rotation = np.array([[0.,0,1], [1,0,0], [0,1,0]])
        for scale, units in [(1e-6,'metres'), (1.,'metres'), (1000.,'millimetres'), (1e6,'relative')]:
            vertices, faces = quad(True); vertices = vertices @ rotation.T * scale + [3*scale, 5*scale, 7*scale]
            coords = quad(True)[0][faces, :2] + .5
            top = self.build(vertices, faces, units=units, face_region_ids=np.zeros(2,np.int64), corner_uv=coords)
            mk = marker(scale)
            for key in ('tangent','bitangent','normal'): mk[key] = (rotation @ np.array(mk[key])).tolist()
            result = st.diagnose_patch(vertices, faces, mk, np.full(2,3,np.int64), top,
                units=units, limits=self.limits).to_dict()['traversals']['approved']['coverage']
            self.assertAlmostEqual(result['union_coverage_fraction'], 1)
            self.assertAlmostEqual(result['patch_surface_area'] / (scale*scale), 1)

    def test_malformed_mesh_arrays_are_refused(self):
        vertices, faces = quad()
        cases = [(vertices.tolist(), faces), (vertices.astype(bool),faces),
            (vertices.astype(object),faces), (vertices,faces.astype(float)),
            (vertices,np.array([[0,1,99]],np.int64)), (vertices,np.array([[0,-1,2]],np.int64)),
            (np.full_like(vertices,np.nan),faces), (vertices[:, :2],faces)]
        for v, f in cases:
            with self.subTest(dtype=str(getattr(v,'dtype',type(v)))), self.assertRaises(ValueError): self.build(v,f)

    def test_array_digest_and_record_snapshots_are_deterministic_and_detached(self):
        vertices, faces = quad(True)
        top = self.build(vertices, faces); again = self.build(vertices.copy(), faces.copy())
        self.assertEqual(top.digest, again.digest)
        record = top.to_dict(); record['counts']['faces'] = 999
        self.assertEqual(top.to_dict()['counts']['faces'], 2)
        vertices[:] = 0
        self.assertEqual(top.neighbors('geometric_candidate',0), (1,))
        with self.assertRaises(ValueError): self.diagnose(vertices,faces,top)

    def test_marker_support_and_shape_errors_never_become_zero_accuracy(self):
        vertices, faces = quad(); top = self.build(vertices,faces)
        for change in [dict(size_object_units=True),dict(size_object_units=0),dict(face_id=True),
                dict(barycentric=[1,1,1]),dict(normal=[0,0,2]),dict(tangent=[0,1,0])]:
            mk = {**marker(), **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                st.diagnose_patch(vertices,faces,mk,np.full(2,3,np.int64),top,units='metres',limits=self.limits)

    def test_mesh_limits_reject_before_processing_and_support_is_explicit(self):
        vertices, faces = quad()
        with self.assertRaises(ValueError): self.build(vertices,faces,limits=replace(self.limits,max_faces=1))
        top = self.build(vertices,faces)
        with self.assertRaises(ValueError):
            st.diagnose_patch(vertices,faces,marker(),np.ones(2,bool),top,units='metres',limits=self.limits)
        result = st.diagnose_patch(vertices,faces,marker(),np.zeros(2,np.int64),top,units='metres',limits=self.limits).to_dict()
        self.assertNotEqual(result['traversals']['index']['status'], 'measured')

    def test_overlap_kernel_detects_multiplicity_and_budget_exhaustion(self):
        triangle = [(-1.,-1.),(1.,-1.),(0.,1.)]
        overlap = st._overlap_analysis([triangle,triangle],[triangle,triangle],self.limits,1e-12)
        self.assertEqual(overlap['status'],'positive_overlap')
        budget = replace(self.limits,max_overlap_candidates=1)
        unavailable = st._overlap_analysis([triangle]*3,[triangle]*3,budget,1e-12)
        self.assertEqual(unavailable['status'],'unavailable_budget')
        self.assertIsNone(unavailable['positive_overlaps'])

    def test_sub_ulp_overlap_is_not_certified_disjoint_by_area_cancellation(self):
        delta = np.spacing(1.)
        p = [(-.5,.125),(.5,.125),(-.5,1.125)]
        q = [(.5-delta,.125),(1.5,.125),(.5-delta,1.125)]
        result = st._overlap_analysis([st._clip_to_square(p),st._clip_to_square(q)],[p,q],self.limits,1e-12)
        self.assertNotEqual(result['status'],'disjoint_interiors_proven', 'true overlap is positive delta squared / 2')

    def test_degeneracy_classification_is_rotation_invariant(self):
        leg = np.sqrt(1.5e-12)
        vertices = np.array([[-.5,-.5,0],[.5,-.5,0],[.5,.5,0],[0,0,0],[leg,0,0],[0,leg,0]])
        faces = np.array([[0,1,2],[3,4,5]],np.int64)
        angle = np.pi/4
        rotation = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        before = self.build(vertices,faces).to_dict()['blocked_faces']
        after = self.build(vertices @ rotation.T,faces).to_dict()['blocked_faces']
        self.assertEqual(before,after)

    def test_unrepresentable_squared_units_fail_explicitly_without_arithmetic_crash(self):
        for scale in (1e-200,1e200):
            vertices,faces = quad(True);vertices *= scale
            with self.subTest(scale=scale):
                try:
                    top = self.build(vertices,faces)
                    result = st.diagnose_patch(vertices,faces,marker(scale),np.full(2,3,np.int64),top,
                        units='metres',limits=self.limits).to_dict()
                except ValueError:
                    continue  # explicit numerical-range refusal is valid
                for tr in result['traversals'].values():
                    if tr['status']=='measured':
                        self.assertGreater(tr['coverage']['nominal_square_area'],0)
                        self.assertTrue(np.isfinite(tr['coverage']['nominal_square_area']))

    def test_actual_topology_output_representation_obeys_byte_cap(self):
        vertices,faces=quad(True)
        try:top=self.build(vertices,faces,limits=replace(self.limits,max_output_bytes=1600))
        except ValueError:return
        raw=json.dumps(top.to_dict(),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        self.assertLessEqual(len(raw),1600)

    def test_replaced_adjacency_cannot_reuse_original_identity(self):
        vertices,faces=quad(True);top=self.build(vertices,faces)
        forged=replace(top,_csr={**top._csr,'approved':top._csr['geometric_candidate']})
        with self.assertRaises(ValueError):self.diagnose(vertices,faces,forged)

    def test_internal_array_cannot_be_mutated_behind_a_retained_digest(self):
        vertices,faces=quad(True);top=self.build(vertices,faces)
        try:top._v.flags.writeable=True
        except ValueError:return  # immutable backing storage is sufficient
        top._v[0,0] += 7
        with self.assertRaises(ValueError):self.diagnose(vertices,faces,top)

    def test_overlap_construction_is_bounded_before_large_bucket_population(self):
        triangle=[(-1.,-1.),(1.,-1.),(0.,1.)]
        limits=replace(self.limits,max_overlap_candidates=1,max_overlap_operations=1)
        count=0
        def bounded_range(*args):
            nonlocal count
            for value in range(*args):
                count+=1
                if count>500:raise AssertionError('construction work exceeded tiny operation budgets')
                yield value
        with mock.patch.object(st,'range',bounded_range,create=True):
            result=st._overlap_analysis([triangle]*40,[triangle]*40,limits,1e-12)
        self.assertEqual(result['status'],'unavailable_budget')


class PrefixBindingTests(unittest.TestCase):
    def fixture(self, clock='timestamp_seconds'):
        document = clip_document(clock_kind=clock)
        document['width'], document['height'] = 4, 3
        document['intrinsics']['K'] = [4.,0,1.5,0,4.,1.,0,0,1.]
        document['onboarding']['selection']['positive'] = [[1.,1.]]
        document['onboarding']['selection']['negative'] = [[2.,2.]]
        clip = wc.ClipSpec.from_dict(document)
        run = wc.RunSpec.from_dict(run_document())
        manifest = {'schema': st.SCHEMA_PREFIX_MANIFEST, 'clip_id': clip.clip_id,
            'run_id': run.run_id, 'units': 'metres', 'frames': []}
        views = {}
        for position in (0,1):
            frame = clip.frames[position]
            rgb = np.full((3,4,3),position,np.uint8)
            mask = np.ones((3,4),bool)
            views[frame.frame_id] = {'rgb':rgb,'mask':mask}
            manifest['frames'].append({'frame_id':frame.frame_id,'clip_position':position,
                'source_frame_index':frame.source_frame_index,'timestamp_s':frame.timestamp_s,
                'clock_kind':clip.clock_kind,'width':4,'height':3,'K':list(clip.intrinsics.K),
                'image':{'array_sha256':st.array_digest(rgb)},
                'observed_mask':{'array_sha256':st.mask_digest(mask),'kind':'observed','producer':'opening_prompt_segmentation'},
                'pose':{'camera_from_object':np.eye(4).tolist(),'units':'metres',
                    'provenance':'controlled_init:opening_only','producer_receipt_sha256':'b'*64},
                'producer_receipt_sha256':'c'*64})
        return clip,run,manifest,views

    def result(self, clip, run, manifest, views):
        return st.validate_prefix_binding(clip,run,manifest,views,limits=st.portable_20k_limits()).to_dict()

    def test_complete_prefix_is_consistent_but_always_unauthenticated(self):
        result = self.result(*self.fixture())
        self.assertEqual(result['status'],'bound_in_memory_unauthenticated')
        self.assertIs(result['authenticated'],False)
        self.assertIs(result['post_prefix_rows_consumed'],False)
        self.assertAlmostEqual(result['duration_s'],.1)

    def test_frame_index_clock_cannot_attest_physical_duration(self):
        result = self.result(*self.fixture('frame_index'))
        self.assertEqual(result['status'],'bound_in_memory_unauthenticated')
        self.assertIsNone(result['duration_s'])
        self.assertEqual(result['duration_status'],'unattested_frame_index_clock')

    def test_supplied_pixel_or_mask_mutation_is_refused(self):
        for field in ('rgb','mask'):
            clip,run,manifest,views = self.fixture()
            views['f0'][field][0,0] = 0 if field=='mask' else 99
            with self.subTest(field=field):
                self.assertEqual(self.result(clip,run,manifest,views)['status'],'refused')

    def test_missing_pose_mask_or_receipt_is_unavailable_not_zero(self):
        for field in ('pose','observed_mask','producer_receipt_sha256'):
            clip,run,manifest,views = self.fixture()
            manifest['frames'][0][field] = None
            with self.subTest(field=field):
                result = self.result(clip,run,manifest,views)
                self.assertEqual(result['status'],'unavailable')
                self.assertIs(result['authenticated'],False)

    def test_projection_masks_and_evaluator_pose_provenance_are_refused(self):
        mutations = [lambda f:f['observed_mask'].__setitem__('kind','projected_silhouette'),
            lambda f:f['observed_mask'].__setitem__('producer','evaluator_masks'),
            lambda f:f['pose'].__setitem__('provenance','controlled_init:ground_truth'),
            lambda f:f['pose'].__setitem__('provenance','onboarding:scored_pose'),
            lambda f:f['pose'].__setitem__('units','millimetres')]
        for i,change in enumerate(mutations):
            clip,run,manifest,views = self.fixture();change(manifest['frames'][0])
            with self.subTest(case=i):self.assertEqual(self.result(clip,run,manifest,views)['status'],'refused')

    def test_scored_frame_duplicate_and_extra_views_are_refused(self):
        for kind in ('future','duplicate','extra'):
            clip,run,manifest,views = self.fixture()
            if kind=='future':
                manifest['frames'][1]['frame_id']='f2';manifest['frames'][1]['clip_position']=2
            elif kind=='duplicate':manifest['frames'][1]=copy.deepcopy(manifest['frames'][0])
            else:views['f3']=views['f0']
            with self.subTest(case=kind):self.assertEqual(self.result(clip,run,manifest,views)['status'],'refused')

    def test_camera_and_matching_clock_conventions_are_bound(self):
        mutations=[lambda f:f['K'].__setitem__(2,2.),lambda f:f.__setitem__('width',5),
            lambda f:f.__setitem__('source_frame_index',0),lambda f:f.__setitem__('timestamp_s',.2),
            lambda f:f.__setitem__('clock_kind','frame_index')]
        for i,change in enumerate(mutations):
            clip,run,manifest,views=self.fixture();change(manifest['frames'][0])
            with self.subTest(case=i):self.assertEqual(self.result(clip,run,manifest,views)['status'],'refused')

    def test_boolean_numeric_pose_and_timestamp_are_not_silently_coerced(self):
        for kind in ('pose','numeric_string_pose','timestamp'):
            clip,run,manifest,views=self.fixture()
            if kind=='pose':manifest['frames'][0]['pose']['camera_from_object']=np.eye(4,dtype=bool).tolist()
            elif kind=='numeric_string_pose':manifest['frames'][0]['pose']['camera_from_object']=np.eye(4).astype(str).tolist()
            else:manifest['frames'][0]['timestamp_s']=False
            with self.subTest(case=kind):self.assertEqual(self.result(clip,run,manifest,views)['status'],'refused')

    def test_no_caller_authentication_switch_or_extra_prefix_fields(self):
        clip,run,manifest,views=self.fixture();manifest['authenticated']=True
        with self.assertRaises(ValueError):self.result(clip,run,manifest,views)

    def test_replaced_schema_and_evaluator_flag_are_refused_before_serialization(self):
        for kind in ('clip_schema','run_schema','evaluator'):
            clip,run,manifest,views=self.fixture()
            if kind=='clip_schema':clip=replace(clip,schema='invalid_schema')
            elif kind=='run_schema':run=replace(run,schema='invalid_schema')
            else:clip=replace(clip,evaluator_inputs_read=True)
            with self.subTest(kind=kind):
                try:result=self.result(clip,run,manifest,views)
                except ValueError:continue
                self.assertEqual(result['status'],'refused')

    def test_missing_nested_pose_receipt_is_unavailable(self):
        clip,run,manifest,views=self.fixture();manifest['frames'][0]['pose']['producer_receipt_sha256']=None
        self.assertEqual(self.result(clip,run,manifest,views)['status'],'unavailable')

    def test_prefix_output_cap_is_enforced(self):
        clip,run,manifest,views=self.fixture()
        with self.assertRaises(ValueError):
            st.validate_prefix_binding(clip,run,manifest,views,limits=replace(st.portable_20k_limits(),max_output_bytes=1))

    def test_large_nested_container_is_refused_without_iterating_it(self):
        class RefuseIteration(list):
            def __iter__(self):raise AssertionError('oversized container traversed before bound')
        clip,run,manifest,views=self.fixture()
        manifest['frames'][0]['K']=RefuseIteration([0]*100000)
        try:result=self.result(clip,run,manifest,views)
        except ValueError:return
        self.assertEqual(result['status'],'refused')

    def test_singular_and_reflected_pose_rotations_are_refused(self):
        for rotation in (np.zeros((3,3)),np.diag([-1.,1.,1.])):
            clip,run,manifest,views=self.fixture();pose=np.eye(4);pose[:3,:3]=rotation
            manifest['frames'][0]['pose']['camera_from_object']=pose.tolist()
            with self.subTest(rotation=rotation.tolist()):self.assertEqual(self.result(clip,run,manifest,views)['status'],'refused')


if __name__ == '__main__':
    unittest.main()
