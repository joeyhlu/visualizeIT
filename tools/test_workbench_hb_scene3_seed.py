"""Codex controls: manufactured annotations only; no real GT, pixels or inference."""
from pathlib import Path
import binascii
import copy
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock
import zlib

import numpy as np
from tools import workbench_hb_scene3_seed as seed
from tools import workbench_hb_scene3_media as media
from tools import test_workbench_hb_scene3_media as fixtures
from tools.test_workbench_hb_metric_inputs import SCRATCH, REVISION, digest, json_bytes

R=[0,-1,0,1,0,0,0,0,1]
T=[125,-50,1000]
def entry(object_id=4,rotation=None,translation=None):
    return {'obj_id':object_id,'cam_R_m2c':list(R if rotation is None else rotation),
        'cam_t_m2c':list(T if translation is None else translation)}

def selected(raw):
    return seed.select_unique_seed(seed.decode_frame_value(raw))


class UniqueSeedNumericTests(unittest.TestCase):
    def test_asymmetric_rotation_and_mm_translation_are_preserved(self):
        result=selected(json_bytes([entry()]))
        np.testing.assert_array_equal(result['pose'],[[0,-1,0,.125],[1,0,0,-.05],[0,0,1,1],[0,0,0,1]])
        self.assertEqual(result['ordinal'],0)

    def test_uniqueness_uses_source_order_not_matching_object_count(self):
        result=selected(json_bytes([entry(2),entry(7),entry(4),entry(3)]))
        self.assertEqual(result['ordinal'],2)
        self.assertEqual(result['matches'],1)

    def test_absent_and_duplicate_selected_object_refuse(self):
        for rows in ([entry(3)],[entry(4),entry(4)]):
            with self.subTest(rows=rows),self.assertRaises(media.mi.Blocked):selected(json_bytes(rows))

    def test_unselected_extreme_numbers_are_opaque_and_never_converted(self):
        other=b'{"obj_id":2,"cam_R_m2c":['+b','.join([b'1e400']*9)+b'],"cam_t_m2c":['+b','.join([b'9'*400]*3)+b']}'
        frame=seed.decode_frame_value(b'['+other+b','+json_bytes(entry())+b']')
        with mock.patch.object(seed,'_convert',wraps=seed._convert) as convert:
            result=seed.select_unique_seed(frame)
        self.assertEqual(convert.call_count,12)
        self.assertEqual(result['other_pose_values_converted'],0)
        self.assertEqual(frame[0]['cam_R_m2c'][0].text,'1e400')

    def test_bad_ids_duplicate_keys_nonfinite_and_strings_are_refused(self):
        bad=[b'[{"obj_id":4,"obj_id":4,"cam_R_m2c":[],"cam_t_m2c":[]}]',
             json_bytes([entry(True)]),json_bytes([entry(4.0)]),json_bytes([entry('4')]),
             json_bytes([entry()]).replace(b'125',b'NaN'),
             json_bytes([entry()]).replace(b'125',b'"125"')]
        for raw in bad:
            with self.subTest(raw=raw[:40]),self.assertRaises((media.mi.InputsError,ValueError)):selected(raw)

    def test_selected_overflow_reflection_singular_and_invalid_lengths_refuse(self):
        bad=[json_bytes([entry()]).replace(b'125',b'1e400'),json_bytes([entry(rotation=[-1,0,0,0,1,0,0,0,1])]),
            json_bytes([entry(rotation=[0]*9)]),json_bytes([entry(rotation=[1,0])])]
        for raw in bad:
            with self.subTest(raw=raw[:40]),self.assertRaises((media.mi.InputsError,ValueError)):selected(raw)

    def test_lexical_frame_capture_ignores_scored_extreme_values_at_every_chunk_boundary(self):
        wanted=json_bytes([entry()])
        document=b'{"0":[{"x":[1,2,3]}],"9":'+wanted+b',"10":{"unselected":1e400}}'
        for chunk in (1,2,7,64,65536):
            scanner=seed.opening.TopLevelFrameExtractor('9')
            for start in range(0,len(document),chunk):scanner.feed(document[start:start+chunk])
            with self.subTest(chunk=chunk):self.assertEqual(scanner.finish(),wanted)

    def test_malformed_utf8_and_deep_nested_values_fail_with_typed_input_error(self):
        for raw in (b'\xff',b'['*2000+b'0'+b']'*2000,b'[]'):
            with self.subTest(raw=raw[:8]),self.assertRaises(media.mi.InputsError):seed.decode_frame_value(raw)


class UniqueSeedIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SCRATCH.mkdir(parents=True,exist_ok=True)
        cls.temp=tempfile.TemporaryDirectory(prefix='hb-seed-codex-',dir=SCRATCH)
        cls.root=Path(cls.temp.name).resolve()
        assert cls.root.is_relative_to(SCRATCH.resolve())
        real_disk=shutil.disk_usage
        if real_disk(cls.root).free < (1<<30)+(12<<20):
            cls.temp.cleanup();raise unittest.SkipTest('Real free-space reserve cannot cover manufactured fixture')
        # Only manufactured scratch budget inputs are deterministic, never production workspace inputs.
        def disk(path):
            actual=real_disk(path)
            if Path(path).resolve().is_relative_to(cls.root):return type(actual)(actual.total,actual.total-(2<<30),2<<30)
            return actual
        cls.patch=mock.patch.object(media.mi.shutil,'disk_usage',side_effect=disk);cls.patch.start()
        cls.addClassCleanup(cls.patch.stop);cls.addClassCleanup(cls.temp.cleanup)
        cls.workspace=cls.root/'workspace';cls.workspace.mkdir()
        original_catalog=fixtures.scene_three_catalog
        cls.annotation_raw=b'{"0":[{"opaque":1e400}],"9":'+json_bytes([entry(2),entry(4),entry(7)])+b',"10":[{"opaque":1e400}]}'
        def catalog(root):
            result=original_catalog(root)
            cls.gt=root/'manufactured-metadata'/seed.GT_MEMBER
            # The established media fixture creates this directory for its camera file.
            # Bind the annotation's known manufactured bytes now, write them after that creation.
            size=len(cls.annotation_raw)
            row={'datasetRevision':REVISION,'archive':media.mi.RGB_ARCHIVE,'sourceMember':seed.GT_MEMBER,
                'allowedRoles':['evaluation'],'sourceFlags':0,'sourceOffset':0,'method':0,'sourceMethod':8,'flags':0,
                'compressedBytes':size,'expandedBytes':size,'sourceCompressedBytes':len(zlib.compress(cls.annotation_raw)),
                'crc32':binascii.crc32(cls.annotation_raw)&0xffffffff,'compressedSHA256':digest(cls.annotation_raw),
                'uncompressedSHA256':digest(cls.annotation_raw),'location':{'kind':'reuse','path':cls.gt.relative_to(root).as_posix()}}
            result['documents']['resources_index'][1]['resources'][seed.GT_MEMBER]=row
            return result
        with mock.patch.object(fixtures,'scene_three_catalog',side_effect=catalog):
            cls.f=fixtures.manufactured_scene_three_inputs(cls.root,cls.workspace)
        cls.gt.write_bytes(cls.annotation_raw)
        f=cls.f
        cls.mp=media.SyntheticMediaPins(f['profile_pins'],f['profile_sha256'],f['camera_k_sha256'],f['association_sha256'],4)
        mc=media.MediaContext(cls.root,cls.workspace,f['scope'],cls.mp)
        cls.allow=media.build_allowlist(mc,profile=f['profile'],expected_profile_sha256=f['profile_sha256'],
            camera_k=f['camera_k'],expected_camera_k_sha256=f['camera_k_sha256'],association_review=f['association'],
            expected_association_review_sha256=f['association_sha256'],out=f['scope']/'allow')
        args={'allowlist':cls.root/cls.allow['allowlist'],'expected_allowlist_sha256':cls.allow['allowlist_sha256']}
        model=media.prepare_model(mc,**args,out=f['scope']/'model')
        video=media.prepare_video(mc,**args,out=f['scope']/'video')
        cls.prereq=media.check_prerequisites(mc,**args,model_receipt=cls.root/model['model_receipt'],
            expected_model_receipt_sha256=model['model_receipt_sha256'],video_receipt=cls.root/video['video_receipt'],
            expected_video_receipt_sha256=video['video_receipt_sha256'],out=f['scope']/'prereq')
        policy={'schema':'workbench_hb_scene3_controlled_seed_policy_v1','timestamp_utc':'2026-10-22T00:00:00Z',
            'reviewer':'Codex manufactured fixture','scene':'000003','object_id':4,'frame_id':'s000009','frame_index':9,
            'policy':'Exactly one CAD4 in opening9 else refuse','frozen_before_annotation_access':True,
            'annotation_values_read':False,'association_sha256':f['association_sha256'],
            'prerequisites_sha256':cls.prereq['prerequisites_receipt_sha256'],'controlled_ground_truth_seed_authorized':True,
            'applies_once_to_frame_id':'s000010','reuse_on_recovery':False,'scored_pose_values_materialized':False,
            'automatic_geometry_annotation_access':False,'principal_point_origin':None,
            'principal_point_origin_review_required_for_assembly':True,'translation':'BOP mm * .001 once',
            'maximum_annotation_bytes':8<<20,'seed_subtree_total_cap_bytes':64<<10,'quality_claim':False,
            'phone_performance_claim':False}
        cls.policy_path=cls.root/'policy.json';cls.policy_path.write_bytes(json_bytes(policy))
        cls.sp=seed.SyntheticSeedPins(cls.mp,cls.prereq['prerequisites_receipt_sha256'],digest(cls.policy_path.read_bytes()))
        cls.context=seed.SeedContext(cls.root,cls.workspace,f['scope'],cls.sp)

    def setUp(self):
        suffix=digest(self._testMethodName.encode())[:8]
        self.plan_dir=self.f['scope']/seed.SEED_SUBTREE/('p-'+suffix)
        self.extract_dir=self.f['scope']/seed.SEED_SUBTREE/('e-'+suffix)

    def tearDown(self):
        for path in (self.plan_dir,self.extract_dir):
            if path.exists():
                assert path.resolve().is_relative_to(self.f['scope'].resolve()) and not path.is_symlink()
                shutil.rmtree(path)

    def plan(self,**changes):
        args={'prerequisites':self.root/self.prereq['prerequisites_receipt'],
            'expected_prerequisites_sha256':self.prereq['prerequisites_receipt_sha256'],
            'policy_review':self.policy_path,'expected_policy_review_sha256':self.sp.policy_review_sha256,'out':self.plan_dir}
        args.update(changes);return seed.plan_seed(self.context,**args)

    def extract(self,plan,**changes):
        args={'plan':self.root/plan['plan'],'expected_plan_sha256':plan['plan_sha256'],'out':self.extract_dir}
        args.update(changes);return seed.extract_unique_seed(self.context,**args)

    def test_metadata_plan_never_opens_annotations_or_payloads(self):
        original=Path.open
        def opening(path,*args,**kwargs):
            if path.resolve()==self.gt.resolve():raise AssertionError('Annotation opened in metadata plan')
            return original(path,*args,**kwargs)
        with mock.patch.object(Path,'open',opening),mock.patch.object(media.mi,'verified_payload_stream',side_effect=AssertionError('payload')):
            result=self.plan()
        doc=json.loads((self.root/result['plan']).read_bytes())
        self.assertEqual(doc['annotation_source']['role'],'evaluation')
        self.assertEqual(doc['flags']['annotation_opens'],0)
        self.assertEqual(doc['label'],'TEST_ONLY')

    def test_actual_manufactured_chain_discloses_four_pass_gt_seed_and_exact_units(self):
        plan=self.plan();result=self.extract(plan)
        out=json.loads((self.root/result['seed']).read_bytes())
        self.assertEqual(result['annotation_opens'],4);self.assertEqual(result['instance_ordinal'],1)
        self.assertIn('controlled_init:ground_truth:HB:',out['seed_label'])
        self.assertTrue(out['flags']['initial_ground_truth_read'])
        self.assertFalse(out['flags']['scored_pose_values_materialized'])
        self.assertFalse(out['reuse_on_recovery']);self.assertEqual(out['applies_once_to_frame_id'],'s000010')
        self.assertIsNone(out['principal_point_origin'])
        np.testing.assert_array_equal(out['seed_pose_m'],[[0,-1,0,.125],[1,0,0,-.05],[0,0,1,1],[0,0,0,1]])
        self.assertIsNone(json.loads(self.f['association'].read_bytes())['instance_ordinal'])

    def test_wrong_prerequisite_or_policy_pin_refuses_before_any_gt_grant(self):
        for key in ('expected_prerequisites_sha256','expected_policy_review_sha256'):
            with self.subTest(key=key),mock.patch.object(seed,'annotation_pass',side_effect=AssertionError('annotation')):
                with self.assertRaises(media.mi.InputsError):self.plan(**{key:'0'*64})

    def test_rehashed_plan_descriptor_drift_refuses_before_annotation_grant(self):
        plan=self.plan();path=self.root/plan['plan'];doc=json.loads(path.read_bytes())
        doc['annotation_source']['expanded_sha256']='0'*64;path.write_bytes(json_bytes(doc))
        with mock.patch.object(seed,'annotation_pass',side_effect=AssertionError('annotation')):
            with self.assertRaises(media.mi.InputsError):self.extract(plan,expected_plan_sha256=digest(path.read_bytes()))

    def test_production_context_cannot_consume_synthetic_inputs(self):
        self.context=seed.SeedContext(self.root,self.workspace,self.f['scope'])
        try:
            with self.assertRaises(media.mi.InputsError):self.plan()
        finally:self.context=type(self).context

    def test_final_byte_change_invalidates_owned_outputs_with_unchanged_source_stat(self):
        plan=self.plan();original_open,original_publish=Path.open,media.publish;published=[False]
        before=os.stat(self.gt)
        def publish(ledger,name,data,**kwargs):
            value=original_publish(ledger,name,data,**kwargs)
            if name==seed.BINDING_NAME:published[0]=True
            return value
        def opening(path,*args,**kwargs):
            if path.resolve()==self.gt.resolve() and published[0] and args and args[0]=='rb':
                with original_open(path,*args,**kwargs) as stream:raw=stream.read()
                # Read-only fault injection models preserved-size/mtime corruption after final publication.
                return io.BytesIO(raw[:-1]+bytes([raw[-1]^1]))
            return original_open(path,*args,**kwargs)
        with mock.patch.object(media,'publish',publish),mock.patch.object(Path,'open',opening):
            with self.assertRaises(media.mi.InputsError):self.extract(plan)
        self.assertEqual(os.stat(self.gt).st_mtime_ns,before.st_mtime_ns)
        self.assertFalse((self.extract_dir/seed.SEED_NAME).exists())
        self.assertFalse((self.extract_dir/seed.BINDING_NAME).exists())
        self.assertTrue((self.extract_dir/('invalidated-'+seed.SEED_NAME)).exists())
        self.assertTrue((self.extract_dir/'failure.json').exists())

    def test_retained_staging_sibling_counts_towards_seed_budget(self):
        self.extract_dir.mkdir(parents=True);(self.extract_dir/'.old.json.staging-fixture').write_bytes(b'x'*(60<<10))
        with mock.patch.object(seed,'annotation_pass',side_effect=AssertionError('annotation')):
            with self.assertRaises(media.mi.Blocked):self.plan()

    def test_changed_plan_scene_and_source_closure_refuse_before_annotation(self):
        plan=self.plan();path=self.root/plan['plan'];original=path.read_bytes()
        for field in ('scene','sources'):
            doc=json.loads(original)
            if field=='scene':doc['scene']='000002'
            else:doc['sources']['tools/workbench_hb_scene3_seed.py']='0'*64
            path.write_bytes(json_bytes(doc))
            try:
                with self.subTest(field=field),mock.patch.object(seed,'annotation_pass',side_effect=AssertionError('annotation')):
                    with self.assertRaises(media.mi.InputsError):self.extract(plan,expected_plan_sha256=digest(path.read_bytes()))
            finally:path.write_bytes(original)

    def test_fresh_free_space_guard_blocks_extraction_before_gt_access(self):
        plan=self.plan()
        actual=shutil.disk_usage(self.root)
        with mock.patch.object(seed.mi.shutil,'disk_usage',return_value=type(actual)(actual.total,actual.total-1,1)), \
                mock.patch.object(seed,'annotation_pass',side_effect=AssertionError('annotation')):
            with self.assertRaises(media.mi.Blocked):self.extract(plan)

    def test_foreign_output_collision_is_never_overwritten_or_owned(self):
        plan=self.plan();original=media.publish
        def publish(ledger,name,data,**kwargs):
            if name==seed.SEED_NAME:(ledger.dir/name).write_bytes(b'foreign bytes')
            return original(ledger,name,data,**kwargs)
        with mock.patch.object(media,'publish',publish):
            with self.assertRaises(media.mi.InputsError):self.extract(plan)
        self.assertEqual((self.extract_dir/seed.SEED_NAME).read_bytes(),b'foreign bytes')
        self.assertFalse((self.extract_dir/('invalidated-'+seed.SEED_NAME)).exists())
        self.assertTrue((self.extract_dir/'failure.json').exists())

    def test_annotation_cap_and_historical_zip_role_are_not_relaxable(self):
        path=self.gt
        index=self.f['catalog']['documents']['resources_index'][1]['resources'][seed.GT_MEMBER]
        for change in ({'allowedRoles':['rgb']},{'sourceMethod':True},{'expandedBytes':(8<<20)+1}):
            bad=copy.deepcopy(index);bad.update(change)
            with self.subTest(change=change),self.assertRaises(media.mi.InputsError):
                seed.validate_annotation_row(bad,root=self.root,canonical=path)
        descriptor={'expanded_bytes':(8<<20)+1,'expanded_sha256':'0'*64,'crc32':0}
        with mock.patch.object(Path,'open',side_effect=AssertionError('oversized annotation opened')):
            with self.assertRaises(media.mi.InputsError):seed.annotation_pass(path,descriptor)

    def test_historical_metadata_role_remains_separate_from_semantic_gt_authorization(self):
        row=copy.deepcopy(self.f['catalog']['documents']['resources_index'][1]['resources'][seed.GT_MEMBER])
        row['allowedRoles']=['metadata']
        descriptor=seed.validate_annotation_row(row,root=self.root,canonical=self.gt)
        self.assertEqual(descriptor['role'],'metadata')
        self.assertEqual(descriptor['semantic_role'],'controlled_ground_truth_annotation')
        self.assertNotEqual(descriptor['role'],'rgb')

    def test_rehashed_noncore_plan_claims_cannot_grant_annotation_access(self):
        plan=self.plan();path=self.root/plan['plan'];original=path.read_bytes()
        changes=[('annotation opens',lambda d:d['reads'].__setitem__('annotation_opens',99)),
            ('planned bytes',lambda d:d['budget'].__setitem__('planned_bytes',0)),
            ('false closure',lambda d:d['closure'].__setitem__('pre_publication_matches',False)),
            ('empty audited reads',lambda d:d['reads'].__setitem__('audited_opens',{}))]
        for label,change in changes:
            doc=json.loads(original);change(doc);path.write_bytes(json_bytes(doc))
            try:
                with self.subTest(label=label),mock.patch.object(seed,'annotation_pass',side_effect=AssertionError('untrusted annotation grant')):
                    with self.assertRaises(media.mi.InputsError):self.extract(plan,expected_plan_sha256=digest(path.read_bytes()))
            finally:path.write_bytes(original)

    def test_failed_publication_with_foreign_quarantine_names_leaves_no_authoritative_owned_output(self):
        plan=self.plan();original=media.publish
        def publish(ledger,name,data,**kwargs):
            result=original(ledger,name,data,**kwargs)
            if name==seed.BINDING_NAME:
                for filename in (seed.SEED_NAME,seed.BINDING_NAME):
                    (ledger.dir/('invalidated-'+filename)).write_bytes(b'foreign quarantine')
                raise media.mi.InputsError('injected postpublication failure with foreign quarantine collision')
            return result
        with mock.patch.object(media,'publish',publish):
            with self.assertRaises(media.mi.InputsError):self.extract(plan)
        for filename in (seed.SEED_NAME,seed.BINDING_NAME):
            with self.subTest(filename=filename):
                self.assertFalse((self.extract_dir/filename).exists(),'failed owned output must be quarantined without replacing foreign bytes')
                self.assertEqual((self.extract_dir/('invalidated-'+filename)).read_bytes(),b'foreign quarantine')
        self.assertTrue((self.extract_dir/'failure.json').exists())

    def owned_failure_fixture(self):
        self.extract_dir.mkdir(parents=True)
        ledger=seed.SeedLedger(self.context.mi(),self.extract_dir,'extract-unique-seed')
        for filename in (seed.SEED_NAME,seed.BINDING_NAME):
            path=self.extract_dir/filename;path.write_bytes(b'manufactured owned bytes')
            st=os.lstat(path);ledger.owned[filename]=(st.st_dev,st.st_ino)
        return ledger

    def test_post_quarantine_lookup_error_never_aborts_remaining_cleanup_or_masks_original(self):
        ledger=self.owned_failure_fixture();original=os.lstat;raised=[False]
        def lookup(path,*args,**kwargs):
            st=original(path,*args,**kwargs)
            if Path(path).name.startswith('invalidated-') and not raised[0]:
                raised[0]=True;raise OSError('injected post-rename lookup failure')
            return st
        with mock.patch.object(seed.os,'lstat',side_effect=lookup):
            seed._failure(ledger,'extract-unique-seed',media.mi.InputsError('original stage failure'))
        self.assertTrue(raised[0])
        self.assertFalse((self.extract_dir/seed.SEED_NAME).exists())
        self.assertFalse((self.extract_dir/seed.BINDING_NAME).exists())
        failure=json.loads((self.extract_dir/'failure.json').read_bytes())
        self.assertIn('original stage failure',failure['reason'])
        self.assertEqual(failure['authoritative_files_remaining'],[])

    def test_quarantine_budget_refusal_is_reported_without_aborting_failure_handler(self):
        ledger=self.owned_failure_fixture()
        with mock.patch.object(seed,'_quarantine',side_effect=media.mi.Blocked('alias budget refused')):
            seed._failure(ledger,'extract-unique-seed',media.mi.InputsError('original stage failure'))
        failure=json.loads((self.extract_dir/'failure.json').read_bytes())
        self.assertIn('original stage failure',failure['reason'])
        self.assertEqual(sorted(failure['authoritative_files_remaining']),sorted([seed.SEED_NAME,seed.BINDING_NAME]))
        self.assertEqual(failure['status'],'failed_authoritative_outputs_remain')

    def test_denied_quarantine_moves_preserve_original_failure_and_owned_bytes(self):
        ledger=self.owned_failure_fixture();rename=seed.os.rename;link=seed.os.link
        names={seed.SEED_NAME,seed.BINDING_NAME}
        def denied(operation):
            def invoke(src,dst,*args,**kwargs):
                if Path(src).name in names:raise PermissionError('injected denied quarantine move')
                return operation(src,dst,*args,**kwargs)
            return invoke
        with mock.patch.object(seed.os,'rename',side_effect=denied(rename)),mock.patch.object(seed.os,'link',side_effect=denied(link)):
            seed._failure(ledger,'extract-unique-seed',media.mi.InputsError('original stage failure'))
        failure=json.loads((self.extract_dir/'failure.json').read_bytes())
        self.assertIn('original stage failure',failure['reason'])
        self.assertEqual(set(failure['authoritative_files_remaining']),names)
        self.assertEqual(failure['status'],'failed_authoritative_outputs_remain')
        self.assertEqual({row['name'] for row in failure['quarantined_outputs']},names)
        for filename in names:self.assertEqual((self.extract_dir/filename).read_bytes(),b'manufactured owned bytes')

    def test_unreadable_authoritative_identity_is_reported_unknown_not_absent(self):
        ledger=self.owned_failure_fixture();original=seed.os.lstat;stderr=io.StringIO()
        names={seed.SEED_NAME,seed.BINDING_NAME}
        def unreadable(path,*args,**kwargs):
            if Path(path).parent==self.extract_dir and Path(path).name in names:
                raise PermissionError('injected unreadable authoritative identity')
            return original(path,*args,**kwargs)
        with mock.patch.object(seed.os,'lstat',side_effect=unreadable),mock.patch.object(seed.sys,'stderr',stderr):
            seed._failure(ledger,'extract-unique-seed',media.mi.InputsError('original stage failure'))
        path=self.extract_dir/'failure.json'
        failure=json.loads(path.read_bytes() if path.exists() else stderr.getvalue().splitlines()[-1])
        self.assertIn('original stage failure',failure['reason'])
        self.assertEqual(set(failure['authoritative_files_uncertain']),names)
        self.assertEqual(failure['status'],'failed_authoritative_outputs_remain')
        self.assertEqual(set(failure['authoritative_states'].values()),{'unknown'})
        for filename in names:self.assertEqual((self.extract_dir/filename).read_bytes(),b'manufactured owned bytes')


if __name__=='__main__':unittest.main()
