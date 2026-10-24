"""Independent offline-report checks; no neural or renderer execution."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import tempfile
import unittest
import numpy as np
from tools import workbench_comparison as comparison

BASE = 'f7d127abe2b8e37b1322a19115343286a6560700c6e02fc6080b4e2426a01086'

def row(fid='f1', role='primary_neural', **kwargs):
    identity = {'role':role, 'base_checkpoint_sha256':BASE,
                'installed_head_sha256':comparison.PRIMARY_HEAD_SHA256} if role == 'primary_neural' else {
                    'role':role,'head':'frozen_base_pose_head','base_checkpoint_sha256':BASE}
    fields = dict(frame_id=fid, pose_state='tracking', render_state='rendered', mask_state='observed',
                  failure_reason=None, metrics={'end_to_end_latency_ms':10.},backend_identity=identity)
    fields.update(kwargs)
    return SimpleNamespace(**fields)

class ComparisonTests(unittest.TestCase):
    def telemetry_report(self, telemetry):
        return {'scope':'regression','runs':[{'run_id':'run-1','clip_id':'clip-1','scope':'regression',
            'mode':'regression','scheduled_frames':2,'run_completeness':'complete','report_status':'completed',
            'neural_comparison':'unverified','roles':{},'paired_frames':None,'run_telemetry':telemetry,'input_files':[]}]}

    def test_legacy_gpu_values_in_markdown_always_keep_unverified_scope(self):
        for value,text in ((True,'true (recorded legacy value)'),(False,'false (recorded legacy value)'),
                (None,'not recorded'),('false','not recorded'),(0,'not recorded'),(1,'not recorded'),([], 'not recorded')):
            report=self.telemetry_report({'gpu_used':value});before=copy.deepcopy(report)
            rendered=comparison.render_markdown(report)
            with self.subTest(value=value):
                self.assertIn('legacy GPU field (run scope unverified): '+text,rendered)
                self.assertNotIn('GPU used ',rendered)
                self.assertNotIn('GPU peak allocated',rendered)
                self.assertEqual(report,before)
        self.assertIn('legacy GPU field (run scope unverified): not recorded',
            comparison.render_markdown(self.telemetry_report({'run_wall_ms':0})))

    def test_memory_scope_labels_preserve_measured_zero_and_unknown(self):
        rendered=comparison.render_markdown(self.telemetry_report({'process_peak_rss_bytes':0,
            'gpu_peak_allocated_bytes':1024,'gpu_peak_reserved_bytes':None}))
        self.assertIn('host process peak RSS (not a per-run peak): 0 B',rendered)
        self.assertIn('process CUDA allocator peak allocated (cumulative; epoch unrecorded): 1024 B',rendered)
        self.assertIn('process CUDA allocator peak reserved (cumulative; epoch unrecorded): not recorded',rendered)
        self.assertIn('Process memory observations do not establish GPU work or per-run memory peaks.',rendered)

    def test_malformed_memory_values_are_never_displayed_as_measurements(self):
        fields=('process_peak_rss_bytes','gpu_peak_allocated_bytes','gpu_peak_reserved_bytes')
        for value in (True,False,-1,1.5,float('inf'),float('nan'),'100',{},2**53):
            with self.subTest(value=value):
                rendered=comparison.render_markdown(self.telemetry_report(dict.fromkeys(fields,value)))
                self.assertEqual(rendered.count(': not recorded'),4) # three memory values and the missing GPU field

    def test_markdown_does_not_modify_exported_raw_resource_json(self):
        report=self.telemetry_report({'gpu_used':False,'process_peak_rss_bytes':0,
            'gpu_peak_allocated_bytes':123,'gpu_peak_reserved_bytes':456,'run_cache_mode':'cached'})
        before=json.dumps(report,sort_keys=True)
        comparison.render_markdown(report)
        self.assertEqual(json.dumps(report,sort_keys=True),before)

    def test_percentile_matches_linear_definition(self):
        data=[0,1,10,12,97]
        for percentile in (0,25,50,95,100):
            self.assertAlmostEqual(comparison.percentile_linear(data,percentile),np.percentile(data,percentile))
        self.assertIsNone(comparison.percentile_linear([],95))

    def test_missing_rows_keep_full_denominator(self):
        result=comparison.summarize_role('primary_neural',['f1','f2'],[row()])
        self.assertEqual(result['pose_availability_fraction'],.5)
        self.assertEqual(result['unavailable_frames_including_missing'],1)
        self.assertEqual(result['missing_frame_ids'],['f2'])

    def test_pose_and_render_have_separate_denominators(self):
        result=comparison.summarize_role('primary_neural',['f1','f2'],[row(),row('f2',render_state='unavailable')])
        self.assertEqual(result['pose_availability_fraction'],1.)
        self.assertEqual(result['raw_artwork_render_fraction'],.5)

    def test_duplicate_rows_are_rejected(self):
        with self.assertRaises(comparison.ComparisonError):
            comparison.summarize_role('primary_neural',['f1'],[row(),row()])

    def test_primary_requested_head_cannot_replace_actual_identity(self):
        for identity in ({},{'installed_head_sha256':'0'*64}):
            with self.subTest(identity=identity),self.assertRaises(comparison.ComparisonError):
                comparison.identity_check('primary_neural',[row(backend_identity=identity)],True,
                                          comparison.PRIMARY_HEAD_SHA256)

    def test_baseline_wrong_base_is_rejected(self):
        bad=row(role='frozen_baseline')
        bad.backend_identity['base_checkpoint_sha256']='0'*64
        with self.assertRaises(comparison.ComparisonError):
            comparison.identity_check('frozen_baseline',[bad],True,comparison.PRIMARY_HEAD_SHA256)

    def test_cache_labels_original_timings(self):
        result=comparison.summarize_role('primary_neural',['f1'],[row()],cache_status='cache_hit',
                                         cached_original={'initialization_ms':30})
        self.assertEqual(result['latency']['source'],'cached_original_run')
        self.assertEqual(result['cached_original_timings'],{'initialization_ms':30})

    def test_primary_wrong_or_missing_base_is_rejected(self):
        for base in (None,'0'*64):
            bad=row()
            bad.backend_identity['base_checkpoint_sha256']=base
            with self.subTest(base=base),self.assertRaises(comparison.ComparisonError):
                comparison.identity_check('primary_neural',[bad],True,comparison.PRIMARY_HEAD_SHA256)

    def test_unconfirmed_stage_remains_unverified(self):
        result=comparison.identity_check('primary_neural',[row()],False,comparison.PRIMARY_HEAD_SHA256)
        self.assertFalse(result['identity_verified'])
        self.assertIsNone(result['installed_head_sha256'])

    def test_existing_output_prevents_any_other_artifact_write(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=comparison.output_paths(Path(directory)/'report',False)
            paths['md'].write_bytes(b'original')
            with self.assertRaises(comparison.ComparisonError):
                comparison.publish(paths,{'json':b'{}','md':b'replacement'})
            self.assertEqual(paths['md'].read_bytes(),b'original')
            self.assertFalse(paths['json'].exists())

    def test_pairing_matches_frame_ids_including_missing(self):
        result=comparison.paired_frames(['f1','f2','f3'],[row('f1'),row('f2')],
                                        [row('f2',role='frozen_baseline')])
        self.assertEqual([result[k] for k in ('pose_both_available','pose_primary_only','pose_baseline_only','pose_neither')],
                         [1,1,0,1])

if __name__=='__main__':unittest.main()
