"""Focused offline report tests: sealing, pooling, denominators and repeats."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import tracking_report as report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(report.canonical(value))


def trial(sequence_id=1, errors=(0., 100.), *, system="current_rigid", repeat=0, failed=False):
    rows, details = [], []
    for frame, error in enumerate(errors, 10):
        pose = [[1.,0.,0.,error*.001],[0.,1.,0.,0.],[0.,0.,1.,.5],[0.,0.,0.,1.]]
        row = dict(frameId=frame, state="lost" if failed else "tracking", failureReason="init_failure" if failed else None,
                   cvCameraFromSourceObject=None if failed else pose, processingSeconds=0. if failed else .01,
                   updateAttempted=not failed, decodeAndHashSeconds=.002)
        detail = dict(frameId=frame, public=not failed, visible=True, accurate=(not failed and error<5.))
        if not failed:
            detail.update(reprojectionPixels=error, addMetres=error*.001, rotationDegrees=error*.1, translationMetres=error*.001)
        rows.append(row); details.append(detail)
    n = len(rows)
    raw = dict(system=system, repeat=repeat, frames=rows, timedUpdateFrames=0 if failed else n,
               setupFailure="init_failure" if failed else None, updateTotalSeconds=0. if failed else n*.01,
               replayWallSeconds=.1, decodeAndHashTotalSeconds=n*.002)
    summary = dict(objectId=sequence_id, name="SYNTHETIC train-smooth-27181", system=system, repeat=repeat,
                   frames=n, publicTrackingFrames=0 if failed else n, visibleFrames=n, visibleTrackingFrames=0 if failed else n,
                   accurateVisibleFrames=sum(row['accurate'] for row in details), hiddenPublicPoseFrames=0,
                   timedUpdateFrames=raw['timedUpdateFrames'], recovery=[])
    return raw, dict(summary=summary, frames=details)


def make_fixture(root, corpus="synthetic", *, repeats=2):
    root.mkdir(parents=True, exist_ok=True)
    sequence_name = "SYNTHETIC train-smooth-27181" if corpus=="synthetic" else "HB object 20"
    sequence = dict(objectId=1, name=sequence_name, trials=[])
    summaries, hashes = [], {}
    for system in report.SYSTEMS:
        for repeat in range(repeats):
            raw, scored = trial(system=system, repeat=repeat)
            scored['summary']['name']=sequence_name
            relative=f"object-000001/{system}-r{repeat}.json"
            write(root/relative,raw)
            write(root/relative.replace('.json','-scored.json'),scored)
            sequence['trials'].append(relative);summaries.append(scored['summary'])
            hashes[relative]=report.sha_file(root/relative)
    freeze=dict(schemaVersion=1,benchmarkId="fixture-v1",dataset="test fixture", evidenceKind="test_fixture",
                inferenceSHA256="1"*64,evaluatorSHA256="2"*64, historicalCommit="a"*40,historicalTrackerSHA256="b"*64,
                sourceHashes={"bench/tracker.py":"3"*64},evaluatorOpenedDuringInference=False,
                hardware=dict(python="3.12 fixture",numpy="fixture",opencv="fixture",pillow="fixture",cpuModel="fixture CPU",
                              logicalCPUs=1,memoryTotal="1024 kB",gpuUsed=False,opencvThreads=1,platform="fixture Linux"),
                configuration=dict(systems=list(report.SYSTEMS),repeats=repeats,warmupFrames=20,initialization="fixture depth-backprojected initialization; first 300 features per view",
                                   thresholds=dict(accurateReprojectionPixels=5.)))
    write(root/'freeze.json',freeze)
    seal=dict(**freeze,freezeSHA256=report.sha_file(root/'freeze.json'),trialHashes=hashes,sequences=[sequence],geometryBindings=[])
    write(root/'inference-seal.json',seal)
    results=dict(benchmarkId="fixture-v1",dataset="test fixture",evidenceKind="test_fixture",
                 inferenceSealSHA256=report.sha_file(root/'inference-seal.json'),sequenceTrials=summaries,
                 aggregate={"deliberately_ignored":"reports must derive metrics from actual frame rows"})
    write(root/'results.json',results)
    return freeze, seal, results


class SealingAndBindingTests(unittest.TestCase):
    def test_unsealed_output_cannot_open_any_results(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(report,"read_json",side_effect=AssertionError("must not read")):
            with self.assertRaisesRegex(ValueError,"not sealed"):
                report.load_corpus(temp,"synthetic")

    def test_both_corpora_must_be_sealed_before_any_scores_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'synthetic').mkdir();(root/'synthetic/inference-seal.json').write_text('{}')
            with patch.object(report,"read_json",side_effect=AssertionError("must not read")):
                with self.assertRaisesRegex(ValueError,"not sealed"):
                    report.generate(root/'synthetic',root/'real',root/'report')

    def test_valid_fixture_loads_every_system_and_repeat(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);make_fixture(root)
            loaded=report.load_corpus(root,"synthetic")
            self.assertEqual(len(loaded['entries']),6)
            self.assertEqual(len(loaded['traceIndex']),6)
            self.assertEqual(loaded['entries'][0]['summary']['scenario'],'smooth')

    def test_tampered_trial_rejected_before_scored_result_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,seal,_=make_fixture(root)
            path=root/next(iter(seal['trialHashes']))
            path.write_bytes(path.read_bytes()+b' ')
            opened=[];real_read=report.read_json
            def reading(path):
                opened.append(Path(path).name)
                return real_read(path)
            with patch.object(report,'read_json',side_effect=reading):
                with self.assertRaisesRegex(ValueError,'Raw prediction'):
                    report.load_corpus(root,'synthetic')
            self.assertNotIn('results.json',opened)
            self.assertTrue(all(not name.endswith('-scored.json') for name in opened))

    def test_results_must_bind_exact_inference_seal(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,_,results=make_fixture(root)
            results['inferenceSealSHA256']='f'*64;write(root/'results.json',results)
            with self.assertRaisesRegex(ValueError,'different inference seal'):
                report.load_corpus(root,'synthetic')

    def test_scored_summary_tampering_and_denominator_changes_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,seal,_=make_fixture(root)
            path=root/next(iter(seal['trialHashes'])).replace('.json','-scored.json')
            scored=report.read_json(path);scored['summary']['frames']=999;write(path,scored)
            with self.assertRaisesRegex(ValueError,'summary differs'):
                report.load_corpus(root,'synthetic')

    def test_escaping_trial_paths_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);outside=root/'outside';outside.write_text('outside');data=root/'data';data.mkdir()
            with self.assertRaises(ValueError):
                report.checked_relative(data,'../outside')


class PoolingAndRepeatTests(unittest.TestCase):
    def entry(self,sequence_id,errors,**kwargs):
        raw,scored=trial(sequence_id,errors,**kwargs)
        sequence=dict(objectId=sequence_id,name='SYNTHETIC train-smooth-27181')
        if kwargs.get('failed'):
            sequence['initializationFailure']='init_failure'
        return report.trial_statistics(raw,scored,sequence,'synthetic')

    def test_percentile_matches_linear_interpolation(self):
        self.assertIsNone(report.percentile([],95))
        self.assertEqual(report.percentile([0.,100.],95),95.)
        self.assertEqual(report.percentile([0.,1.,100.],50),1.)

    def test_conditional_errors_pool_frames_not_sequence_medians(self):
        entries=[self.entry(1,(0.,100.)),self.entry(2,(1.,))]
        value=report.aggregate(entries)
        self.assertEqual(value['reprojectionPixelsMedian'],1.)
        self.assertEqual(value['conditionalPublicPoseFrames'],3)
        self.assertAlmostEqual(value['native5pxVisibleCoverage'],2/3)
        self.assertEqual(value['uniqueScheduledObjectFrames'],3)

    def test_initialization_failures_keep_denominators_without_fake_speed(self):
        value=report.aggregate([self.entry(1,(0.,0.),failed=True)])
        self.assertEqual(value['scheduledFramesWithRepeats'],2)
        self.assertEqual(value['skippedUpdateFrames'],2)
        self.assertEqual(value['initializationFailureSequences'],1)
        self.assertEqual(value['native5pxVisibleCoverage'],0.)
        self.assertIsNone(value['updateFPS'])
        self.assertIsNone(value['replayFPS'])
        self.assertIsNone(value['reprojectionPixelsMedian'])

    def test_failed_sequence_cannot_inflate_attempted_update_fps(self):
        value=report.aggregate([self.entry(1,(0.,0.)),self.entry(2,(0.,0.),failed=True)])
        self.assertEqual(value['scheduledFramesWithRepeats'],4)
        self.assertEqual(value['timedUpdateFrames'],2)
        self.assertEqual(value['updateFPS'],100.)
        self.assertEqual(value['replayFPS'],20.)
        self.assertEqual(value['native5pxVisibleCoverage'],.5)

    def test_timing_differences_do_not_change_outcome_digest(self):
        first=self.entry(1,(0.,1.),repeat=0);second=self.entry(1,(0.,1.),repeat=1)
        second['summary']['updateTotalSeconds']*=2
        second['times']=[value*2 for value in second['times']]
        value=report.determinism([first,second])
        self.assertEqual(value['identicalSequenceSystemGroups'],1)
        self.assertEqual(value['changedFramesAgainstFirstRepeat'],0)

    def test_pose_variation_is_detected_across_repeats(self):
        first=self.entry(1,(0.,1.),repeat=0);second=self.entry(1,(0.,2.),repeat=1)
        value=report.determinism([first,second])
        self.assertEqual(value['identicalSequenceSystemGroups'],0)
        self.assertEqual(value['changedFramesAgainstFirstRepeat'],1)
        self.assertGreater(value['maxPoseMatrixElementAbsoluteDifference'],0.)

    def test_exact_synthetic_name_parser_preserves_scenario_and_split(self):
        self.assertEqual(report.scenario_from_name('SYNTHETIC test-full_occlusion-123','synthetic'),('full_occlusion','test'))
        self.assertEqual(report.scenario_from_name('HB object 20','real'),(None,None))
        for bad in ('SYNTHETIC train-full_occlusion_extra-123','train-full_occlusion-123','SYNTHETIC nope-smooth-123'):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                report.scenario_from_name(bad,'synthetic')

    def test_inconsistent_5px_flag_rejected(self):
        raw,scored=trial(errors=(10.,));scored['frames'][0]['accurate']=True;scored['summary']['accurateVisibleFrames']=1
        with self.assertRaisesRegex(ValueError,'5 px coverage flag'):
            report.trial_statistics(raw,scored,dict(objectId=1,name='SYNTHETIC train-smooth-27181'),'synthetic')


class ArtifactTests(unittest.TestCase):
    def test_complete_fixture_report_json_csv_markdown_and_plots(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);make_fixture(root/'synthetic');make_fixture(root/'real','real')
            value=report.generate(root/'synthetic',root/'real',root/'published')
            self.assertEqual(len(value['aggregate']),6)
            text=(root/'published'/f'tracking-results-{report.DATE}.md').read_text()
            self.assertIn('Neural GoTrack/adapted-head tracking quality is not measured',text)
            self.assertIn('SHARED SIMPLIFIED INITIALIZATION',text)
            self.assertTrue(all(row['initializationMode']==report.SIMPLIFIED_MODE for row in value['aggregate']))
            self.assertIn('Native `<5 px` coverage',text)
            self.assertIn('Conditional pose errors',text)
            from PIL import Image
            for name in ('coverage','speed'):
                with Image.open(root/'published'/f'tracking-{name}-{report.DATE}.png') as image:
                    self.assertEqual(image.size,(1050,570))
            self.assertGreater((root/'published'/f'tracking-sequences-{report.DATE}.csv').stat().st_size,200)


class PairedTradeoffTests(unittest.TestCase):
    def entry(self,system,errors,withheld=(),repeat=0,recovery=None):
        raw,scored=trial(errors=errors,system=system,repeat=repeat)
        for index in withheld:
            raw['frames'][index].update(state='lost',failureReason='withheld',cvCameraFromSourceObject=None)
            scored['frames'][index].update(public=False,accurate=False)
            for metric in report.ERRORS:
                scored['frames'][index].pop(metric,None)
        summary=scored['summary'];details=scored['frames']
        summary['publicTrackingFrames']=summary['visibleTrackingFrames']=sum(row['public'] for row in details)
        summary['accurateVisibleFrames']=sum(row['accurate'] for row in details)
        summary['recovery']=recovery or []
        return report.trial_statistics(raw,scored,dict(objectId=1,name='SYNTHETIC train-smooth-27181'),'synthetic')

    def test_shared_improvements_withheld_good_and_new_wrong_balance_exactly(self):
        old=self.entry('original_rigid',(10.,1.,1.,10.,0.,0.,0.),withheld=(4,5,6))
        new=self.entry('current_prototype',(2.,10.,0.,0.,1.,10.,0.),withheld=(2,3,6))
        value=report.paired_tradeoff([old,new],'original_rigid','current_prototype')
        self.assertEqual(value['visibleUniqueObjectFrames'],7)
        self.assertEqual(value['bothPublicVisibleFrames'],2)
        self.assertEqual(value['withheldBaselineQualifiedFrames'],1)
        self.assertEqual(value['withheldBaselineWrongFrames'],1)
        self.assertEqual(value['addedQualifiedWhereBaselineWithheld'],1)
        self.assertEqual(value['addedWrongWhereBaselineWithheld'],1)
        self.assertEqual(value['bothWithheldVisibleFrames'],1)
        self.assertEqual(value['qualifiedVisibleDelta'],0)
        self.assertEqual(value['wrongVisiblePublicDelta'],0)

    def test_better_shared_errors_can_coexist_with_worse_whole_stream_coverage(self):
        old=self.entry('original_rigid',(1.,1.,10.))
        new=self.entry('current_prototype',(0.,0.,2.),withheld=(0,1))
        value=report.paired_tradeoff([old,new],'original_rigid','current_prototype')
        self.assertEqual(value['sharedQualifiedDelta'],1)
        self.assertEqual(value['withheldBaselineQualifiedFrames'],2)
        self.assertEqual(value['qualifiedVisibleDelta'],-1)
        self.assertEqual(value['wrongVisiblePublicDelta'],-1)
        self.assertLess(value['sharedCandidateReprojectionP95'],value['sharedBaselineReprojectionP95'])

    def test_unique_repeat_zero_only_and_paired_recovery_slowdown(self):
        old=self.entry('original_rigid',(1.,),recovery=[dict(returnFrameId=20,observableFramesToTwoAccurate=4)])
        new=self.entry('current_prototype',(1.,),recovery=[dict(returnFrameId=20,observableFramesToTwoAccurate=41)])
        other=[self.entry('original_rigid',(1.,),repeat=1),self.entry('current_prototype',(100.,),repeat=1)]
        value=report.paired_tradeoff([old,new]+other,'original_rigid','current_prototype')
        self.assertEqual(value['scheduledUniqueObjectFrames'],1)
        self.assertEqual(value['qualifiedVisibleDelta'],0)
        self.assertEqual(value['pairedRecovery'][0]['addedObservableFrames'],37)

    def test_candidate_recovery_failure_and_mismatched_sequence_sets(self):
        old=self.entry('original_rigid',(1.,),recovery=[dict(returnFrameId=20,observableFramesToTwoAccurate=4)])
        new=self.entry('current_prototype',(1.,),recovery=[dict(returnFrameId=20,observableFramesToTwoAccurate=None)])
        value=report.paired_tradeoff([old,new],'original_rigid','current_prototype')
        self.assertEqual(value['candidateRecoveryFailuresAfterBaselineSuccess'],1)
        with self.assertRaisesRegex(ValueError,'same sequence set'):
            report.paired_tradeoff([old],'original_rigid','current_prototype')


class InitializationDiagnosticTests(unittest.TestCase):
    def retag_production(self,root,source_conflict=False):
        freeze=report.read_json(root/'freeze.json')
        freeze['configuration']['initializationMode']='production_mesh'
        freeze['configuration']['initialization']='exact production mesh-ray initialization; ORB nfeatures=2000'
        if source_conflict:
            freeze['sourceHashes']['bench/tracker.py']='f'*64
        write(root/'freeze.json',freeze)
        seal=report.read_json(root/'inference-seal.json');seal.update(freeze);seal['freezeSHA256']=report.sha_file(root/'freeze.json');write(root/'inference-seal.json',seal)
        results=report.read_json(root/'results.json');results['inferenceSealSHA256']=report.sha_file(root/'inference-seal.json');write(root/'results.json',results)

    def test_separate_prefix_keeps_its_production_mode_and_does_not_pool_full_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);make_fixture(root/'simple','real',repeats=1);make_fixture(root/'production','real',repeats=1)
            self.retag_production(root/'production')
            diagnostic=report.initialization_diagnostic(root/'simple',root/'production')
            self.assertEqual(len(diagnostic['aggregate']),6)
            self.assertEqual({row['initializationMode'] for row in diagnostic['aggregate']},{report.SIMPLIFIED_MODE,report.PRODUCTION_MODE})
            self.assertTrue(all(row['initializationMode']==report.PRODUCTION_MODE for row in diagnostic['productionPairedTradeoffs']))

    def test_prefix_comparison_rejects_extra_source_confound(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);make_fixture(root/'simple','real',repeats=1);make_fixture(root/'production','real',repeats=1)
            self.retag_production(root/'production',source_conflict=True)
            with self.assertRaisesRegex(ValueError,'source/data confound'):
                report.initialization_diagnostic(root/'simple',root/'production')


if __name__=='__main__':
    unittest.main()
