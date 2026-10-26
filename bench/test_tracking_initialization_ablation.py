"""Mode dispatch and label-bound prefix tests; no real tracker inference."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import tracking_benchmark as benchmark
from .prototype_benchmark import digest, write_json
from . import test_tracking_benchmark as fixtures

POINTS=fixtures.POINTS
prediction=fixtures.prediction


class InitializationAblationTests(unittest.TestCase):
    def fixture(self, dataset):
        return fixtures.InferenceIsolationTests().fixture(dataset)

    def test_invalid_mode_and_nonpositive_prefix_rejected_before_opening_data(self):
        for kwargs in (dict(initialization='unknown'),dict(max_scored_frames=0),dict(max_scored_frames=-1)):
            with tempfile.TemporaryDirectory() as temp,self.subTest(kwargs=kwargs),self.assertRaisesRegex(ValueError,'Invalid benchmark configuration'):
                benchmark.run_inference(Path(temp)/'not-read',Path(temp)/'out',**kwargs)

    def test_production_calls_original_mesh_builder_without_budget_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dataset=root/'data';dataset.mkdir();value,bank=self.fixture(dataset)
            bank=(np.resize(bank[0],(4001,3)),np.resize(bank[1],(4001,32)),bank[2],bank[3],bank[4])
            original_mesh=object();transform=np.eye(4);scan=SimpleNamespace(original=original_mesh)
            with patch.object(benchmark,'read_ply',return_value=(scan,POINTS,transform)), \
                 patch.object(benchmark,'build_bank',return_value=bank) as mesh_builder, \
                 patch.object(benchmark,'build_depth_bank') as depth_builder, \
                 patch.object(benchmark,'load_rgb',return_value=np.zeros((8,8,3),dtype=np.uint8)), \
                 patch.object(benchmark,'make_tracker',return_value=object()), \
                 patch.object(benchmark,'update_tracker',side_effect=lambda s,t,r,*a:prediction(r['frameId'])):
                benchmark.run_inference(dataset,root/'out',systems=['current_rigid'],repeats=1,warmup=0,
                                        initialization='production_mesh',max_scored_frames=1)
            mesh_builder.assert_called_once();depth_builder.assert_not_called()
            args=mesh_builder.call_args.args
            self.assertEqual(args[0],dataset)
            self.assertEqual(args[1],value['objects'][0]['onboarding'])
            self.assertIs(args[2],original_mesh);self.assertIs(args[3],transform)
            seal=benchmark.json_read(root/'out'/'inference-seal.json')
            self.assertEqual(seal['configuration']['initializationMode'],'production_mesh')
            self.assertEqual(seal['sequences'][0]['anchors'],4001)
            self.assertEqual(seal['sequences'][0]['seedViewAnchors'],16)

    def test_depth_mode_dispatches_only_budget_builder(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dataset=root/'data';dataset.mkdir();_,bank=self.fixture(dataset)
            with patch.object(benchmark,'read_ply',return_value=(None,POINTS,None)), \
                 patch.object(benchmark,'build_bank') as mesh_builder, \
                 patch.object(benchmark,'build_depth_bank',return_value=bank) as depth_builder, \
                 patch.object(benchmark,'load_rgb',return_value=np.zeros((8,8,3),dtype=np.uint8)), \
                 patch.object(benchmark,'make_tracker',return_value=object()), \
                 patch.object(benchmark,'update_tracker',side_effect=lambda s,t,r,*a:prediction(r['frameId'])):
                benchmark.run_inference(dataset,root/'out',systems=['current_rigid'],repeats=1,warmup=0,
                                        initialization='depth_budget',max_scored_frames=1)
            depth_builder.assert_called_once();mesh_builder.assert_not_called()

    def test_prefix_is_bound_evaluated_and_does_not_mutate_source_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dataset=root/'data';dataset.mkdir();value,bank=self.fixture(dataset)
            before=(dataset/'inference.json').read_bytes();updates=[];states=[object(),object()]
            def update(s,t,r,*a):updates.append((t,r['frameId']));return prediction(r['frameId'])
            with patch.object(benchmark,'read_ply',return_value=(None,POINTS,None)), \
                 patch.object(benchmark,'build_depth_bank',return_value=bank), \
                 patch.object(benchmark,'load_rgb',return_value=np.zeros((8,8,3),dtype=np.uint8)), \
                 patch.object(benchmark,'make_tracker',side_effect=states), \
                 patch.object(benchmark,'update_tracker',side_effect=update):
                benchmark.run_inference(dataset,root/'out',systems=['current_rigid'],repeats=1,warmup=20,
                                        max_scored_frames=1)
                benchmark.run_evaluation(dataset,root/'out')
            self.assertEqual((dataset/'inference.json').read_bytes(),before)
            self.assertEqual([r[1] for r in updates],[10,10])
            self.assertIsNot(updates[0][0],updates[1][0])
            seal=benchmark.json_read(root/'out'/'inference-seal.json')
            self.assertEqual(seal['configuration']['maxScoredFrames'],1)
            self.assertEqual([r['frameId'] for r in seal['geometryBindings'][0]['frames']],[10])
            results=benchmark.json_read(root/'out'/'results.json')
            self.assertEqual(results['aggregate']['current_rigid']['uniqueFrames'],1)
            self.assertEqual(results['sequenceTrials'][0]['frames'],1)
            self.assertEqual(len(value['objects'][0]['frames']),2)

    def test_prefix_evaluator_still_rejects_mismatched_first_frame_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dataset=root/'data';dataset.mkdir();value,bank=self.fixture(dataset)
            evaluator=benchmark.json_read(dataset/'evaluator.json')
            evaluator['objects'][0]['frames'][0]['calibration']['cam_K'][0]+=1
            write_json(dataset/'evaluator.json',evaluator)
            value['evaluatorManifestSHA256']=digest(dataset/'evaluator.json');write_json(dataset/'inference.json',value)
            with patch.object(benchmark,'read_ply',return_value=(None,POINTS,None)), \
                 patch.object(benchmark,'build_depth_bank',return_value=bank), \
                 patch.object(benchmark,'load_rgb',return_value=np.zeros((8,8,3),dtype=np.uint8)), \
                 patch.object(benchmark,'make_tracker',return_value=object()), \
                 patch.object(benchmark,'update_tracker',side_effect=lambda s,t,r,*a:prediction(r['frameId'])):
                benchmark.run_inference(dataset,root/'out',systems=['current_rigid'],repeats=1,warmup=0,
                                        max_scored_frames=1)
                with self.assertRaisesRegex(ValueError,'Evaluator geometry/RGB/calibration binding differs'):
                    benchmark.run_evaluation(dataset,root/'out')

    def test_insufficient_production_bank_retains_all_prefix_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);dataset=root/'data';dataset.mkdir();_,bank=self.fixture(dataset)
            bank=tuple(a[:11] if i in (0,1,3,4) else a for i,a in enumerate(bank))
            with patch.object(benchmark,'read_ply',return_value=(SimpleNamespace(original=object()),POINTS,np.eye(4))), \
                 patch.object(benchmark,'build_bank',return_value=bank), \
                 patch.object(benchmark,'update_tracker') as update:
                benchmark.run_inference(dataset,root/'out',systems=['current_rigid'],repeats=1,warmup=0,
                                        initialization='production_mesh',max_scored_frames=1)
                benchmark.run_evaluation(dataset,root/'out')
            update.assert_not_called()
            result=benchmark.json_read(root/'out'/'results.json')['aggregate']['current_rigid']
            self.assertEqual(result['uniqueFrames'],1);self.assertEqual(result['skippedUpdateFrames'],1)
            self.assertEqual(result['initializationFailureSequences'],1)
            self.assertEqual(result['accurateVisibleAvailability'],0.)
            self.assertIsNone(result['updateFramesPerSecond'])

    def test_added_source_closure_names_include_mesh_dependencies(self):
        self.assertTrue({'bench/renderer.py','bench/model.py','bench/obj_model.py'}.issubset(benchmark.source_hashes()))


if __name__=='__main__':unittest.main()
