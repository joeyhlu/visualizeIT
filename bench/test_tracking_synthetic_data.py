"""Contracts for synthetic benchmark geometry, labels and reproducibility."""
import gzip
import json
from pathlib import Path
import tempfile
import tarfile
import unittest

import numpy as np
from PIL import Image

from . import tracking_synthetic_data as data


class SyntheticTrackingDataTests(unittest.TestCase):
    def test_pose_is_finite_proper_cv_metres_and_nonplanar(self):
        for seed in data.SPLIT_SEEDS:
            mesh, _ = data.mesh_for(seed)
            self.assertEqual(mesh.positions.shape, (54, 3))
            self.assertEqual(mesh.triangles.shape, (48, 3))
            self.assertGreater(np.linalg.svd(mesh.positions-mesh.positions.mean(0), compute_uv=False)[-1], .1)
            for scenario in data.SCENARIOS:
                for frame in (0,9,10,80,199):
                    rotation,translation=data.pose_at(frame,200,scenario,seed)
                    np.testing.assert_allclose(rotation@rotation.T,np.eye(3),atol=1e-12)
                    self.assertAlmostEqual(np.linalg.det(rotation),1.,places=12)
                    self.assertGreater((mesh.positions@rotation.T+translation)[:,2].min(),.03)

    def test_stress_events_do_not_pollute_onboarding(self):
        for scenario in data.SCENARIOS:
            for index in range(10):
                self.assertIsNone(data.occluder(index,200,scenario))
        self.assertEqual(data.occluder(80,200,'full_occlusion'),(0,0,320,240))
        self.assertIsNotNone(data.occluder(80,200,'partial_occlusion'))
        self.assertIsNone(data.occluder(199,200,'full_occlusion'))
        r,t=data.pose_at(100,200,'exit_reentry',data.SPLIT_SEEDS[0])
        self.assertGreater(t[0],.4)

    def test_generation_labels_isolated_hashed_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            first=data.generate(root/'first',frame_count=20,sequence_limit=1,artifact_dir=root/'artifacts')
            second=data.generate(root/'second',frame_count=20,sequence_limit=1)
            self.assertEqual(first['inferenceSHA256'],second['inferenceSHA256'])
            self.assertEqual(first['evaluatorSHA256'],second['evaluatorSHA256'])
            inference=json.loads((root/'first/inference.json').read_text())
            evaluator=json.loads((root/'first/evaluator.json').read_text())
            self.assertEqual(inference['evidenceKind'],'synthetic_stress')
            self.assertEqual(first['scoredFrames'],10)
            obj=inference['objects'][0]
            self.assertEqual(len(obj['onboarding']),10)
            for row in obj['onboarding']:
                self.assertTrue({'evaluationPose','depth','onboardingMask'}<=set(row))
            forbidden={'evaluationPose','depth','evaluationMask','evaluationInfo','stress'}
            for row in obj['frames']:
                self.assertFalse(forbidden&set(row))
                path=root/'first'/row['rgb']['path']
                self.assertEqual(data.digest(path),row['rgb']['sha256'])
            for row in evaluator['objects'][0]['frames']:
                self.assertTrue(forbidden<=set(row))
                mask=np.asarray(Image.open(root/'first'/row['evaluationMask']['path']))>0
                depth=np.asarray(Image.open(root/'first'/row['depth']['path']))
                self.assertEqual(mask.shape,(240,320))
                self.assertTrue((depth[~mask]==0).all())
                self.assertTrue(0<=row['evaluationInfo']['visib_fract']<=1)
            annotations=[json.loads(x) for x in gzip.decompress((root/'artifacts/synthetic-annotations.jsonl.gz').read_bytes()).splitlines()]
            self.assertEqual(len(annotations),20)
            self.assertEqual(sum(x['phase']=='evaluation' for x in annotations),10)
            self.assertEqual(data.verify_dataset(root/'first')['verifiedResources'],61)
            data.generate(root/'first',frame_count=20,sequence_limit=1,artifact_dir=root/'reexport')
            self.assertEqual(data.digest(root/'artifacts/synthetic-annotations.jsonl.gz'),
                             data.digest(root/'reexport/synthetic-annotations.jsonl.gz'))
            with self.assertRaisesRegex(ValueError,'outside dataset root'):
                data.package_dataset(root/'first',root/'first/self.tar.gz')
            packed=data.package_dataset(root/'first',root/'first.tar.gz')
            again=data.package_dataset(root/'first',root/'second.tar.gz')
            self.assertEqual(packed['sha256'],again['sha256'])
            with tarfile.open(root/'first.tar.gz') as bundle:
                self.assertIn('synthetic-v1/LICENSE.txt',bundle.getnames())
                self.assertIn('synthetic-v1/README.md',bundle.getnames())
                self.assertTrue(all(p.startswith('synthetic-v1/') for p in bundle.getnames()))
            with self.assertRaises(ValueError):
                data.generate(root/'first',frame_count=21,sequence_limit=1)
            damaged=root/'first'/obj['frames'][0]['rgb']['path']
            damaged.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Media/model hash mismatch'):
                data.verify_dataset(root/'first')

    def test_invalid_frame_counts_are_rejected(self):
        for count in (0,19,1001,22.5):
            with self.assertRaises(ValueError):
                data.config(count)
        for count in (0,25,1.5):
            with self.assertRaises(ValueError):
                data.config(200,count)


if __name__=='__main__':
    unittest.main()
