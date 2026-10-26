"""Model-free checks for native recorded SAM/pose dataset registration."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from . import tracking_offline_dataset as data
from . import workbench_contract as wc
from .workbench_neural import glb_mesh,MetricBundle
from .workbench_cli import WorkbenchCLIError


def ply(path):
    path.write_text('''ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
property uchar alpha
element face 2
property list uchar int vertex_indices
end_header
0 0 0 0 0 1 255 0 0 255
1000 0 0 0 0 1 0 255 0 255
1000 1000 0 0 0 1 0 0 255 255
0 1000 0 0 0 1 255 255 255 255
3 0 1 2
3 0 2 3
''')


def records():
    k=[537.4799,0,318.8965,0,536.1447,238.3781,0,0,1]
    def row(i):
        return dict(frameId=i,calibration=dict(cam_K=k,depth_scale=1),rgb=dict(path=f'{i:06d}.png',sha256='a'*64))
    seed=row(9);seed['evaluationPose']=dict(cam_R_m2c=np.eye(3).ravel().tolist(),cam_t_m2c=[100,-200,900])
    return dict(objectId=20,sceneId='000001',frames=[row(i) for i in range(10,310)]),seed


class GeometryAndPromptTests(unittest.TestCase):
    def test_ply_to_glb_preserves_source_axes_topology_metric_scale_and_colors(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ply(root/'source.ply');audit=data.convert_ply_to_glb(root/'source.ply',root/'mesh.glb')
            vertices,faces=glb_mesh(root/'mesh.glb')
            np.testing.assert_array_equal(vertices,[[0,0,0],[1,0,0],[1,1,0],[0,1,0]])
            np.testing.assert_array_equal(faces,[[0,1,2],[0,2,3]])
            self.assertEqual(audit['scale'],.001);self.assertTrue(audit['sourceVertexColorsPreserved'])
            data.convert_ply_to_glb(root/'source.ply',root/'second.glb');self.assertEqual(data.digest(root/'mesh.glb'),data.digest(root/'second.glb'))

    def test_seed_translation_is_metres_without_axis_remapping(self):
        _,seed=records();pose=data.source_pose(seed)
        np.testing.assert_array_equal(np.asarray(pose)[:3,:3],np.eye(3))
        np.testing.assert_allclose(np.asarray(pose)[:3,3],[.1,-.2,.9])

    def test_prompt_positive_inside_authorized_mask_negative_outside(self):
        mask=np.zeros((480,640),np.uint8);mask[100:180,240:300]=255
        positive,negative=data.prompt_points(mask)
        self.assertTrue(mask[positive[0][1],positive[0][0]])
        self.assertFalse(mask[negative[0][1],negative[0][0]])
        with self.assertRaises(ValueError):data.prompt_points(np.zeros((480,640),np.uint8))


class ClipProtocolTests(unittest.TestCase):
    def test_native_ids_source_order_clock_and_scored_label_isolation(self):
        obj,seed=records();clip=data.build_clip(obj,seed,'unit_fixture_reviewer')
        selection=wc.Selection('f000009',((270,140),),((16,16),),True,'unit_fixture_reviewer')
        clip=wc.ClipSpec.from_dict({**clip.to_dict(),'onboarding':wc.Onboarding(('f000009',),selection).to_dict()})
        self.assertEqual(clip.frames[0].frame_id,'f000009');self.assertEqual(clip.frames[1].source_frame_index,10)
        self.assertEqual(clip.scoring_frame_ids[-1],'f000309');self.assertEqual(len(clip.scoring_frame_ids),300)
        self.assertEqual(clip.clock_kind,'frame_index');self.assertEqual(clip.intrinsics.status,'dataset_reference')
        self.assertTrue(all(frame.timestamp_s is None and frame.media_time_s is None for frame in clip.frames))
        for frame in clip.to_dict()['frames']:
            self.assertFalse({'pose','evaluationPose','depth','mask','evaluationMask','visibility'}&set(frame))
        self.assertFalse(wc.check_onboarding(clip,clip.onboarding,clip.scoring_frame_ids)['prefix_bound_attested'])

    def test_static_clip_rejects_hidden_varying_calibration(self):
        obj,seed=records();obj['frames'][2]['calibration']['cam_K']=list(seed['calibration']['cam_K']);obj['frames'][2]['calibration']['cam_K'][0]+=1
        with self.assertRaisesRegex(ValueError,'varying calibration'):data.build_clip(obj,seed,'unit')

    def test_case_preparation_must_not_forge_observed_sam_mask(self):
        template=dict(schema='offline_sam_observed_case_template_v1',status='blocked_until_actual_SAM_mask',cases=[])
        with self.assertRaises(ValueError):data.validate_case_template(template,Path('/unused'))
        template['cases']=[dict(case_id='c',family='f',rgb_path='rgb.png',rgb_sha256='a'*64,mask_path='gt-mask.png',mask_sha256='b'*64,
            segmentation=dict(producer='ground_truth',manifest_sha256='c'*64),K=records()[1]['calibration']['cam_K'],principal_point_origin='integer_centers',
            initial_pose_m=np.eye(4).tolist(),initialization_source='onboarding:seed',glb_path='mesh.glb',glb_sha256='d'*64,obj_id=20,metric_provenance='dataset_metric_cad')]
        with self.assertRaisesRegex(ValueError,'counterfeit'):data.validate_case_template(template,Path('/unused'))

    def test_wrong_source_manifest_is_rejected_before_preparation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'source').mkdir();(root/'source/inference.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'exact pinned'):data.prepare(root/'source',root/'output',root/'artifacts')


class LosslessInputTests(unittest.TestCase):
    def test_lossless_runtime_representation_binds_encoded_and_pixel_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);array=np.zeros((48,64,3),np.uint8);array[::2]=[23,140,250]
            Image.fromarray(array).save(root/'source.png',compress_level=0)
            result=data.lossless_frame(root/'source.png',root/'runtime.png')
            self.assertTrue(result['exact_uint8_rgb_equal']);self.assertLess(result['runtime_encoded_bytes'],result['original_encoded_bytes'])
            self.assertNotEqual(result['runtime_encoded_sha256'],result['original_encoded_sha256'])
            np.testing.assert_array_equal(np.asarray(Image.open(root/'runtime.png')),array)
            self.assertEqual(result,data.lossless_frame(root/'source.png',root/'runtime.png'))

    def test_existing_runtime_pixel_change_is_rejected_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);Image.fromarray(np.zeros((4,4,3),np.uint8)).save(root/'source.png');Image.fromarray(np.ones((4,4,3),np.uint8)).save(root/'runtime.png')
            with self.assertRaisesRegex(ValueError,'changed decoded'):data.lossless_frame(root/'source.png',root/'runtime.png')


if __name__=='__main__':unittest.main()
