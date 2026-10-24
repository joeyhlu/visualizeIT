"""Failure-path tests for persistent RGBA evidence, without model execution."""
from pathlib import Path
import tempfile,unittest
from unittest import mock
import numpy as np
from bench import workbench_image_io as imageio
from bench.vision import cv2

SCRATCH=Path(__file__).resolve().parents[1]/'.cache/video-workbench/scratch'

class ImageEvidenceTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='image-io-test-',dir=SCRATCH)
        self.folder=Path(self.temp.name).resolve()
        self.addCleanup(self.cleanup)
        self.image=np.zeros((9,13,4),np.uint8)
        self.image[2:7,3:10]=[227,19,63,255]
        self.image[0,0]=[7,11,29,128]
    def cleanup(self):
        assert self.folder.is_relative_to(SCRATCH.resolve())
        self.temp.cleanup()
    def test_exact_color_and_alpha_roundtrip(self):
        path=self.folder/'rgba.png';record=imageio.write_png_rgba(path,self.image)
        self.assertTrue(path.is_file());self.assertGreater(record['bytes'],0)
        np.testing.assert_array_equal(imageio.decode_png_rgba(path.read_bytes()),self.image)
        self.assertEqual(list(self.folder.iterdir()),[path])
    def test_encoder_false_is_explicit_and_no_path_exists(self):
        path=self.folder/'absent.png'
        with mock.patch.object(cv2,'imencode',return_value=(False,None)):
            with self.assertRaises(imageio.ImageWriteError):imageio.write_png_rgba(path,self.image)
        self.assertFalse(path.exists());self.assertEqual(list(self.folder.iterdir()),[])
    def test_disk_write_error_and_zero_write_leave_no_registered_file(self):
        for result in ['error','zero']:
            with self.subTest(result=result):
                patch=mock.patch.object(imageio.os,'write',side_effect=OSError('manufactured ENOSPC')) if result=='error' else mock.patch.object(imageio.os,'write',return_value=0)
                with patch,self.assertRaises(imageio.ImageWriteError):imageio.write_png_rgba(self.folder/'failed.png',self.image)
                self.assertEqual(list(self.folder.iterdir()),[])
    def test_partial_os_writes_are_completed_and_verified(self):
        original=imageio.os.write
        with mock.patch.object(imageio.os,'write',side_effect=lambda fd,data:original(fd,data[:7])):
            imageio.write_png_rgba(self.folder/'partial.png',self.image)
        np.testing.assert_array_equal(imageio.decode_png_rgba((self.folder/'partial.png').read_bytes()),self.image)
    def test_existing_evidence_is_never_overwritten(self):
        path=self.folder/'existing.png';path.write_bytes(b'preserved original')
        with self.assertRaises(imageio.ImageWriteError):imageio.write_png_rgba(path,self.image)
        self.assertEqual(path.read_bytes(),b'preserved original')
    def test_publication_failure_cleans_only_owned_temp(self):
        keep=self.folder/'keep.txt';keep.write_text('untouched')
        with mock.patch.object(imageio.os,'link',side_effect=PermissionError('manufactured denial')):
            with self.assertRaises(imageio.ImageWriteError):imageio.write_png_rgba(self.folder/'failed.png',self.image)
        self.assertEqual(list(self.folder.iterdir()),[keep]);self.assertEqual(keep.read_text(),'untouched')
    def test_wrong_dtype_shape_empty_and_missing_channel_are_rejected(self):
        for arr in [self.image.astype(float),self.image[:,:,:3],self.image[:0],[[1]]]:
            with self.subTest(shape=getattr(arr,'shape',None)),self.assertRaises(imageio.ImageWriteError):imageio.encode_png_rgba(arr)
    def test_wrong_dimensions_and_non_png_are_rejected(self):
        raw=imageio.encode_png_rgba(self.image)
        with self.assertRaises(imageio.ImageWriteError):imageio.decode_png_rgba(raw,expected_shape=(3,4))
        with self.assertRaises(imageio.ImageWriteError):imageio.decode_png_rgba(b'not a PNG')
    def test_output_cap_checked_before_temp_creation(self):
        with self.assertRaises(imageio.ImageWriteError):imageio.write_png_rgba(self.folder/'over.png',self.image,max_bytes=1)
        self.assertEqual(list(self.folder.iterdir()),[])
    def test_temp_readback_corruption_is_not_published(self):
        read=Path.read_bytes
        def corrupt(path):return b'corrupted' if path.suffix=='.tmp' else read(path)
        with mock.patch.object(Path,'read_bytes',corrupt),self.assertRaises(imageio.ImageWriteError):imageio.write_png_rgba(self.folder/'bad.png',self.image)
        self.assertEqual(list(self.folder.iterdir()),[])

    def test_failed_final_read_or_corruption_removes_only_owned_publication(self):
        original = Path.read_bytes
        for mode in ['unreadable', 'corrupt']:
            path = self.folder / (mode + '.json')
            def faulty(candidate):
                if candidate == path:
                    if mode == 'unreadable': raise OSError('manufactured final read failure')
                    return b'corrupt'
                return original(candidate)
            with self.subTest(mode=mode), mock.patch.object(Path, 'read_bytes', faulty):
                with self.assertRaises(imageio.ImageWriteError):
                    imageio.publish_new_bytes(path, b'{"status":"completed_evidence_recovery"}')
            self.assertFalse(path.exists())
            self.assertEqual(list(self.folder.iterdir()), [])

    def test_external_replacement_is_preserved_on_final_read_failure(self):
        path = self.folder / 'report.json'; original = Path.read_bytes
        def replace(candidate):
            if candidate == path:
                path.unlink(); path.write_bytes(b'external replacement')
                raise OSError('manufactured concurrent replacement')
            return original(candidate)
        with mock.patch.object(Path, 'read_bytes', replace), self.assertRaises(imageio.ImageWriteError):
            imageio.publish_new_bytes(path, b'owned report')
        self.assertEqual(path.read_bytes(), b'external replacement')
        self.assertEqual(list(self.folder.iterdir()), [path])

    def test_neural_overlay_failure_preserves_all_computed_rows_and_blocks_cache(self):
        # Manufactured backend and pixels exercise the real Pipeline/NeuralSession
        # control flow; no tracking model, CUDA, media decoder or geometry is used.
        from contextlib import nullcontext
        from types import SimpleNamespace
        from bench import workbench_cli as cli,workbench_contract as wc,workbench_neural as wn,workbench_cache as cache,workbench_geometry as geometry
        from bench.test_workbench_contract import clip_document
        from bench.test_workbench_neural import FakeBackend
        clip=wc.ClipSpec.from_dict(clip_document())
        ws=cli.Workspace(self.folder/'workspace');ws.init()
        run=cli.build_run_spec(clip,mode='smoke',roles=('primary_neural',))
        admission=wn.ServingAdmission(True,(),{'families':['fixture'],'contract_sha256':'c'*64})
        pipe=cli.Pipeline(ws,clip,run,cli.RunOptions(admission=admission,metric_bundle='manufactured',head_path='manufactured',device='cpu'),progress=lambda _:None)
        pipe.run_dir.mkdir(parents=True)
        rgb=np.zeros((clip.height,clip.width,3),np.uint8);mask=np.ones(rgb.shape[:2],bool)
        rgba=np.zeros((*mask.shape,4),np.uint8);rgba[0,0]=[20,30,40,255]
        artwork=SimpleNamespace(sha256='a'*64,render=lambda *a:{'rgba':rgba,'stats':{'rendered_pixels':1}})
        seed_pose=np.eye(4);seed_pose[2,3]=1
        bundle=wn.MetricBundle('fixture',Path('NEVER-READ-manufactured.glb'),'d'*64,1,
                              'integer_centers',seed_pose,'onboarding:fixture','f1','dataset_metric_cad')
        plan={'bundle':bundle,'seed_target':run.scheduled_frame_ids[0],'warmup':[],'artwork':artwork,'artwork_reason':None}
        reader=SimpleNamespace(read=lambda frame:(rgb,{'frame_id':frame.frame_id,
                              'source_frame_index':frame.source_frame_index,
                              'decoder_media_time_s':frame.media_time_s,'image_sha256':geometry.array_sha256(rgb)}))
        masks=SimpleNamespace(load=lambda fid,shape:(mask,{'mask_frame_id':fid,'mask_sha256':geometry.mask_sha256(mask)}))
        store=mock.Mock();store.lookup.return_value=None;backend=FakeBackend()
        backend.bundle=bundle;backend.seed_target_frame_id=run.scheduled_frame_ids[0]
        with mock.patch.object(wn,'neural_preflight',return_value={}),mock.patch.object(pipe,'_neural_plan',return_value=plan),\
             mock.patch.object(pipe,'_input_identity',return_value={}),mock.patch.object(cli,'sha256_file',return_value=wn.PRIMARY['export_sha256']),\
             mock.patch.object(wn,'GoTrackMetricBackend',return_value=backend),mock.patch('bench.workbench_telemetry.gpu_stage',side_effect=lambda *a:nullcontext()),\
             mock.patch.object(imageio,'write_png_rgba',side_effect=imageio.ImageWriteError('manufactured encoder failure')),\
             mock.patch.object(pipe,'_cache_store') as commit:
            pipe._neural('primary_neural',reader,masks,(cache,store))
        self.assertEqual(len(pipe.results),len(run.scheduled_frame_ids));self.assertEqual(backend.calls,len(pipe.results))
        self.assertTrue(all(r.pose_state=='tracking' and r.pose is not None and r.render_state=='rendered' for r in pipe.results))
        self.assertTrue(all('overlay' not in r.evidence and 'artifact_write_error' in r.evidence for r in pipe.results))
        self.assertEqual(pipe.stages['primary_neural']['status'],'completed')
        self.assertEqual(pipe.stages['primary_neural_artifacts']['status'],'failed');commit.assert_not_called()
        self.assertEqual(pipe._finish()['status'],'failed')

if __name__=='__main__':unittest.main()
