"""Independent PowerShell launch-plan checks: these never start the service."""
from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'tools/run-workbench.ps1'
PWSH=shutil.which('pwsh')

@unittest.skipUnless(PWSH,'PowerShell is unavailable')
class LauncherTests(unittest.TestCase):
    def run_plan(self,*args,script=SCRIPT):
        return subprocess.run([PWSH,'-NoProfile','-File',str(script),'-Plan',*args],capture_output=True,text=True,timeout=30)

    def test_reviewed_preset_preserves_exact_six_case_configuration(self):
        result=self.run_plan('-ReviewedSixCase');self.assertEqual(result.returncode,0,result.stderr)
        plan=json.loads(result.stdout);self.assertEqual(plan['mode'],'plan')
        command=plan['commands'][0];args=command['arguments']
        self.assertIn('quality-windows',command['executable'])
        self.assertEqual(args.count('--metric-bundle-for'),3)
        self.assertEqual(args.count('--execution-control-for'),3)
        self.assertEqual(args.count('--masks-for'),6)
        self.assertEqual(args[args.index('--host')+1],'127.0.0.1')
        self.assertEqual(args[args.index('--device')+1],'cuda')
        self.assertIn('attempt-08',args[args.index('--admission')+1])

    def test_path_with_spaces_is_one_argument(self):
        mapping='clip-a=C:\\path with spaces\\bundle.json'
        result=self.run_plan('-MetricBundleFor',mapping);self.assertEqual(result.returncode,0,result.stderr)
        args=json.loads(result.stdout)['commands'][0]['arguments']
        self.assertEqual(args[args.index('--metric-bundle-for')+1],mapping)

    def test_preset_rejects_explicit_model_override(self):
        result=self.run_plan('-ReviewedSixCase','-Head','older.pt')
        self.assertEqual(result.returncode,2);self.assertIn('override',result.stderr)

    def test_duplicate_and_conflicting_geometry_rejected(self):
        for args in [('-MetricBundleFor','clip-a=x,clip-a=y'),
                     ('-MetricBundleFor','clip-a=x','-ExecutionControlFor','clip-a=y')]:
            result=self.run_plan(*args);self.assertEqual(result.returncode,2,result.stdout)

    def test_malformed_mapping_and_bad_port_rejected(self):
        for args in [('-MasksFor','missing-separator'),('-Port','65536'),('-MetricBundleClip','clip-a')]:
            result=self.run_plan(*args);self.assertEqual(result.returncode,2,result.stdout)

    def fixture(self,temp):
        root=Path(temp);script=root/'tools/run-workbench.ps1';script.parent.mkdir();script.write_bytes(SCRIPT.read_bytes())
        return root,script

    def test_missing_preset_assets_never_fall_back(self):
        with tempfile.TemporaryDirectory() as temp:
            root,script=self.fixture(temp)
            result=self.run_plan('-ReviewedSixCase',script=script)
            self.assertEqual(result.returncode,2);self.assertIn('assets are missing',result.stderr)

    def test_wrong_head_hash_rejected_after_all_assets_are_present(self):
        with tempfile.TemporaryDirectory() as temp:
            root,script=self.fixture(temp)
            files=['.cache/quality-windows/Scripts/python.exe','.cache/video-workbench/models/flow-terminal-20261021/head.pt',
                '.cache/video-workbench-tools/colmap-3.11.1/bin/colmap.exe']
            for name in ['mug','keyboard','ranch']:files.append(f'.cache/video-workbench/admission/bundle-{name}-v1.json')
            for oid in ['000002','000004','000018']:
                for name in ['model.glb','metric-bundle.json','execution-control-sidecar.json','emission-receipt.json']:
                    files.append(f'.cache/video-workbench/derived/execution-controls/v1/object-{oid}/{name}')
            clips=['dev-mug','dev-keyboard','dev-ranch','dev-ycb-003-cracker-box','dev-ycb-005-tomato-soup-can','dev-ycb-040-large-marker']
            for clip in clips:
                mask='sam2-workbench-02' if clip=='dev-keyboard' else 'sam2-workbench-01'
                files.append(f'.cache/video-workbench/masks/{clip}/{mask}/manifest.json')
            for name in files:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
            (root/'.cache/video-workbench/admission/attempt-08').mkdir()
            result=self.run_plan('-ReviewedSixCase',script=script)
            self.assertEqual(result.returncode,2,result.stdout)
            self.assertIn('SHA-256',result.stderr);self.assertIn('refusing to start',result.stderr)

if __name__=='__main__':unittest.main()
