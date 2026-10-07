"""Export label-free inference bundles and a separate, pending annotation set."""
import json
from pathlib import Path
import shutil
import numpy as np
from .quality_assets import CACHE, ROOT, BUDGET, digest
from .show3d_experiment import reference_pose
from .vision import cv2
from .storage import write_artifact


def prepare():
    reports = json.loads((ROOT/'artifacts/video-60/report.json').read_text())['objects']
    assets = ROOT/'.cache/show3d'
    output = ROOT/'artifacts/model-quality'; output.mkdir(parents=True, exist_ok=True)
    annotation_root = output/'annotations'; annotation_root.mkdir(exist_ok=True)
    held_out = {'keyboard': (0, 610), 'mug': (0, 610), 'ranch': (250, 860)}
    summary = []
    for r in reports:
        alias = r['name']; start = r['sourceStart']; first = start+10
        source = assets/('scenes/'+r['scene'])
        for name, sha in r['sourceHashes'].items():
            if digest(assets/name) != sha: raise ValueError('Original source changed: '+name)
        bundle = CACHE/'inputs'/alias; bundle.mkdir(parents=True, exist_ok=True)
        for src, name in ((source/'headset0.mp4', 'source.mp4'), (assets/f'models/obj_{r["modelId"]:06d}.glb', 'object.glb')):
            target = bundle/name
            if not target.exists(): shutil.copyfile(src, target)
            if digest(target) != digest(src): raise ValueError('Inference asset copy changed')
        cal = json.loads((source/'camera_calibration/headset0.json').read_text())
        records_path = next(name for name in r['sourceHashes'] if name.endswith('object_pose.json'))
        records = json.loads((assets/records_path).read_text())
        initial = reference_pose(records[str(start+9)], cal, start+9)
        pose = np.eye(4); pose[:3, :3], pose[:3, 3] = initial
        # Only the final permitted onboarding pose crosses the inference boundary.
        # The main complete mode explicitly discards this field.
        manifest = dict(schema_version=1, object=alias, object_id=r['modelId'], video='source.mp4', asset='object.glb',
            native_resolution=r['nativeResolution'], intrinsics=np.array(r['cameraCalibration']).reshape(3, 3).tolist(),
            setup_frame_id=start+9, frame_ids=list(range(first, first+240)),
            controlled_initial_pose=pose.tolist(), selection=None,
            units='metres', coordinates='OpenCV camera-from-object, column vectors',
            source_hashes={name: digest(bundle/name) for name in ('source.mp4', 'object.glb')},
            initialization_scope='Controlled mode allows final onboarding pose only; complete mode supplies no object pose.',
            additional_window=dict(setup_frames=list(range(*[held_out[alias][0], held_out[alias][0]+10])),
                                   eval_frames=list(range(held_out[alias][0]+10, held_out[alias][1])),
                                   status='preselected before candidate inference; source-only initialization visibility review pending'))
        manifest_path = bundle/'input.json'
        if manifest_path.exists():
            prior = json.loads(manifest_path.read_text()); manifest['selection'] = prior.get('selection')
        manifest_path.write_text(json.dumps(manifest, indent=2))
        cap = cv2.VideoCapture(str(bundle/'source.mp4'))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start+9); ok, image = cap.read()
        if not ok: raise ValueError('Missing setup image')
        cv2.imwrite(str(bundle/'setup.png'), image)
        # The evenly spaced list is frozen; hard-frame choices must be reviewed
        # from original footage before predictions are visible.
        uniform = np.rint(np.linspace(first, first+239, 30)).astype(int).tolist()
        proposed_hard = [first+i for i in (5, 19, 38, 59, 84, 108, 137, 166, 199, 231)]
        ids = sorted(set(uniform+proposed_hard))
        while len(ids) < 40:
            ids.append(next(i for i in range(first, first+240) if i not in ids)); ids.sort()
        directory = annotation_root/alias; directory.mkdir(exist_ok=True)
        frames = []
        for fid in ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid); ok, image = cap.read()
            if not ok: raise ValueError('Incomplete source footage')
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 96])
            if not ok: raise ValueError('Annotation preview encoding failed')
            write_artifact(directory/f'{fid}.jpg', encoded.tobytes())
            frames.append(dict(frame_id=fid, image=f'{fid}.jpg', group='uniform' if fid in uniform else 'hard_candidate',
                status='pending', visible_object=[], overlapping_hands=[], landmarks=[], review_notes='',
                provenance='Independent image annotation required; no model/reference projection supplied.'))
        cap.release()
        annotation = dict(schema_version=1, object=alias, resolution=r['nativeResolution'],
                          selection_status='hard candidates need image-only difficulty review',
                          operator=None, source_sha256=manifest['source_hashes']['source.mp4'], frames=frames)
        annotation_path = directory/'annotations.json'
        if not annotation_path.exists(): write_artifact(annotation_path, json.dumps(annotation, indent=2))
        summary.append(dict(object=alias, source_start=start, frames=240, setup=start+9, annotations=40,
                            annotation_status='pending', inference_status='not_run'))
    if sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file()) > BUDGET: raise ValueError('Model cache budget exceeded')
    write_artifact(output/'manifest.json', json.dumps(dict(schema_version=1, objects=summary, original_scored_frames=720,
        runtime='Linux model inference deferred: user requested no further Docker startup attempts',
        inference_complete=False, independent_accuracy_scored=False), indent=2))
    print('720-frame windows preserved; 120 annotation candidates prepared, NOT annotated.', flush=True)


if __name__ == '__main__': prepare()
