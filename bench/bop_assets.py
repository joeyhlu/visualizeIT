"""Acquire small contiguous real YCB-Video sequences with separate evaluation annotations."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .remote_zip import RemoteZip
from .ycb_assets import ROOT

TARGETS = {5: 'Mustard bottle', 14: 'Mug', 15: 'Power drill'}
MAX_BYTES = 384*1024*1024
RESERVE = 512*1024*1024


def acquire_masks(output):
    manifests = [(p,json.loads(p.read_text())) for p in sorted(output.glob('object-*.json'))]
    if manifests and all('evaluationMask' in frame for _, m in manifests for frame in m['frames']):
        return
    revision = (output/'revision.txt').read_text().strip()
    archive = RemoteZip(f'https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/{revision}/ycbv_test_all.zip')
    work = []
    for _, manifest in manifests:
        for frame in manifest['frames']:
            member = f'test/{manifest["sceneId"]}/mask_visib/{frame["frameId"]:06d}_{frame["instanceId"]:06d}.png'
            if archive.entries[member]['size'] > 1024*1024: raise ValueError('Evaluation mask exceeds small-member bound.')
            work.append((frame,member))
    def fetch(item):
        frame,member = item; path = output/member
        raw = path.read_bytes() if path.exists() else archive.read(member)
        if not path.exists(): cache_write(output,path,raw)
        frame['evaluationMask'] = dict(path=member,sha256=hashlib.sha256(raw).hexdigest())
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch,work))
    for path,manifest in manifests:
        manifest['license'] = 'MIT per BOP YCB-Video listing; original YCB scan attribution CC BY 4.0 retained'
        manifest['authors'] = 'Yu Xiang, Tanner Schmidt, Venkatraman Narayanan, Dieter Fox; BOP conversion by BOP authors; original YCB scans by Calli et al.'
        cache_write(output,path,(json.dumps(manifest,indent=2)+'\n').encode())
    print(f'Acquired {len(work)} small evaluation-only visibility masks; tracker does not receive them.',flush=True)


def cache_write(root, path, data):
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Output outside dataset cache.')
    used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if used+len(data) > MAX_BYTES or shutil.disk_usage(root).free-len(data) < RESERVE:
        raise ValueError('Dataset/disk budget reached; partial cache retained for retry.')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def acquire(output, frames=120, include_limited_motion=False):
    output.mkdir(parents=True, exist_ok=True)
    completed = [output/f'object-{obj:06d}.json' for obj in TARGETS]
    if all(p.exists() for p in completed) and (output/'selection.json').exists():
        manifests = [json.loads(p.read_text()) for p in completed]
        if all(len(m['frames']) == frames and 'model' in m for m in manifests) and all(
            (output/asset['path']).exists() for m in manifests for asset in [m['model']]+[
                f[key] for f in m['frames'] for key in ('rgb','depth','evaluationMask') if key in f]):
            if not include_limited_motion and not all(m['selectionGatePassed'] for m in manifests):
                raise ValueError('Cached bottle/mug clips fail the motion gate; use explicitly labelled limited-motion controls.')
            for manifest in manifests:
                for asset in [manifest['model']]+[f[key] for f in manifest['frames'] for key in ('rgb','depth','evaluationMask') if key in f]:
                    if hashlib.sha256((output/asset['path']).read_bytes()).hexdigest() != asset['sha256']:
                        raise ValueError('Completed cache hash mismatch; refusing silently changed input.')
            acquire_masks(output)
            print('Completed RGB/depth/model cache hashes verified.', flush=True)
            return
    revision_file = output/'revision.txt'
    if revision_file.exists():
        revision = revision_file.read_text().strip()
    else:
        with urllib.request.urlopen('https://huggingface.co/api/datasets/bop-benchmark/ycbv', timeout=30) as response:
            revision = json.load(response)['sha']
        revision_file.write_text(revision+'\n')
    base = f'https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/{revision}/'
    archive = RemoteZip(base+'ycbv_test_all.zip')
    scenes = sorted({p.split('/')[1] for p in archive.entries if p.startswith('test/') and p.endswith('/scene_gt.json')})
    pending = dict(TARGETS)
    selections = {}
    for scene in scenes:
        if not pending:
            break
        prefix = f'test/{scene}/'
        metadata = {}
        for name in ('scene_gt.json', 'scene_gt_info.json', 'scene_camera.json'):
            path = output/prefix/name
            raw = path.read_bytes() if path.exists() else archive.read(prefix+name)
            if not path.exists():
                cache_write(output, path, raw)
            metadata[name] = json.loads(raw)
        gt, info, calibration = (metadata[n] for n in ('scene_gt.json', 'scene_gt_info.json', 'scene_camera.json'))
        keys = sorted(map(int, gt))
        for object_id in list(pending):
            candidates = []
            for frame in keys:
                instances = [i for i, pose in enumerate(gt[str(frame)]) if pose['obj_id'] == object_id]
                if len(instances) == 1:
                    candidates.append((frame, instances[0]))
            chosen = None
            best = None
            for start in range(max(0, len(candidates)-frames+1)):
                clip = candidates[start:start+frames]
                if clip[-1][0]-clip[0][0] != frames-1:
                    continue
                visible = [info[str(f)][i]['visib_fract'] for f, i in clip]
                if sum(v >= .5 for v in visible) < frames//2:
                    continue
                directions = []
                for frame, instance in clip:
                    pose = gt[str(frame)][instance]
                    rotation = np.array(pose['cam_R_m2c']).reshape(3, 3)
                    direction = -rotation.T@np.array(pose['cam_t_m2c'])
                    directions.append(direction/np.linalg.norm(direction))
                angle = np.rad2deg(np.arccos(np.clip(np.min(np.array(directions)@np.array(directions).T), -1, 1)))
                if best is None or angle > best[2]:
                    best = (clip, visible, float(angle))
                if angle >= 15:
                    chosen = (clip, visible, float(angle)); break
            if chosen is None:
                if best is not None and (object_id not in selections or best[2] > selections[object_id][2]):
                    selections[object_id] = (*best, scene, metadata)
                continue
            selections[object_id] = (*chosen, scene, metadata)
            del pending[object_id]
    unresolved = dict(pending)
    if not include_limited_motion:
        selections = {identifier: values for identifier, values in selections.items() if values[2] >= 15}
    for object_id, values in selections.items():
            clip, visibility, angle, scene, metadata = values
            prefix = f'test/{scene}/'
            gt, info, calibration = (metadata[n] for n in ('scene_gt.json', 'scene_gt_info.json', 'scene_camera.json'))
            manifest = dict(schemaVersion=1, objectId=object_id, name=TARGETS[object_id], sceneId=scene,
                            datasetRevision=revision, source='https://bop.felk.cvut.cz/datasets/',
                            onboardingFrames=10, viewChangeDegrees=angle,
                            license='MIT per BOP YCB-Video listing; original YCB scan attribution CC BY 4.0 retained',
                            authors='Yu Xiang, Tanner Schmidt, Venkatraman Narayanan, Dieter Fox; BOP conversion by BOP authors; original YCB scans by Calli et al.',
                            selectionGatePassed=angle >= 15,
                            selectionFailureReason=None if angle >= 15 else f'No {frames}-frame clip meeting 15 degree view-change requirement; limited-motion control only',
                            frameRate=None, temporalUnits='frame index; no FPS inferred', frames=[])
            manifest['metadataSHA256'] = {key: hashlib.sha256((output/prefix/key).read_bytes()).hexdigest() for key in metadata}
            for number, instance in clip:
                assets = {}
                for kind, suffix in (('rgb', 'png'), ('depth', 'png')):
                    member = f'{prefix}{kind}/{number:06d}.{suffix}'
                    if member not in archive.entries and kind == 'rgb':
                        member = f'{prefix}{kind}/{number:06d}.jpg'
                    path = output/member
                    raw = path.read_bytes() if path.exists() else archive.read(member)
                    digest = hashlib.sha256(raw).hexdigest() if path.exists() else cache_write(output, path, raw)
                    assets[kind] = dict(path=member, sha256=digest)
                manifest['frames'].append(dict(frameId=number, instanceId=instance, **assets,
                                                calibration=calibration[str(number)],
                                                evaluationPose=gt[str(number)][instance],
                                                evaluationInfo=info[str(number)][instance]))
                if (number-clip[0][0])%30 == 0:
                    print(f'{TARGETS[object_id]} scene {scene}: frame {number}', flush=True)
            cache_write(output, output/f'object-{object_id:06d}.json', (json.dumps(manifest, indent=2)+'\n').encode())
            print(f'{TARGETS[object_id]}: selected {frames} consecutive frames, {angle:.1f} degree view change; selection gate {angle >= 15}', flush=True)
    # Discovery annotations are already embedded in selected manifests with hashes.
    for path in (output/'test').glob('*/scene_*.json'):
        if path.name in ('scene_gt.json', 'scene_gt_info.json', 'scene_camera.json') and path.resolve().is_relative_to(output.resolve()):
            path.unlink()
    models = RemoteZip(base+'ycbv_models.zip')
    for object_id in selections:
        member = f'models/obj_{object_id:06d}.ply'
        path = output/member
        if not path.exists():
            cache_write(output, path, models.read(member))
        manifest_path = output/f'object-{object_id:06d}.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['model'] = dict(path=member, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    selection_report = dict(qualifyingObjects=[i for i, v in selections.items() if v[2] >= 15],
                            failedSelectionObjects=list(unresolved), limitedMotionControls=include_limited_motion,
                            thresholds=dict(frames=frames, viewChangeDegrees=15, halfFramesVisibility=.5))
    cache_write(output, output/'selection.json', (json.dumps(selection_report, indent=2)+'\n').encode())
    acquire_masks(output)
    # Full scene annotations are discovery intermediates. Selected annotations and
    # source hashes are retained in manifests; remove only this tool's named cache files.
    for path in (output/'test').glob('*/scene_*.json'):
        if path.name in ('scene_gt.json', 'scene_gt_info.json', 'scene_camera.json') and path.resolve().is_relative_to(output.resolve()):
            path.unlink()
    print(f'Completed bounded acquisition; {sum(p.stat().st_size for p in output.rglob("*") if p.is_file())/1024**2:.1f} MiB cached', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'.cache'/'bop')
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--include-limited-motion', action='store_true', help='Fetch best failed-selection clips as explicitly labelled controls, without passing the gate.')
    args = parser.parse_args()
    if args.frames < 20 or args.frames > 120:
        parser.error('Use 20–120 frames.')
    acquire(args.output, args.frames, args.include_limited_motion)
