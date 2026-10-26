#!/usr/bin/env python3
"""Read persisted authorized-onboarding banks; never rerun features/tracking.

Nearest model VERTEX distances are a discretization-sensitive proxy, not exact
point-to-surface distances or physical ground truth. This script never opens
scored frames, evaluator annotations, benchmark scores or trained weights.
Requires NumPy and SciPy; no PyTorch import or model forward pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import scipy
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
ARRAY_NAMES = ('referencePoints', 'referenceDescriptors', 'previousGray', 'seedPixels', 'seedPoints')
OBJECTS = (20, 28)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def display(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name


def load_vertices_mm(path):
    """Independent ASCII PLY vertex read, retaining exact source millimetres."""
    with path.open(encoding='ascii') as stream:
        require(stream.readline().strip() == 'ply', 'Expected PLY')
        require(stream.readline().strip() == 'format ascii 1.0', 'Expected ASCII PLY')
        properties, count, element = [], 0, None
        for line in stream:
            words = line.split()
            if words[:2] == ['element', 'vertex']:
                count, element = int(words[2]), 'vertex'
            elif words and words[0] == 'element':
                element = words[1]
            elif words and words[0] == 'property' and element == 'vertex':
                properties.append(words[-1])
            elif words and words[0] == 'end_header':
                break
        require(4 <= count <= 1000000, 'Invalid/oversized model')
        columns = [properties.index(name) for name in ('x', 'y', 'z')]
        vertices = np.loadtxt(stream, dtype=np.float64, usecols=columns, max_rows=count)
    require(vertices.shape == (count, 3) and np.isfinite(vertices).all(), 'Invalid model vertices')
    return vertices


def geometry_stats(points_metres, tree):
    points = np.asarray(points_metres, dtype=np.float64) * 1000
    require(points.ndim == 2 and points.shape[1] == 3 and len(points) > 0 and np.isfinite(points).all(),
            'Invalid bank geometry')
    distances, _ = tree.query(points, k=1, workers=1)
    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    return dict(count=len(points), nearestModelVertexDistanceP50MM=float(np.median(distances)),
        nearestModelVertexDistanceP95MM=float(np.percentile(distances, 95)),
        nearestModelVertexDistanceMaxMM=float(np.max(distances)),
        nearestModelVertexDistanceAbove10MMFraction=float(np.mean(distances > 10)),
        exactCoordinateDuplicateFraction=1-len(np.unique(points, axis=0))/len(points),
        occupiedOneMMSourceGridCells=len(np.unique(np.floor(points).astype(np.int64), axis=0)),
        axisExtentMM=np.ptp(points, axis=0).tolist(),
        rmsRadiusFromCentroidMM=float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))),
        principalPopulationStdMM=(singular/np.sqrt(len(points))).tolist(),
        wholeBankNonplanaritySmallestOverLargest=float(singular[-1]/max(singular[0], 1e-15)))


def load_verified_bank(run, object_id):
    # Progress and NPZ contain bank provenance only; no scored/evaluator file is read.
    progress_path = run/'progress.json'
    progress = json.loads(progress_path.read_text())
    row = next(row for row in progress['sequences'] if row['objectId'] == object_id)
    artifact = row['bankArtifact']
    path = (run/artifact['path']).resolve()
    require(path.is_relative_to(run.resolve()) and path.is_file(), 'Bank path escaped run root')
    require(path.stat().st_size <= 32*1024**2, 'Bank compressed-size budget exceeded')
    require(sha(path) == artifact['sha256'], 'Bank NPZ differs from progress hash')
    with zipfile.ZipFile(path) as archive:
        require(set(archive.namelist()) == {name+'.npy' for name in ARRAY_NAMES}, 'Unexpected bank NPZ arrays')
        require(sum(item.file_size for item in archive.infolist()) <= 64*1024**2, 'Bank expansion budget exceeded')
    with np.load(path, allow_pickle=False) as archive:
        arrays = [np.array(archive[name], copy=True) for name in ARRAY_NAMES]
    array_hash = hashlib.sha256(b''.join(np.ascontiguousarray(array).tobytes() for array in arrays)).hexdigest()
    require(array_hash == row['bankSHA256'], 'Ordered raw bank arrays differ from progress hash')
    require(len(arrays[0]) == row['anchors'] and len(arrays[3]) == row['seedViewAnchors'], 'Bank counts differ from progress')
    require(arrays[1].shape == (len(arrays[0]),32) and len(arrays[3]) == len(arrays[4]), 'Bank array shapes disagree')
    return arrays, dict(npzPath=display(path), npzSHA256=artifact['sha256'],
        orderedArraysSHA256=array_hash, progressPath=display(progress_path), progressSHA256=sha(progress_path),
        archiveHashVerified=True, orderedArrayHashVerified=True)


def audit(models, lock_path, simplified, production, output):
    started = time.perf_counter()
    models, lock_path, simplified, production, output = [Path(path).resolve()
        for path in (models, lock_path, simplified, production, output)]
    require(not output.exists(), 'Choose a fresh output JSON; existing evidence is preserved')
    require(not any(output.is_relative_to(path) for path in (models, simplified, production)),
            'Audit output must not modify source models/banks')
    script_hash = sha(Path(__file__))
    lock = json.loads(lock_path.read_text())
    model_hashes = {row['member']: row['sha256'] for row in lock['resources']}
    comparisons = []
    for object_id in OBJECTS:
        model_path = models/f'obj_{object_id:06d}.ply'
        require(sha(model_path) == model_hashes[f'models/{model_path.name}'], 'Model differs from acquisition lock')
        vertices = load_vertices_mm(model_path)
        tree = cKDTree(vertices)
        case = dict(objectId=object_id, modelPath=display(model_path), modelSHA256=sha(model_path),
                    sourceModelVertices=len(vertices), initializations={})
        for name, run in (('simplified_depth_budget', simplified), ('original_production_mesh', production)):
            arrays, provenance = load_verified_bank(run, object_id)
            points, descriptors, gray, pixels, seed_points = arrays
            case['initializations'][name] = dict(provenance=provenance,
                anchorGeometry=geometry_stats(points, tree), seedViewGeometry=geometry_stats(seed_points, tree),
                exactDescriptorDuplicateFraction=1-len(np.unique(descriptors, axis=0))/len(descriptors),
                previousGrayShape=list(gray.shape), seedPixelCount=len(pixels))
        comparisons.append(case)
    result = dict(schema='visualizeit-tracking-bank-geometry-audit/1', scriptSHA256=script_hash,
        numpyVersion=np.__version__, scipyVersion=scipy.__version__, objects=comparisons,
        acquiredModelLockSHA256=sha(lock_path),
        scope='Persisted first-ten authorized-onboarding banks only; no ORB, PnP, tracker/model forward pass, rasterization, scored frames, evaluator labels or benchmark scores read.',
        methods=dict(nearest='Single-thread cKDTree query against every exact source-model vertex; anchor metres converted to model millimetres.',
            nonplanarity='Smallest/largest centered point singular value for whole anchor/seed sets, not per-frame inlier solver gates.',
            duplicates='Exact coordinate/descriptor equality; one-mm grid occupancy is descriptive and grid-origin dependent.'),
        limitations=['Nearest-vertex distance is NOT exact point-to-triangle surface distance or independent physical truth.',
            'Original ray anchors lie on triangles; their nonzero nearest-vertex distance includes mesh-discretization floor.',
            'Depth anchors may differ from the scan due to sensor noise, registration, scan shape, source poses or mixed-depth pixels.',
            'Anchor count, locations, spatial coverage, feature selection and geometry all change together; no causal quality-versus-quantity isolation or fix claim.',
            'No unseen/generalization, full production pipeline, GoTrack quality, accuracy or speed is established by these proxies.'],
        auditWallSeconds=time.perf_counter()-started)
    require(sha(Path(__file__)) == script_hash, 'Audit script changed during execution')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,default=ROOT/'.cache/tracking-benchmark/real-v1/models')
    parser.add_argument('--source-lock',type=Path,default=ROOT/'.cache/tracking-benchmark/real-v1/sources.lock.json')
    parser.add_argument('--simplified',type=Path,default=ROOT/'artifacts/tracking-benchmark/real-prefix-simplified-v2')
    parser.add_argument('--production',type=Path,default=ROOT/'artifacts/tracking-benchmark/real-prefix-production-v2')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/tracking-benchmark/bank-geometry-audit.json')
    args=parser.parse_args()
    result=audit(args.models,args.source_lock,args.simplified,args.production,args.output)
    for case in result['objects']:
        print('HB object',case['objectId'])
        for name,row in case['initializations'].items():
            g=row['anchorGeometry']; s=row['seedViewGeometry']
            print(name,'anchors',g['count'],'seed',s['count'],'vertex P50/P95 mm',g['nearestModelVertexDistanceP50MM'],g['nearestModelVertexDistanceP95MM'])
    print('ScriptSHA256',result['scriptSHA256'],'auditSeconds',result['auditWallSeconds'])


if __name__=='__main__':
    main()
