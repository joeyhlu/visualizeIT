"""Bounded loopback HTTP smoke for the real production geometric frame service.

This is API/serialization/pipeline evidence, never a pose-accuracy benchmark,
live-camera test, neural-model test, or FPS claim. It reads inference.json only,
uses exactly ten authorized RGB-D onboarding records to prepare ordinary service
assets, then sends at most twenty label-free prerecorded RGB JPEG requests per
corpus to the unmodified production service on an ephemeral loopback port.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import http.client
import io
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image

from .ply_model import read_ply
from .prototype_benchmark import digest, write_json
from .prototype_service import (
    MAX_ASSET_BYTES, MAX_ASSET_UNCOMPRESSED_BYTES, MAX_IMAGE_PIXELS,
    MAX_JPEG_BYTES, MAX_REQUEST_BYTES, PrototypeService, create_server,
    load_asset_registry, load_calibration_registry,
)
from .prototype_tracking import is_proper_rotation
from .tracking_benchmark import build_depth_bank, checked_path, validate_manifest

ROOT = Path(__file__).resolve().parents[1]
MAX_PROBE_FRAMES = 20
MAX_DATASETS = 2
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_SOURCE_IMAGE_BYTES = 32 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 15.
JPEG_QUALITY = 85
WIRE_FIELDS = frozenset({
    'sessionId', 'frameId', 'logicalIndex', 'captureTimestampSeconds', 'timestampKind',
    'objectId', 'referenceId', 'calibrationId', 'width', 'height', 'rgbJPEGBase64',
})
IDENTITY_FIELDS = ('sessionId', 'objectId', 'referenceId', 'calibrationId',
                   'frameId', 'logicalIndex', 'timestampKind', 'captureTimestampSeconds')
SOURCE_FILES = ('bench/tracking_service_probe.py', 'bench/tracking_benchmark.py',
                'bench/prototype_service.py', 'bench/prototype_tracking.py',
                'bench/tracker.py', 'bench/ply_model.py')


def _strict_json(raw):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    value = json.loads(raw, parse_constant=reject)
    return value


def _bounded_integer(value, label, minimum=1, maximum=MAX_PROBE_FRAMES):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{label} must be an integer between {minimum} and {maximum}')
    return value


def _rgb_image(dataset, record):
    path = checked_path(dataset, record['rgb'])
    if path.stat().st_size > MAX_SOURCE_IMAGE_BYTES:
        raise ValueError('RGB source exceeds 32 MiB local-read budget')
    with Image.open(path) as source:
        if source.format not in ('PNG', 'JPEG'):
            raise ValueError('RGB source must be PNG or JPEG')
        width, height = source.size
        if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            raise ValueError('RGB dimensions exceed production service limits')
        source.load()
        return source.convert('RGB')


def prepare_asset(dataset, assets_root, *, object_id=None, frame_count=MAX_PROBE_FRAMES):
    """Export the standard NPZ/JSON contract using only authorized onboarding.

    No evaluator manifest is opened, and no evaluation depth/mask/pose is read.
    The service receives no last-frame seed: this probe tests its actual cold
    descriptor-bank acquisition, unlike seeded offline tracker benchmarks.
    """
    _bounded_integer(frame_count, 'frame_count')
    dataset, assets_root = Path(dataset).resolve(), Path(assets_root).resolve()
    if assets_root == dataset or assets_root.is_relative_to(dataset):
        raise ValueError('Service assets must not modify the source dataset')
    if assets_root.exists():
        raise ValueError('Choose a fresh production asset directory')
    manifest_path = dataset / 'inference.json'
    manifest = _strict_json(manifest_path.read_text(encoding='utf-8'))
    validate_manifest(manifest)
    candidates = manifest['objects']
    if object_id is not None:
        _bounded_integer(object_id, 'object_id', maximum=2**53-1)
        candidates = [obj for obj in candidates if obj['objectId'] == object_id]
    if not candidates:
        raise ValueError('Requested object/sequence is absent from inference manifest')
    obj = candidates[0]
    records = obj['frames'][:frame_count]
    # Fail before bank creation if a corpus cannot use a stable service camera.
    images = [_rgb_image(dataset, row) for row in records]
    first_size = images[0].size
    first_k = np.asarray(records[0]['calibration']['cam_K'], dtype=float).reshape(3, 3)
    for row, image in zip(records, images):
        current_k = np.asarray(row['calibration']['cam_K'], dtype=float).reshape(3, 3)
        if image.size != first_size or not np.array_equal(current_k, first_k):
            raise ValueError('Probe requires stable per-sequence image dimensions/intrinsics')
    scan, source_positions, transform = read_ply(checked_path(dataset, obj['model']), obj['name'])
    bank = build_depth_bank(dataset, obj)
    bank_points, bank_descriptors = bank[:2]
    if len(bank_points) < 12:
        raise ValueError('insufficient_onboarding_references')
    synthetic = str(manifest.get('evidenceKind', '')).startswith('synthetic')
    evidence_kind = 'synthetic_control' if synthetic else 'dataset_reference'
    calibration_kind = 'synthetic' if synthetic else 'dataset_reference'
    directory = assets_root / f"object-{obj['objectId']:06d}"
    directory.mkdir(parents=True)
    asset_path = directory / 'tracking-asset.npz'
    arrays = dict(referencePoints=bank_points, referenceDescriptors=bank_descriptors,
        modelPositions=source_positions, positions=scan.original.positions,
        normals=scan.original.normals, triangles=scan.original.triangles,
        uv=scan.original.uv, benchFromSource=transform, schemaVersion=np.asarray(1,dtype=np.int32))
    if sum(array.nbytes for array in arrays.values()) > MAX_ASSET_UNCOMPRESSED_BYTES:
        raise ValueError('Prepared arrays exceed production asset expansion budget')
    np.savez_compressed(asset_path, **arrays)
    if asset_path.stat().st_size > MAX_ASSET_BYTES:
        raise ValueError('Prepared NPZ exceeds production asset budget')
    asset_hash = digest(asset_path)
    metadata = dict(objectId=obj['objectId'], name=obj['name'], evidenceKind=evidence_kind,
        assetPath=asset_path.name, assetSHA256=asset_hash, units='metres',
        sourceBasis='Original source-object model axes; source PLY millimetres converted to metres',
        onboardingFrameIds=[row['frameId'] for row in obj['onboarding']],
        inferenceManifestSHA256=digest(manifest_path), sourceModelSHA256=obj['model']['sha256'],
        initializationAdapter='shared_masked_depth_bank_first_10_max_300_ORB_anchors_per_view',
        inputEvidenceKind=manifest.get('evidenceKind'))
    write_json(directory / 'asset.json', metadata)
    calibration_id = f"service-probe-{obj['objectId']}-{asset_hash[:16]}"
    calibration_path = assets_root / 'calibrations.json'
    write_json(calibration_path, dict(schemaVersion=1, calibrations=[dict(
        calibrationId=calibration_id, width=first_size[0], height=first_size[1],
        K=first_k.tolist(), kind=calibration_kind, objectId=obj['objectId'],
        name='Prerecorded dataset-reference calibration; not physically measured live capture')]))
    # Exercise the actual service loaders, including source/basis mapping checks.
    loaded_assets = load_asset_registry(assets_root)
    loaded_calibrations = load_calibration_registry(calibration_path)
    if loaded_assets[obj['objectId']].reference_id != asset_hash or calibration_id not in loaded_calibrations:
        raise ValueError('Production registry readback identity mismatch')
    return dict(dataset=dataset, manifest=manifest, object=obj, records=records,
        assetsRoot=assets_root, calibrationsPath=calibration_path, referenceId=asset_hash,
        calibrationId=calibration_id, calibrationKind=calibration_kind,
        width=first_size[0], height=first_size[1], referenceAnchors=len(bank_points),
        inferenceManifestSHA256=digest(manifest_path))


def make_packet(prepared, record, session):
    """Serialize only RGB and explicit service identity/calibration metadata."""
    image = _rgb_image(prepared['dataset'], record)
    encoded = io.BytesIO()
    image.save(encoded, format='JPEG', quality=JPEG_QUALITY, optimize=False)
    jpeg = encoded.getvalue()
    if not jpeg or len(jpeg) > MAX_JPEG_BYTES:
        raise ValueError('JPEG exceeds production service budget')
    packet = dict(sessionId=session, frameId=record['frameId'],
        logicalIndex=record.get('logicalIndex',record['frameId']),
        captureTimestampSeconds=None, timestampKind='frame_index_no_fps',
        objectId=prepared['object']['objectId'], referenceId=prepared['referenceId'],
        calibrationId=prepared['calibrationId'], width=image.width, height=image.height,
        rgbJPEGBase64=base64.b64encode(jpeg).decode('ascii'))
    if set(packet) != WIRE_FIELDS:
        raise ValueError('Internal packet contains non-whitelisted fields')
    body = json.dumps(packet, allow_nan=False, separators=(',',':')).encode()
    if len(body) > MAX_REQUEST_BYTES:
        raise ValueError('Request exceeds production service budget')
    return packet, body


def interpret_response(status, payload, packet):
    """Check API contracts without comparing a response to any reference pose."""
    if not isinstance(payload,dict):
        return dict(contractValid=False, issues=['non_object_response'])
    issues=[]
    if status != 200:
        issues.append('non_200_response')
    if any(type(payload.get(k)) is not type(packet[k]) or payload.get(k) != packet[k]
           for k in IDENTITY_FIELDS):
        issues.append('response_identity_mismatch')
    state=payload.get('poseState')
    if state not in ('lost','limited','recovering','tracking'):
        issues.append('invalid_pose_state')
    processing=payload.get('processingSeconds')
    if type(processing) not in (int,float) or not math.isfinite(processing) or processing < 0:
        issues.append('invalid_processing_time')
    pose=payload.get('cameraFromObject')
    if state=='tracking':
        try:
            scalar_types_ok=(isinstance(pose,list) and len(pose)==4 and all(
                isinstance(row,list) and len(row)==4 and all(type(x) in (int,float) for x in row) for row in pose))
            matrix=np.asarray(pose,dtype=float)
            proper=(scalar_types_ok and (matrix.shape==(4,4) and np.isfinite(matrix).all()
                    and np.allclose(matrix[3],[0,0,0,1],atol=1e-8,rtol=0)
                    and is_proper_rotation(matrix[:3,:3])))
        except (TypeError,ValueError):
            proper=False
        if not proper or payload.get('units')!='metres' or payload.get('cameraConvention')!='CV X right, Y down, Z forward; column vectors':
            issues.append('invalid_public_pose')
    elif pose is not None or payload.get('projectionSegments') is not None or payload.get('renderState')!='suppressed':
        issues.append('stale_pose_or_overlay_in_nontracking_response')
    if payload.get('freshCameraFrameVerified') is not False:
        issues.append('prerecorded_frame_claimed_live_camera_freshness')
    return dict(contractValid=not issues, issues=issues, poseState=state,
        failureReason=payload.get('failureReason'), processingSeconds=processing,
        publicPosePresent=pose is not None, liveCameraFreshnessVerified=False)


def _exchange(port, method, endpoint, body=None):
    connection=http.client.HTTPConnection('127.0.0.1',port,timeout=REQUEST_TIMEOUT_SECONDS)
    started=time.perf_counter()
    try:
        connection.request(method,endpoint,body=body,headers={
            'Content-Type':'application/json','Accept':'application/json','Connection':'close'})
        response=connection.getresponse()
        raw=response.read(MAX_RESPONSE_BYTES+1)
        elapsed=time.perf_counter()-started
        if len(raw)>MAX_RESPONSE_BYTES:
            raise ValueError('Response exceeds probe budget')
        return response.status,_strict_json(raw.decode('utf-8')),elapsed
    finally:
        connection.close()


def probe_prepared(prepared):
    """Start, exercise, and close an actual loopback production HTTP service."""
    service=PrototypeService(prepared['assetsRoot'],prepared['calibrationsPath'])
    server=create_server('127.0.0.1',0,service)
    thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True)
    rows=[]; statuses=[]; health=None; session=f"probe-{prepared['object']['objectId']}-{prepared['referenceId'][:12]}"
    thread.start()
    try:
        status,health,_=_exchange(server.server_port,'GET','/healthz')
        if status!=200 or health.get('status')!='ok' or health.get('assetsLoaded')!=1:
            raise ValueError('Production service health check failed')
        for record in prepared['records']:
            packet,body=make_packet(prepared,record,session)
            status,payload,elapsed=_exchange(server.server_port,'POST','/v1/frames',body)
            contract=interpret_response(status,payload,packet)
            if not math.isfinite(elapsed) or elapsed<0:
                raise ValueError('Nonfinite request time')
            rows.append(dict(frameId=record['frameId'],logicalIndex=packet['logicalIndex'],
                sourceRGBSHA256=record['rgb']['sha256'],httpStatus=status,
                requestSeconds=elapsed,**contract))
            statuses.append(status)
        status,service_status,_=_exchange(server.server_port,'GET','/v1/status')
        if status!=200:
            raise ValueError('Production status endpoint failed')
    finally:
        server.shutdown(); server.server_close(); thread.join(2.)
    closed=not thread.is_alive() and server.socket.fileno()==-1
    if not closed:
        raise ValueError('Probe failed to close its loopback server')
    if digest(prepared['dataset']/'inference.json')!=prepared['inferenceManifestSHA256']:
        raise ValueError('Inference manifest changed during HTTP probe')
    obj=prepared['object']; manifest=prepared['manifest']
    return dict(dataset=manifest.get('dataset'),benchmarkId=manifest.get('benchmarkId'),
        inputEvidenceKind=manifest.get('evidenceKind'),objectId=obj['objectId'],objectName=obj['name'],
        inferenceManifestSHA256=prepared['inferenceManifestSHA256'],
        referenceId=prepared['referenceId'],referenceAnchors=prepared['referenceAnchors'],
        assetsRoot=str(prepared['assetsRoot'].relative_to(ROOT)) if prepared['assetsRoot'].is_relative_to(ROOT) else str(prepared['assetsRoot']),
        calibrationKind=prepared['calibrationKind'],wireClock='frame_index_no_fps',
        sourceClockNormalizedForPrerecordedReplay=True,serverHost='127.0.0.1',ephemeralPort=True,
        serverClosed=closed,health=health,framesSent=len(rows),
        allRequestsHTTP200=all(s==200 for s in statuses),
        allFrameContractsValid=all(row['contractValid'] for row in rows),
        stateCounts=dict(Counter(row.get('poseState') for row in rows)),
        failureReasonCounts=dict(Counter(row.get('failureReason') for row in rows if row.get('failureReason'))),
        trackingResponses=sum(row.get('poseState')=='tracking' for row in rows),
        acceptedFrameRequests=service_status.get('acceptedFrameRequests'),frames=rows,
        evaluatorOpened=False,accuracyMeasured=False,liveFPSMeasured=False,
        initializationScope='Exactly ten authorized annotated RGB-D onboarding frames; service cold-starts from immutable descriptor bank without offline temporal seed')


def run_probe(datasets, output, *, frame_count=MAX_PROBE_FRAMES, assets_root=None):
    """Probe one or two corpora; do not overwrite earlier evidence/assets."""
    _bounded_integer(frame_count,'frame_count')
    if not isinstance(datasets,(list,tuple)) or not 1<=len(datasets)<=MAX_DATASETS:
        raise ValueError('Probe requires one or two dataset roots')
    output=Path(output).resolve()
    if any(output==Path(d).resolve() or output.is_relative_to(Path(d).resolve()) for d in datasets):
        raise ValueError('Probe output must not modify a source dataset')
    if output.exists():
        raise ValueError('Choose a fresh probe output file')
    assets_root=Path(assets_root).resolve() if assets_root else output.with_name(output.stem+'-assets')
    if assets_root.exists():
        raise ValueError('Choose a fresh probe assets root')
    frozen={name:digest(ROOT/name) for name in SOURCE_FILES}
    reports=[]
    for index,dataset in enumerate(datasets,1):
        prepared=prepare_asset(dataset,assets_root/f'dataset-{index:02d}',frame_count=frame_count)
        reports.append(probe_prepared(prepared))
    if frozen!={name:digest(ROOT/name) for name in SOURCE_FILES}:
        raise ValueError('Probe source changed during execution')
    result=dict(schema='visualizeit-tracking-service-probe/1',sourceSHA256=frozen,datasets=reports,
        allContractsPassed=all(r['allFrameContractsValid'] and r['serverClosed'] for r in reports),
        evaluatorOpened=False,accuracyMeasured=False,liveFPSMeasured=False,
        trainedNeuralInferenceMeasured=False,framesPerDatasetCap=MAX_PROBE_FRAMES,
        scope='Bounded prerecorded RGB-over-loopback-HTTP contract smoke of unchanged current geometric production service; not accuracy, live-camera, phone, or FPS evidence')
    write_json(output,result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',action='append',type=Path,
        help='Inference dataset root, repeat at most twice; default synthetic-v1 and real-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/tracking-benchmark/service-probe-v1.json')
    parser.add_argument('--assets-root',type=Path)
    parser.add_argument('--frames',type=int,default=MAX_PROBE_FRAMES,choices=range(1,MAX_PROBE_FRAMES+1))
    args=parser.parse_args(argv)
    datasets=args.dataset or [ROOT/'.cache/tracking-benchmark/synthetic-v1',ROOT/'.cache/tracking-benchmark/real-v1']
    result=run_probe(datasets,args.output,frame_count=args.frames,assets_root=args.assets_root)
    print(json.dumps({key:result[key] for key in ('allContractsPassed','accuracyMeasured','liveFPSMeasured')},indent=2))


if __name__=='__main__':
    main()
