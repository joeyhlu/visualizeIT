"""Export dataset-reference calibrations for local replay, never phone calibration."""
import argparse
import json
from pathlib import Path

from bench.prototype_benchmark import load_rgb, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a fresh calibration registry output.')
    manifest = json.loads((args.dataset / 'inference.json').read_text(encoding='utf-8'))
    kind = manifest.get('evidenceKind', 'dataset_reference')
    if kind not in ('dataset_reference', 'synthetic_control'):
        raise ValueError('Only explicit dataset or synthetic replay is supported.')
    rows = []
    for obj in manifest['objects']:
        frames = obj['frames']
        first = frames[0]
        k = first['calibration']['cam_K']
        if any(frame['calibration']['cam_K'] != k for frame in frames):
            raise ValueError('Changing intrinsics require separate replay calibration identities.')
        rgb = load_rgb(args.dataset, first)
        height, width = rgb.shape[:2]
        rows.append(dict(calibrationId=f"dataset-{obj['objectId']}",
                         kind='synthetic' if kind == 'synthetic_control' else 'dataset_reference',
                         objectId=obj['objectId'], width=width, height=height, cam_K=k))
    write_json(args.output, dict(schemaVersion=1, calibrations=rows))
    print('Wrote replay-only calibration registry:', args.output)


if __name__ == '__main__':
    main()
