"""Score manually annotated, 720p fixture footage; no camera results are bundled.

One row per visible physical reference point per frame. Invalid tracking must
remain in the CSV so that hiding the overlay cannot improve availability.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered)-1)*fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper]-ordered[lower])*(position-lower)


def evaluate(rows, minimum_availability=.9):
    if not 0 <= minimum_availability <= 1:
        raise ValueError('Availability must be between zero and one.')
    errors = []
    frames = {}
    seen = set()
    count = 0
    for row in rows:
        frame = str(row['frame'])
        point = str(row['point'])
        if (frame, point) in seen:
            raise ValueError('Duplicate frame/point annotation.')
        seen.add((frame, point))
        valid_text = str(row['valid']).lower()
        if valid_text not in ('0', '1', 'true', 'false'):
            raise ValueError('valid must be 0/1 or true/false.')
        valid = valid_text in ('1', 'true')
        if frame in frames and frames[frame] != valid:
            raise ValueError('All annotations in a frame must share tracking validity.')
        frames[frame] = valid
        expected = [float(row[k]) for k in ('expected_x', 'expected_y')]
        if not all(math.isfinite(v) for v in expected):
            raise ValueError('Reference coordinates must be finite.')
        if valid:
            observed = [float(row[k]) for k in ('observed_x', 'observed_y')]
            if not all(math.isfinite(v) for v in observed):
                raise ValueError('Valid tracking requires finite observed coordinates.')
            errors.append(math.hypot(observed[0]-expected[0], observed[1]-expected[1]))
        count += 1
    if not frames:
        raise ValueError('No annotated frames.')
    availability = sum(frames.values())/len(frames)
    median = percentile(errors, .5) if errors else None
    p95 = percentile(errors, .95) if errors else None
    return {'annotatedFrames': len(frames), 'annotatedPoints': count,
            'validPoints': len(errors), 'trackingAvailability': availability,
            'medianErrorPixels': median, 'p95ErrorPixels': p95,
            'minimumAvailability': minimum_availability,
            'attachmentGatePassed': bool(errors and median < 5 and p95 < 10
                                         and availability >= minimum_availability),
            'scope': 'Attachment only; appearance, occlusion and performance gates are separate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotations', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--minimum-availability', type=float, default=.9)
    args = parser.parse_args()
    with args.annotations.open(newline='') as source:
        result = evaluate(list(csv.DictReader(source)), args.minimum_availability)
    encoded = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
