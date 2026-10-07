"""Plot recorded synthetic experiments; this command never generates camera observations."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.cache'/'plotting'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache'/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = dict(exact_pose_control='Known pose control', translation_3mm='Position offset: 3 mm',
              translation_5mm='Position offset: 5 mm', yaw_2deg='Rotation offset: 2°',
              scale_plus_2percent='Geometry scale: +2%', focal_plus_2percent='Focal length: +2%',
              surface_lift_2mm='Surface lift: 2 mm', surface_lift_0_25mm='Surface lift: 0.25 mm',
              lost_3frames='Hide 3 of 24 frames', stale_world_anchor='Object moves; anchor stays')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts'/'bench')
    args = parser.parse_args()
    report = json.loads((args.output/'report.json').read_text())
    cases = list(report['scenarios'].items())
    labels = LABELS.copy()
    labels['lost_3frames'] = f'Hide 3 of {report["orbitFrames"]} frames'
    y = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=(13, 8.5), facecolor='#f0f4f1')
    ax.set_facecolor('#f0f4f1')
    ax.barh(y-.17, [case['medianErrorPixels'] for _, case in cases], height=.32, label='Median', color='#247d70')
    ax.barh(y+.17, [case['p95ErrorPixels'] for _, case in cases], height=.32, label='95th percentile', color='#dfa456')
    ax.axvline(5, color='#247d70', linestyle='--', linewidth=1, label='Median target: <5 px')
    ax.axvline(10, color='#b77725', linestyle='--', linewidth=1, label='95th percentile target: <10 px')
    ax.set_yticks(y, [labels[name] for name, _ in cases])
    ax.invert_yaxis()
    ax.set_xlabel('Attachment error (pixels in 1280 × 720 synthetic frames)')
    ax.grid(axis='x', alpha=.18)
    ax.set_axisbelow(True)
    for i, (_, score) in enumerate(cases):
        ax.text(score['p95ErrorPixels']+.5, i+.17, f'{score["p95ErrorPixels"]:.1f}', va='center', fontsize=9)
    ax.legend(loc='upper right', frameon=False)
    for edge in ('top', 'right', 'left'):
        ax.spines[edge].set_visible(False)
    fig.subplots_adjust(left=.27, right=.96, top=.85, bottom=.18)
    fig.text(.045, .94, 'How small mapping errors become visible sliding', fontsize=23, weight='bold', color='#173e38')
    fig.text(.045, .892, f'{report["benchmarkShape"].title()} fixture · {report["orbitFrames"]}-view orbit · focal length 850 px · horizontal orbit radius 36 cm', fontsize=11)
    loss = report['scenarios']['lost_3frames']
    fig.text(.045, .115, f'Hidden frames still count: zero error while visible, but only {loss["trackingAvailability"]:.1%} availability.', fontsize=11)
    fig.text(.045, .07, 'SIMULATION — injected errors, exact reference geometry, no real camera pose estimator.', fontsize=12, weight='bold', color='#92501e')
    fig.savefig(args.output/'error-sensitivity.png', dpi=140)
    print(args.output/'error-sensitivity.png')


if __name__ == '__main__':
    main()
