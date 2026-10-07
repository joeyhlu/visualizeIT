"""Build a portable interactive report from an existing simulation result."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(output):
    report = json.loads((output/'report.json').read_text())
    if not report.get('synthetic') or report.get('physicalValidationPassed'):
        raise ValueError('This report template is only for explicitly synthetic, unvalidated results.')
    # Prevent embedded source data from closing the script element.
    payload = json.dumps(report).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = (ROOT/'bench'/'report.html').read_text(encoding='utf-8')
    (output/'index.html').write_text(template.replace('__REPORT_JSON__', payload), encoding='utf-8')
    print(output/'index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts'/'bench')
    build(parser.parse_args().output)
