"""First inference-array mismatch between two opt-in traces; no labels read."""
import argparse
import json
from pathlib import Path
from .quality_assets import save_result


def compare(short, long):
    for index, a in enumerate(short):
        if index >= len(long): return dict(index=index, reason='long_trace_shorter')
        b = long[index]
        if a['stage'] != b['stage']: return dict(index=index, reason='different_stage', short_stage=a['stage'], long_stage=b['stage'])
        different = [name for name in a['arrays'] if a['arrays'][name] != b['arrays'].get(name)]
        if different: return dict(index=index, stage=a['stage'], arrays=different,
                                  short={n:a['arrays'][n] for n in different}, long={n:b['arrays'].get(n) for n in different})
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('short','long','output'): parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    read = lambda path: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    short, long = read(a.short), read(a.long)
    report = dict(short_records=len(short), long_records=len(long), first_mismatch=compare(short,long))
    save_result(a.output,report); print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()
