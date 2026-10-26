"""Postprocess completed, sealed CPU tracking benchmarks without opening datasets.

This tool never runs tracking, opens an evaluator manifest or changes a frozen
trial. It refuses results until an inference seal exists and verifies all raw
prediction hashes before reading scored output. Conditional errors are pooled
from actual public-frame scalar errors, never from per-sequence medians.

Usage::

    python -m bench.tracking_report \
      --synthetic-output artifacts/tracking-benchmark/synthetic-full-v1 \
      --real-output artifacts/tracking-benchmark/real-full-v1 \
      --destination docs/benchmark
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import tarfile

DATE = "2026-10-26"
ROOT = Path(__file__).resolve().parents[1]
DATASET_RELEASE_URL = "https://github.com/joeyhlu/visualizeIT/releases/tag/tracking-dataset-v1"
DATASET_RELEASE_SOURCE_COMMIT = "c10cb8b0dd464b5f12e892c0f2c277569ae872be"
FROZEN_SOURCE_SNAPSHOT_COMMIT = DATASET_RELEASE_SOURCE_COMMIT
SIMPLIFIED_MODE = "simplified_depth_backprojected_top300_shared_bank"
PRODUCTION_MODE = "production_mesh_ray_bank_all_detected_orb_up_to2000_per_view"
SYSTEMS = ("original_rigid", "current_rigid", "current_prototype")
LABELS = {"original_rigid": "Original rigid", "current_rigid": "Current rigid", "current_prototype": "Current prototype"}
SCENARIOS = ("smooth", "fast_motion", "motion_blur", "illumination", "partial_occlusion", "full_occlusion", "exit_reentry", "low_texture")
SCENARIO_PATTERN = re.compile(r"^SYNTHETIC (train|dev|test)-(" + "|".join(SCENARIOS) + r")-([0-9]+)$")
ERRORS = ("reprojectionPixels", "addMetres", "rotationDegrees", "translationMetres")


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()


def sha_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def percentile(values, percent):
    """NumPy's default linear quantile, implemented without another dependency."""
    if not values:
        return None
    values = sorted(values)
    position = (len(values)-1)*percent/100
    low, high = math.floor(position), math.ceil(position)
    return values[low]+(values[high]-values[low])*(position-low)


def ratio(numerator, denominator):
    return numerator/denominator if denominator else None


def checked_relative(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Invalid trial path")
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Missing or escaping trial path")
    return path


def scenario_from_name(name, corpus):
    if corpus != "synthetic":
        return None, None
    match = SCENARIO_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Unknown synthetic sequence name; refusing guessed stress slice: {name}")
    return match.group(2), match.group(1)


def longest_run(values):
    current = longest = 0
    for value in values:
        current = current+1 if value else 0
        longest = max(longest, current)
    return longest


def normalized_outcome(row):
    return {"frameId": row["frameId"], "state": row["state"],
            "pose": row.get("cvCameraFromSourceObject"),
            "reason": row.get("failureReason", row.get("reason"))}


def trial_statistics(raw, scored, sequence, corpus, initialization_mode=SIMPLIFIED_MODE):
    frames, details, summary = raw["frames"], scored["frames"], scored["summary"]
    if [row["frameId"] for row in frames] != [row["frameId"] for row in details]:
        raise ValueError("Scored and sealed raw frames differ")
    if len(frames) != summary["frames"] or len(frames) == 0:
        raise ValueError("Scored frame denominator differs or is empty")
    if any((row["state"] == "tracking") != detail["public"] for row, detail in zip(frames, details)):
        raise ValueError("Scored public states differ from sealed predictions")
    public = sum(detail["public"] for detail in details)
    visible = sum(detail["visible"] for detail in details)
    visible_public = sum(detail["public"] and detail["visible"] for detail in details)
    good = sum(detail["accurate"] for detail in details)
    for name, expected in (("publicTrackingFrames", public), ("visibleFrames", visible),
                           ("visibleTrackingFrames", visible_public), ("accurateVisibleFrames", good)):
        if summary[name] != expected:
            raise ValueError(f"Scored summary disagrees with actual frame rows: {name}")
    errors = {metric: [] for metric in ERRORS}
    for detail in details:
        if detail["accurate"] and not (detail["public"] and detail["visible"]):
            raise ValueError("Coverage flag lacks a public visible frame")
        if detail["public"]:
            for metric in ERRORS:
                value = detail.get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("Public-frame metric is missing/nonfinite/negative")
                errors[metric].append(value)
            if detail["accurate"] != (detail["visible"] and detail["reprojectionPixels"] < 5.):
                raise ValueError("Native 5 px coverage flag disagrees with scalar error")
    attempted = [row for row in frames if row.get("updateAttempted")]
    if len(attempted) != raw["timedUpdateFrames"] or summary["timedUpdateFrames"] != len(attempted):
        raise ValueError("Timed update denominator differs")
    times = [row["processingSeconds"] for row in attempted]
    if any(not math.isfinite(value) or value < 0 for value in times):
        raise ValueError("Invalid attempted update time")
    if not math.isclose(sum(times), raw["updateTotalSeconds"], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Raw attempted update times do not sum to trial total")
    scenario, split = scenario_from_name(sequence["name"], corpus)
    outcomes = [normalized_outcome(row) for row in frames]
    item = dict(corpus=corpus, objectId=sequence["objectId"], name=sequence["name"],
                initializationMode=initialization_mode,
                system=raw["system"], repeat=raw["repeat"], scenario=scenario, split=split,
                scheduledFrames=len(frames), visibleFrames=visible, publicTrackingFrames=public,
                nonPublicOutputFrames=len(frames)-public,
                visibleTrackingFrames=visible_public, native5pxVisibleFrames=good,
                allFrameTrackingAvailability=ratio(public, len(frames)),
                visibleTrackingAvailability=ratio(visible_public, visible), native5pxVisibleCoverage=ratio(good, visible),
                timedUpdateFrames=len(attempted), skippedUpdateFrames=len(frames)-len(attempted),
                updateTotalSeconds=sum(times), decodeAndHashSeconds=raw.get("decodeAndHashTotalSeconds", 0.),
                replayWallSeconds=raw["replayWallSeconds"],
                updateFPS=ratio(len(attempted), sum(times)),
                replayFPS=ratio(len(attempted), raw["replayWallSeconds"]) if attempted else None,
                setupFailure=raw.get("setupFailure"), initializationFailure=sequence.get("initializationFailure"),
                hiddenPublicPoseFrames=summary["hiddenPublicPoseFrames"], recovery=summary["recovery"],
                longestPublicLossRun=longest_run(not row["public"] for row in details),
                longestVisible5pxCoverageGap=longest_run(row["visible"] and not row["accurate"] for row in details),
                failures=dict(Counter(row.get("failureReason", row.get("reason")) or "unspecified"
                                      for row in frames if row["state"] != "tracking")),
                outcomeSHA256=hashlib.sha256(canonical(outcomes)).hexdigest())
    for metric, values in errors.items():
        item[metric+"Median"] = percentile(values, 50)
        item[metric+"P95"] = percentile(values, 95)
    return dict(summary=item, errors=errors, times=times, outcomes=outcomes,
                visibility=[(row["frameId"],row["visible"]) for row in details], scoredFrames=details)


def load_corpus(output, corpus, allow_production=False):
    """All inference-integrity checks precede ANY score/result file reads."""
    output = Path(output)
    seal_path = output/"inference-seal.json"
    if not seal_path.is_file():
        raise ValueError(f"Inference is not sealed; refusing score/results reads: {output}")
    seal = read_json(seal_path)
    freeze_path = output/"freeze.json"
    if sha_file(freeze_path) != seal["freezeSHA256"]:
        raise ValueError("Freeze differs from inference seal")
    freeze = read_json(freeze_path)
    if freeze.get("evaluatorOpenedDuringInference") is not False:
        raise ValueError("Inference label-isolation declaration is missing/false")
    if freeze["configuration"]["thresholds"]["accurateReprojectionPixels"] != 5.:
        raise ValueError("This report requires the frozen native 5 px threshold")
    for key, value in freeze.items():
        if seal.get(key) != value:
            raise ValueError("Seal metadata differs from frozen configuration")
    for relative, expected in seal["trialHashes"].items():
        if sha_file(checked_relative(output, relative)) != expected:
            raise ValueError("Raw prediction differs from inference seal")
    results_path = output/"results.json"
    if not results_path.is_file():
        raise ValueError(f"Inference sealed but evaluation has not completed: {output}")
    results = read_json(results_path)
    if results["inferenceSealSHA256"] != sha_file(seal_path) or results["benchmarkId"] != seal["benchmarkId"]:
        raise ValueError("Results belong to a different inference seal")
    summaries = {(row["objectId"], row["system"], row["repeat"]): row for row in results["sequenceTrials"]}
    if len(summaries) != len(results["sequenceTrials"]):
        raise ValueError("Duplicate scored summary identity")
    entries, index, expected_paths = [], [], set()
    repeats = freeze["configuration"]["repeats"]
    systems = freeze["configuration"]["systems"]
    if tuple(systems) != SYSTEMS:
        raise ValueError("Publishable comparison requires all three frozen systems in standard order")
    for sequence in seal["sequences"]:
        seen = set()
        for relative in sequence["trials"]:
            expected_paths.add(relative)
            raw = read_json(checked_relative(output, relative))
            key = (sequence["objectId"], raw["system"], raw["repeat"])
            if key in seen or raw["system"] not in systems or raw["repeat"] not in range(repeats):
                raise ValueError("Duplicate/unknown trial system or repeat")
            seen.add(key)
            scored_relative = relative[:-5]+"-scored.json" if relative.endswith(".json") else ""
            scored_path = checked_relative(output, scored_relative)
            scored = read_json(scored_path)
            if scored["summary"] != summaries.get(key):
                raise ValueError("Scored file summary differs from results.json")
            description=freeze['configuration'].get('initialization','')
            if 'depth-backprojected' in description and 'first 300' in description:
                initialization_mode=SIMPLIFIED_MODE
            elif allow_production and freeze['configuration'].get('initializationMode')=='production_mesh':
                initialization_mode=PRODUCTION_MODE
            else:
                raise ValueError('This report is specifically for the frozen shared simplified-initialization full runs')
            entries.append(trial_statistics(raw, scored, sequence, corpus, initialization_mode))
            index.append(dict(corpus=corpus, objectId=key[0], system=key[1], repeat=key[2],
                              predictionPath=relative, predictionSHA256=seal["trialHashes"][relative],
                              scoredPath=scored_relative, scoredSHA256=sha_file(scored_path)))
        if len(seen) != len(systems)*repeats:
            raise ValueError("Missing system/repeat trial")
    if expected_paths != set(seal["trialHashes"]) or len(entries) != len(summaries):
        raise ValueError("Sealed/raw/scored trial identity sets differ")
    # All systems and repeats must share frame identities and visibility labels.
    for _, group in group_entries(entries, ("objectId",)).items():
        expected = [row["frameId"] for row in group[0]["outcomes"]]
        if any([row["frameId"] for row in item["outcomes"]] != expected for item in group[1:]):
            raise ValueError("System/repeat trial frame identities differ")
        if any(item["visibility"] != group[0]["visibility"] for item in group[1:]):
            raise ValueError("System/repeat visibility labels differ")
    return dict(corpus=corpus, freeze=freeze, entries=entries, traceIndex=index,
                provenance=dict(benchmarkId=results["benchmarkId"], dataset=results["dataset"],
                    evidenceKind=results["evidenceKind"], inferenceManifestSHA256=seal["inferenceSHA256"],
                    evaluatorManifestSHA256=seal["evaluatorSHA256"], inferenceSealSHA256=sha_file(seal_path),
                    resultsSHA256=sha_file(results_path), freezeSHA256=sha_file(freeze_path),
                    historicalCommit=seal["historicalCommit"], historicalTrackerSHA256=seal["historicalTrackerSHA256"],
                    sourceHashes=seal["sourceHashes"], hardware=seal["hardware"], configuration=seal["configuration"]))


def group_entries(entries, fields):
    groups = defaultdict(list)
    for entry in entries:
        groups[tuple(entry["summary"][field] for field in fields)].append(entry)
    return groups


def determinism(entries):
    groups = group_entries(entries, ("objectId", "system"))
    details, max_pose_difference, compared = [], 0., 0
    for (object_id, system), group in sorted(groups.items()):
        group = sorted(group, key=lambda entry: entry["summary"]["repeat"])
        first = group[0]["outcomes"]
        changed_frames = 0
        for entry in group[1:]:
            for baseline, row in zip(first, entry["outcomes"]):
                changed_frames += baseline != row
                compared += 1
                if baseline["pose"] is not None and row["pose"] is not None:
                    max_pose_difference = max(max_pose_difference, max(abs(a-b) for line_a, line_b in zip(baseline["pose"], row["pose"]) for a,b in zip(line_a,line_b)))
        hashes = sorted({entry["summary"]["outcomeSHA256"] for entry in group})
        details.append(dict(objectId=object_id, system=system, repeatCount=len(group), distinctOutcomeHashes=len(hashes),
                            allRepeatsIdentical=len(hashes)==1, changedFramesVersusFirst=changed_frames,
                            outcomeSHA256ByRepeat=[entry["summary"]["outcomeSHA256"] for entry in group]))
    return dict(sequenceSystemGroups=len(details), identicalSequenceSystemGroups=sum(row["allRepeatsIdentical"] for row in details),
                comparedFramesAgainstFirstRepeat=compared, changedFramesAgainstFirstRepeat=sum(row["changedFramesVersusFirst"] for row in details),
                maxPoseMatrixElementAbsoluteDifference=max_pose_difference, sequenceGroups=details,
                definition="Exact state/frame/reason/SE(3) numeric values; excludes clocks, session IDs, timings and internal statistics. Matrix-element difference mixes dimensionless rotation and metre translation; it is not a physical pose-error metric.")


def aggregate(entries):
    rows = [entry["summary"] for entry in entries]
    total, visible = sum(row["scheduledFrames"] for row in rows), sum(row["visibleFrames"] for row in rows)
    timed = sum(row["timedUpdateFrames"] for row in rows)
    updates = sum(row["updateTotalSeconds"] for row in rows)
    replay = sum(row["replayWallSeconds"] for row in rows if row["timedUpdateFrames"])
    errors = {metric: [value for entry in entries for value in entry["errors"][metric]] for metric in ERRORS}
    times = [value for entry in entries for value in entry["times"]]
    first_repeats = [row for row in rows if row["repeat"] == 0]
    events = [event for row in rows for event in row["recovery"]]
    delays = [event["observableFramesToTwoAccurate"] for event in events if event["observableFramesToTwoAccurate"] is not None]
    failures = Counter()
    for row in rows:
        failures.update(row["failures"])
    repeat_rows = []
    for repeat in sorted({row["repeat"] for row in rows}):
        selected = [row for row in rows if row["repeat"] == repeat]
        n, v = sum(row["timedUpdateFrames"] for row in selected), sum(row["visibleFrames"] for row in selected)
        repeat_rows.append(dict(repeat=repeat, native5pxVisibleCoverage=ratio(sum(row["native5pxVisibleFrames"] for row in selected), v),
                                updateFPS=ratio(n, sum(row["updateTotalSeconds"] for row in selected))))
    if len({row['initializationMode'] for row in rows}) != 1:
        raise ValueError('Do not aggregate different initialization modes')
    result = dict(corpus=rows[0]["corpus"], system=rows[0]["system"],initializationMode=rows[0]['initializationMode'],sequences=len(first_repeats), sequenceTrials=len(rows),
                  uniqueScheduledObjectFrames=sum(row["scheduledFrames"] for row in first_repeats),
                  uniqueVisibleObjectFrames=sum(row["visibleFrames"] for row in first_repeats),
                  scheduledFramesWithRepeats=total, visibleFramesWithRepeats=visible,
                  publicTrackingFramesWithRepeats=sum(row["publicTrackingFrames"] for row in rows),
                  nonPublicOutputFramesWithRepeats=sum(row["nonPublicOutputFrames"] for row in rows),
                  native5pxVisibleFramesWithRepeats=sum(row["native5pxVisibleFrames"] for row in rows),
                  allFrameTrackingAvailability=ratio(sum(row["publicTrackingFrames"] for row in rows), total),
                  visibleTrackingAvailability=ratio(sum(row["visibleTrackingFrames"] for row in rows), visible),
                  native5pxVisibleCoverage=ratio(sum(row["native5pxVisibleFrames"] for row in rows), visible),
                  timedUpdateFrames=timed, skippedUpdateFrames=total-timed,
                  initializationFailureSequences=len({row["objectId"] for row in rows if row["initializationFailure"]}),
                  setupFailureTrials=sum(bool(row["setupFailure"]) for row in rows),
                  updateTotalSeconds=updates, decodeAndHashSeconds=sum(row["decodeAndHashSeconds"] for row in rows),
                  replayWallSecondsAttemptedTrials=replay, updateFPS=ratio(timed, updates), replayFPS=ratio(timed,replay),
                  updateMedianMilliseconds=1000*percentile(times,50) if times else None,
                  updateP95Milliseconds=1000*percentile(times,95) if times else None,
                  conditionalPublicPoseFrames=sum(len(entry["errors"][ERRORS[0]]) for entry in entries),
                  hiddenPublicPoseFrames=sum(row["hiddenPublicPoseFrames"] for row in rows),
                  recoveryEventsWithRepeats=len(events), successfulRecoveryEventsWithRepeats=len(delays),
                  failedRecoveryEventsWithRepeats=len(events)-len(delays),
                  recoveryObservableFramesMedian=percentile(delays,50), recoveryObservableFramesP95=percentile(delays,95),
                  longestPublicLossRun=max(row["longestPublicLossRun"] for row in rows),
                  longestVisible5pxCoverageGap=max(row["longestVisible5pxCoverageGap"] for row in rows),
                  failures=dict(failures.most_common()), repeatVariation=repeat_rows,
                  determinism=determinism(entries))
    for metric, values in errors.items():
        result[metric+"Median"] = percentile(values,50)
        result[metric+"P95"] = percentile(values,95)
    return result


def initialization_diagnostic(simplified_output,production_output):
    """A separately sealed small paired diagnostic; never merge with full runs."""
    simplified=load_corpus(simplified_output,'real')
    production=load_corpus(production_output,'real',allow_production=True)
    a,b=simplified['provenance'],production['provenance']
    for key in ('inferenceManifestSHA256','evaluatorManifestSHA256','sourceHashes','historicalCommit'):
        if a[key]!=b[key]:
            raise ValueError('Initialization diagnostic has an extra source/data confound: '+key)
    config_a={key:value for key,value in a['configuration'].items() if key not in ('initialization','initializationMode')}
    config_b={key:value for key,value in b['configuration'].items() if key not in ('initialization','initializationMode')}
    if config_a!=config_b:
        raise ValueError('Initialization diagnostic configurations differ beyond bank policy')
    seal_a=read_json(Path(simplified_output)/'inference-seal.json')
    seal_b=read_json(Path(production_output)/'inference-seal.json')
    if seal_a['geometryBindings']!=seal_b['geometryBindings']:
        raise ValueError('Initialization diagnostic RGB/model/calibration/frame bindings differ')
    rows=[]
    banks=[]
    for label,corpus,seal in (('simplified',simplified,seal_a),('production_mesh',production,seal_b)):
        for (_,),entries in sorted(group_entries(corpus['entries'],('system',)).items()):
            row=aggregate(entries);row['diagnosticMode']=label
            row['wrongVisiblePublicFramesWithRepeats']=sum(entry['summary']['visibleTrackingFrames']-entry['summary']['native5pxVisibleFrames'] for entry in entries)
            rows.append(row)
        banks.append(dict(mode=label,objects=[dict(objectId=row['objectId'],anchors=row.get('anchors'),
            initializationSeconds=row.get('initializationSeconds'),initializationFailure=row.get('initializationFailure')) for row in seal['sequences']]))
    return dict(scope='Separate paired prefix diagnostic only; not a production full-corpus run',
                aggregate=rows,bankSetup=banks,provenance={'simplified':a,'production_mesh':b},
                productionPairedTradeoffs=[paired_tradeoff(production['entries'],'original_rigid',candidate) for candidate in ('current_rigid','current_prototype')],
                traceIndex=simplified['traceIndex']+production['traceIndex'],
                caveat='Both anchor geometry and detector feature cap differ between adapters; this does not isolate either component alone')


def paired_tradeoff(entries, baseline_system, candidate_system):
    """Unique-frame repeat-0 quality/abstention accounting on paired outputs.

    'Wrong' below means a public, reference-visible pose at >=5 native pixels
    mean-model reprojection error. It is NOT a human/detection false positive.
    """
    baseline = {entry["summary"]["objectId"]:entry for entry in entries
                if entry["summary"]["system"]==baseline_system and entry["summary"]["repeat"]==0}
    candidate = {entry["summary"]["objectId"]:entry for entry in entries
                 if entry["summary"]["system"]==candidate_system and entry["summary"]["repeat"]==0}
    if not baseline or set(baseline)!=set(candidate):
        raise ValueError("Paired repeat-0 systems do not have the same sequence set")
    counts = Counter()
    shared_errors = {"baseline":[],"candidate":[]}
    recovery=[]
    for object_id in sorted(baseline):
        left,right=baseline[object_id],candidate[object_id]
        if left["visibility"]!=right["visibility"]:
            raise ValueError("Paired reference frame/visibility labels differ")
        for old,new in zip(left["scoredFrames"],right["scoredFrames"]):
            counts["scheduledUniqueObjectFrames"]+=1
            if not old["visible"]:
                continue
            counts["visibleUniqueObjectFrames"]+=1
            counts["baselineQualifiedVisibleFrames"]+=old["accurate"]
            counts["candidateQualifiedVisibleFrames"]+=new["accurate"]
            counts["baselineWrongVisiblePublicFrames"]+=old["public"] and not old["accurate"]
            counts["candidateWrongVisiblePublicFrames"]+=new["public"] and not new["accurate"]
            if old["public"] and new["public"]:
                counts["bothPublicVisibleFrames"]+=1
                counts["sharedBaselineQualifiedFrames"]+=old["accurate"]
                counts["sharedCandidateQualifiedFrames"]+=new["accurate"]
                counts["sharedBaselineWrongCandidateQualified"]+=not old["accurate"] and new["accurate"]
                counts["sharedBaselineQualifiedCandidateWrong"]+=old["accurate"] and not new["accurate"]
                shared_errors["baseline"].append(old["reprojectionPixels"])
                shared_errors["candidate"].append(new["reprojectionPixels"])
            elif old["public"]:
                counts["withheldBaselineQualifiedFrames" if old["accurate"] else "withheldBaselineWrongFrames"]+=1
            elif new["public"]:
                counts["addedQualifiedWhereBaselineWithheld" if new["accurate"] else "addedWrongWhereBaselineWithheld"]+=1
            else:
                counts["bothWithheldVisibleFrames"]+=1
        old_events={event["returnFrameId"]:event for event in left["summary"]["recovery"]}
        new_events={event["returnFrameId"]:event for event in right["summary"]["recovery"]}
        if set(old_events)!=set(new_events):
            raise ValueError("Paired recovery event identities differ")
        for frame in old_events:
            a=old_events[frame]["observableFramesToTwoAccurate"]
            b=new_events[frame]["observableFramesToTwoAccurate"]
            recovery.append(dict(objectId=object_id,name=left["summary"]["name"],returnFrameId=frame,
                                 baselineObservableFrames=a,candidateObservableFrames=b,
                                 addedObservableFrames=b-a if a is not None and b is not None else None))
    fields=("scheduledUniqueObjectFrames","visibleUniqueObjectFrames","baselineQualifiedVisibleFrames","candidateQualifiedVisibleFrames",
            "baselineWrongVisiblePublicFrames","candidateWrongVisiblePublicFrames","bothPublicVisibleFrames",
            "sharedBaselineQualifiedFrames","sharedCandidateQualifiedFrames","sharedBaselineWrongCandidateQualified",
            "sharedBaselineQualifiedCandidateWrong","withheldBaselineQualifiedFrames","withheldBaselineWrongFrames",
            "addedQualifiedWhereBaselineWithheld","addedWrongWhereBaselineWithheld","bothWithheldVisibleFrames")
    result={name:counts[name] for name in fields}
    modes={entry['summary']['initializationMode'] for entry in list(baseline.values())+list(candidate.values())}
    if len(modes)!=1:
        raise ValueError('Paired tracker comparison must share one initialization mode')
    result.update(corpus=next(iter(baseline.values()))["summary"]["corpus"],initializationMode=next(iter(modes)),baselineSystem=baseline_system,candidateSystem=candidate_system,
                  repeat=0,qualifiedVisibleDelta=counts["candidateQualifiedVisibleFrames"]-counts["baselineQualifiedVisibleFrames"],
                  wrongVisiblePublicDelta=counts["candidateWrongVisiblePublicFrames"]-counts["baselineWrongVisiblePublicFrames"],
                  sharedQualifiedDelta=counts["sharedCandidateQualifiedFrames"]-counts["sharedBaselineQualifiedFrames"],
                  sharedBaselineReprojectionP95=percentile(shared_errors["baseline"],95),
                  sharedCandidateReprojectionP95=percentile(shared_errors["candidate"],95),
                  baselineHiddenPublicPoseFrames=sum(entry["summary"]["hiddenPublicPoseFrames"] for entry in baseline.values()),
                  candidateHiddenPublicPoseFrames=sum(entry["summary"]["hiddenPublicPoseFrames"] for entry in candidate.values()),
                  pairedRecovery=recovery,
                  candidateRecoveryFailuresAfterBaselineSuccess=sum(row["baselineObservableFrames"] is not None and row["candidateObservableFrames"] is None for row in recovery),
                  candidateRecoverySuccessesAfterBaselineFailure=sum(row["baselineObservableFrames"] is None and row["candidateObservableFrames"] is not None for row in recovery),
                  definition="Repeat 0, unique paired frames only; wrong means visible public mean-model reprojection >=5 native pixels, not human/detection false positive")
    if result["qualifiedVisibleDelta"] != result["sharedQualifiedDelta"]-result["withheldBaselineQualifiedFrames"]+result["addedQualifiedWhereBaselineWithheld"]:
        raise ValueError("Paired coverage/abstention accounting does not balance")
    return result


def source_context(corpora):
    """Separate the measured source identities from later report-only commits."""
    def git(*args):
        result=subprocess.run(['git',*args],cwd=ROOT,check=True,capture_output=True)
        return result.stdout
    context=dict(reportToolSHA256=sha_file(Path(__file__)),datasetReleaseURL=DATASET_RELEASE_URL,
                 datasetReleaseSourceCommit=DATASET_RELEASE_SOURCE_COMMIT)
    try:
        context['reportWorkingTreeBaseCommit']=git('rev-parse','HEAD').decode().strip()
        anchor=git('log','-1','--format=%H',FROZEN_SOURCE_SNAPSHOT_COMMIT,'--','bench/tracking_benchmark.py').decode().strip()
        context['benchmarkHarnessLastChangeCommit']=anchor
        context['frozenSourceSnapshotCommit']=FROZEN_SOURCE_SNAPSHOT_COMMIT
        context['benchmarkHarnessAtAnchorSHA256']=hashlib.sha256(git('show',FROZEN_SOURCE_SNAPSHOT_COMMIT+':bench/tracking_benchmark.py')).hexdigest()
        context['workingBenchmarkHarnessSHA256']=sha_file(ROOT/'bench/tracking_benchmark.py')
        context['allFrozenHarnessHashesMatchAnchor']=all(corpus['provenance']['sourceHashes']['bench/tracking_benchmark.py']==context['benchmarkHarnessAtAnchorSHA256'] for corpus in corpora)
        context['workingHarnessMatchesFrozen']=context['workingBenchmarkHarnessSHA256']==context['benchmarkHarnessAtAnchorSHA256']
    except (subprocess.CalledProcessError,FileNotFoundError,KeyError) as error:
        context['gitContextUnavailable']=type(error).__name__
    neural=ROOT/'docs/benchmark/tracking-neural-cpu-v1/summary.json'
    if neural.is_file():
        summary=read_json(neural)
        context['separateNeuralDiagnostic']=dict(summarySHA256=sha_file(neural),summaryPath='tracking-neural-cpu-v1/README.md',
            fullNetworkExecuted=summary.get('fullNetworkExecuted'),networkForwards=summary.get('networkForwards'),
            referenceExportParity=summary.get('referenceExportParity'),newScoredBenchmarkFrames=summary.get('newScoredBenchmarkFrames'),
            elapsedSeconds=summary.get('supervision',{}).get('elapsedSeconds'),peakSampledRSSBytes=summary.get('supervision',{}).get('peakSampledRSSBytes'),
            workbenchAdmissionGranted=summary.get('workbenchAdmissionGranted'))
    return context


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key, value in row.items() if not isinstance(value,(dict,list))))
    with Path(path).open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key,value in row.items() if key in fields})


def fmt(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def percent(value):
    return "—" if value is None else f"{100*value:.2f}%"


def plots(destination, aggregate_rows):
    from PIL import Image, ImageDraw, ImageFont
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf",22)
        font = ImageFont.truetype("DejaVuSans.ttf",16)
    except OSError:
        title_font, font = ImageFont.load_default(), ImageFont.load_default()
    colors = {"original_rigid":"#697586", "current_rigid":"#3158cf", "current_prototype":"#098a7b"}
    names = []
    for metric, title, suffix in (("native5pxVisibleCoverage", "Visible-frame <5 px coverage — SIMPLIFIED INIT", "coverage"),
                                   ("updateFPS", "CPU tracking update speed — SIMPLIFIED INIT", "speed")):
        image = Image.new("RGB", (1050,570), "#ffffff")
        draw = ImageDraw.Draw(image)
        draw.text((26,17), title, fill="#14202e", font=title_font)
        draw.text((26,49), "Shared simplified bank; not an exact production-pipeline, phone or neural benchmark.", fill="#4d5b6b", font=font)
        for panel, corpus in enumerate(("synthetic","real")):
            y = 96+panel*213
            draw.text((26,y), "SYNTHETIC stress" if corpus=="synthetic" else "REAL HB dataset reference", fill="#14202e", font=title_font)
            values = [row for row in aggregate_rows if row["corpus"] == corpus]
            maximum = 1. if suffix=="coverage" else max((row[metric] or 0 for row in values),default=1.)*1.15 or 1.
            for j, system in enumerate(SYSTEMS):
                row = next(row for row in values if row["system"]==system)
                yy = y+43+j*44
                draw.text((26,yy+4), LABELS[system], fill="#243449", font=font)
                draw.rectangle((230,yy,925,yy+29), fill="#eef1f5")
                value = row[metric]
                if value is not None:
                    width = max(0,min(695,round(695*value/maximum)))
                    draw.rectangle((230,yy,230+width,yy+29),fill=colors[system])
                label = percent(value) if suffix=="coverage" else fmt(value)
                draw.text((937,yy+4), label,fill="#243449",font=font)
        draw.text((26,538), "SIMPLIFIED INIT: depth-backprojected top 300 ORB/view, not production mesh-ray initialization.",fill="#4d5b6b",font=font)
        name=f"tracking-{suffix}-{DATE}.png"
        image.save(destination/name, optimize=True)
        names.append(name)
    return names


def markdown(corpora, aggregates, stresses, pairs, filenames, context):
    lines=[f"# Shared simplified-initialization tracking experiment — {DATE}", "", "## Scope and measured comparison", "",
           "**These full-corpus results use SHARED SIMPLIFIED INITIALIZATION, not the exact full production pipeline.** The adapter backprojects authorized masked RGB-D using known onboarding poses and keeps only the first 300 ORB features per view (at most 3,000 anchors). The production bank instead uses mesh-ray surface intersections and all detected features up to its configured 2,000 per view. A common simplified bank controls the comparison but may materially change absolute quality and the relative ranking; do not generalize this experiment to the unmeasured full-production initialization.", "",
           "This is **initialized RGB rigid tracking**, not automatic object discovery, independent physical-attachment accuracy, or phone/live-camera performance. Real HB footage and explicitly synthetic stress tests are reported separately. All three systems receive the same simplified immutable bank from ten authorized masked RGB-D/pose onboarding frames; scored updates receive RGB and calibration only.", "",
           ("**Initialization sensitivity is confirmed by the separately sealed prefix diagnostic below.** Its limited production-mesh results must not replace or be extrapolated into the immutable simplified full-corpus tables; a full-corpus production-initialization comparison is not reported here." if context.get('initializationDiagnostic') else
            "**Production-initialization interpretation is provisional:** a separate paired real-HB prefix diagnostic compares simplified versus exact production mesh-ray banks on the first three objects, 110 scored frames each, three systems, one repeat. Its result is pending; it does not retroactively change these immutable full-corpus results."), ""]
    real_base=next(row for row in aggregates if row['corpus']=='real' and row['system']=='original_rigid')
    real_current=next(row for row in aggregates if row['corpus']=='real' and row['system']=='current_prototype')
    lines += [f"**Main finding under the simplified bank:** real HB native-5-pixel visible coverage is only {percent(real_base['native5pxVisibleCoverage'])} for the original and {percent(real_current['native5pxVisibleCoverage'])} for the current wrapper. This does not establish reliable real-world tracking, but it is also **not a measured verdict on the full-production bank**. On synthetic footage, the current code exposes fewer wrong-threshold poses but also withholds usable outputs; the paired analysis below separates those effects.", "",
              f"Complete licensed real and synthetic media, annotation packs and checksums are published in the [tracking dataset prerelease]({DATASET_RELEASE_URL}) (release target `{DATASET_RELEASE_SOURCE_COMMIT}`). The small committed recipe/examples are not the full media corpus.", ""]
    diagnostic=context.get('initializationDiagnostic')
    if diagnostic:
        first=diagnostic['aggregate'][0]
        lines += ["## Separate paired initialization diagnostic — limited real prefix", "",
                  f"The first {first['sequences']} HB objects ({', '.join(str(row['objectId']) for row in diagnostic['bankSetup'][0]['objects'])}) use the same RGB/model/calibration/frame subset: {first['uniqueScheduledObjectFrames']} scheduled object-frames, {first['uniqueVisibleObjectFrames']} reference-visible, {diagnostic['provenance']['simplified']['configuration']['repeats']} repeat, and identical source/configuration except initialization policy. These rows are **not pooled into** the full-corpus tables.", "",
                  "| Bank / tracker | Qualified <5 px / visible frames | Wrong-threshold visible public poses | Public / all frames | Init failures | Skipped updates |", "|---|---:|---:|---:|---:|---:|"]
        for row in diagnostic['aggregate']:
            lines.append(f"| {row['diagnosticMode']} / {LABELS[row['system']]} | {row['native5pxVisibleFramesWithRepeats']}/{row['visibleFramesWithRepeats']} ({percent(row['native5pxVisibleCoverage'])}) | {row['wrongVisiblePublicFramesWithRepeats']} | {row['publicTrackingFramesWithRepeats']}/{row['scheduledFramesWithRepeats']} | {row['initializationFailureSequences']} | {row['skippedUpdateFrames']} |")
        lines += ["", "The production mesh-ray bank materially changes this prefix: it removes the shared initialization failure and changes coverage. This confirms that the earlier simplified percentages cannot be presented as full-production-pipeline quality. It does **not** establish performance on the remaining objects, later frames, other datasets, or neural tracking.", "",
                  "The adapters change both point geometry and the feature cap; this is not an isolated causal test of either detail. Production setup is substantially more expensive and is excluded from update FPS. Per-object bank evidence:", "",
                  "| Bank | Object | Anchors | Initialization seconds | Failure |", "|---|---:|---:|---:|---|"]
        for bank in diagnostic['bankSetup']:
            for row in bank['objects']:
                lines.append(f"| {bank['mode']} | {row['objectId']} | {row['anchors'] if row['anchors'] is not None else '—'} | {fmt(row['initializationSeconds'])} | {row['initializationFailure'] or 'none'} |")
        lines += ["", f"[Initialization-prefix CSV]({filenames['initializationCSV']}); full provenance, paired-output accounting, trace hashes and bank metadata are in aggregate JSON. Both prefix inference seals and all raw predictions were checked before reading their scored output.", ""]
    lines += ['## Full-corpus shared simplified-initialization results','']
    for corpus in corpora:
        name=corpus["corpus"]
        rows=[row for row in aggregates if row["corpus"]==name]
        baseline=next(row for row in rows if row["system"]=="original_rigid")
        first=rows[0]
        lines += ["### "+("Synthetic stress corpus" if name=="synthetic" else "Real HB validation corpus"), "",
                  f"{first['sequences']} sequences; {first['uniqueScheduledObjectFrames']:,} unique scheduled object-frames; {first['uniqueVisibleObjectFrames']:,} reference-visible object-frames (visibility ≥0.5). Metrics pool all repeats; unique counts do not multiply repeated observations.", "",
                  "| System | <5 px visible coverage | Public pose / all frames | Public pose / visible frames | Update FPS | Replay FPS | Init failures / sequences | Skipped updates |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {LABELS[row['system']]} | {percent(row['native5pxVisibleCoverage'])} | {percent(row['allFrameTrackingAvailability'])} | {percent(row['visibleTrackingAvailability'])} | {fmt(row['updateFPS'])} | {fmt(row['replayFPS'])} | {row['initializationFailureSequences']}/{row['sequences']} | {row['skippedUpdateFrames']:,}/{row['scheduledFramesWithRepeats']:,} |")
        lines += ["", "Relative to the exact original tracker:"]
        for row in rows[1:]:
            delta=100*(row["native5pxVisibleCoverage"]-baseline["native5pxVisibleCoverage"]) if row["native5pxVisibleCoverage"] is not None and baseline["native5pxVisibleCoverage"] is not None else None
            speed=ratio(row["updateFPS"],baseline["updateFPS"]) if row["updateFPS"] is not None and baseline["updateFPS"] else None
            lines.append(f"- **{LABELS[row['system']]}:** {fmt(delta)} percentage-point change in native-5-pixel visible coverage; {fmt(speed)}× attempted-update throughput. A stricter public-pose policy may lower availability; speed is not a quality score.")
        lines += ["", "**Conditional pose errors — public outputs only.** These exclude lost/uninitialized frames; read them together with coverage above, not as unconditional accuracy.", "",
                  "| System | Public frame samples | Mean-model reprojection px P50 / P95 | ADD m P50 / P95 | Rotation ° P50 / P95 | Translation m P50 / P95 |", "|---|---:|---:|---:|---:|---:|"]
        for row in rows:
            cells=[f"{fmt(row[metric+'Median'], 4 if 'Metres' in metric else 2)} / {fmt(row[metric+'P95'],4 if 'Metres' in metric else 2)}" for metric in ERRORS]
            lines.append(f"| {LABELS[row['system']]} | {row['conditionalPublicPoseFrames']:,} | "+" | ".join(cells)+" |")
        lines += ["", "| System | Recovery events / recovered / failed | Successful recovery delay P50 / P95, observable frames | Longest public loss run | Longest visible <5 px coverage gap | Hidden public poses |", "|---|---:|---:|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {LABELS[row['system']]} | {row['recoveryEventsWithRepeats']} / {row['successfulRecoveryEventsWithRepeats']} / {row['failedRecoveryEventsWithRepeats']} | {fmt(row['recoveryObservableFramesMedian'],1)} / {fmt(row['recoveryObservableFramesP95'],1)} | {row['longestPublicLossRun']} | {row['longestVisible5pxCoverageGap']} | {row['hiddenPublicPoseFrames']} |")
        lines += ["", "Recovery starts only after a reference visibility-zero interval; success needs two consecutive visible public outputs below 5 px before another hidden interval. Events above include repeats. No observed event means **not tested**, not perfect recovery.", ""]
    lines += ["## Under simplified initialization, is the current code simply worse?", "",
              "A smaller error among accepted poses can coexist with **fewer usable poses overall**. The table below isolates that tradeoff on repeat 0 only (unique paired object-frames, not repeat-multiplied counts). ‘Wrong’ is shorthand for a public pose on a reference-visible frame at **≥5 px mean-model reprojection error**—an error-threshold proxy, **not** a human-verified attachment or detector false positive.", "",
              "| Corpus / baseline → candidate | Qualified visible outputs | Wrong-threshold visible outputs | Both-public subset / P95 px baseline → candidate | Net qualified outputs |",
              "|---|---:|---:|---:|---:|"]
    for row in pairs:
        lines.append(f"| {row['corpus']} / {LABELS[row['baselineSystem']]} → {LABELS[row['candidateSystem']]} | {row['baselineQualifiedVisibleFrames']:,} → {row['candidateQualifiedVisibleFrames']:,} | {row['baselineWrongVisiblePublicFrames']:,} → {row['candidateWrongVisiblePublicFrames']:,} | {row['bothPublicVisibleFrames']:,} / {fmt(row['sharedBaselineReprojectionP95'])} → {fmt(row['sharedCandidateReprojectionP95'])} | {row['qualifiedVisibleDelta']:+,} |")
    lines += ["", "| Corpus / baseline → candidate | Baseline-qualified outputs withheld | Baseline-wrong outputs withheld | New qualified / wrong outputs where baseline withheld | Net qualified change within shared public subset |", "|---|---:|---:|---:|---:|"]
    for row in pairs:
        lines.append(f"| {row['corpus']} / {LABELS[row['baselineSystem']]} → {LABELS[row['candidateSystem']]} | {row['withheldBaselineQualifiedFrames']:,} | {row['withheldBaselineWrongFrames']:,} | {row['addedQualifiedWhereBaselineWithheld']:,} / {row['addedWrongWhereBaselineWithheld']:,} | {row['sharedQualifiedDelta']:+,} |")
    lines.append("")
    for corpus in ("synthetic","real"):
        row=next(pair for pair in pairs if pair['corpus']==corpus and pair['baselineSystem']=='original_rigid' and pair['candidateSystem']=='current_prototype')
        lines.append(f"- **{corpus}, wrapper versus original:** {row['qualifiedVisibleDelta']:+,} qualified visible outputs overall. Within their {row['bothPublicVisibleFrames']:,} mutually public visible frames, qualified outputs change by {row['sharedQualifiedDelta']:+,}; this shared-output comparison excludes the {row['withheldBaselineQualifiedFrames']:,} baseline-qualified frames that the wrapper withholds. It also exposes {row['addedWrongWhereBaselineWithheld']:,} wrong-threshold frames where the original withheld. Lower accepted-only error is therefore not proof of better whole-stream utility.")
        if corpus=='real':
            lines.append(f"  - On this real shared-public subset, the P95 tail changes from {fmt(row['sharedBaselineReprojectionP95'])} to {fmt(row['sharedCandidateReprojectionP95'])} px; it does **not** reproduce the synthetic tail-error improvement. The current rigid and wrapper have the same measured public-pose coverage/error counts here; that does not imply all diagnostic states are byte-identical.")
        delays=[event for event in row['pairedRecovery'] if event['addedObservableFrames'] is not None and event['addedObservableFrames']>0]
        if delays:
            event=max(delays,key=lambda event:event['addedObservableFrames'])
            lines.append(f"  - Largest observed successful-recovery slowdown: `{event['name']}`, return frame {event['returnFrameId']}: original {event['baselineObservableFrames']} → wrapper {event['candidateObservableFrames']} observable frames (requires two consecutive qualified outputs).")
        if row['candidateRecoveryFailuresAfterBaselineSuccess']:
            lines.append(f"  - The wrapper does not achieve qualified recovery in {row['candidateRecoveryFailuresAfterBaselineSuccess']} repeat-0 events where the original does.")
        extra=next(pair for pair in pairs if pair['corpus']==corpus and pair['baselineSystem']=='current_rigid' and pair['candidateSystem']=='current_prototype')
        lines.append(f"- **{corpus}, additional wrapper policy versus current rigid:** {extra['wrongVisiblePublicDelta']:+,} wrong-threshold visible outputs and {extra['qualifiedVisibleDelta']:+,} qualified outputs. This is a measurable safety/availability tradeoff, not evidence that the whole current implementation is useless or universally superior.")
    lines += ["", "The preferred operating point depends on the application cost of a wrong exposed pose versus an unavailable overlay; no independent human utility or physical-attachment assessment was performed. Fully hidden public-pose counts are reported separately above and are not inferred from this visible-frame proxy.", "",
              "## Synthetic stress slices", "", "Scenario names are parsed from the exact generator sequence identity (not missing `tags` metadata). Train/dev/test seed groups remain separately identifiable in the CSV; this benchmark fits no new tracker parameters.", "",
              "| Scenario | Original rigid <5 px | Current rigid <5 px | Current prototype <5 px |", "|---|---:|---:|---:|"]
    for scenario in SCENARIOS:
        selected={row['system']:row for row in stresses if row['scenario']==scenario}
        if selected:
            lines.append(f"| {scenario} | "+" | ".join(percent(selected[system]['native5pxVisibleCoverage']) for system in SYSTEMS)+" |")
    lines += ["", "## Repeat stability and failures", "", "Timing-free digests compare exact frame/state/reason/pose values, excluding clocks, session IDs, timing and internal diagnostics. These are repeated executions of the same footage, not independent data samples or confidence intervals.", "",
              "| Corpus / system | Identical sequence groups | Changed frame comparisons | Repeat coverage range | Repeat update FPS range |", "|---|---:|---:|---:|---:|"]
    for row in aggregates:
        d=row['determinism']; cv=[r['native5pxVisibleCoverage'] for r in row['repeatVariation'] if r['native5pxVisibleCoverage'] is not None]; speed=[r['updateFPS'] for r in row['repeatVariation'] if r['updateFPS'] is not None]
        lines.append(f"| {row['corpus']} / {LABELS[row['system']]} | {d['identicalSequenceSystemGroups']}/{d['sequenceSystemGroups']} | {d['changedFramesAgainstFirstRepeat']:,}/{d['comparedFramesAgainstFirstRepeat']:,} | {percent(min(cv)) if cv else '—'}–{percent(max(cv)) if cv else '—'} | {fmt(min(speed)) if speed else '—'}–{fmt(max(speed)) if speed else '—'} |")
    lines += ["", "Most frequent withheld-output reasons (all repeats; not silently dropped):"]
    for row in aggregates:
        common="; ".join(f"`{str(reason).replace('`','')}`: {count:,}" for reason,count in list(row['failures'].items())[:3]) or "none"
        lines.append(f"- {row['corpus']} / {LABELS[row['system']]}: {common}.")
    lines += ["", "## Reproduction and boundaries", "",
              "- Native `<5 px` coverage is the fraction of reference-visible frames with a public pose whose **mean corresponding-model-point reprojection error** is strictly below 5 input pixels. It is not a landmark attachment tolerance, human-annotated accuracy, detector accuracy, or an official BOP score.",
              "- Evaluation uses the same deterministic geometry-only sample of at most 512 source-model vertices per object. ADD is mean corresponding-point distance in metres; rotation/translation are pose errors. Symmetries are **not** matched, and dataset-reference poses can contain error.",
              "- Attempted-update FPS = timed calls / summed update time. Replay FPS additionally includes RGB decode/hash and loop overhead for trials with attempted updates. Both exclude onboarding, disposable warmup, evaluation, rendering and export. Setup/initialization failures keep scheduled-frame denominators but are excluded from speed numerators; skipped updates are reported explicitly.",
              "- Original means the exact repository historical **update implementation**, loaded from `git show`, not a reimplementation. Current rigid and current prototype also preserve their production update methods. **All three initialization paths are replaced by the shared simplified adapter**; this is not a full-production-pipeline comparison.",
              "- **Neural GoTrack/adapted-head tracking quality is not measured by these tables.** A separate finite model diagnostic is described below; execution/parity is not video-tracking quality or serving admission.",
              "- Synthetic images are not real evidence. Real HB is a previously used project dataset, with unknown pretrained exposure; these clips do not demonstrate unseen-object generalization. Full RGB/depth/mask/model media are acquired from the pinned CC0 source or the published dataset release; committed real examples and complete annotation packs are a subset/different representation, not the complete media archive.", "",
              "### Frozen benchmark source versus later publication/report commits", "",
              f"The full-run harness last changed in commit `{context.get('benchmarkHarnessLastChangeCommit','unavailable')}` and is snapshotted from `{context.get('frozenSourceSnapshotCommit','unavailable')}`; its frozen SHA-256 is `{context.get('benchmarkHarnessAtAnchorSHA256','unavailable')}`. Both corpora match that source anchor: `{context.get('allFrozenHarnessHashesMatchAnchor')}`. The current checkout matches the old harness: `{context.get('workingHarnessMatchesFrozen')}`; later initialization-ablation code must not be mistaken for what ran in these full results. The report was generated from working-tree base `{context.get('reportWorkingTreeBaseCommit','unavailable')}`. Source hashes in each inference seal, not the eventual PR head, identify what actually ran; report/source snapshots are hashed separately.", ""]
    neural=context.get('separateNeuralDiagnostic')
    if neural and neural.get('fullNetworkExecuted'):
        lines += ["### Separate trained-model execution diagnostic (not tracking quality)", "",
                  f"The [finite CPU neural diagnostic]({neural['summaryPath']}) completed {neural['networkForwards']} full-network forwards in {fmt(neural['elapsedSeconds'])} seconds, with {fmt(neural['peakSampledRSSBytes']/1024**3)} GiB peak sampled process-group RSS. Saved-reference versus trained-export numerical parity: `{neural['referenceExportParity']}`. It used two authorized onboarding queries with known seed poses/masks and contributes **{neural['newScoredBenchmarkFrames']} new scored video-tracking frames**. Workbench admission granted: `{neural['workbenchAdmissionGranted']}`. It is an execution/parity check, not evidence of better tracking, independent attachment accuracy or live/neural FPS.", ""]
    for corpus in corpora:
        p=corpus['provenance']; h=p['hardware']; c=p['configuration']
        lines += [f"### {corpus['corpus']} runtime and seals", "",
                  f"- Hardware: {h.get('cpuModel')}; {h.get('logicalCPUs')} logical CPUs; RAM `{h.get('memoryTotal')}`; GPU used: `{h.get('gpuUsed')}`; OpenCV threads: `{h.get('opencvThreads')}`.",
                  f"- Versions: Python `{h.get('python','').splitlines()[0]}`; NumPy `{h.get('numpy')}`; OpenCV `{h.get('opencv')}`; Pillow `{h.get('pillow')}`. Platform: `{h.get('platform')}`.",
                  f"- Protocol: {c['repeats']} full repeats; {c['warmupFrames']} disposable warmup frames per sequence/system/repeat; serial systems/sequences. Initialization: {c['initialization']}.",
                  f"- Original commit: `{p['historicalCommit']}`; original tracker SHA-256: `{p['historicalTrackerSHA256']}`.",
                  "", "| Artifact | SHA-256 |", "|---|---|"]
        for label,key in (("Inference manifest","inferenceManifestSHA256"),("Evaluator manifest","evaluatorManifestSHA256"),("Inference seal","inferenceSealSHA256"),("Results","resultsSHA256")):
            lines.append(f"| {label} | `{p[key]}` |")
        lines.append("")
    lines += ["## Compact artifacts", "", f"![Native-pixel visible coverage]({filenames['plots'][0]})", "", f"![Attempted CPU update throughput]({filenames['plots'][1]})", "",
              f"- [Aggregate JSON]({filenames['json']}) — pooled metrics, repeat variation, hardware, versions and source/data hashes.",
              f"- [Aggregate CSV]({filenames['aggregateCSV']}); [per-sequence/repeat CSV]({filenames['sequenceCSV']}); [stress-slice CSV]({filenames['stressCSV']}).",
              f"- [Paired-output tradeoff CSV]({filenames['pairedCSV']}) — unique repeat-0 shared-output and abstention accounting.",
              f"- [Trace hash index]({filenames['index']}) — exact hashes of local raw/scored traces, without committing all bulky trace files.", "",
              "Shared-VM timings are descriptive of this run, not dedicated-hardware latency guarantees. No merge, deployment, physical-phone test, paid compute, or new model training is implied.", ""]
    return "\n".join(lines)


def generate(synthetic_output, real_output, destination, archive_path=None, initialization_outputs=None):
    # Both seals must exist before either corpus's result scores are read.
    for output in (synthetic_output, real_output):
        if not (Path(output)/"inference-seal.json").is_file():
            raise ValueError(f"Inference is not sealed; report deferred: {output}")
    corpora=[load_corpus(synthetic_output,"synthetic"), load_corpus(real_output,"real")]
    aggregates, sequences, stresses, pairs = [], [], [], []
    for corpus in corpora:
        for (_,), entries in sorted(group_entries(corpus['entries'],('system',)).items()):
            aggregates.append(aggregate(entries))
        sequences.extend(entry['summary'] for entry in corpus['entries'])
        for baseline,candidate in (("original_rigid","current_rigid"),("original_rigid","current_prototype"),("current_rigid","current_prototype")):
            pairs.append(paired_tradeoff(corpus['entries'],baseline,candidate))
        if corpus['corpus']=='synthetic':
            for (scenario,system), entries in sorted(group_entries(corpus['entries'],('scenario','system')).items()):
                row=aggregate(entries);row['scenario']=scenario;stresses.append(row)
    order={system:i for i,system in enumerate(SYSTEMS)}
    aggregates.sort(key=lambda row:(("synthetic","real").index(row['corpus']),order[row['system']]))
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    filenames=dict(markdown=f"tracking-results-{DATE}.md", json=f"tracking-aggregate-{DATE}.json",
                   aggregateCSV=f"tracking-aggregate-{DATE}.csv",sequenceCSV=f"tracking-sequences-{DATE}.csv",
                   stressCSV=f"tracking-stress-{DATE}.csv",pairedCSV=f"tracking-paired-{DATE}.csv", index=f"tracking-trace-index-{DATE}.json")
    filenames['plots']=plots(destination,aggregates)
    context=source_context(corpora)
    if initialization_outputs:
        context['initializationDiagnostic']=initialization_diagnostic(*initialization_outputs)
        filenames['initializationCSV']=f'tracking-initialization-prefix-{DATE}.csv'
    result=dict(schemaVersion=1,reportDate=DATE,reportToolSHA256=sha_file(Path(__file__)),
                metric="Native input-pixel mean-model reprojection <5 px on dataset-reference visibility>=0.5; not generic accuracy",
                corpora={corpus['corpus']:corpus['provenance'] for corpus in corpora},
                publicationContext=context,aggregate=aggregates, sequenceTrials=sequences, stressSlices=stresses, pairedTradeoffs=pairs)
    (destination/filenames['json']).write_bytes(canonical(result))
    write_csv(destination/filenames['aggregateCSV'],aggregates)
    write_csv(destination/filenames['sequenceCSV'],sequences)
    write_csv(destination/filenames['stressCSV'],stresses)
    write_csv(destination/filenames['pairedCSV'],pairs)
    if context.get('initializationDiagnostic'):
        write_csv(destination/filenames['initializationCSV'],context['initializationDiagnostic']['aggregate'])
    index=dict(schemaVersion=1, reportToolSHA256=result['reportToolSHA256'],
               traces=[row for corpus in corpora for row in corpus['traceIndex']],
               note="Paths are relative to their corpus result directory; raw/scored traces remain local unless separately published.")
    (destination/filenames['index']).write_bytes(canonical(index))
    (destination/filenames['markdown']).write_text(markdown(corpora,aggregates,stresses,pairs,filenames,context),encoding='utf-8')
    print(json.dumps(dict(status="complete",destination=str(destination),report=filenames['markdown'],
                          scoredSequenceTrials=len(sequences),artifacts=len(filenames)-1+len(filenames['plots']))),flush=True)
    if archive_path:
        result['rawResultsArchive']=package_results(corpora,(synthetic_output,real_output),destination,filenames,archive_path)
    return result


def package_results(corpora,outputs,destination,filenames,archive_path):
    """Archive only named CPU evidence/source/report files, never data or weights."""
    archive_path=Path(archive_path)
    if archive_path.exists() or archive_path.with_name(archive_path.name+'.part').exists():
        raise ValueError('Raw-results archive already exists; retain it or choose a fresh path')
    files,inline={},{ }
    allowed_top={'freeze.json','inference-seal.json','progress.json','results.json'}
    trial_pattern=re.compile(r'object-[0-9]{6}/(?:original_rigid|current_rigid|current_prototype)-r[0-9]+(?:-scored)?\.json$')
    for corpus,output in zip(corpora,outputs):
        output=Path(output).resolve()
        for path in sorted(output.rglob('*')):
            if not path.is_file():
                continue
            relative=path.relative_to(output).as_posix()
            if path.is_symlink() or not path.resolve().is_relative_to(output) or (relative not in allowed_top and trial_pattern.fullmatch(relative) is None):
                raise ValueError('Unexpected raw result file; inspect before packaging: '+relative)
            files[corpus['corpus']+'-full-v1/'+relative]=path
        log=output.with_suffix('.log')
        if not log.is_file():
            raise ValueError('Missing complete benchmark log: '+str(log))
        text=log.read_text(encoding='utf-8')
        if re.search(r'(?:ghp_|github_pat_|sk-proj-|AKIA)[A-Za-z0-9_\-]{12,}',text):
            raise ValueError('Credential-like content in benchmark log; inspect separately')
        files['logs/'+corpus['corpus']+'-full-v1.log']=log
    source_hashes=corpora[0]['provenance']['sourceHashes']
    if any(corpus['provenance']['sourceHashes']!=source_hashes for corpus in corpora[1:]):
        raise ValueError('Corpora have different frozen source sets; package separately')
    for relative,expected in source_hashes.items():
        raw=subprocess.run(['git','show',FROZEN_SOURCE_SNAPSHOT_COMMIT+':'+relative],cwd=ROOT,check=True,capture_output=True).stdout
        if hashlib.sha256(raw).hexdigest()!=expected:
            raise ValueError('Frozen source unavailable for snapshot: '+relative)
        inline['source-snapshot/frozen-c10cb8b/'+relative]=raw
    for relative in ('bench/tracking_report.py','bench/test_tracking_report.py','bench/requirements-tracking.txt'):
        path=ROOT/relative
        if path.is_file():
            files['source-snapshot/report/'+relative]=path
    report_names=[value for key,value in filenames.items() if key!='plots']+filenames['plots']
    for name in report_names:
        files['report/'+name]=Path(destination)/name
    old=corpora[0]['provenance']['historicalCommit']
    original=subprocess.run(['git','show',old+':bench/tracker.py'],cwd=ROOT,check=True,capture_output=True).stdout
    if hashlib.sha256(original).hexdigest()!=corpora[0]['provenance']['historicalTrackerSHA256']:
        raise ValueError('Original tracker snapshot differs from frozen hash')
    inventory=[dict(path=name,bytes=path.stat().st_size,sha256=sha_file(path)) for name,path in sorted(files.items())]
    inventory.extend(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in sorted(inline.items()))
    old_path='source-snapshot/bench/original-tracker-'+old+'.py'
    inventory.append(dict(path=old_path,bytes=len(original),sha256=hashlib.sha256(original).hexdigest()))
    readme=f'''# Complete VisualizeIT SHARED SIMPLIFIED-INITIALIZATION experiment evidence

Run date: {DATE}. Both corpora completed inference and separate evaluation.
IMPORTANT: all 8,160 unique full-corpus scored object-frames used a SHARED
SIMPLIFIED INITIALIZATION BANK: masked RGB-D depth backprojection with known
onboarding poses, first 300 ORB features/view, at most 3,000 anchors. This is
NOT the exact production mesh-ray/all-detected-features initialization and is
NOT a full-production-pipeline benchmark. A separate initialization diagnostic
must not be conflated with these original immutable full-corpus results.

Contains every sealed raw prediction, every scored trace, freeze/seal/results/
progress JSON, both complete run logs, compact reports/plots, frozen benchmark
source, the exact original historical tracker, and the pinned CPU requirements.
The six frozen core source files are retrieved from commit
{FROZEN_SOURCE_SNAPSHOT_COMMIT}, each checked against the original inference
seal hash. They are NOT taken from the later, modified current harness.
No private media, input images, model weights, credentials or GPU artifacts are
included. Input dataset media are a separate licensed release:
{DATASET_RELEASE_URL}

MANIFEST.json lists byte counts and SHA-256 for all payload files. The manifest
does not self-hash; its own hash and the complete archive hash are in the adjacent
archive receipt. All tar ownership/mode/time fields are fixed and gzip mtime=0.

Public benchmark report: report/tracking-results-{DATE}.md
Raw predictions are under synthetic-full-v1/ and real-full-v1/.
Scored traces have the -scored.json suffix. Each inference-seal.json binds all
prediction hashes and freeze metadata; each results.json binds its inference
seal. Keep failures and initialization failures in scheduled-frame denominators.
These are initialized geometric CPU tracking results, not trained GoTrack,
automatic detection, independent human attachment accuracy or phone performance.
'''.encode()
    inventory.append(dict(path='README.md',bytes=len(readme),sha256=hashlib.sha256(readme).hexdigest()))
    manifest=canonical(dict(schemaVersion=1,reportDate=DATE,payloadFiles=sorted(inventory,key=lambda row:row['path']),
                            corpusSeals={corpus['corpus']:corpus['provenance']['inferenceSealSHA256'] for corpus in corpora},
                            originalCommit=old,sourceHashes=source_hashes,frozenSourceSnapshotCommit=FROZEN_SOURCE_SNAPSHOT_COMMIT,
                            initializationMode=SIMPLIFIED_MODE,datasetReleaseURL=DATASET_RELEASE_URL,
                            scope='Complete SHARED SIMPLIFIED-INITIALIZATION CPU raw/scored results, logs and code only; not full production pipeline; no input media, private data, credentials or model weights'))
    archive_path.parent.mkdir(parents=True,exist_ok=True)
    temporary=archive_path.with_name(archive_path.name+'.part')
    def add(archive,name,raw=None,path=None):
        info=tarfile.TarInfo('tracking-cpu-results-v1/'+name)
        info.mode=0o644;info.mtime=info.uid=info.gid=0;info.uname=info.gname=''
        info.size=len(raw) if raw is not None else path.stat().st_size
        if raw is not None:
            archive.addfile(info,io.BytesIO(raw))
        else:
            with path.open('rb') as source:
                archive.addfile(info,source)
    with temporary.open('wb') as target,gzip.GzipFile(filename='',mode='wb',fileobj=target,mtime=0,compresslevel=1) as compressed:
        with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.USTAR_FORMAT) as archive:
            for name,path in sorted(files.items()):
                add(archive,name,path=path)
            for name,raw in sorted(inline.items()):
                add(archive,name,raw=raw)
            add(archive,old_path,raw=original)
            add(archive,'README.md',raw=readme)
            add(archive,'MANIFEST.json',raw=manifest)
    temporary.replace(archive_path)
    receipt=dict(schemaVersion=1,filename=archive_path.name,bytes=archive_path.stat().st_size,sha256=sha_file(archive_path),
                 initializationMode=SIMPLIFIED_MODE,frozenSourceSnapshotCommit=FROZEN_SOURCE_SNAPSHOT_COMMIT,
                 manifestSHA256=hashlib.sha256(manifest).hexdigest(),payloadFiles=len(inventory),
                 rawAndScoredSequenceTrials=sum(len(corpus['entries'])*2 for corpus in corpora),
                 corpusSeals={corpus['corpus']:corpus['provenance']['inferenceSealSHA256'] for corpus in corpora},
                 publicationStatus='prepared_locally_not_uploaded')
    archive_path.with_suffix(archive_path.suffix+'.json').write_bytes(canonical(receipt))
    archive_path.with_suffix(archive_path.suffix+'.manifest.json').write_bytes(manifest)
    print(json.dumps(receipt),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-output',required=True,type=Path)
    parser.add_argument('--real-output',required=True,type=Path)
    parser.add_argument('--destination',required=True,type=Path)
    parser.add_argument('--archive',type=Path,help='Optional complete raw CPU-results tar.gz; never uploads')
    parser.add_argument('--initialization-simplified-output',type=Path)
    parser.add_argument('--initialization-production-output',type=Path)
    args=parser.parse_args()
    if bool(args.initialization_simplified_output)!=bool(args.initialization_production_output):
        parser.error('Supply both paired initialization output directories, or neither')
    initialization_outputs=(args.initialization_simplified_output,args.initialization_production_output) if args.initialization_simplified_output else None
    generate(args.synthetic_output,args.real_output,args.destination,args.archive,initialization_outputs)


if __name__=='__main__':
    main()
