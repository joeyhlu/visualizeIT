"""Fail-closed projection of the real WorkbenchService status for the phone UI.

Standard library only. This module never imports torch/numpy, never loads models,
never runs preflight and never touches the network itself: the caller passes the
already-fetched, size-bounded JSON value from the fixed loopback GET /api/status.

The projection only reports what upstream metadata states. It is not proof that
a model is installed correctly or that inference works. Live tracking is always
reported unsupported (code ``live_neural_contract_unsupported``).
"""
from __future__ import annotations

import re

SCHEMA = "visualizeit_phone_tracking_status_v1"
EXPECTED_SERVICE = "visualizeit-workbench"
EXPECTED_SCHEMA = "workbench_service_v2"
# Hard pin: the locally verified GoTrack metric head (80,165,526 bytes).
PINNED_HEAD_SHA256 = "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311"
LIVE_REASON = "live_neural_contract_unsupported"
MAX_BLOCKERS = 16
MAX_CLIP_COUNT = 100000

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_CODE = re.compile(r"[a-z][a-z0-9_.:-]{0,63}\Z")

STATES = ("disconnected", "reachable_blocked", "recorded_configured")
# Codes this module itself can emit. Upstream codes pass only the strict ASCII pattern.
LOCAL_CODES = frozenset({
    "workbench_unreachable", "invalid_status", "unexpected_service", "unexpected_schema",
    "model_metadata_missing", "model_mismatch", "neural_blockers_malformed", "neural_blocked",
    "admission_metadata_missing", "admission_not_admitted", "head_metadata_missing",
    "head_not_configured", "metric_bundle_metadata_missing", "metric_bundle_not_configured",
    "unrecognized_blocker", "installed_metadata_invalid", "pipeline_metadata_missing", "pipeline_disabled",
})

MESSAGES = {
    "disconnected": "Workbench is not reachable. Local design creation still works.",
    "reachable_blocked": "Workbench is reachable, but recorded tracking is not ready. Reachable is not the same as model ready.",
    "recorded_configured": "Workbench metadata reports the pinned head configured for recorded clips only. Live phone tracking is unavailable.",
}


def _hex(value):
    return value if type(value) is str and _HEX64.fullmatch(value) else None


def _base(state):
    return {
        "schema": SCHEMA,
        "state": state,
        "service_reachable": state != "disconnected",
        "requested_head_sha256": PINNED_HEAD_SHA256,
        "reported_export_sha256": None,
        "installed_head_sha256": None,
        "model_match": False,
        "recorded_admitted": False,
        "head_configured": None,
        "metric_bundle_configured": None,
        "registered_clip_count": None,
        "blockers": [],
        "upstream_blockers": [],
        "recorded_supported": False,
        "live_supported": False,
        "live_reason": LIVE_REASON,
        "quality_claim": False,
        "message": MESSAGES[state],
    }


def disconnected_status():
    result = _base("disconnected")
    result["blockers"] = ["workbench_unreachable"]
    return result


def _add(blockers, code):
    if code in LOCAL_CODES and code not in blockers and len(blockers) < MAX_BLOCKERS:
        blockers.append(code)


def sanitize_tracking_status(raw):
    """Return an allowlisted projection. Never raises; malformed input is blocked."""
    try:
        return _sanitize(raw)
    except Exception:  # defensive: any surprise fails closed without echoing data
        result = _base("reachable_blocked")
        result["blockers"] = ["invalid_status"]
        return result


def _sanitize(raw):
    result = _base("reachable_blocked")
    blockers = result["blockers"]
    if type(raw) is not dict:
        _add(blockers, "invalid_status")
        return result
    if raw.get("service") != EXPECTED_SERVICE:
        _add(blockers, "unexpected_service")
    if raw.get("schema") != EXPECTED_SCHEMA:
        _add(blockers, "unexpected_schema")

    model = raw.get("primary_model")
    reported = _hex(model.get("export_sha256")) if type(model) is dict else None
    result["reported_export_sha256"] = reported
    # Installed hash is only ever an upstream-reported, validated value; never the pin.
    raw_installed = model.get("installed_head_sha256") if type(model) is dict else None
    installed = _hex(raw_installed)
    result["installed_head_sha256"] = installed
    installed_ok = True
    if raw_installed is not None and installed is None:
        _add(blockers, "installed_metadata_invalid")  # present but malformed: fail closed
        installed_ok = False
    elif installed is not None and installed != PINNED_HEAD_SHA256:
        _add(blockers, "model_mismatch")  # a different (e.g. older) head is installed
        installed_ok = False
    # null installed stays unknown; it is not proof that the pinned head is installed.
    if reported is None:
        _add(blockers, "model_metadata_missing")
    elif reported != PINNED_HEAD_SHA256:
        _add(blockers, "model_mismatch")
    elif installed_ok:
        result["model_match"] = True

    capabilities = raw.get("capabilities")
    neural = capabilities.get("neural_blocked") if type(capabilities) is dict else None
    neural_ok = False
    if type(neural) is not list:
        _add(blockers, "neural_blockers_malformed")
    elif neural:
        _add(blockers, "neural_blocked")
        upstream = result["upstream_blockers"]
        for item in neural[:MAX_BLOCKERS]:
            code = item if type(item) is str and _CODE.fullmatch(item) else "unrecognized_blocker"
            if code not in upstream:
                upstream.append(code)
    else:
        neural_ok = True

    # Read-only catalog flag; this projection never turns execution on.
    pipeline = capabilities.get("pipeline") if type(capabilities) is dict else None
    if type(pipeline) is not bool:
        _add(blockers, "pipeline_metadata_missing")
    elif pipeline is not True:
        _add(blockers, "pipeline_disabled")
    pipeline_ok = pipeline is True

    admission = raw.get("admission")
    admitted = admission.get("admitted") if type(admission) is dict else None
    if type(admitted) is not bool:
        _add(blockers, "admission_metadata_missing")
    elif admitted is not True:
        _add(blockers, "admission_not_admitted")
    result["recorded_admitted"] = admitted is True

    defaults = raw.get("defaults")
    for key, missing, off in (("head_configured", "head_metadata_missing", "head_not_configured"),
                              ("metric_bundle_configured", "metric_bundle_metadata_missing", "metric_bundle_not_configured")):
        value = defaults.get(key) if type(defaults) is dict else None
        if type(value) is not bool:
            _add(blockers, missing)
        else:
            result[key] = value
            if value is not True:
                _add(blockers, off)

    count = raw.get("registered_clip_count")
    if type(count) is int and 0 <= count <= MAX_CLIP_COUNT:
        result["registered_clip_count"] = count

    if (not blockers and neural_ok and pipeline_ok and result["model_match"] and admitted is True
            and result["head_configured"] is True and result["metric_bundle_configured"] is True):
        result["state"] = "recorded_configured"
        result["recorded_supported"] = True
        result["message"] = MESSAGES["recorded_configured"]
    return result
