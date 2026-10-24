"use strict";
// Local workbench client. No external dependencies, no remote URLs.
(() => {
  const $ = (id) => document.getElementById(id);
  const state = {
    clip: null, frames: [], points: [], mode: 1, cursor: null, selectionId: null,
    run: null, rows: {}, order: [], index: 0, playing: false, timer: null, job: null,
  };

  async function api(path, options = {}) {
    const res = await fetch(path, { credentials: "same-origin", ...options });
    const type = res.headers.get("Content-Type") || "";
    const body = type.includes("application/json") ? await res.json() : null;
    if (!res.ok) throw new Error((body && body.error) || `HTTP ${res.status}`);
    return body;
  }
  const postJSON = (path, value) => api(path, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(value),
  });
  function setStatus(el, text, kind = "") { el.textContent = text; el.className = `status ${kind}`; }
  const fmt = (v) => (v === null || v === undefined ? "unavailable" : String(v));

  // ------------------------------------------------------------- status/clips
  async function loadStatus() {
    try {
      const s = await api("/api/status");
      const c = s.capabilities || {};
      const adm = s.admission || {};
      const dev = s.runtime_device || {};
      const lines = [
        `Clips: ${s.registered_clip_count}. Primary model: ${s.primary_model ? s.primary_model.artifact_id + " (" + s.primary_model.export_sha256.slice(0, 12) + "…)" : "unavailable"}`,
        `Serving admission: ${adm.admitted ? "VALID (re-verified now; " + (adm.scope || "execution compatibility only") + "; no quality claim)"
          : adm.configured ? "BLOCKED — " + (adm.reasons || []).slice(0, 3).join("; ") : "not configured"}`,
        `Device: ${fmt(dev.configured_device)} · torch ${fmt(dev.torch_version)} (${fmt(dev.torch_build)} build)` +
          `${dev.CUDA_VISIBLE_DEVICES != null ? " · CUDA_VISIBLE_DEVICES=" + dev.CUDA_VISIBLE_DEVICES : ""}`,
        `Segmentation: ${c.segmentation ? "runtime present (runs only when you press Segment)" : "blocked " + (c.segmentation_blocked || []).join(", ")}`,
        `COLMAP: ${c.colmap ? "available" : "unavailable — " + (c.colmap_reason || "")}`,
        `Model loading blockers: ${(c.neural_blocked || []).join(", ") || "none"}. Prepared geometry is checked for the selected clip below.`,
        `Default observed mask set: ${(s.defaults && s.defaults.mask_set) || "none configured"}` +
          ` (per-clip mappings win; the global set applies only to clips that contain it)`,
        `Prepared metric controls: ${controlOverview(s.metric_controls)}`,
        `Inputs ${s.budget.new_inputs_bytes.toLocaleString()} / ${s.budget.new_inputs_limit.toLocaleString()} B; derived ${s.budget.derived_bytes.toLocaleString()} / ${s.budget.derived_limit.toLocaleString()} B; free ${s.budget.free_bytes.toLocaleString()} B`,
      ];
      state.defaultMaskSet = (s.defaults && s.defaults.mask_set) || null;
      const blocked = !s.budget.within_budget || (adm.configured && !adm.admitted);
      setStatus($("service-status"), lines.join("\n"), blocked ? "blocked" : "");
    } catch (e) { setStatus($("service-status"), `Service unavailable: ${e.message}`, "failed"); }
  }

  const short = (sha) => (sha ? sha.slice(0, 12) + "…" : "none");
  const isExecution = (c) => c && c.control_kind === "execution_only_supplemental_cad";
  const executionLabels = (c) => (isExecution(c) && c.labels
    ? ["control", "placement", "correspondence", "uv", "automatic_rgb_geometry", "admission", "primary_model"]
        .filter((k) => c.labels[k]).map((k) => c.labels[k]).join("; ")
    : "");
  function controlOverview(mc) {
    if (!mc) return "unknown";
    const parts = Object.entries(mc.per_clip || {}).map(([id, c]) =>
      `${id}: ${c.ready ? (c.label || "prepared metric CAD control") : (c.status || "unavailable")}` +
      `${c.default_mask_set ? ", masks " + c.default_mask_set : ""}`);
    const g = mc.global || {};
    if (g.configured) parts.push(`legacy global bundle ${g.error ? "BLOCKED (" + g.error + ")" : short(g.bundle_sha256)} ` +
      (g.bound_clip_id ? `bound to ${g.bound_clip_id} only` : "UNBOUND — not used for any clip; configure --metric-bundle-for CLIP=PATH or --metric-bundle-clip CLIP_ID"));
    return parts.join("; ") || "none configured (neural roles stay blocked; automatic preparation is a separate experimental path)";
  }

  function showClipControl(detail) {
    const c = detail.prepared_control || {};
    const m = detail.masks || {};
    // The clip's own default replaces any value left over from another clip.
    $("run-masks").value = m.default || "";
    $("run-masks-list").replaceChildren(...(m.available || []).map((name) => {
      const o = document.createElement("option"); o.value = name; return o;
    }));
    $("run-masks-note").textContent = m.default
      ? `Default mask set ${m.default} (${m.source === "per_clip" ? "per-clip mapping" : "global default"})` +
        (m.present ? "." : " — NOT FOUND for this clip; masks will be unavailable.")
      : `No default mask set for this clip${(m.available || []).length ? "; available: " + m.available.join(", ") : ""}.`;
    const text = c.ready && isExecution(c)
      ? `Neural roles: ${c.label} (bundle ${short(c.bundle_sha256)}, GLB ${short(c.glb_sha256)}, ` +
        `sidecar ${short(c.sidecar_sha256)}, ${c.family}, seed ${c.seed_frame_id}). ${executionLabels(c)}. ` +
        `${c.validation || ""}.`
      : c.ready
      ? `Neural roles: prepared metric CAD control (${c.source === "per_clip" ? "per-clip mapping" : "legacy bundle explicitly bound to this clip"}; ` +
        `bundle ${short(c.bundle_sha256)}, GLB ${short(c.glb_sha256)}, ${c.family}, seed ${c.seed_frame_id}). ` +
        `Serving admission and hashes are re-verified when the job starts. Principal-point origin is the bundle's declaration (not independently validated).`
      : `Neural roles: ${c.label || "unavailable"} [${c.status || "unavailable"}]` +
        `${(c.reasons || []).length ? " — " + c.reasons.join("; ") : ""}` +
        `${isExecution(c) ? ". " + executionLabels(c) : ""}. Relative geometry can still be scheduled separately.`;
    setStatus($("clip-control"), text, c.ready ? "" : "blocked");
  }

  function selectedRoles() {
    const roles = [];
    if ($("run-neural").checked) roles.push("primary_neural", "frozen_baseline");
    if ($("run-relative").checked) roles.push("relative_geometry");
    return roles;
  }

  async function loadClips(selectId) {
    const data = await api("/api/clips");
    const sel = $("clip-select");
    sel.replaceChildren(...data.clips.map((c) => {
      const o = document.createElement("option");
      o.value = c.clip_id;
      o.textContent = `${c.title} — ${c.frame_count} frames, ${c.clock_kind}, K ${c.intrinsics}${c.selection_ready ? ", reviewed selection" : ""}`;
      return o;
    }));
    if (selectId) sel.value = selectId;
  }

  async function openClip(clipId) {
    clearProvenance();
    const detail = await api(`/api/clips/${encodeURIComponent(clipId)}`);
    state.clip = detail; state.frames = detail.clip.frames; state.points = []; state.selectionId = null;
    const s = detail.summary;
    $("clip-summary").textContent = `${s.title}: ${s.width}×${s.height}, ${s.frame_count} frames (${s.scoring_frames} scoring), ` +
      `clock ${s.clock_kind}${s.clock_kind === "frame_index" ? " (cannot attest the 10 s onboarding bound)" : ""}, ` +
      `intrinsics ${s.intrinsics}, role ${s.evidence_role}.`;
    showClipControl(detail);
    const opts = state.frames.map((f, i) => {
      const o = document.createElement("option");
      o.value = String(i);
      o.textContent = `${f.frame_id} (source ${f.source_frame_index}${f.timestamp_s !== null ? ", " + f.timestamp_s.toFixed(3) + " s" : ""})`;
      return o;
    });
    $("key-start").replaceChildren(...opts.map((o) => o.cloneNode(true)));
    $("key-end").replaceChildren(...opts);
    const ob = detail.clip.onboarding;
    const pos = Object.fromEntries(state.frames.map((f, i) => [f.frame_id, i]));
    if (ob) {
      $("key-start").value = String(pos[ob.keyframe_ids[0]]);
      $("key-end").value = String(pos[ob.keyframe_ids[ob.keyframe_ids.length - 1]]);
      if (ob.selection) {
        ob.selection.positive.forEach(([x, y]) => state.points.push({ x, y, label: 1 }));
        ob.selection.negative.forEach(([x, y]) => state.points.push({ x, y, label: 0 }));
      }
    } else { $("key-start").value = "0"; $("key-end").value = "0"; }
    $("select-panel").hidden = false; $("run-panel").hidden = false;
    updateKeyframes(); renderPoints();
    const scoring = detail.clip.scoring_frame_ids;
    $("run-first").replaceChildren(...scoring.map((fid) => { const o = document.createElement("option"); o.value = fid; o.textContent = fid; return o; }));
    $("run-select").replaceChildren(...detail.runs.map((r) => {
      const o = document.createElement("option"); o.value = r.run_id; o.textContent = `${r.run_id} — ${r.mode}, ${r.status}`; return o;
    }));
  }

  // ---------------------------------------------------------------- selection
  function keyRange() {
    const a = Number($("key-start").value), b = Number($("key-end").value);
    return [Math.min(a, b), Math.max(a, b)];
  }
  function updateKeyframes() {
    const [a, b] = keyRange();
    const n = b - a + 1;
    $("key-count").textContent = `${n} frame(s)${n > 24 ? " — too many (max 24)" : ""}; selection is reviewed on the last onboarding frame.`;
    const frame = state.frames[b];
    $("select-img").src = state.clip.frame_url_template.replace("{frame_id}", encodeURIComponent(frame.frame_id));
    const svg = $("select-overlay");
    svg.setAttribute("viewBox", `-0.5 -0.5 ${state.clip.clip.width} ${state.clip.clip.height}`);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    renderPoints();
  }
  function clientToSource(clientX, clientY) {
    // object-fit: contain letterboxing correction; source pixels use integer centres.
    const img = $("select-img"), rect = img.getBoundingClientRect();
    const W = state.clip.clip.width, H = state.clip.clip.height;
    const scale = Math.min(rect.width / W, rect.height / H);
    const offX = (rect.width - W * scale) / 2, offY = (rect.height - H * scale) / 2;
    const x = (clientX - rect.left - offX) / scale - 0.5, y = (clientY - rect.top - offY) / scale - 0.5;
    if (x < -0.5 || y < -0.5 || x >= W - 0.5 || y >= H - 0.5) return null;
    return { x: Math.round(x * 100) / 100, y: Math.round(y * 100) / 100 };
  }
  function renderPoints() {
    const svg = $("select-overlay"); const ns = "http://www.w3.org/2000/svg";
    const W = state.clip ? state.clip.clip.width : 1;
    const r = Math.max(4, W / 120);
    const nodes = state.points.map((p) => {
      const c = document.createElementNS(ns, "circle");
      c.setAttribute("cx", p.x); c.setAttribute("cy", p.y); c.setAttribute("r", r);
      c.setAttribute("fill", p.label ? "#12c45a" : "#e8423a"); c.setAttribute("stroke", "#fff"); c.setAttribute("stroke-width", r / 3);
      return c;
    });
    if (state.cursor) {
      const h = document.createElementNS(ns, "path");
      const { x, y } = state.cursor;
      h.setAttribute("d", `M${x - 3 * r} ${y}H${x + 3 * r}M${x} ${y - 3 * r}V${y + 3 * r}`);
      h.setAttribute("stroke", "#ffbf47"); h.setAttribute("stroke-width", r / 2);
      nodes.push(h);
    }
    svg.replaceChildren(...nodes);
    $("points-list").replaceChildren(...state.points.map((p, i) => {
      const li = document.createElement("li");
      li.textContent = `${p.label ? "positive" : "negative"} at (${p.x}, ${p.y}) source px `;
      const b = document.createElement("button"); b.type = "button"; b.textContent = "Remove";
      b.setAttribute("aria-label", `Remove point ${i + 1}`);
      b.addEventListener("click", () => { state.points.splice(i, 1); renderPoints(); });
      li.append(b); return li;
    }));
  }
  function setMode(label) {
    state.mode = label;
    $("mode-pos").setAttribute("aria-pressed", String(label === 1));
    $("mode-neg").setAttribute("aria-pressed", String(label === 0));
  }
  function addPoint(p) { if (p) { state.points.push({ ...p, label: state.mode }); renderPoints(); } }

  async function saveSelection() {
    const [a, b] = keyRange();
    if (b - a + 1 > 24) throw new Error("At most 24 onboarding frames.");
    const keyframe_ids = state.frames.slice(a, b + 1).map((f) => f.frame_id);
    const body = {
      clip_id: state.clip.clip.clip_id, keyframe_ids, frame_id: keyframe_ids[keyframe_ids.length - 1],
      positive: state.points.filter((p) => p.label).map((p) => [p.x, p.y]),
      negative: state.points.filter((p) => !p.label).map((p) => [p.x, p.y]),
    };
    const receipt = await postJSON("/api/selections", body);
    state.selectionId = receipt.selection_id;
    const att = receipt.prefix_attestation;
    setStatus($("select-status"), `Saved selection ${receipt.selection_id}. Prefix bound ${att.prefix_bound_attested ? "attested (" + att.prefix_duration_s.toFixed(2) + " s)" : "not attestable (frame-index clock)"}.`);
    return receipt;
  }

  async function segment() {
    if (!state.selectionId) await saveSelection();
    const [, b] = keyRange();
    const next = state.clip.clip.scoring_frame_ids.find((fid) => state.frames.findIndex((f) => f.frame_id === fid) > b);
    const job = await postJSON("/api/segment", {
      clip_id: state.clip.clip.clip_id, selection_id: state.selectionId, mode: "focused",
      first_frame: next, count: Number($("segment-count").value) || 10,
    });
    $("run-masks").value = job.mask_set || "";
    await followJob(job, $("select-status"));
  }

  // ------------------------------------------------------------------- jobs
  async function followJob(job, el) {
    state.job = job; $("job-cancel").disabled = false;
    const terminal = new Set(["blocked", "failed", "completed", "cancelled"]);
    let current = job;
    while (!terminal.has(current.status)) {
      setStatus(el, `Job ${current.job_id.slice(0, 8)}: ${current.status}…`);
      await new Promise((r) => setTimeout(r, 1000));
      current = await api(`/api/jobs/${current.job_id}`);
    }
    $("job-cancel").disabled = true; state.job = null;
    const kind = current.status === "completed" ? "" : (current.status === "blocked" ? "blocked" : "failed");
    const mc = current.metric_control;
    const context = current.roles ? `\nRoles scheduled: ${current.roles.join(", ")}; mask set ${current.mask_set || "none"}` +
      (mc && mc.status !== "not_requested" ? `; neural control: ${mc.ready ? (mc.label || "prepared metric CAD control") + ", bundle " + short(mc.bundle_sha256) : (mc.label || mc.status)}` : "") : "";
    setStatus(el, `Job ${current.job_id.slice(0, 8)}: ${current.status}` +
      (current.blocked_reason ? `\n${current.blocked_reason}` : "") + (current.error ? `\n${current.error}` : "") + context, kind);
    return current;
  }

  async function startRun() {
    const roles = selectedRoles();
    if (!roles.length) throw new Error("Choose at least one role: the paired neural comparison and/or automatic relative geometry.");
    clearProvenance();
    const body = {
      clip_id: state.clip.clip.clip_id, mode: $("run-mode").value, artwork_kind: $("run-artwork").value,
      mask_set: $("run-masks").value || undefined, roles,
    };
    if (state.selectionId) body.selection_id = state.selectionId;
    if (body.mode === "focused") { body.first_frame = $("run-first").value; body.count = Number($("run-count").value); }
    const created = await postJSON("/api/runs", body);
    const job = await postJSON("/api/jobs", { run_id: created.run_id });
    const done = await followJob(job, $("job-status"));
    try { await openRun(created.run_id); } catch (e) {
      const res = await api(`/api/jobs/${done.job_id}/results`);
      showRun({ run: { run: created.run }, report: {}, frame_results: res.frame_results });
    }
  }

  // ----------------------------------------------------------------- viewer
  async function openRun(runId) { clearProvenance(); showRun(await api(`/api/runs/${encodeURIComponent(runId)}`)); }

  const ALL_ROLES = ["primary_neural", "frozen_baseline", "relative_geometry"];
  const ROLE_FIGURES = { relative_geometry: "fig-relative", primary_neural: "fig-primary", frozen_baseline: "fig-baseline" };

  // Scheduled roles come from the run itself (RunSpec.roles), then the report,
  // never from which rows happen to exist on the current frame.
  function scheduledRoles(detail) {
    const run = (detail.run && (detail.run.run || detail.run)) || {};
    if (Array.isArray(run.roles) && run.roles.length) return run.roles.filter((r) => ALL_ROLES.includes(r));
    const reported = Object.keys((detail.report && detail.report.roles) || {});
    if (reported.length) return reported.filter((r) => ALL_ROLES.includes(r));
    return ALL_ROLES.filter((r) => (detail.frame_results || []).some((row) => row.backend_role === r));
  }

  function applyRoleLayout(roles) {
    for (const [role, id] of Object.entries(ROLE_FIGURES)) $(id).hidden = !roles.includes(role);
    // Original + observed mask + one artwork pane per scheduled role, on one row when width allows.
    $("panes").style.setProperty("--pane-count", String(2 + roles.filter((r) => ROLE_FIGURES[r]).length));
    const neural = roles.filter((r) => r !== "relative_geometry");
    $("pane-neural").hidden = neural.length === 0;
    const missing = ALL_ROLES.filter((r) => !roles.includes(r));
    $("roles-context").textContent = `Scheduled: ${roles.join(", ") || "none"}; Not scheduled: ${missing.join(", ") || "none"}` +
      (neural.length === 0 ? " (neural roles not scheduled in this run)" : "");
  }

  function showRun(detail) {
    clearProvenance();
    resetReplay();
    state.run = detail;
    const run = detail.run.run || detail.run;
    state.order = run.scheduled_frame_ids;
    state.roles = scheduledRoles(detail);
    state.rows = {};
    for (const row of detail.frame_results) {
      (state.rows[row.frame_id] ||= {})[row.backend_role] = row;
    }
    $("view-panel").hidden = false;
    applyRoleLayout(state.roles);
    $("frame-slider").max = String(Math.max(0, state.order.length - 1));
    renderEvidence(detail); renderBookmarks(); show(0);
    renderProvenance(detail);   // only once the loaded run's data is ready
  }

  // ------------------------------------------------- recorded run provenance
  // Replay identity comes ONLY from the loaded run's recorded RunSpec.config
  // (GET /api/runs/{id} -> run.json -> run.run.config). The create-run form
  // (#run-masks etc.) is never read here, and nothing is inferred from mask-set names.
  function provenanceStrip() {
    let el = $("run-provenance");
    if (!el) {
      el = document.createElement("div");
      el.id = "run-provenance"; el.className = "hint"; el.setAttribute("role", "note");
      el.setAttribute("aria-label", "Recorded run provenance");
      el.style.overflowWrap = "anywhere";   // long SHA-256 values wrap instead of widening the panel
      el.hidden = true;
      $("view-h").insertAdjacentElement("afterend", el);
    }
    return el;
  }

  function clearProvenance() { const el = provenanceStrip(); el.replaceChildren(); el.hidden = true; }

  function renderProvenance(detail) {
    const el = provenanceStrip();
    el.replaceChildren();
    const spec = (detail && detail.run && (detail.run.run || detail.run)) || null;
    const config = spec && spec.config && typeof spec.config === "object" && !Array.isArray(spec.config) ? spec.config : null;
    const recorded = (key) => {
      if (!config) return "unknown (no recorded run configuration)";
      if (!(key in config)) return "not recorded";
      const v = config[key];
      return typeof v === "string" && v ? v : "unrecognised recorded value";
    };
    const line = (label, value, mono) => {
      const row = document.createElement("span");
      row.style.display = "block";
      row.append(document.createTextNode(`${label}: `));
      const val = document.createElement("span");
      if (mono) val.className = "mono";
      val.textContent = value;
      row.append(val);
      el.append(row);
    };
    const runIdValue = spec && typeof spec.run_id === "string" && spec.run_id ? spec.run_id : "unknown";
    line("Loaded run", runIdValue, true);
    line("Recorded observed mask set", recorded("mask_set"), true);
    line("Recorded segmentation prompt SHA-256", recorded("segmentation_prompt_sha256"), true);
    if (config && config.development_diagnostic_only === true) {
      const note = typeof config.segmentation_diagnostic === "string" && config.segmentation_diagnostic
        ? `Development diagnostic (recorded segmentation: ${config.segmentation_diagnostic})` : "Development diagnostic";
      line("Run class", note, false);
    }
    renderRecoveryProvenance(detail, config, line);
    renderTimingProvenance(detail, spec, line);
    const src = document.createElement("span");
    src.style.display = "block";
    src.textContent = "Recorded configuration for this replay. Attachment accuracy: unverified.";
    el.append(src);
    el.hidden = false;
  }

  // Evidence-recovery provenance comes ONLY from the recorded RunSpec.config flag and the
  // loaded report's `recovery` block. Nothing is inferred from run IDs or the form.
  const RECOVERY_STATUS = "completed_evidence_recovery";
  function renderRecoveryProvenance(detail, config, line) {
    if (!config || config.evidence_recovery_only !== true) return;
    const report = detail && detail.report;
    const rec = report && report.recovery;
    if (!rec || typeof rec !== "object" || Array.isArray(rec) || report.status !== RECOVERY_STATUS || rec.status !== RECOVERY_STATUS) {
      line("Evidence recovery", "flagged in the recorded config, but no completed recovery report (status unknown)", false);
      return;
    }
    const str = (v) => (typeof v === "string" && v ? v : "not recorded");
    const count = (v) => (Number.isInteger(v) && v >= 0 ? String(v) : "not recorded");
    line("Evidence recovery", `saved predictions reused from source run ${str(rec.source_neural_run_id)} `
      + `(recorded source status ${str(rec.source_report_status)}, exit ${count(rec.source_exit_code)})`, false);
    const none = rec.new_neural_execution === false && rec.new_neural_frames === 0 && rec.stage_cache_hit === false && rec.gpu_used === false;
    line("Neural source check", none
      ? "no new neural inference, no stage-cache hit, no GPU (0 new neural frames); head identities are reused from the source rows"
      : "not confirmed by the recorded report", false);
    line("Overlay evidence", `${count(rec.overlays_copied)} copied from the source run; ${count(rec.overlays_regenerated)} regenerated `
      + "by deterministic CPU re-render matching the recorded render stats (historical bytes not provable)", false);
    const wall = rec.current_cpu_wall_ms;
    line("Recorded CPU recovery wall (not inference)",
      typeof wall === "number" && Number.isFinite(wall) && wall >= 0 ? `${(wall / 1000).toFixed(1)} s` : "not recorded", false);
  }

  // Timing/cache provenance comes ONLY from the loaded report's telemetry
  // (detail.report.telemetry). Nothing is inferred from run IDs, mask names or the form.
  // Resource labels are presentation only: a legacy gpu_used boolean proves neither per-run GPU
  // work nor idleness, and memory peaks are process-scoped observations (not per-run peaks).
  const CACHE_MODES = ["cached", "cold", "partially_cached", "artifact_recovery"];
  const MEMORY_SCOPE_NOTE = "Process memory observations do not establish GPU work or per-run memory peaks.";
  const MEMORY_FIELDS = [
    ["process_peak_rss_bytes", "Host process peak RSS (not a per-run peak)"],
    ["gpu_peak_allocated_bytes", "Process CUDA allocator peak allocated (cumulative; epoch unrecorded)"],
    ["gpu_peak_reserved_bytes", "Process CUDA allocator peak reserved (cumulative; epoch unrecorded)"],
  ];
  // Only a real JSON boolean is shown; 0/1/strings are never coerced.
  function legacyGpuLabel(v) {
    return v === true ? "true (recorded legacy value)" : v === false ? "false (recorded legacy value)" : "not recorded";
  }
  // Valid non-negative safe-integer byte counts only (zero is a valid zero).
  function recordedBytesLabel(v) {
    return typeof v === "number" && Number.isSafeInteger(v) && v >= 0 ? `${v} B` : "not recorded";
  }
  function renderTimingProvenance(detail, spec, line) {
    const tel = detail && detail.report && detail.report.telemetry;
    if (!tel || typeof tel !== "object" || Array.isArray(tel)) {
      line("Timing provenance", "not recorded (no report telemetry)", false);
      return;
    }
    const obj = (v) => (v && typeof v === "object" && !Array.isArray(v) ? v : null);
    const cache = obj(tel.cache), cached = obj(tel.cached_original_timings);
    const frames = Array.isArray(tel.frames) ? tel.frames : null;
    line("Recorded cache mode", CACHE_MODES.includes(tel.run_cache_mode) ? tel.run_cache_mode : "unknown (not recorded or unrecognised)", true);
    const wall = tel.run_wall_ms;
    line("Recorded pipeline wall (excludes outer command validation)",
      typeof wall === "number" && Number.isFinite(wall) && wall >= 0 ? `${(wall / 1000).toFixed(1)} s` : "not recorded", false);
    line("Legacy GPU field; run scope unverified", legacyGpuLabel(tel.gpu_used), false);
    for (const [key, label] of MEMORY_FIELDS) line(label, recordedBytesLabel(tel[key]), false);
    line("Memory scope", MEMORY_SCOPE_NOTE, false);
    let roles = spec && Array.isArray(spec.roles) ? spec.roles.filter((r) => typeof r === "string") : null;
    if (!roles || !roles.length) roles = [...new Set([...Object.keys(cache || {}), ...Object.keys(cached || {})])];
    const historical = [], fresh = [], unknown = [];
    for (const role of roles) {
      const entry = cache && obj(cache[role]);
      const hit = entry !== null && entry.status === "cache_hit";
      const history = cached && obj(cached[role]);
      if (hit && history && history.source === "cached_original_run") historical.push(role);
      else if (!hit && entry && frames && frames.some((f) => f && f.role === role)) fresh.push(role);
      else unknown.push(role);
    }
    const parts = [];
    if (historical.length) parts.push(`historical from cached original run (${historical.join(", ")}); not new inference`);
    if (fresh.length) parts.push(`recorded frame timings from this run (${fresh.join(", ")})`);
    if (unknown.length) parts.push(`unknown (${unknown.join(", ")})`);
    line("Per-frame timings", parts.join("; ") || "not recorded", false);
  }

  function frameUrl(fid) { return state.clip.frame_url_template.replace("{frame_id}", encodeURIComponent(fid)); }
  function runId() { const r = state.run.run.run || state.run.run; return r.run_id; }

  function setImage(id, url) {
    // Never leave a broken-image icon: hide the element until a real source exists.
    const img = $(id);
    if (!url) { img.hidden = true; img.removeAttribute("src"); return; }
    img.hidden = false;
    img.onerror = () => { img.hidden = true; img.removeAttribute("src"); };
    img.src = url;
  }

  // Recorded render exclusion counts (evidence.render_stats of THIS role's row on THIS frame).
  // Diagnostic display only: counts are shown as recorded; no cause or accuracy is inferred.
  const RENDER_COUNT_FIELDS = [
    ["rendered_pixels", "rendered"], ["mask_rejected_pixels", "excluded by mask"],
    ["backfacing_rejected_pixels", "back-facing"], ["overlap_rejected_pixels", "overlap"],
  ];
  function recordedCount(stats, key) {
    const v = stats[key];
    return typeof v === "number" && Number.isFinite(v) && v >= 0 ? String(v) : "not recorded";
  }
  function renderCountsText(row) {
    if (!row) return "Recorded pixels: not recorded (no result row for this frame)";
    const stats = row.evidence && row.evidence.render_stats;
    if (!stats || typeof stats !== "object" || Array.isArray(stats)) return "Recorded pixels: not recorded for this frame";
    return "Recorded pixels: " + RENDER_COUNT_FIELDS.map(([key, label]) => `${label} ${recordedCount(stats, key)}`).join("; ");
  }
  function renderCountsElement(role) {
    const figure = $(ROLE_FIGURES[role]);
    let el = figure.querySelector(":scope > .render-counts");
    if (!el) {
      el = document.createElement("p");
      el.className = "render-counts";
      figure.appendChild(el);   // below the image pane, never over the object
    }
    return el;
  }

  function overlayPane(role, imgId, noteId, rows, fid) {
    const r = rows[role];
    renderCountsElement(role).textContent = renderCountsText(r);
    if (r && r.render_state === "rendered" && r.pose && r.evidence && r.evidence.overlay) {
      setImage(imgId, `/api/runs/${encodeURIComponent(runId())}/frames/${encodeURIComponent(fid)}/overlay?role=${role}`);
    } else if (r && r.render_state === "rendered" && r.pose && replayEligible(role, fid)) {
      // Original artwork file unavailable: optional RAM-only CPU replay of the SAVED pose (never new tracking).
      setImage(imgId, null);
      $(noteId).textContent = `${REPLAY_LABEL} Loading…`;
      scheduleReplay(role, imgId, noteId, fid);
    } else {
      setImage(imgId, null);
      $(noteId).textContent = r ? `Artwork ${r.render_state}: ${r.failure_reason || (r.evidence && r.evidence.render) || "no rendered pixels for the current validated pose"}`
        : "No result row for this scheduled role on this frame.";
    }
  }

  // ------------------------------------------- RAM-only artwork replay (saved pose)
  // One debounced request per visible eligible pane; superseded requests are aborted, every
  // response is checked against the CURRENT run/frame/role and generation before it is applied,
  // and blob URLs are revoked on every frame change. Not real-time; not the original file.
  const REPLAY_LABEL = "CPU replay from saved pose; original artwork file unavailable. Historical bytes cannot be verified.";
  const replay = { gen: 0, controllers: [], urls: [], pending: [], timer: null };
  function resetReplay() {
    replay.gen += 1;
    clearTimeout(replay.timer); replay.timer = null; replay.pending = [];
    for (const c of replay.controllers) c.abort();
    replay.controllers = [];
    for (const u of replay.urls) URL.revokeObjectURL(u);
    replay.urls = [];
  }
  function replayEligible(role, fid) {
    const src = state.run && state.run.artwork_sources;
    if (!src || !Array.isArray(src.rows)) return false;
    return src.rows.some((x) => x.role === role && x.frame_id === fid && x.ram_replay === "eligible");
  }
  function scheduleReplay(role, imgId, noteId, fid) {
    replay.pending.push({ role, imgId, noteId, fid });
    clearTimeout(replay.timer);
    const gen = replay.gen, run = runId();
    replay.timer = setTimeout(() => {
      const jobs = replay.pending; replay.pending = []; replay.timer = null;
      runReplayJobs(jobs, gen, run);
    }, 300);
  }
  // The server renders one replay at a time, so the current frame's panes are requested
  // SEQUENTIALLY (never self-inducing "busy"); a busy reply from another client gets a small,
  // cancel-aware bounded retry. Any frame/run change ends the whole sequence.
  async function runReplayJobs(jobs, gen, run) {
    for (const job of jobs) {
      if (!replayCurrent(job, gen, run)) return;
      await fetchReplay(job, gen, run);
    }
  }
  const REPLAY_BUSY_RETRIES = 3;
  const REPLAY_BUSY_DELAY_MS = 500;
  function replayDelay(ms, gen) {
    return new Promise((resolve) => {
      const t = setTimeout(() => resolve(gen === replay.gen), ms);
      replay.controllers.push({ abort: () => { clearTimeout(t); resolve(false); } });
    });
  }
  function replayCurrent(job, gen, run) {
    return gen === replay.gen && state.run && run === runId() && state.order[state.index] === job.fid;
  }
  async function fetchReplay(job, gen, run) {
    const url = `/api/runs/${encodeURIComponent(run)}/frames/${encodeURIComponent(job.fid)}/artwork-replay?role=${encodeURIComponent(job.role)}`;
    for (let attempt = 0; attempt <= REPLAY_BUSY_RETRIES; attempt += 1) {
      if (!replayCurrent(job, gen, run)) return;
      const controller = new AbortController(); replay.controllers.push(controller);
      let res, body;
      try {
        res = await fetch(url, { credentials: "same-origin", signal: controller.signal });
        body = await res.json();
      } catch (err) {
        if (err && err.name === "AbortError") return;
        if (replayCurrent(job, gen, run)) $(job.noteId).textContent = `${REPLAY_LABEL} Replay request failed.`;
        return;
      }
      if (!replayCurrent(job, gen, run)) return;
      if (res.status === 429 && body && body.status === "busy" && attempt < REPLAY_BUSY_RETRIES) {
        $(job.noteId).textContent = `${REPLAY_LABEL} Waiting for another replay to finish…`;
        if (!(await replayDelay(REPLAY_BUSY_DELAY_MS, gen))) return;
        continue;
      }
      if (!res.ok || !body || body.status !== "ok" || body.run_id !== run || body.frame_id !== job.fid ||
          body.role !== job.role || body.source !== "cpu_replay_of_saved_pose" || !body.png || typeof body.png.base64 !== "string") {
        $(job.noteId).textContent = `${REPLAY_LABEL} Replay unavailable (${(body && body.status) || res.status}): ${(body && body.reason) || "no reason given"}`;
        return;
      }
      const bin = atob(body.png.base64); const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
      const blobUrl = URL.createObjectURL(new Blob([bytes], { type: "image/png" }));
      if (!replayCurrent(job, gen, run)) { URL.revokeObjectURL(blobUrl); return; }
      replay.urls.push(blobUrl);
      setImage(job.imgId, blobUrl);
      $(job.noteId).textContent = `${REPLAY_LABEL} Not the original file and not new tracking.`;
      return;
    }
  }

  function show(i) {
    if (!state.order.length) return;
    state.index = Math.max(0, Math.min(i, state.order.length - 1));
    const fid = state.order[state.index];
    const rows = state.rows[fid] || {};
    // Abort/forget any pending artwork replay and revoke its blob URLs before anything else.
    resetReplay();
    // Clear frame-bound overlays first so no stale render survives a frame change.
    for (const id of ["pane-mask", "pane-art", "pane-primary", "pane-baseline"]) setImage(id, null);
    for (const id of ["pane-mask-note", "pane-art-note", "pane-primary-note", "pane-baseline-note"]) $(id).textContent = "";
    // Clear every role's recorded counts so no earlier frame's values survive (e.g. recovering rows without stats).
    for (const role of Object.keys(ROLE_FIGURES)) renderCountsElement(role).textContent = "";
    $("frame-slider").value = String(state.index);
    const any = Object.values(rows)[0];
    $("frame-label").textContent = `${fid} · source ${any ? any.source_frame_index : "?"}` +
      (any && any.timestamp_s !== null ? ` · ${any.timestamp_s.toFixed(3)} s` : "") + ` · ${state.index + 1}/${state.order.length}`;
    for (const id of ["pane-original", "pane-art-base", "pane-primary-base", "pane-baseline-base"]) setImage(id, frameUrl(fid));
    const observed = Object.values(rows).find((r) => r.mask_state === "observed" && r.mask_frame_id === fid);
    if (observed) setImage("pane-mask", `/api/runs/${encodeURIComponent(runId())}/frames/${encodeURIComponent(fid)}/mask`);
    else $("pane-mask-note").textContent = "No observed mask for this frame (never substituted by a projected silhouette).";
    const roles = state.roles || ALL_ROLES;
    if (roles.includes("relative_geometry")) overlayPane("relative_geometry", "pane-art", "pane-art-note", rows, fid);
    if (roles.includes("primary_neural")) overlayPane("primary_neural", "pane-primary", "pane-primary-note", rows, fid);
    if (roles.includes("frozen_baseline")) overlayPane("frozen_baseline", "pane-baseline", "pane-baseline-note", rows, fid);
    const neural = $("pane-neural");
    neural.replaceChildren(...["primary_neural", "frozen_baseline"].filter((role) => roles.includes(role)).map((role) => {
      const r = rows[role]; const p = document.createElement("p");
      const label = role === "primary_neural" ? "New head (primary)" : "Frozen baseline";
      // A scheduled role is always shown, even when it has no row for this frame.
      if (!r) { p.textContent = `${label}: no result row for this frame`; return p; }
      const t = r.pose ? r.pose.slice(0, 3).map((row) => row[3].toFixed(4)).join(", ") : null;
      p.textContent = `${label}: pose ${r.pose_state}` + (t ? ` — t = [${t}] ${r.pose_units}` : ` — ${r.failure_reason}`);
      return p;
    }));
    $("state-rows").replaceChildren(...Object.entries(rows).map(([role, r]) => {
      const tr = document.createElement("tr");
      for (const value of [role, r.mask_state, r.pose_state, r.render_state, r.failure_reason || ""]) {
        const td = document.createElement("td"); td.textContent = value;
        if (value === "tracking" || value === "rendered") td.className = "state-tracking";
        else if (["blocked", "lost", "unavailable", "rejected"].includes(value)) td.className = "state-bad";
        tr.append(td);
      }
      return tr;
    }));
  }

  function play(on) {
    state.playing = on; $("play").setAttribute("aria-pressed", String(on)); $("play").textContent = on ? "Pause" : "Play";
    clearTimeout(state.timer);
    const step = () => {
      if (!state.playing) return;
      if (state.index >= state.order.length - 1) { play(false); return; }
      show(state.index + 1);
      const img = $("pane-original");
      const go = () => { state.timer = setTimeout(step, 150); };
      if (img.complete) go(); else { img.addEventListener("load", go, { once: true }); img.addEventListener("error", go, { once: true }); }
    };
    if (on) step();
  }

  // -------------------------------------------------------------- bookmarks
  function bookmarkKey() { return `workbench-bookmarks-${runId()}`; }
  function autoEvents() {
    const out = [];
    for (const role of ["relative_geometry", "primary_neural"]) {
      let prev = null;
      state.order.forEach((fid, i) => {
        const r = (state.rows[fid] || {})[role]; if (!r) return;
        const tracking = r.pose_state === "tracking";
        if (prev !== null && prev && !tracking) out.push({ i, kind: "loss", text: `${role}: ${r.failure_reason}`, auto: true });
        if (prev !== null && !prev && tracking) out.push({ i, kind: "recovery", text: role, auto: true });
        prev = tracking;
      });
    }
    return out;
  }
  function renderBookmarks() {
    const user = JSON.parse(localStorage.getItem(bookmarkKey()) || "[]");
    const all = [...autoEvents(), ...user].sort((a, b) => a.i - b.i);
    $("bookmarks").replaceChildren(...all.map((b) => {
      const li = document.createElement("li"); const btn = document.createElement("button");
      btn.type = "button"; btn.textContent = `${state.order[b.i]} — ${b.kind}${b.auto ? " (auto)" : ""}`;
      btn.addEventListener("click", () => show(b.i));
      li.append(btn, document.createTextNode(` ${b.text || ""}`)); return li;
    }));
  }
  function addBookmark() {
    const list = JSON.parse(localStorage.getItem(bookmarkKey()) || "[]");
    list.push({ i: state.index, kind: $("note-kind").value, text: $("note-text").value.slice(0, 200) });
    localStorage.setItem(bookmarkKey(), JSON.stringify(list)); $("note-text").value = ""; renderBookmarks();
  }

  function renderEvidence(detail) {
    const report = detail.report || {};
    const dl = $("evidence"); const items = [];
    const add = (k, v) => { const dt = document.createElement("dt"); dt.textContent = k; const dd = document.createElement("dd"); dd.textContent = v; items.push(dt, dd); };
    const primary = report.primary || {};
    add("Run", `${runId()} (${report.mode || "?"}, status ${report.status || "unknown"})`);
    add("Primary model", `${fmt(primary.artifact_id)} · export ${fmt(primary.export_sha256)} · ${fmt(primary.export_bytes)} bytes`);
    const acc = report.accuracy || {};
    add("Accuracy", `${fmt(acc.status)}; median ${fmt(acc.median_px_720)} px, p95 ${fmt(acc.p95_px_720)} px (gates verified: ${fmt(acc.gates_verified)})`);
    for (const [role, v] of Object.entries(report.roles || {})) {
      const first = detail.frame_results.find((r) => r.backend_role === role);
      const bid = first ? first.backend_identity || {} : {};
      const headLabel = role === "frozen_baseline" && bid.head === "frozen_base_pose_head" ?
        "pretrained base head (no separate export)" : `installed head ${fmt(bid.installed_head_sha256)}`;
      const installed = role === "relative_geometry" ? "" :
        ` · ${headLabel} · head tensors ${fmt(bid.installed_head_state_sha256)}`;
      add(role, `availability ${v.availability === null ? "unavailable" : (100 * v.availability).toFixed(1) + "%"} of ${v.scheduled_frames} scheduled${installed}; ` +
        `identity ${first ? JSON.stringify(bid) : "unavailable"}`);
    }
    const stages = report.stages || {};
    for (const [name, s] of Object.entries(stages)) add(`stage ${name}`, stageLabel(name, s, stages, report));
    dl.replaceChildren(...items);
  }

  // A "<role>_load" stage is written once as "running" when loading starts and is
  // never rewritten. Present it from the role's terminal outcome instead of as busy;
  // no timing is invented and the stored artifact is not changed.
  const TERMINAL = new Set(["completed", "cache_hit", "blocked", "failed", "cancelled"]);
  function stageLabel(name, s, stages, report) {
    const reason = s.reason ? ` — ${s.reason}` : "";
    if (s.status === "running" && name.endsWith("_load")) {
      const role = stages[name.slice(0, -"_load".length)];
      if (role && TERMINAL.has(role.status)) {
        return `load finished (role ${role.status}; start event recorded as running)${reason}`;
      }
      if (TERMINAL.has(report.status)) {
        return `load started; run ended ${report.status} without a terminal role stage${reason}`;
      }
    }
    return `${s.status}${reason}`;
  }

  // ------------------------------------------------------------------ wiring
  function bind() {
    $("clip-load").addEventListener("click", () => openClip($("clip-select").value).catch((e) => setStatus($("service-status"), e.message, "failed")));
    $("key-start").addEventListener("change", updateKeyframes);
    $("key-end").addEventListener("change", () => { state.points = []; state.selectionId = null; updateKeyframes(); });
    $("mode-pos").addEventListener("click", () => setMode(1));
    $("mode-neg").addEventListener("click", () => setMode(0));
    $("points-clear").addEventListener("click", () => { state.points = []; renderPoints(); });
    $("select-stage").addEventListener("click", (e) => addPoint(clientToSource(e.clientX, e.clientY)));
    $("select-stage").addEventListener("keydown", (e) => {
      if (!state.clip) return;
      const W = state.clip.clip.width, H = state.clip.clip.height;
      state.cursor ||= { x: Math.floor(W / 2), y: Math.floor(H / 2) };
      const d = e.shiftKey ? 10 : 1;
      const moves = { ArrowLeft: [-d, 0], ArrowRight: [d, 0], ArrowUp: [0, -d], ArrowDown: [0, d] };
      if (moves[e.key]) {
        e.preventDefault();
        state.cursor.x = Math.max(0, Math.min(W - 1, state.cursor.x + moves[e.key][0]));
        state.cursor.y = Math.max(0, Math.min(H - 1, state.cursor.y + moves[e.key][1]));
        renderPoints();
      } else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); addPoint({ ...state.cursor }); }
      else if (e.key.toLowerCase() === "p") setMode(1);
      else if (e.key.toLowerCase() === "n") setMode(0);
    });
    $("selection-save").addEventListener("click", () => saveSelection().catch((e) => setStatus($("select-status"), e.message, "failed")));
    $("segment-go").addEventListener("click", () => segment().catch((e) => setStatus($("select-status"), e.message, "failed")));
    $("run-go").addEventListener("click", () => startRun().catch((e) => setStatus($("job-status"), e.message, "failed")));
    $("job-cancel").addEventListener("click", () => state.job && postJSON(`/api/jobs/${state.job.job_id}/cancel`, {}).catch(() => {}));
    $("run-open").addEventListener("click", () => openRun($("run-select").value).catch((e) => setStatus($("job-status"), e.message, "failed")));
    $("prev").addEventListener("click", () => { play(false); show(state.index - 1); });
    $("next").addEventListener("click", () => { play(false); show(state.index + 1); });
    $("play").addEventListener("click", () => play(!state.playing));
    $("frame-slider").addEventListener("input", (e) => { play(false); show(Number(e.target.value)); });
    $("note-add").addEventListener("click", addBookmark);
    $("import-go").addEventListener("click", async () => {
      const file = $("import-file").files[0];
      if (!file) return;
      try {
        const res = await api("/api/import", { method: "POST", body: file, headers: {
          "Content-Type": file.type || "application/octet-stream", "X-Filename": file.name,
          "X-Clip-Id": $("import-clip-id").value, "X-Intrinsics": $("import-k").value || "unavailable" } });
        setStatus($("service-status"), `Imported ${res.original_name}: ${res.registration_status}${res.blocked_reason ? " — " + res.blocked_reason : ""}`,
          res.registration_status === "registered" ? "" : "blocked");
        await loadClips(res.clip_id);
      } catch (e) { setStatus($("service-status"), e.message, "failed"); }
    });
  }

  bind(); loadStatus(); loadClips().catch((e) => setStatus($("service-status"), e.message, "failed"));
})();
