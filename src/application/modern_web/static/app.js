/* ------------------------------------------------------------------ *
 * Selective Sound Removal — front-end controller
 *
 * Drives the three-step flow against the FastAPI back-end:
 *   1. upload         -> POST /api/upload   (returns a file_id)
 *   2. analyze        -> POST /api/detect   (returns detected sources)
 *   3. apply & render -> POST /api/process  (returns cleaned audio/video + stems)
 *
 * Each detected source carries its own action — keep / reduce / remove —
 * which maps to a per-source strength sent to the back-end.
 * ------------------------------------------------------------------ */
"use strict";

const state = {
  models: [],
  model: null,
  file: null,        // { file_id, filename, is_video, duration }
  detected: [],      // [{ name, score }]
  ranked: [],        // [{ name, score }]
  hasDetectionHead: false,
  selections: {},    // name -> { action: 'keep'|'reduce'|'remove', strength }
};

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ //
// Small UI utilities
// ------------------------------------------------------------------ //
function showOverlay(title, sub) {
  $("overlayTitle").textContent = title;
  $("overlaySub").textContent = sub || "";
  $("overlay").classList.remove("hidden");
}
function hideOverlay() { $("overlay").classList.add("hidden"); }

let toastTimer = null;
function toast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), 4200);
}

function prettify(name) { return name.replace(/_/g, " "); }

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

// ------------------------------------------------------------------ //
// Step 1 — models
// ------------------------------------------------------------------ //
async function loadModels() {
  try {
    const data = await api("/api/models");
    state.models = data.models;
    const sel = $("modelSelect");
    sel.innerHTML = "";
    data.models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = m.name;
      if (m.is_default) opt.selected = true;
      sel.appendChild(opt);
    });
    const def = data.models.find((m) => m.is_default) || data.models[0];
    setModel(def.name);
  } catch (err) {
    $("modelBadge").textContent = "No models found";
    toast("Could not load models: " + err.message);
  }
}

function setModel(name) {
  state.model = name;
  const m = state.models.find((x) => x.name === name);
  $("modelBadge").textContent = m ? `${m.num_classes} classes` : name;
  $("modelBadge").classList.remove("badge-muted");
  $("modelBadge").classList.add("badge-soft");
  $("modelMeta").innerHTML = m
    ? `Active: <strong>${name}</strong> · ${m.num_classes} sound classes`
    : "";
}

// ------------------------------------------------------------------ //
// Step 2 — upload
// ------------------------------------------------------------------ //
const AUDIO_ICON = '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 12h2l2-6 3 16 3-12 2 8 2-4h4"/></svg>';
const VIDEO_ICON = '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="5" width="14" height="14" rx="2"/><path d="M16 9l6-3v12l-6-3"/></svg>';

async function handleFile(file) {
  if (!file) return;
  resetDownstream();
  showOverlay("Uploading…", file.name);
  const fd = new FormData();
  fd.append("file", file);
  try {
    const info = await api("/api/upload", { method: "POST", body: fd });
    state.file = info;
    renderFileBox(info, file);
    $("analyzeBtn").disabled = false;
  } catch (err) {
    toast("Upload failed: " + err.message);
  } finally {
    hideOverlay();
  }
}

function renderFileBox(info, rawFile) {
  $("fileIcon").innerHTML = info.is_video ? VIDEO_ICON : AUDIO_ICON;
  $("fileName").textContent = info.filename || "uploaded file";
  const bits = [];
  bits.push(info.is_video ? "Video" : "Audio");
  if (info.duration) bits.push(`${info.duration.toFixed(1)} s`);
  $("fileSub").textContent = bits.join(" · ");
  $("fileBox").classList.remove("hidden");
  $("dropzone").classList.add("hidden");

  // Local preview straight from the chosen file (no extra round-trip).
  const url = URL.createObjectURL(rawFile);
  const wrap = $("previewWrap");
  wrap.innerHTML = "";
  const el = document.createElement(info.is_video ? "video" : "audio");
  el.src = url;
  el.controls = true;
  wrap.appendChild(el);
  wrap.classList.remove("hidden");
}

function clearFile() {
  state.file = null;
  $("fileBox").classList.add("hidden");
  $("dropzone").classList.remove("hidden");
  $("previewWrap").classList.add("hidden");
  $("previewWrap").innerHTML = "";
  $("fileInput").value = "";
  $("analyzeBtn").disabled = true;
  resetDownstream();
}

function resetDownstream() {
  state.detected = [];
  state.ranked = [];
  state.selections = {};
  $("detectCard").classList.add("hidden");
  $("resultCard").classList.add("hidden");
  $("sourcesCard").classList.add("hidden");
}

// ------------------------------------------------------------------ //
// Step 2b — analyze / detect
// ------------------------------------------------------------------ //
async function analyze() {
  if (!state.file || !state.model) return;
  showOverlay("Analyzing…",
    "Running the model for each sound class. The first run can take up to a minute.");
  try {
    const data = await api("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_id: state.file.file_id, model: state.model }),
    });
    state.detected = data.detected || [];
    state.ranked = data.ranked || [];
    state.hasDetectionHead = !!data.has_detection_head;
    renderDetected();
  } catch (err) {
    toast("Analysis failed: " + err.message);
  } finally {
    hideOverlay();
  }
}

function defaultSelection() {
  // Detected sources default to "remove"; the user dials any back to keep/reduce.
  return { action: "remove", strength: 1.0 };
}

function renderDetected() {
  const grid = $("soundGrid");
  grid.innerHTML = "";
  state.selections = {};

  state.detected.forEach((d) => {
    state.selections[d.name] = defaultSelection();
    grid.appendChild(buildSoundCard(d));
  });

  $("noDetect").classList.toggle("hidden", state.detected.length > 0);
  $("bulkRow").classList.toggle("hidden", state.detected.length === 0);
  $("detectCard").classList.remove("hidden");
  updateProcessBtn();
  $("detectCard").scrollIntoView({ behavior: "smooth", block: "start" });
}

function buildSoundCard(d) {
  const sel = state.selections[d.name];
  const card = document.createElement("div");
  card.className = "sound-card act-" + sel.action;
  card.dataset.name = d.name;

  const confPct = state.hasDetectionHead
    ? Math.round(Math.min(1, d.score) * 100)
    : Math.round(Math.min(1, d.score) * 100);

  card.innerHTML = `
    <div class="sound-top">
      <div>
        <div class="sound-name">${prettify(d.name)}</div>
        <div class="conf">
          <div class="conf-bar"><div class="conf-fill" style="width:${confPct}%"></div></div>
          <span class="conf-val">${state.hasDetectionHead ? confPct + "% likely" : "score " + d.score}</span>
        </div>
      </div>
      <div class="seg" role="group" aria-label="Action for ${prettify(d.name)}">
        <button data-act="keep">Keep</button>
        <button data-act="reduce">Reduce</button>
        <button data-act="remove">Remove</button>
      </div>
    </div>
    <div class="reduce-row hidden">
      <span class="muted">Reduction</span>
      <input type="range" min="10" max="90" step="5" value="50" />
      <span class="amt">50%</span>
    </div>`;

  const segButtons = card.querySelectorAll(".seg button");
  const reduceRow = card.querySelector(".reduce-row");
  const slider = card.querySelector('input[type="range"]');
  const amt = card.querySelector(".amt");

  function paint() {
    segButtons.forEach((b) => b.classList.toggle("active", b.dataset.act === sel.action));
    card.className = "sound-card act-" + sel.action;
    reduceRow.classList.toggle("hidden", sel.action !== "reduce");
  }

  segButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      sel.action = btn.dataset.act;
      if (sel.action === "remove") sel.strength = 1.0;
      else if (sel.action === "keep") sel.strength = 0.0;
      else sel.strength = slider.value / 100;
      paint();
      updateProcessBtn();
    });
  });
  slider.addEventListener("input", () => {
    amt.textContent = slider.value + "%";
    sel.strength = slider.value / 100;
  });

  paint();
  return card;
}

function setAllActions(action) {
  state.detected.forEach((d) => {
    const sel = state.selections[d.name];
    sel.action = action;
    sel.strength = action === "remove" ? 1.0 : action === "keep" ? 0.0 : 0.5;
  });
  renderDetectedKeepScroll();
}

// Re-render cards in place without yanking the scroll position.
function renderDetectedKeepScroll() {
  const grid = $("soundGrid");
  grid.innerHTML = "";
  state.detected.forEach((d) => grid.appendChild(buildSoundCard(d)));
  updateProcessBtn();
}

function updateProcessBtn() {
  const active = Object.values(state.selections).some((s) => s.action !== "keep");
  $("processBtn").disabled = !active;
}

// ------------------------------------------------------------------ //
// Step 3 — process
// ------------------------------------------------------------------ //
async function process() {
  const sounds = Object.entries(state.selections)
    .filter(([, s]) => s.action !== "keep")
    .map(([name, s]) => ({ name, strength: s.strength }));
  if (!sounds.length) { toast("Pick at least one source to reduce or remove."); return; }

  showOverlay("Rendering…",
    `Separating and recombining ${sounds.length} source${sounds.length > 1 ? "s" : ""}.`);
  try {
    const data = await api("/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        file_id: state.file.file_id, model: state.model, sounds,
      }),
    });
    renderResult(data);
  } catch (err) {
    toast("Render failed: " + err.message);
  } finally {
    hideOverlay();
  }
}

function renderResult(data) {
  $("beforeAudio").src = data.original_url;
  $("afterAudio").src = data.clean_url;
  $("downloadClean").href = data.clean_url;

  const videoPane = $("videoPane");
  if (data.video_url) {
    $("afterVideo").src = data.video_url;
    $("downloadVideo").href = data.video_url;
    videoPane.classList.remove("hidden");
  } else {
    videoPane.classList.add("hidden");
  }
  $("resultCard").classList.remove("hidden");

  renderSources(data.stems);
  $("resultCard").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderSources(stems) {
  const list = $("stemList");
  list.innerHTML = "";
  (stems || []).forEach((s) => {
    const row = document.createElement("div");
    row.className = "stem-item";
    row.innerHTML = `<div class="stem-name">${prettify(s.name)}</div>`;
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.src = s.url;
    row.appendChild(audio);
    list.appendChild(row);
  });

  // Detection-score panel for inspection.
  $("scoreMode").textContent = state.hasDetectionHead
    ? "(detection-head probability)"
    : "(mask-energy heuristic)";
  const sl = $("scoreList");
  sl.innerHTML = "";
  const max = state.ranked.length ? Math.max(...state.ranked.map((r) => r.score)) : 1;
  state.ranked.forEach((r) => {
    const pct = max > 0 ? Math.round((r.score / max) * 100) : 0;
    const row = document.createElement("div");
    row.className = "score-row";
    row.innerHTML =
      `<span class="nm">${prettify(r.name)}</span>` +
      `<span class="track"><span style="width:${pct}%"></span></span>` +
      `<span class="vv">${r.score}</span>`;
    sl.appendChild(row);
  });

  $("sourcesCard").classList.remove("hidden");
}

// ------------------------------------------------------------------ //
// Wiring
// ------------------------------------------------------------------ //
function init() {
  loadModels();

  $("modelSelect").addEventListener("change", (e) => setModel(e.target.value));

  const dz = $("dropzone");
  const input = $("fileInput");
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  input.addEventListener("change", (e) => handleFile(e.target.files[0]));

  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  $("clearFile").addEventListener("click", clearFile);
  $("analyzeBtn").addEventListener("click", analyze);
  $("processBtn").addEventListener("click", process);

  document.querySelectorAll("[data-bulk]").forEach((btn) =>
    btn.addEventListener("click", () => setAllActions(btn.dataset.bulk)));
}

document.addEventListener("DOMContentLoaded", init);
