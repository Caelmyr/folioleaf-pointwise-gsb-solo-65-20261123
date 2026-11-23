import {q, qa, e, btn, badge, empty, option, img, textField, selectField, stat, table, api, request, n, percent, date} from "./widgets.js";

let editor = null;

const MODE_LANDMARK = "landmark";
const MODE_EXCLUDE = "exclude";

function photoChoices(tile, selected) {
  const ready = tile.photos.filter(p => p.registered_url);
  return ready.map(p => option(p.id, `${p.role_label} · ${p.name}`, selected)).join("");
}

function markerSvg(pairs, circles, width = 1000, height = 1000) {
  const dots = pairs.map((pair, index) => {
    const n = index + 1;
    const one = pair.first
      ? `<circle class="rp-one" cx="${pair.first[0] * width}" cy="${pair.first[1] * height}" r="11"/>
         <text class="rp-label" x="${pair.first[0] * width + 16}" y="${pair.first[1] * height + 6}">${n}</text>` : "";
    const two = pair.second
      ? `<circle class="rp-two" cx="${pair.second[0] * width}" cy="${pair.second[1] * height}" r="11"/>
         <text class="rp-label" x="${pair.second[0] * width + 16}" y="${pair.second[1] * height + 6}">${n}</text>` : "";
    return one + two;
  }).join("");
  const rings = circles.map((c, i) => {
    const cx = c.center[0] * width, cy = c.center[1] * height, r = c.radius * width;
    return `<circle class="rp-exclude" cx="${cx}" cy="${cy}" r="${r}"/><line class="rp-exclude-x" x1="${cx-r}" y1="${cy-r}" x2="${cx+r}" y2="${cy+r}"/>
      <text class="rp-label ex" x="${cx + r + 6}" y="${cy + 4}">E${i + 1} · ${c.day === "first" ? "day 1" : "day 2"}</text>`;
  }).join("");
  return `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" class="rp-svg" aria-hidden="true">${dots}${rings}</svg>`;
}

function stageHtml(side, photo, state) {
  const circles = state.circles.filter(c => c.day === side);
  return `<div class="rp-stage" data-side="${side}">
    <div class="rp-head"><strong>${side === "first" ? "Day 1 (first visit)" : "Day 2 (later visit)"}</strong>
      <span class="soft">${e(photo ? photo.role_label : "")}</span></div>
    <div class="capture-drawing" data-rp-stage="${side}">
      ${photo ? img(photo.registered_url, `Registered ${side} plane`) : empty("Choose a registered capture.")}
      ${photo ? markerSvg(state.pairs, circles) : ""}
    </div></div>`;
}

function editorBody(state, tile) {
  const first = tile.photos.find(p => p.id === state.firstId);
  const second = tile.photos.find(p => p.id === state.secondId);
  const complete = state.pairs.filter(p => p.first && p.second).length;
  return `
  <div class="form-columns">
    ${textField("Comparison title", "rp-title", state.title, 'required maxlength="90"')}
    ${selectField("Exclusion radius (fraction of plane)", "rp-radius",
      [0.02, 0.03, 0.045, 0.06, 0.09].map(r => option(String(r), `${Math.round(r * 100)}%`, String(state.radius))).join(""))}
    ${selectField("Day 1 capture", "rp-first", photoChoices(tile, state.firstId))}
    ${selectField("Day 2 capture", "rp-second", photoChoices(tile, state.secondId))}
  </div>
  <div class="rp-mode actions">
    ${btn("Mark vein-crossing / notch pairs", "rp-mode-landmark", state.mode === MODE_LANDMARK ? "tiny primary" : "tiny secondary")}
    ${btn("Mark hidden-area exclusion", "rp-mode-exclude", state.mode === MODE_EXCLUDE ? "tiny primary" : "tiny secondary")}
  </div>
  <p class="notice">Mark the same recognizable point on both days with the same number. A point on only one day stays pending and is not saved. Never bridge a broken vein to its nearest segment, and keep two parallel veins as separate numbered points.</p>
  <div class="rp-grid">${stageHtml("first", first, state)}${stageHtml("second", second, state)}</div>
  <div class="rp-summary soft">${complete} complete pair(s) · ${state.pairs.length - complete} pending · ${state.circles.length} exclusion(s)</div>
  <div class="actions">
    ${btn("Undo last point", "rp-undo", "tiny ghost")}
    ${btn("Clear all marks", "rp-clear", "tiny danger")}
    ${btn("Preview alignment", "rp-preview", "tiny secondary")}
  </div>
  <div id="rp-preview"></div>`;
}

function refresh() {
  const body = q("#rp-body");
  if (!body || !editor) return;
  body.innerHTML = editorBody(editor.state, editor.tile);
  bindStageClicks();
}

function bindStageClicks() {
  qa("[data-rp-stage]").forEach(stage => {
    stage.addEventListener("click", event => {
      const rect = stage.getBoundingClientRect();
      const u = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
      const v = Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height));
      const side = stage.dataset.rpStage;
      const point = [Number(u.toFixed(6)), Number(v.toFixed(6))];
      const s = editor.state;
      if (s.mode === MODE_EXCLUDE) {
        s.circles.push({day: side, center: point, radius: Number(q("#rp-radius").value)});
      } else {
        // Fill the most recent pair still missing this side; otherwise open a new one.
        let pending = [...s.pairs].reverse().find(p => !p[side]);
        if (!pending) {
          pending = {first: null, second: null};
          s.pairs.push(pending);
        }
        pending[side] = point;
      }
      refresh();
    });
  });
}

export function openRepeatEditor(modal, tile, onSaved) {
  const registered = tile.photos.filter(p => p.registered_url);
  if (registered.length < 2) throw Error("Register at least two captures before building a repeat visit.");
  const diffuse = registered.find(p => p.role === "diffuse") || registered[0];
  const revisit = registered.find(p => p.role === "revisit") || registered[1];
  editor = {
    tile,
    onSaved,
    state: {
      title: `${tile.name} · repeat visit`,
      firstId: diffuse.id,
      secondId: revisit.id,
      mode: MODE_LANDMARK,
      radius: 0.03,
      pairs: [],
      circles: [],
    },
  };
  modal("Repeat-visit local alignment", `<div id="rp-body"></div>`, "Save comparison", async () => {
    const s = editor.state;
    const pairs = s.pairs.filter(p => p.first && p.second);
    if (pairs.length < 4) throw Error("Mark at least four complete day-1/day-2 landmark pairs.");
    const payload = {
      title: q("#rp-title").value,
      first_photo_id: s.firstId,
      second_photo_id: s.secondId,
      landmarks: pairs.map(p => ({first: p.first, second: p.second})),
      exclusions: s.circles,
    };
    const saved = await api(`/api/specimens/${tile.id}/repeats`, {method: "POST", body: payload});
    await onSaved();
    return saved;
  }, true);
  refresh();
}

export function bindRepeatActions(doAction) {
  doAction["rp-mode-landmark"] = () => { editor.state.mode = MODE_LANDMARK; refresh(); };
  doAction["rp-mode-exclude"] = () => { editor.state.mode = MODE_EXCLUDE; refresh(); };
  doAction["rp-undo"] = () => {
    const s = editor.state;
    if (s.mode === MODE_EXCLUDE) { s.circles.pop(); refresh(); return; }
    const last = s.pairs[s.pairs.length - 1];
    if (!last) return;
    // Remove the most recently placed point of the last pair, then drop empty pairs.
    if (last.second) last.second = null;
    else if (last.first) last.first = null;
    if (!last.first && !last.second) s.pairs.pop();
    refresh();
  };
  doAction["rp-clear"] = () => { editor.state.pairs = []; editor.state.circles = []; refresh(); };
  doAction["rp-preview"] = async () => {
    const s = editor.state;
    const pairs = s.pairs.filter(p => p.first && p.second);
    if (pairs.length < 4) throw Error("Mark at least four complete pairs before previewing.");
    const token = Math.random();
    editor.previewToken = token;
    const payload = {
      title: q("#rp-title").value || "preview",
      first_photo_id: s.firstId,
      second_photo_id: s.secondId,
      landmarks: pairs.map(p => ({first: p.first, second: p.second})),
      exclusions: s.circles,
    };
    const res = await request(`/api/specimens/${editor.tile.id}/repeats/preview`, {method: "POST", body: payload});
    const uri = URL.createObjectURL(await res.blob());
    if (editor.previewToken !== token) { URL.revokeObjectURL(uri); return; }
    const metrics = JSON.parse(res.headers.get("X-FolioLeaf-Repeat"));
    const box = q("#rp-preview");
    if (box.dataset.blob) URL.revokeObjectURL(box.dataset.blob);
    box.dataset.blob = uri;
    box.innerHTML = `<figure class="stage">${img(uri, "Repeat-visit preview")}</figure>
      <div class="metrics">${stat("Strong blade evidence", percent(metrics.evidence_percent.strong))}
      ${stat("Uncertain", percent(metrics.evidence_percent.uncertain))}
      ${stat("Change threshold ΔE", n(metrics.change_threshold_delta_e))}
      ${stat("Changed pixels", n(metrics.changed_pixels, 0))}</div>`;
  };
}

export function wireRepeatInputs() {
  // Capture select/radius changes inside the editor without closing it.
  document.addEventListener("change", event => {
    if (!editor) return;
    const id = event.target.id;
    if (id === "rp-first") editor.state.firstId = event.target.value;
    else if (id === "rp-second") editor.state.secondId = event.target.value;
    else if (id === "rp-radius") editor.state.radius = Number(event.target.value);
    else return;
    refresh();
  });
}

function repeatDetail(record) {
  const base = `/api/repeats/${record.id}`;
  const m = record.metrics;
  const rows = m.components.map(c => [
    c.key,
    c.kind === "lightened" ? "Lightened (e.g. fading mark)" : c.kind === "darkened" ? "Darkened (e.g. new mark)" : "Hue change",
    n(c.pixels, 0), n(c.median_lightness_change), n(c.median_delta_e), c.center.map(v => n(v, 3)).join(", "),
  ]);
  return `<section class="card stack">
    <div class="title-row"><div><h3>${e(record.title)}</h3>
      <small>${e(date(record.created_at))} · ${m.landmark_count} landmark pairs</small></div>
      ${badge("Saved comparison", "ready")}</div>
    <div class="actions">${["overlay", "change", "tradeoff"].map(layer =>
      btn({overlay: "Warp + evidence", change: "Color change", tradeoff: "Trade-off sheet"}[layer],
        "repeat-layer", (window.__repeatLayer || "overlay") === layer ? "tiny primary" : "tiny secondary",
        null, `data-value="${layer}" data-id="${record.id}"`)).join("")}</div>
    <figure class="stage">${img(`${base}/image/${window.__repeatLayer || "overlay"}?v=${record.version}`, "Repeat-visit comparison")}</figure>
    <div class="legend"><span>Green ring = darkened · amber = lightened · purple = hue · grey hatch = uncertain · dark hatch = no evidence</span></div>
    <div class="metrics">
      ${stat("Strong evidence", percent(m.evidence_percent.strong))}
      ${stat("Uncertain", percent(m.evidence_percent.uncertain))}
      ${stat("No evidence", percent(m.evidence_percent.no_evidence))}
      ${stat("Median landmark shift", n(m.median_displacement, 4))}
      ${stat("ΔE threshold", n(m.change_threshold_delta_e))}
      ${stat("Changed area", `${n(m.changed_area)} ${m.area_unit}`)}
    </div>
    <h3>Detected tissue changes (strong-evidence blade only)</h3>
    ${rows.length ? table(["Ref", "Direction", "Pixels", "Median ΔL*", "Median ΔE", "Center (x, y)"], rows) : empty("No color change exceeded the local noise floor inside well-supported blade tissue.")}
    <p class="notice">${e(m.uncertain_note)}</p>
    <details><summary>Global Lab offset and per-landmark shift</summary>
      ${table(["Property", "Value"], [
        ["Global Lab offset", m.global_lab_offset.map(v => n(v, 3)).join(", ")],
        ["Robust ΔE median", n(m.robust_delta_e_median)],
        ["Per-landmark shift", m.landmark_displacement.map(v => n(v, 4)).join(", ")],
      ])}</details>
  </section>`;
}

export function repeatTab(tile, repeats, activeId, doAction) {
  const active = repeats.find(r => r.id === activeId) || repeats[0] || null;
  if (active) window.__repeatLayer = window.__repeatLayer || "overlay";
  const list = repeats.length ? `<div class="divider-list scroll-list">${repeats.map(r => `
    <div class="row"><div class="grow"><strong class="row-title">${e(r.title)}</strong>
      <small>${e(date(r.created_at))} · ${r.metrics.landmark_count} pairs · ${r.metrics.components.length} change(s)</small></div>
      ${btn("Open", "repeat-open", "tiny secondary", null, `data-id="${r.id}"`)}</div>`).join("")}</div>`
    : empty("Build a comparison between two registered visits using matched vein crossings.");
  return `<div class="grid-detail">
    ${active ? repeatDetail(active) : empty("No repeat-visit comparison yet.")}
    <div class="stack">
      <section class="card stack"><h3>Repeat visits</h3>
        <p class="soft">Warp day 2 onto day 1 with local landmark pairs. Parallel veins and broken veins are protected by a fold check, and unsupported tissue is never called a color change.</p>
        ${btn("New repeat comparison", "repeat-new", "primary", "compare")}
      </section>
      <section class="card stack"><h3>Saved comparisons</h3>${list}</section>
      <section class="card stack"><h3>Compare a drawn area</h3>
        <p class="soft">Open a region drawn on day 1 to see the same physical area on both days, with comparable and missing-evidence edges marked.</p>
        ${active ? btn("Compare region area in Geometry", "repeat-area", "secondary", "frame", `data-id="${active.id}"`) : ""}
      </section>
    </div></div>`;
}
