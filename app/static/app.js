/* Anuvaad Lab – dashboard + live arena.
   Works in two modes:
   live   – served by app/server.py; translations are produced on demand.
   static – plain files (GitHub Pages / artifact); the arena replays stored outputs. */
const $ = (s, el = document) => el.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const colorOf = (k) => css(`--c-${k}`) || "#888";
const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const FLAG_TEXT = {
  number_mismatch: "number changed", negation_lost: "negation lost", repetition: "repetition",
  length_anomaly: "length anomaly", untranslated: "untranslated", wrong_script: "wrong script", low_fidelity: "low fidelity", empty: "empty",
};
const ARENA_DEFAULT = ["transformer", "mt5", "llama_lora", "gemma_lora"];
let S = null;            // summary.json
let LIVE = false;
const state = { arena: new Set(ARENA_DEFAULT), tempMetric: "chrf", probeCat: "fidelity", shown: 15 };

async function getJSON(url) { const r = await fetch(url); if (!r.ok) throw new Error(r.status); return r.json(); }

async function boot() {
  try { S = await getJSON("/api/summary"); LIVE = true; }
  catch { S = await getJSON("summary.json"); }
  if (LIVE) {
    try { const st = await getJSON("/api/status"); LIVE = st.systems.some((s) => s.available); } catch { LIVE = false; }
  }
  const m = $("#mode");
  m.textContent = LIVE ? "Live models" : "Stored outputs";
  m.classList.toggle("live", LIVE);
  Chart.defaults.font.family = css("--ui");
  Chart.defaults.color = css("--ink-2");
  Chart.defaults.borderColor = css("--line");
  facts(); arena(); board(); charts(); temperature(); responsible(); explorer();
}

const sysList = () => Object.keys(S.systems);
const label = (k) => S.systems[k]?.label ?? k;

/* ------------------------------------------------------------------ hero facts */
function facts() {
  const sys = S.systems;
  const best = Object.keys(sys).reduce((a, b) => (sys[a].chrf > sys[b].chrf ? a : b));
  const d = S.dataset;
  const items = [
    ["Training pairs", `${(d.train.pairs / 1000).toFixed(0)}k <small>of ${(d.raw_counts.train / 1e6).toFixed(2)}M</small>`],
    ["Test sentences", `${d.test.pairs}`],
    ["Best chrF", `${fmt(sys[best].chrf)} <small>${esc(sys[best].label.split(" (")[0])}</small>`],
    ["Systems compared", `${Object.keys(sys).length} <small>4 models, ${Object.keys(sys).length} setups</small>`],
  ];
  $("#facts").innerHTML = items.map(([t, v]) => `<div><dt>${t}</dt><dd>${v}</dd></div>`).join("");
}

/* ------------------------------------------------------------------ arena */
function arena() {
  const keys = ["transformer", "mt5", "llama_zs", "llama_icl", "llama_lora", "gemma_zs", "gemma_icl", "gemma_lora"].filter((k) => S.systems[k]);
  $("#toggles").innerHTML = keys.map((k) => `<label class="toggle ${state.arena.has(k) ? "" : "off"}">
      <input type="checkbox" value="${k}" ${state.arena.has(k) ? "checked" : ""}><i style="background:${colorOf(k)}"></i>${esc(label(k))}</label>`).join("");
  $("#toggles").addEventListener("change", (e) => {
    const k = e.target.value; e.target.checked ? state.arena.add(k) : state.arena.delete(k);
    e.target.closest(".toggle").classList.toggle("off", !e.target.checked);
  });
  const probes = Object.values(S.probes)[0]?.items ?? [];
  const pick = ["g_doctor_F", "g_nurse_amb", "t00", "t05", "t12", "t14"];
  const examples = pick.map((id) => probes.find((p) => p.id === id)).filter(Boolean).map((p) => p.en);
  const chipText = (s) => (s.length > 44 ? s.slice(0, 42) + "…" : s);
  $("#chips").innerHTML = examples.map((e) => `<button class="chip" data-t="${esc(e)}">${esc(chipText(e))}</button>`).join("");
  $("#chips").addEventListener("click", (e) => { if (e.target.dataset.t) { $("#src").value = e.target.dataset.t; run(); } });
  $("#temp").addEventListener("input", (e) => ($("#tval").textContent = Number(e.target.value).toFixed(1)));
  $("#go").addEventListener("click", run);
  $("#src").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run(); });
  if (!LIVE) {
    $("#arena-note").textContent = "This copy replays stored outputs: pick one of the example sentences. Run the local server to translate anything.";
    $("#temp").disabled = true;
    if (examples.length) $("#src").value = examples[0];
  }
  run();
}

function storedLookup(text) {
  const hits = {};
  for (const [k, p] of Object.entries(S.probes)) {
    const it = p.items.find((x) => x.en.trim() === text.trim());
    if (it) hits[k] = { key: k, label: p.label, hyp: it.hi, ms: null, ...it };
  }
  return hits;
}

async function run() {
  const text = $("#src").value.trim();
  if (!text) return;
  const keys = [...state.arena];
  const list = $("#outputs");
  list.innerHTML = keys.map((k) => outRow(k, null)).join("");
  $("#go").disabled = true;
  try {
    if (LIVE) {
      const r = await fetch("/api/translate", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, systems: keys, temperature: Number($("#temp").value) }) });
      if (!r.ok) throw new Error(`Server returned ${r.status}`);
      const { results } = await r.json();
      const maxMs = Math.max(...results.map((x) => x.ms || 0), 1);
      list.innerHTML = results.map((x) => outRow(x.key, x, maxMs)).join("");
    } else {
      const hits = storedLookup(text);
      list.innerHTML = keys.map((k) => outRow(k, hits[k] ?? { error: "No stored output for this sentence. Pick an example above, or run the local server for live translation." })).join("");
    }
  } catch (err) {
    list.innerHTML = keys.map((k) => outRow(k, { error: `Translation failed: ${err.message}. Check that the server is running.` })).join("");
  } finally { $("#go").disabled = false; }
}

function outRow(k, x, maxMs = 1) {
  const s = S.systems[k];
  const head = `<i class="stripe" style="background:${colorOf(k)}"></i>
    <div class="who"><b>${esc(label(k))}</b><span>${esc(s?.family ?? "")}</span></div>`;
  if (!x) return `<li class="out">${head}<div class="text"><div class="skeleton"></div></div><div class="meta"></div></li>`;
  if (x.error) return `<li class="out">${head}<div class="text err">${esc(x.error)}</div><div class="meta"></div></li>`;
  const flags = Object.entries(x.flags || {}).filter(([, v]) => v).map(([f]) => `<span class="badge flag">${FLAG_TEXT[f] ?? f}</span>`);
  const doms = (x.domains || []).map((d) => `<span class="badge domain">${d.replace("_", " ")}</span>`);
  const lat = x.ms != null ? `<div class="lat"><span style="width:${Math.max(2, (100 * x.ms) / maxMs)}px;background:${colorOf(k)}"></span>${fmt(x.ms, 0)} ms</div>` : "";
  return `<li class="out">${head}
    <div class="text" lang="hi">${esc(x.hyp)}</div>
    <div class="meta">${lat}
      <div>Source–output similarity <b>${fmt(x.similarity ?? x.sim, 2)}</b></div>
      <div class="badges"><span class="badge risk-${x.risk}">${x.risk} risk</span>${flags.join("")}${doms.join("")}</div>
    </div></li>`;
}

/* ------------------------------------------------------------------ leaderboard */
const COLS = [
  { k: "label", t: "System" },
  { k: "bleu", t: "BLEU", bar: 1, ci: "bleu_ci" },
  { k: "chrf", t: "chrF", bar: 1, ci: "chrf_ci" },
  { k: "chrfpp", t: "chrF++", d: 1 },
  { k: "ter", t: "TER (lower is better)", d: 1, low: 1 },
  { k: "params", t: "Params", f: (v) => `${fmt(v / 1e6, 0)}M` },
  { k: "trainable_params", t: "Trained", f: (v) => (v ? `${fmt(v / 1e6, v < 1e7 ? 1 : 0)}M` : "none") },
  { k: "train_time_sec", t: "Train (min)", f: (v) => (v ? fmt(v / 60, 1) : "–") },
  { k: "ms_per_sentence", t: "ms / sent", d: 0, low: 1 },
  { k: "hallucination_rate", t: "Flagged %", d: 1, low: 1 },
];
let sortKey = "chrf", sortDir = -1;
function board() {
  const keys = sysList().sort((a, b) => sortDir * ((S.systems[a][sortKey] ?? -1e9) > (S.systems[b][sortKey] ?? -1e9) ? 1 : -1));
  const best = sysList().reduce((a, b) => (S.systems[a].chrf > S.systems[b].chrf ? a : b));
  const max = { bleu: Math.max(...keys.map((k) => S.systems[k].bleu_ci[1])), chrf: Math.max(...keys.map((k) => S.systems[k].chrf_ci[1])) };
  const th = COLS.map((c) => `<th class="${c.k === "label" ? "" : "num"}" data-k="${c.k}" ${c.k === sortKey ? `aria-sort="${sortDir < 0 ? "descending" : "ascending"}"` : ""}>${c.t}</th>`).join("");
  const rows = keys.map((k) => {
    const r = S.systems[k];
    const tds = COLS.map((c) => {
      if (c.k === "label") return `<td><div class="sys"><i style="background:${colorOf(k)}"></i>${esc(r.label)}</div><div class="arch">${esc(r.family)} · ${esc(r.adaptation)}${r.decoding !== "greedy" ? ` · ${r.decoding}` : ""}</div></td>`;
      if (c.bar) {
        const m = max[c.k], [lo, hi] = r[c.ci];
        return `<td class="num"><div class="metric"><b>${fmt(r[c.k], 1)}</b><div class="track">
          <span class="fill" style="width:${(100 * r[c.k]) / m}%;background:${colorOf(k)}"></span>
          <span class="ci" style="left:${(100 * lo) / m}%;width:${(100 * (hi - lo)) / m}%"></span></div></div></td>`;
      }
      return `<td class="num">${c.f ? c.f(r[c.k]) : fmt(r[c.k], c.d)}</td>`;
    }).join("");
    return `<tr class="${k === best ? "best" : ""}">${tds}</tr>`;
  }).join("");
  $("#board").innerHTML = `<thead><tr>${th}</tr></thead><tbody>${rows}</tbody>`;
  $("#board thead").onclick = (e) => {
    const k = e.target.closest("th")?.dataset.k; if (!k) return;
    const low = COLS.find((c) => c.k === k)?.low;
    sortDir = k === sortKey ? -sortDir : (low ? 1 : -1); sortKey = k; board();
  };
  const sig = S.significance;
  if (sig?.best) {
    const others = Object.entries(sig.vs).filter(([, v]) => v.p_chrf < 0.05).length;
    $("#sig-note").textContent = `Paired bootstrap resampling (1,000 samples): ${label(sig.best)} has the highest chrF and is significantly better (p < 0.05) than ${others} of the other ${Object.keys(sig.vs).length} setups.` +
      (S.reference_baseline ? ` For calibration, the same reference-free checks flag ${fmt(S.reference_baseline.hallucination_rate)}% of the human reference translations.` : "");
  }
}

/* ------------------------------------------------------------------ charts */
const logTicks = { callback: (v) => ([1, 10, 100, 1000, 10000].includes(v) ? v.toLocaleString() : "") };
function charts() {
  const keys = sysList();
  const sys = S.systems;
  const base = { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } };
  new Chart($("#c-pareto"), {
    type: "bubble",
    data: { datasets: keys.map((k) => ({ label: sys[k].label, backgroundColor: colorOf(k) + "cc", borderColor: colorOf(k),
      data: [{ x: sys[k].ms_per_sentence, y: sys[k].chrf, r: 5 + 2.2 * Math.sqrt(sys[k].params / 1e7) }] })) },
    options: { ...base, plugins: { legend: { display: true, position: "right", labels: { boxWidth: 10 } },
        tooltip: { callbacks: { label: (c) => `${c.dataset.label}: chrF ${fmt(c.raw.y)}, ${fmt(c.raw.x, 0)} ms` } } },
      scales: { x: { type: "logarithmic", ticks: logTicks, title: { display: true, text: "ms per sentence (log)" } }, y: { title: { display: true, text: "chrF" } } } },
  });
  const uniq = keys.filter((k) => !k.endsWith("_beam"));
  const bar = (id, vals, opts = {}) => new Chart($(id), {
    type: "bar", data: { labels: uniq.map(label), datasets: [{ data: vals, backgroundColor: uniq.map(colorOf), borderRadius: 3 }] },
    options: { ...base, indexAxis: "y", scales: { x: opts.x ?? {}, y: { ticks: { font: { size: 11 } } } } },
  });
  bar("#c-params", uniq.map((k) => sys[k].params / 1e6), { x: { type: "logarithmic", ticks: logTicks } });
  bar("#c-train", uniq.map((k) => (sys[k].train_time_sec || 0) / 60));
  const fk = uniq.filter((k) => !k.endsWith("_icl") && !k.endsWith("_zs"));
  new Chart($("#c-fert"), {
    type: "bar",
    data: { labels: fk.map((k) => label(k).split(" (")[0]), datasets: [
      { label: "English", data: fk.map((k) => sys[k].fertility_en), backgroundColor: css("--line") },
      { label: "Hindi", data: fk.map((k) => sys[k].fertility_hi), backgroundColor: fk.map(colorOf) }] },
    options: { ...base, plugins: { legend: { display: true } } },
  });
  const bands = [0, 20, 40, 60, 80, 101];
  const bl = ["0–20", "20–40", "40–60", "60–80", "80–100"];
  const pk = S.primary.filter((k) => sys[k]);
  new Chart($("#c-dist"), {
    type: "bar",
    data: { labels: bl, datasets: pk.map((k) => ({ label: label(k).split(" (")[0], backgroundColor: colorOf(k), borderRadius: 2,
      data: bl.map((_, i) => (100 * sys[k].sentence_chrf.filter((v) => v >= bands[i] && v < bands[i + 1]).length) / sys[k].sentence_chrf.length) })) },
    options: { ...base, plugins: { legend: { display: true } }, scales: { y: { title: { display: true, text: "% of sentences" } }, x: { title: { display: true, text: "sentence chrF" } } } },
  });
}

/* ------------------------------------------------------------------ temperature */
let tChart;
function temperature() {
  const T = S.temperature;
  const keys = Object.keys(T);
  if (!keys.length) return;
  const metrics = { chrf: "chrF", bleu: "BLEU", hallucination_rate: "Flagged %", self_bleu: "Self-BLEU (lower = more diverse)", distinct_2: "Distinct bigrams %", semantic_fidelity: "Source–output similarity" };
  $("#temp-tabs").innerHTML = Object.entries(metrics).map(([k, t]) => `<button class="tab" role="tab" data-m="${k}" aria-selected="${k === state.tempMetric}">${t}</button>`).join("");
  $("#temp-tabs").onclick = (e) => {
    const m = e.target.dataset.m; if (!m) return; state.tempMetric = m;
    [...$("#temp-tabs").children].forEach((b) => b.setAttribute("aria-selected", b.dataset.m === m)); draw();
  };
  const temps = T[keys[0]].sweep.map((r) => r.temperature);
  function draw() {
    tChart?.destroy();
    tChart = new Chart($("#c-temp"), {
      type: "line",
      data: { labels: temps.map((t) => t.toFixed(1)), datasets: keys.map((k) => ({ label: T[k].label, borderColor: colorOf(k), backgroundColor: colorOf(k),
        data: T[k].sweep.map((r) => r[state.tempMetric]), tension: 0.25, pointRadius: 4 })) },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { boxWidth: 10 } } },
        scales: { x: { title: { display: true, text: "sampling temperature" } }, y: { title: { display: true, text: metrics[state.tempMetric] } } } },
    });
  }
  draw();
  let sentIdx = 0;
  const showSamples = () => {
    const t = temps[Number($("#tslider").value)];
    $("#tsv").textContent = t.toFixed(1);
    const first = T[keys[0]].samples[String(t)] ?? T[keys[0]].samples[t.toFixed(1)];
    if (!first) return;
    const s = first[sentIdx];
    $("#tsamples").innerHTML = `<p class="ts-src">${esc(s.src)}<br><span lang="hi">${esc(s.ref)}</span> (reference)</p>` +
      keys.map((k) => { const smp = (T[k].samples[String(t)] ?? T[k].samples[t.toFixed(1)])[sentIdx];
        return `<div class="ts-row"><i style="background:${colorOf(k)}"></i><div><b>${esc(T[k].label)}</b><p lang="hi">${esc(smp.hyp)}</p></div></div>`; }).join("") +
      `<button class="chip" id="tnext">Next sentence</button>`;
    $("#tnext").onclick = () => { sentIdx = (sentIdx + 1) % first.length; showSamples(); };
  };
  $("#tslider").addEventListener("input", showSamples);
  showSamples();
}

/* ------------------------------------------------------------------ responsible AI */
function responsible() {
  const P = S.probes;
  const keys = Object.keys(P);
  $("#gender").innerHTML = keys.map((k) => {
    const g = P[k].gender;
    const dbar = (v) => { const w = Math.min(50, Math.abs(v) / 2); return `<div class="dbar"><i style="left:${v >= 0 ? 50 : 50 - w}%;width:${w}%;background:${colorOf(k)}"></i></div>`; };
    return `<article class="gcard" style="--c:${colorOf(k)}"><h4>${esc(P[k].label)}</h4><div class="gstat">
      <span>Follows the pronoun</span><b>${fmt(g.accuracy, 0)}%</b>
      <span>ΔG, male minus female accuracy</span><b>${fmt(g.delta_G, 0)}</b>${dbar(g.delta_G)}
      <span>ΔS, stereotypical minus anti-stereotypical</span><b>${fmt(g.delta_S, 0)}</b>${dbar(g.delta_S)}
      <span>No pronoun, chose masculine</span><b>${fmt(g.ambiguous_masc_pct, 0)}%</b>
      <span>Gender not detectable</span><b>${fmt(g.undetermined_pct, 0)}%</b></div></article>`;
  }).join("");

  const amb = P[keys[0]].items.filter((i) => i.category === "gender" && !i.gold);
  const sym = { M: ["♂", "gM", "masculine"], F: ["♀", "gF", "feminine"], "?": ["?", "gU", "not detectable"] };
  const gOf = (k, id) => {
    const it = P[k].items.find((i) => i.id === id);
    const w = (it.hi || "").replace(/[।,.!?]/g, " ").split(/\s+/);
    const F = ["गई", "गयी", "गईं", "आई", "आयी", "बैठी", "चली", "लौटी", "थी", "रही", "हुई", "करती", "सकती", "चुकी", "थकी"];
    const M = ["गया", "गए", "गये", "आया", "आए", "आये", "बैठा", "चला", "चले", "लौटा", "था", "रहा", "हुआ", "करता", "सकता", "चुका", "थका"];
    const f = w.filter((x) => F.includes(x)).length, m = w.filter((x) => M.includes(x)).length;
    return [f > m ? "F" : m > f ? "M" : "?", it.hi];
  };
  $("#amb").innerHTML = `<thead><tr><th>Occupation</th><th>Stereotype</th>${keys.map((k) => `<th>${esc(P[k].label)}</th>`).join("")}</tr></thead><tbody>` +
    amb.map((it) => `<tr><td>${esc(it.occupation)}</td><td class="stereo">${it.stereotype === "M" ? "male" : "female"}</td>${keys.map((k) => {
      const [g, hi] = gOf(k, it.id); const [s, c, t] = sym[g];
      return `<td class="g ${c}" title="${esc(hi)}"><span aria-label="${t}">${s}</span></td>`; }).join("")}</tr>`).join("") + "</tbody>";

  const cats = { fidelity: "Facts and numbers", cultural: "Idioms and culture", sensitive: "Sensitive content", toxic: "Toxic or biased input" };
  $("#probe-tabs").innerHTML = Object.entries(cats).map(([c, t]) => `<button class="tab" role="tab" data-c="${c}" aria-selected="${c === state.probeCat}">${t}</button>`).join("");
  $("#probe-tabs").onclick = (e) => {
    const c = e.target.dataset.c; if (!c) return; state.probeCat = c;
    [...$("#probe-tabs").children].forEach((b) => b.setAttribute("aria-selected", b.dataset.c === c)); drawProbes();
  };
  function drawProbes() {
    const items = P[keys[0]].items.filter((i) => i.category === state.probeCat);
    $("#probes").innerHTML = items.map((it) => `<div class="probe"><p>${esc(it.en)}</p><ul>${keys.map((k) => {
      const x = P[k].items.find((i) => i.id === it.id);
      const flags = Object.entries(x.flags).filter(([, v]) => v).map(([f]) => `<span class="badge flag">${FLAG_TEXT[f]}</span>`).join("");
      return `<li><i style="background:${colorOf(k)};align-self:stretch"></i><span class="lbl">${esc(P[k].label)}</span>
        <span class="hi" lang="hi">${esc(x.hi)}</span><span class="badges">${flags}<small>sim ${fmt(x.sim, 2)}</small></span></li>`; }).join("")}</ul></div>`).join("");
  }
  drawProbes();
}

/* ------------------------------------------------------------------ explorer */
let XR = null;
async function explorer() {
  const keys = S.primary.filter((k) => S.systems[k]);
  const load = (k) => (LIVE ? getJSON(`/api/translations/${k}`) : fetch(`translations/${k}.jsonl`).then((r) => r.text()).then((t) => t.trim().split("\n").map((l) => JSON.parse(l))));
  try {
    const all = await Promise.all(keys.map(load));
    XR = all[0].map((row, i) => ({ id: i, src: row.src, ref: row.ref,
      hyps: Object.fromEntries(keys.map((k, j) => [k, all[j][i]])) }));
    XR.forEach((r) => { const c = keys.map((k) => r.hyps[k].chrf); r.mean = c.reduce((a, b) => a + b) / c.length; r.spread = Math.max(...c) - Math.min(...c);
      r.flagged = keys.some((k) => r.hyps[k].hallucination); });
  } catch { $("#rows").innerHTML = `<p class="note">Test-set translations are not available in this copy.</p>`; $("#more").hidden = true; return; }
  const draw = () => {
    const q = $("#q").value.trim().toLowerCase();
    let rows = XR.filter((r) => (!q || (r.src + r.ref + Object.values(r.hyps).map((h) => h.hyp).join(" ")).toLowerCase().includes(q)) && (!$("#flagged").checked || r.flagged));
    const s = $("#sort").value;
    if (s === "spread") rows = [...rows].sort((a, b) => b.spread - a.spread);
    if (s === "worst") rows = [...rows].sort((a, b) => a.mean - b.mean);
    const tint = (v) => `background:color-mix(in srgb, ${v >= 60 ? css("--leaf") : v >= 35 ? css("--marigold") : css("--vermilion")} 18%, transparent)`;
    $("#rows").innerHTML = rows.slice(0, state.shown).map((r) => `<article class="xrow"><header><p>${esc(r.src)}</p><p class="ref" lang="hi">${esc(r.ref)}</p></header><ul>${
      keys.map((k) => { const h = r.hyps[k]; const fl = Object.entries(h.flags).filter(([, v]) => v).map(([f]) => `<span class="badge flag">${FLAG_TEXT[f]}</span>`).join("");
        return `<li><i style="background:${colorOf(k)};align-self:stretch"></i><span class="lbl">${esc(label(k))}</span><span><span class="hi" lang="hi">${esc(h.hyp)}</span> <span class="badges" style="display:inline-flex">${fl}</span></span><span class="score" style="${tint(h.chrf)}">${fmt(h.chrf, 0)}</span></li>`; }).join("")}</ul></article>`).join("") ||
      `<p class="note">No sentences match. Clear the filter or untick “Only sentences with a guardrail flag”.</p>`;
    $("#more").hidden = rows.length <= state.shown;
  };
  ["#q", "#flagged", "#sort"].forEach((s) => $(s).addEventListener("input", () => { state.shown = 15; draw(); }));
  $("#more").onclick = () => { state.shown += 15; draw(); };
  draw();
}

boot().catch((e) => { document.querySelector("main").insertAdjacentHTML("afterbegin", `<p class="note" style="padding:24px 0">Could not load results (${esc(e.message)}). Run <code>python -m src.report</code> first.</p>`); });
