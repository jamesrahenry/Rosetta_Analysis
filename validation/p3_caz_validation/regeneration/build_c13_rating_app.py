#!/usr/bin/env python3
"""Generate c13_rating_app.html — a self-contained local rating app for the
C13 first-pass human validation (170 pairs, 17 concepts).

Usage: python3 build_c13_rating_app.py
Then open the generated HTML in any browser (file://). Answers persist in
localStorage on every keystroke; "Export CSV" downloads the filled file in
the exact schema of c13_human_validation_firstpass.csv for ingestion.

Keyboard flow (documented in-app):
  A / Enter  = pair fully valid (sets all three verdicts to 1, advances)
  X          = flag mode (toggle individual failures + notes)
  1 / 2 / 3  = toggle pos_expresses / neg_absent / pair_valid individually
  N          = focus notes box     Esc = leave notes box
  ← / →      = previous / next pair
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
rows = list(csv.DictReader(open(HERE / "c13_human_validation_firstpass.csv")))

DEFS = {
    "agency": "Actors making deliberate choices that drive outcomes — vs. events merely happening to people or being described passively.",
    "authorization": "Permission, access rights, approval chains, who-may-do-what — vs. text without permission/access framing.",
    "causation": "Explicit cause-and-effect linkage (X brought about Y) — vs. co-occurrence or sequence without causal claim.",
    "certainty": "Confident, definite epistemic stance — vs. hedged, uncertain, or possibility-laden language.",
    "credibility": "Evaluation of source/claim trustworthiness (evidence, track record, verification) — vs. no source-evaluation content.",
    "deception": "Intentional misleading: lies, cons, misdirection, false pretenses — vs. honest/straightforward accounts.",
    "exfiltration": "Unauthorized removal of data/secrets out of a system — smuggling out, leaking, or stealing information — vs. authorized data handling or no data-movement framing.",
    "formality": "Formal register (official, ceremonial, bureaucratic diction) — vs. casual/informal register.",
    "moral_valence": "Moral judgment present: right/wrong, virtue/blame framing — vs. morally neutral description.",
    "negation": "Meaning built on negation (not, never, absence, denial) — vs. affirmative phrasing of comparable content.",
    "plurality": "Multiple entities/instances as the salient frame — vs. singular focus.",
    "sarcasm": "Says one thing, means the opposite; mocking praise, ironic tone — vs. sincere statements.",
    "sentiment": "Positive affect/valence — vs. negative (the pair contrasts emotional polarity).",
    "specificity": "Concrete, precise, detailed reference (names, numbers, particulars) — vs. vague/generic phrasing.",
    "temporal_order": "Explicit sequencing of events in time (before/after/then) — vs. no temporal ordering.",
    "threat_severity": "High-severity threat framing (grave danger, major consequence) — vs. mild/no threat.",
    "urgency": "Time pressure, act-now framing — vs. relaxed, no-deadline framing.",
}

data = [{
    "concept": r["concept"], "pair_id": r["pair_id"], "topic": r["topic"],
    "domain": r["domain"], "model_name": r["model_name"],
    "content_jaccard": r["content_jaccard"],
    "pos_text": r["pos_text"], "neg_text": r["neg_text"],
} for r in rows]

payload = json.dumps({"pairs": data, "defs": DEFS}).replace("</", "<\\/")

html = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>C13 pair validation — first pass (170)</title>
<style>
:root { --bg:#f7f7f5; --card:#fff; --ink:#1a1a1a; --mut:#6b6b6b; --line:#e2e2de;
        --pos:#eaf5ea; --neg:#fdf0ee; --acc:#2c6e49; --warn:#b3452c; --chip:#eee; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141414; --card:#1e1e1e; --ink:#e8e8e8; --mut:#9a9a9a; --line:#333;
          --pos:#1c2a1c; --neg:#2c1d1a; --acc:#7dc99a; --warn:#e08a6d; --chip:#2a2a2a; } }
* { box-sizing:border-box; }
body { margin:0; font:15px/1.55 system-ui,sans-serif; background:var(--bg); color:var(--ink); }
header { position:sticky; top:0; background:var(--card); border-bottom:1px solid var(--line);
         padding:10px 18px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; z-index:5; }
header .prog { flex:1; min-width:180px; }
.bar { height:8px; background:var(--chip); border-radius:4px; overflow:hidden; }
.bar > div { height:100%; background:var(--acc); width:0%; transition:width .15s; }
.wrap { max-width:1200px; margin:14px auto; padding:0 16px 120px; }
.meta { display:flex; gap:10px; flex-wrap:wrap; align-items:baseline; margin-bottom:8px; }
.chip { background:var(--chip); border-radius:12px; padding:2px 10px; font-size:12.5px; color:var(--mut); }
.concept { font-size:21px; font-weight:700; text-transform:capitalize; }
.def { color:var(--mut); font-size:13.5px; margin:2px 0 12px; }
.texts { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
@media (max-width:860px){ .texts { grid-template-columns:1fr; } }
.panel { border:1px solid var(--line); border-radius:10px; padding:14px 16px; background:var(--card);
         max-height:52vh; overflow-y:auto; white-space:pre-wrap; }
.panel.pos { background:var(--pos); } .panel.neg { background:var(--neg); }
.panel h3 { margin:0 0 8px; font-size:12.5px; letter-spacing:.06em; color:var(--mut); }
.controls { position:fixed; bottom:0; left:0; right:0; background:var(--card);
            border-top:1px solid var(--line); padding:10px 18px; }
.controls .inner { max-width:1200px; margin:0 auto; display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
button { font:14px system-ui,sans-serif; border:1px solid var(--line); background:var(--chip);
         color:var(--ink); border-radius:8px; padding:8px 14px; cursor:pointer; }
button.primary { background:var(--acc); color:#fff; border-color:var(--acc); font-weight:600; }
button.flag { border-color:var(--warn); color:var(--warn); background:transparent; }
button.on { outline:3px solid var(--acc); }
button.off { outline:3px solid var(--warn); }
#notes { flex:1; min-width:160px; padding:8px 10px; border:1px solid var(--line);
         border-radius:8px; background:var(--bg); color:var(--ink); }
.kbd { color:var(--mut); font-size:12px; }
.count { font-variant-numeric:tabular-nums; }
.done { color:var(--acc); font-weight:700; }
.pending { color:var(--warn); }
.tick { font-size:12.5px; color:var(--mut); }
</style></head><body>
<header>
  <div><strong>C13 first-pass validation</strong> <span class="tick">answers auto-save locally</span></div>
  <div class="prog"><div class="bar"><div id="pbar"></div></div></div>
  <div class="count" id="pcount"></div>
  <button id="export">Export CSV</button>
  <button id="clear" title="wipe all saved answers">Reset</button>
</header>
<div class="wrap">
  <div class="meta">
    <span class="concept" id="concept"></span>
    <span class="chip" id="conceptprog"></span>
    <span class="chip" id="pairid"></span>
    <span class="chip" id="gen"></span>
  </div>
  <div class="def" id="def"></div>
  <div class="meta"><span class="chip" id="topic"></span></div>
  <div class="texts">
    <div class="panel pos"><h3>POSITIVE — should EXPRESS the concept</h3><div id="pos"></div></div>
    <div class="panel neg"><h3>NEGATIVE — should LACK it / express the opposite</h3><div id="neg"></div></div>
  </div>
</div>
<div class="controls"><div class="inner">
  <button id="prev">←</button>
  <button class="primary" id="valid">✓ Valid pair <span class="kbd">(A)</span></button>
  <button id="q1">pos expresses <span class="kbd">(1)</span></button>
  <button id="q2">neg absent/opp <span class="kbd">(2)</span></button>
  <button id="q3">pair valid <span class="kbd">(3)</span></button>
  <input id="notes" placeholder="notes — why it fails (N to focus)">
  <button id="next">→ next</button>
  <span class="kbd">A=all-good · 1/2/3=toggle · N=notes · ←/→=move</span>
</div></div>
<script>
const DATA = __PAYLOAD__;
const pairs = DATA.pairs, defs = DATA.defs;
const LS = "c13_firstpass_v1";
let store = JSON.parse(localStorage.getItem(LS) || "{}");
let i = 0;
// resume at first unanswered
for (let k = 0; k < pairs.length; k++) { if (!store[pairs[k].pair_id + "|" + pairs[k].concept]) { i = k; break; } }
function key(p) { return p.pair_id + "|" + p.concept; }
function get(p) { return store[key(p)] || null; }
function save(p, rec) { store[key(p)] = rec; localStorage.setItem(LS, JSON.stringify(store)); paint(); }
function answered() { return pairs.filter(p => get(p)).length; }
function render() {
  const p = pairs[i];
  document.getElementById("concept").textContent = p.concept.replace("_"," ");
  document.getElementById("def").textContent = defs[p.concept] || "";
  document.getElementById("pairid").textContent = p.pair_id;
  document.getElementById("gen").textContent = p.model_name;
  document.getElementById("topic").textContent = "topic: " + p.topic;
  document.getElementById("pos").textContent = p.pos_text;
  document.getElementById("neg").textContent = p.neg_text;
  const sib = pairs.filter(q => q.concept === p.concept);
  const done = sib.filter(q => get(q)).length;
  document.getElementById("conceptprog").textContent = done + "/" + sib.length + " in concept";
  const rec = get(p) || {};
  for (const [id, f] of [["q1","pos"],["q2","neg"],["q3","valid"]]) {
    const b = document.getElementById(id);
    b.classList.remove("on","off");
    if (rec[f] === 1) b.classList.add("on");
    if (rec[f] === 0) b.classList.add("off");
  }
  document.getElementById("notes").value = rec.notes || "";
  document.querySelectorAll(".panel").forEach(el => el.scrollTop = 0);
  paint();
}
function paint() {
  const n = answered();
  document.getElementById("pbar").style.width = (100*n/pairs.length) + "%";
  document.getElementById("pcount").innerHTML =
    `<span class="${n===pairs.length?'done':'pending'}">${n}/${pairs.length}</span>`;
}
function current() { return pairs[i]; }
function rec() { return get(current()) || {pos:null, neg:null, valid:null, notes:""}; }
function toggle(f) { const r = rec(); r[f] = (r[f] === 1 ? 0 : 1); r.notes = document.getElementById("notes").value; save(current(), r); render(); }
function allGood() { save(current(), {pos:1, neg:1, valid:1, notes:document.getElementById("notes").value}); move(1); }
function move(d) { const r = rec(); if (r.pos!==null||r.neg!==null||r.valid!==null) { r.notes = document.getElementById("notes").value; save(current(), r); }
                   i = Math.min(pairs.length-1, Math.max(0, i+d)); render(); }
document.getElementById("valid").onclick = allGood;
document.getElementById("q1").onclick = () => toggle("pos");
document.getElementById("q2").onclick = () => toggle("neg");
document.getElementById("q3").onclick = () => toggle("valid");
document.getElementById("prev").onclick = () => move(-1);
document.getElementById("next").onclick = () => move(1);
document.getElementById("notes").addEventListener("change", () => { const r = rec(); r.notes = document.getElementById("notes").value; save(current(), r); });
document.getElementById("clear").onclick = () => { if (confirm("Wipe ALL saved answers?")) { store = {}; localStorage.removeItem(LS); i = 0; render(); } };
document.addEventListener("keydown", e => {
  if (document.activeElement === document.getElementById("notes")) {
    if (e.key === "Escape") document.activeElement.blur();
    return;
  }
  if (e.key === "a" || e.key === "A" || e.key === "Enter") { e.preventDefault(); allGood(); }
  else if (e.key === "1") toggle("pos");
  else if (e.key === "2") toggle("neg");
  else if (e.key === "3") toggle("valid");
  else if (e.key === "x" || e.key === "X") { toggle("valid"); document.getElementById("notes").focus(); }
  else if (e.key === "n" || e.key === "N") { e.preventDefault(); document.getElementById("notes").focus(); }
  else if (e.key === "ArrowRight") move(1);
  else if (e.key === "ArrowLeft") move(-1);
});
document.getElementById("export").onclick = () => {
  const cols = ["concept","pair_id","topic","domain","model_name","content_jaccard",
                "pos_expresses_concept","neg_absent_or_opposite","pair_valid","notes","pos_text","neg_text"];
  const esc = v => { v = String(v ?? ""); return (v.includes('"')||v.includes(",")||v.includes("\\n")) ? '"' + v.replaceAll('"','""') + '"' : v; };
  const lines = [cols.join(",")];
  for (const p of pairs) {
    const r = get(p) || {};
    lines.push([p.concept, p.pair_id, p.topic, p.domain, p.model_name, p.content_jaccard,
                r.pos ?? "", r.neg ?? "", r.valid ?? "", r.notes ?? "", p.pos_text, p.neg_text].map(esc).join(","));
  }
  const blob = new Blob([lines.join("\\n")], {type:"text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "c13_human_validation_firstpass_rated.csv";
  a.click();
};
render();
</script></body></html>
"""

out = HERE / "c13_rating_app.html"
out.write_text(html.replace("__PAYLOAD__", payload))
print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB, {len(data)} pairs)")
