"""
build_rating_page.py - generate an HTML rating page for manual quality eval.

For each condition in CONDITIONS, samples N_PER_CONDITION outputs from
outputs/raw_outputs.jsonl (deterministic via random.seed(SEED_SAMPLE)),
renders each widget via the existing widget_html renderer, and puts them
all on one HTML page in **shuffled** order so the rater is blind to which
condition each card belongs to.

Each card has a 1-5 Likert rating with explicit anchors:
  1 = broken / unusable
  2 = correct content but poorly structured
  3 = ok with some flaws
  4 = good, minor issues
  5 = excellent
Plus an optional notes field. Ratings save to localStorage so the page can be
re-opened. An "Export CSV" button downloads the ratings with full per-card
metadata (condition, question_id, etc.) for downstream analysis.

Output: outputs/renders/rating_page.html
"""
from __future__ import annotations

import json
import random
import sys
from html import escape
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))   # widget_html, widget_spec
sys.path.insert(0, str(HERE))          # metrics, axtree_to_widget
from widget_html import render_widget, _CSS  # noqa: E402
from widget_spec import Widget, WidgetType, WidgetHint  # noqa: E402
import metrics as M  # noqa: E402

# ---- config -------------------------------------------------------------
# (type, shots) tuples — the curated comparison set
CONDITIONS = [
    ("t1", "zs"), ("t1", "fs"),
    ("t2", "zs"), ("t2", "fs"),
    ("t3_real", "zs"), ("t3_real", "fs"),
    ("t4", "zs"), ("t4", "fs"),
]
N_PER_CONDITION = 5            # how many outputs to sample per condition
SEED_MODE_PICK = "fixed"        # which seed_mode to sample from (less noise)
SEED_SAMPLE = 42                # rng seed for reproducible sampling/shuffling

RAW_PATH = HERE / "outputs" / "raw_outputs.jsonl"
QUESTIONS_PATH = HERE / "eval_set" / "questions.jsonl"
OUT_PATH = HERE / "outputs" / "renders" / "rating_page.html"


def dict_to_widget(d):
    """generation-format dict -> Widget dataclass."""
    if not isinstance(d, dict):
        return Widget(type=WidgetType.BLOCK, hints=set(), name="", children=[])
    try:
        wtype = WidgetType(d.get("type", "block"))
    except Exception:
        wtype = WidgetType.BLOCK
    hints = set()
    for h in (d.get("hints") or []):
        try:
            hints.add(WidgetHint(h))
        except Exception:
            pass
    return Widget(
        type=wtype, hints=hints, name=(d.get("name") or ""),
        children=[dict_to_widget(c) for c in (d.get("children") or [])],
    )


def load_questions():
    qs = {}
    for line in open(QUESTIONS_PATH, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            qs[r["id"]] = r["question"]
    return qs


def load_raw_outputs():
    rows = []
    for line in open(RAW_PATH, encoding="utf-8"):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ---- HTML / CSS ---------------------------------------------------------
EXTRA_CSS = """
body { background: #f4f4f6; max-width: 1100px; margin: 0 auto; padding: 16px; }
.header { background: #fff; padding: 16px 20px; border: 1px solid #ddd;
          border-radius: 8px; position: sticky; top: 8px; z-index: 100;
          box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
.header h1 { margin: 0 0 4px 0; font-size: 18px; }
.header .sub { color: #666; font-size: 13px; }
.header .stats { font-family: ui-monospace, monospace; font-size: 12px;
                 color: #444; margin-top: 8px; }
.header button { font-size: 13px; padding: 6px 12px; border-radius: 6px;
                 border: 1px solid #555; background: #fff; cursor: pointer;
                 margin-right: 6px; }
.header button:hover { background: #f0f0f0; }
.header button.primary { background: #3d5a80; color: #fff; border-color: #2a3f5a; }
.header button.primary:hover { background: #2a3f5a; }
.progress-wrap { margin-top: 8px; background: #e8eaef; height: 6px;
                 border-radius: 3px; overflow: hidden; }
.progress-bar { height: 100%; background: #4c9f70; width: 0%;
                transition: width 0.2s; }

.card { margin: 18px 0; padding: 16px 18px; background: #fff;
        border: 1px solid #ddd; border-radius: 8px; }
.card .question { font-size: 15px; font-weight: 600; margin-bottom: 4px; }
.card .meta { color: #999; font-size: 11px; font-family: ui-monospace, monospace;
              margin-bottom: 10px; }
.card .meta.revealed { color: #225; }
.widget-area { background: #fafafa; padding: 10px; border-radius: 6px;
               border: 1px solid #eee; min-height: 60px; }
.rate-row { margin-top: 14px; display: flex; align-items: center; gap: 10px; }
.rate-row .label { font-size: 13px; color: #555; margin-right: 6px; }
.rate-btn { padding: 6px 14px; border: 1px solid #888; border-radius: 6px;
            background: #fff; cursor: pointer; font-size: 14px;
            font-family: ui-monospace, monospace; min-width: 40px; }
.rate-btn:hover { background: #f0f0f0; }
.rate-btn.selected-1 { background: #c14953; color: #fff; border-color: #8a1c1c; }
.rate-btn.selected-2 { background: #d77654; color: #fff; border-color: #a04020; }
.rate-btn.selected-3 { background: #e0a458; color: #fff; border-color: #a06e2e; }
.rate-btn.selected-4 { background: #88b86b; color: #fff; border-color: #4a7a3a; }
.rate-btn.selected-5 { background: #4c9f70; color: #fff; border-color: #1a7a4c; }
.notes-row { margin-top: 8px; }
.notes-row input { width: 100%; padding: 6px 10px; border: 1px solid #ccc;
                   border-radius: 4px; font-size: 12px; }
.error { background: #fde8e8; color: #8a1c1c; padding: 10px;
         border-radius: 4px; font-family: ui-monospace, monospace;
         font-size: 12px; }
"""

PAGE_JS = """
const STORAGE_KEY = 'zsfs_ratings_v1';
function loadState() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); }
  catch(e) { return {}; }
}
function saveState(s) { localStorage.setItem(STORAGE_KEY, JSON.stringify(s)); }
let STATE = loadState();

function rate(cardId, score) {
  STATE[cardId] = STATE[cardId] || {};
  STATE[cardId].rating = score;
  saveState(STATE);
  refresh();
}
function setNotes(cardId, txt) {
  STATE[cardId] = STATE[cardId] || {};
  STATE[cardId].notes = txt;
  saveState(STATE);
  refresh();
}
function refresh() {
  const cards = document.querySelectorAll('.card');
  let done = 0;
  cards.forEach(c => {
    const id = c.dataset.cardId;
    const r = (STATE[id] || {}).rating;
    c.querySelectorAll('.rate-btn').forEach(b => {
      b.className = 'rate-btn';
      if (r && parseInt(b.dataset.score) === r) b.className = `rate-btn selected-${r}`;
    });
    if (r) done++;
  });
  document.getElementById('done-count').textContent = done;
  document.getElementById('total-count').textContent = cards.length;
  document.getElementById('progress-bar').style.width =
    (100 * done / cards.length).toFixed(0) + '%';
}
function toggleReveal() {
  document.querySelectorAll('.meta').forEach(m => m.classList.toggle('revealed'));
  document.querySelectorAll('.meta-condition').forEach(m =>
    m.style.display = m.style.display === 'none' ? 'inline' : 'none');
}
function exportCSV() {
  const rows = [['card_id', 'question_id', 'type', 'shots', 'seed_mode',
                 'run_idx', 'rating', 'notes', 'question']];
  document.querySelectorAll('.card').forEach(c => {
    const id = c.dataset.cardId;
    const meta = JSON.parse(c.dataset.meta);
    const r = STATE[id] || {};
    rows.push([id, meta.qid, meta.type, meta.shots, meta.seed_mode,
               meta.run_idx, r.rating || '', r.notes || '', meta.question]);
  });
  const csv = rows.map(r =>
    r.map(c => '"' + String(c).replace(/"/g, '""') + '"').join(',')).join('\\n');
  const blob = new Blob([csv], {type: 'text/csv;charset=utf-8'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'manual_ratings.csv';
  a.click();
  URL.revokeObjectURL(url);
}
function clearRatings() {
  if (!confirm('Clear all ratings?')) return;
  STATE = {};
  localStorage.removeItem(STORAGE_KEY);
  document.querySelectorAll('.notes-row input').forEach(i => i.value = '');
  refresh();
}
window.addEventListener('DOMContentLoaded', refresh);
"""


def render_card(card_id, question, rendered_widget_html, meta, display_idx, total):
    """One rating card."""
    meta_json = json.dumps(meta)
    safe_q = escape(question)
    return f"""
<div class="card" data-card-id="{escape(card_id)}" data-meta='{escape(meta_json)}'>
  <div class="question">Q: {safe_q}</div>
  <div class="meta">
    <span>Card {display_idx} / {total}</span>
    <span class="meta-condition" style="display:none; margin-left:14px; color:#225;">
      [card_id={escape(card_id)} · type={escape(meta['type'])} / shots={escape(meta['shots'])} /
       seed_mode={escape(meta['seed_mode'])} / run_idx={meta['run_idx']}]
    </span>
  </div>
  <div class="widget-area">{rendered_widget_html}</div>
  <div class="rate-row">
    <span class="label">Rate this output:</span>
    <button class="rate-btn" data-score="1" onclick="rate('{escape(card_id)}', 1)" title="broken / unusable">1</button>
    <button class="rate-btn" data-score="2" onclick="rate('{escape(card_id)}', 2)" title="correct content but poorly structured">2</button>
    <button class="rate-btn" data-score="3" onclick="rate('{escape(card_id)}', 3)" title="ok with some flaws">3</button>
    <button class="rate-btn" data-score="4" onclick="rate('{escape(card_id)}', 4)" title="good, minor issues">4</button>
    <button class="rate-btn" data-score="5" onclick="rate('{escape(card_id)}', 5)" title="excellent">5</button>
  </div>
  <div class="notes-row">
    <input type="text" placeholder="optional notes…"
           oninput="setNotes('{escape(card_id)}', this.value)">
  </div>
</div>
"""


# ---- main ---------------------------------------------------------------
def main():
    rng = random.Random(SEED_SAMPLE)

    questions = load_questions()
    raw = load_raw_outputs()

    # bucket rows by (type, shots) for sampling
    buckets = {}
    for r in raw:
        key = (r.get("type"), r.get("shots"))
        if key not in [(t, s) for t, s in CONDITIONS]:
            continue
        if r.get("seed_mode") != SEED_MODE_PICK:
            continue
        buckets.setdefault(key, []).append(r)

    # sample N per condition (deterministic)
    cards = []
    skipped = []
    for cond in CONDITIONS:
        bucket = buckets.get(cond, [])
        if not bucket:
            skipped.append(cond)
            continue
        rng.shuffle(bucket)
        for r in bucket[:N_PER_CONDITION]:
            qid = r["question_id"]
            question = questions.get(qid, f"(missing question for {qid})")
            widget_dict, info = M.parse_output(r)
            if widget_dict is None:
                rendered = f'<div class="error">hard error: {escape(", ".join(info.get("hard_error_msgs", [])[:2]))}</div>'
            else:
                try:
                    rendered = render_widget(dict_to_widget(widget_dict))
                except Exception as e:
                    rendered = f'<div class="error">render error: {escape(str(e))}</div>'
            card_id = f"{qid}_{r['type']}_{r['shots']}_{r['seed_mode']}_{r['run_idx']}"
            meta = {
                "card_id": card_id, "qid": qid,
                "type": r["type"], "shots": r["shots"],
                "seed_mode": r["seed_mode"], "run_idx": r["run_idx"],
                "question": question,
            }
            cards.append((card_id, question, rendered, meta))

    # shuffle ALL cards together so rater is blind to condition
    rng.shuffle(cards)

    if skipped:
        print(f"  skipped conditions (no data): {skipped}")
    print(f"  sampled {len(cards)} cards across {len(CONDITIONS) - len(skipped)} conditions"
          f" ({N_PER_CONDITION}/condition, seed_mode={SEED_MODE_PICK!r})")

    total = len(cards)
    card_html = "\n".join(
        render_card(cid, q, rendered, meta, idx, total)
        for idx, (cid, q, rendered, meta) in enumerate(cards, start=1))

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Manual rating page</title>
<style>{_CSS}{EXTRA_CSS}</style>
</head><body>
<div class="header">
  <h1>Manual quality rating — widget generation experiment</h1>
  <div class="sub">{len(cards)} samples ({N_PER_CONDITION}/condition × {len(CONDITIONS) - len(skipped)} conditions) at seed_mode={SEED_MODE_PICK!r}. Order is shuffled — you don't see the condition until you click "reveal" at the end.</div>
  <div class="stats">
    rated <span id="done-count">0</span> / <span id="total-count">{len(cards)}</span>
    <div class="progress-wrap"><div class="progress-bar" id="progress-bar"></div></div>
  </div>
  <div style="margin-top:10px;">
    <button class="primary" onclick="exportCSV()">Export ratings (CSV)</button>
    <button onclick="toggleReveal()">Reveal / hide conditions</button>
    <button onclick="clearRatings()">Clear all ratings</button>
  </div>
  <div class="sub" style="margin-top:8px;">
    <b>Rate each output 1-5</b>:
    <b>1</b> = broken / unusable (structural fail, doesn't answer the question, malformed)
    · <b>2</b> = correct content but poorly structured (right info, bad tree layout)
    · <b>3</b> = ok with some flaws (acceptable structure and content, noticeable issues)
    · <b>4</b> = good, minor issues (clean structure, clear content, small nitpicks)
    · <b>5</b> = excellent (well-structured, rich, useful UI)
  </div>
</div>
{card_html}
<script>{PAGE_JS}</script>
</body></html>"""

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"\nwrote {OUT_PATH}  ({len(html):,} chars, {len(cards)} cards)")
    print(f"  open in browser, rate, then click 'Export ratings (CSV)'")


if __name__ == "__main__":
    main()
