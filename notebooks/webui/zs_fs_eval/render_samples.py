"""
render_samples.py - render selected model-generated widgets as HTML for
visual inspection.

For each question listed in QUESTION_IDS, picks one output per (type, shots)
condition (at seed_mode=SEED_MODE, run_idx=RUN_IDX) and renders it via the
existing widget_html renderer from `notebooks/webui/`. Each question gets a
2x2 comparison page (t1/zs | t1/fs | t2/zs | t2/fs), plus a top-level index.

Outputs:
  outputs/renders/{qid}.html   - one side-by-side page per question
  outputs/renders/index.html   - links to all of them

Just open `outputs/renders/index.html` in a browser.

Edit QUESTION_IDS / SEED_MODE / RUN_IDX at the top to render different samples.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent

# pull in the existing renderer from notebooks/webui/
sys.path.insert(0, str(HERE.parent))
from widget_html import render_widget, _CSS  # noqa: E402
from widget_spec import Widget, WidgetType, WidgetHint  # noqa: E402

sys.path.insert(0, str(HERE))
import metrics as M  # noqa: E402


# ---- config -------------------------------------------------------------
QUESTION_IDS = ["q01", "q08", "q11"]  # which questions to render
SEED_MODE = "fixed"                   # which seed mode to pick samples from
RUN_IDX = 0                           # which run index inside the seed mode

CONDITIONS = [("t1", "zs"), ("t1", "fs"), ("t2", "zs"), ("t2", "fs")]

RAW_PATH = HERE / "outputs" / "raw_outputs.jsonl"
QUESTIONS_PATH = HERE / "eval_set" / "questions.jsonl"
RENDERS_DIR = HERE / "outputs" / "renders"


# ---- generation-format dict -> Widget dataclass ------------------------
def dict_to_widget(d):
    return Widget(
        type=WidgetType(d.get("type", "block")),
        hints={WidgetHint(h) for h in (d.get("hints") or [])},
        name=(d.get("name") or ""),
        children=[dict_to_widget(c) for c in (d.get("children") or [])],
    )


# ---- HTML assembly -----------------------------------------------------
EXTRA_CSS = """
body { background: #f6f6f8; }
.header { max-width: 1400px; margin: 0 auto 20px; padding: 12px 16px;
          background: #fff; border: 1px solid #ddd; border-radius: 6px; }
.header h1 { margin: 0 0 6px 0; font-size: 18px; }
.header .q { font-size: 14px; color: #444; }
.compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
                max-width: 1400px; margin: 0 auto; }
.pane { background: #fff; border: 1px solid #ddd; border-radius: 6px;
        padding: 12px; min-height: 200px; }
.pane h2 { font-family: ui-monospace, monospace; font-size: 13px;
           margin: 0 0 4px 0; padding-bottom: 6px;
           border-bottom: 1px solid #eee; color: #333; letter-spacing: 0.3px; }
.pane .meta { font-size: 11px; color: #888; margin: 0 0 10px 0;
              font-family: ui-monospace, monospace; }
.pane .error { background: #fde8e8; color: #8a1c1c; padding: 10px;
               border-radius: 4px; font-family: ui-monospace, monospace;
               font-size: 12px; white-space: pre-wrap; }
.pane .widget-area { background: #fafafa; padding: 8px; border-radius: 4px; }
.back { display: inline-block; margin-top: 8px; font-size: 13px; color: #225; }
.index-list { max-width: 700px; margin: 0 auto; list-style: none; padding: 0; }
.index-list li { padding: 10px 14px; background: #fff; border: 1px solid #ddd;
                 border-radius: 6px; margin-bottom: 8px; }
.index-list li a { text-decoration: none; color: #225; font-weight: 600; }
.index-list li .q { display: block; color: #555; font-size: 13px;
                    margin-top: 4px; font-weight: 400; }
"""


def render_pane(typ, shots, raw_row):
    head = f'<h2>{typ}/{shots}</h2>'
    if raw_row is None:
        return ('<div class="pane">' + head +
                '<div class="error">no output found for this condition</div></div>')
    w, info = M.to_widget(raw_row.get("output", ""))
    meta = f'<div class="meta">status={info["status"]} | parse={info["parse_strategy"]}</div>'
    if info["status"] == "hard_error":
        msg = "; ".join(info["hard_error_msgs"][:2]) or "(no msg)"
        return ('<div class="pane">' + head + meta +
                f'<div class="error">HARD ERROR: {msg}</div></div>')
    try:
        widget = dict_to_widget(w)
        body = render_widget(widget)
    except Exception as e:  # noqa: BLE001
        return ('<div class="pane">' + head + meta +
                f'<div class="error">render error: {e}</div></div>')
    return ('<div class="pane">' + head + meta +
            f'<div class="widget-area">{body}</div></div>')


def render_comparison(qid, question, picks):
    panes = "\n".join(render_pane(t, s, picks.get((t, s))) for t, s in CONDITIONS)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{qid}: widget renders</title>
<style>{_CSS}{EXTRA_CSS}</style></head>
<body>
<div class="header">
  <h1>{qid}: side-by-side widget renders ({SEED_MODE}, run {RUN_IDX})</h1>
  <div class="q">{question}</div>
  <a class="back" href="index.html">&larr; back to index</a>
</div>
<div class="compare-grid">
{panes}
</div></body></html>"""


def render_index(items):
    li = "\n".join(
        f'<li><a href="{qid}.html">{qid}</a><span class="q">{q}</span></li>'
        for qid, q in items)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Widget render index</title>
<style>{_CSS}{EXTRA_CSS}</style></head>
<body>
<div class="header">
  <h1>Rendered widget samples</h1>
  <div class="q">seed_mode={SEED_MODE}, run_idx={RUN_IDX}</div>
</div>
<ul class="index-list">{li}</ul>
</body></html>"""


# ---- main --------------------------------------------------------------
def main():
    questions = {}
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                questions[r["id"]] = r["question"]

    by_key = {}
    with open(RAW_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_key[(r["question_id"], r["type"], r["shots"],
                    r["seed_mode"], r["run_idx"])] = r

    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for qid in QUESTION_IDS:
        if qid not in questions:
            print(f"  skip {qid}: not in questions.jsonl")
            continue
        q = questions[qid]
        picks = {(t, s): by_key.get((qid, t, s, SEED_MODE, RUN_IDX))
                 for t, s in CONDITIONS}
        html = render_comparison(qid, q, picks)
        (RENDERS_DIR / f"{qid}.html").write_text(html, encoding="utf-8")
        items.append((qid, q))
        found = sum(1 for v in picks.values() if v is not None)
        print(f"  wrote {qid}.html ({found}/{len(CONDITIONS)} conditions found)")

    (RENDERS_DIR / "index.html").write_text(render_index(items), encoding="utf-8")
    print(f"\nopen in browser:")
    print(f"  {RENDERS_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
