"""
render_few_shot.py - render eval_set/few_shot_examples.jsonl as one HTML page
for visual inspection.

For each few-shot example, shows:
  - the user question
  - the plain-text answer (Type 2 stage-1 input)
  - the hand-authored widget tree (Type 1 / Type 2 gold), rendered
  - the AXtree converted via axtree_to_widget, rendered side-by-side

Side-by-side rendering lets you verify that the widget and AXtree forms
produce structurally equivalent UIs. The raw JSON for each form is included
in a collapsed <details> block so you can spot-check the source.

Output: outputs/renders/few_shot.html  (open in any browser)
"""
from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))   # for widget_html, widget_spec
sys.path.insert(0, str(HERE))          # for axtree_to_widget
from widget_html import render_widget, _CSS  # noqa: E402
from widget_spec import Widget, WidgetType, WidgetHint  # noqa: E402
from axtree_to_widget import axtree_to_widget  # noqa: E402

FS_PATH = HERE / "eval_set" / "few_shot_examples.jsonl"
OUT_PATH = HERE / "outputs" / "renders" / "few_shot.html"


def dict_to_widget(d):
    """generation-format dict -> Widget dataclass (so we can call render_widget)."""
    return Widget(
        type=WidgetType(d.get("type", "block")),
        hints={WidgetHint(h) for h in (d.get("hints") or [])},
        name=(d.get("name") or ""),
        children=[dict_to_widget(c) for c in (d.get("children") or [])],
    )


EXTRA_CSS = """
body { background: #f4f4f6; }
.header { max-width: 1500px; margin: 0 auto 20px; padding: 16px 20px;
          background: #fff; border: 1px solid #ddd; border-radius: 6px; }
.header h1 { margin: 0 0 4px 0; font-size: 20px; }
.header .sub { font-size: 13px; color: #666; }
.toc { max-width: 1500px; margin: 0 auto 16px; padding: 8px 16px;
       background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;
       font-size: 13px; }
.toc a { margin-right: 16px; color: #225; text-decoration: none; }
.example { max-width: 1500px; margin: 0 auto 28px; padding: 18px 20px;
           background: #fff; border: 1px solid #ddd; border-radius: 8px; }
.example h2 { margin: 0 0 12px 0; font-size: 18px;
              padding-bottom: 8px; border-bottom: 1px solid #eee; }
.example h2 .id { font-family: ui-monospace, monospace; color: #888;
                  margin-right: 10px; font-size: 14px; }
.example h3 { margin: 14px 0 6px 0; font-size: 13px; color: #555;
              font-family: ui-monospace, monospace; letter-spacing: 0.3px;
              text-transform: uppercase; font-weight: 600; }
.plain-text { background: #fafafa; border-left: 3px solid #ccc;
              padding: 10px 14px; margin: 0; font-size: 13px;
              line-height: 1.55; color: #333; white-space: pre-wrap; }
.render-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
               margin-top: 8px; }
.render-panel { background: #fbfbfd; border: 1px solid #e6e6ec;
                border-radius: 6px; padding: 10px; }
.render-panel h3 { margin-top: 4px; }
.render-panel .badge { display: inline-block; margin-left: 8px;
                       padding: 1px 6px; border-radius: 3px;
                       font-size: 10px; background: #eef0f5;
                       color: #555; font-weight: 400;
                       text-transform: none; letter-spacing: 0; }
.widget-area { background: #fff; padding: 8px; border-radius: 4px;
               border: 1px solid #eee; min-height: 80px; }
details { margin-top: 10px; font-size: 12px; }
details summary { cursor: pointer; color: #557; user-select: none; }
details pre { background: #f6f7fa; padding: 8px; border-radius: 4px;
              overflow-x: auto; font-size: 11px; line-height: 1.35;
              max-height: 320px; overflow-y: auto; }
.stats { font-size: 11px; color: #888; font-family: ui-monospace, monospace;
         margin: 4px 0 0 0; }
"""


def count_nodes(d):
    if not isinstance(d, dict):
        return 0
    return 1 + sum(count_nodes(c) for c in (d.get("children") or []))


def max_depth(d, depth=0):
    if not isinstance(d, dict):
        return depth
    kids = d.get("children") or []
    if not kids:
        return depth
    return max(max_depth(c, depth + 1) for c in kids)


def render_example(ex):
    widget = ex["widget"]
    axtree = ex["axtree"]
    ax_as_widget = axtree_to_widget(axtree)

    widget_html = render_widget(dict_to_widget(widget))
    ax_html = render_widget(dict_to_widget(ax_as_widget))

    widget_stats = f"{count_nodes(widget)} nodes, depth {max_depth(widget)}"
    ax_stats = (f"{count_nodes(axtree)} axtree nodes → "
                f"{count_nodes(ax_as_widget)} widget nodes, depth {max_depth(ax_as_widget)}")

    widget_json = escape(json.dumps(widget, ensure_ascii=False, indent=2))
    axtree_json = escape(json.dumps(axtree, ensure_ascii=False, indent=2))
    converted_json = escape(json.dumps(ax_as_widget, ensure_ascii=False, indent=2))

    return f"""
<section class="example" id="{ex['id']}">
  <h2><span class="id">{ex['id']}</span>{escape(ex['question'])}</h2>

  <h3>plain-text answer  <span class="badge">Type 2 stage-1 input</span></h3>
  <p class="plain-text">{escape(ex['text'])}</p>

  <div class="render-grid">

    <div class="render-panel">
      <h3>hand-authored widget  <span class="badge">T1 / T2 gold</span></h3>
      <p class="stats">{widget_stats}</p>
      <div class="widget-area">{widget_html}</div>
      <details><summary>raw widget JSON</summary><pre>{widget_json}</pre></details>
    </div>

    <div class="render-panel">
      <h3>AXtree → widget  <span class="badge">T3 gold, post-conversion</span></h3>
      <p class="stats">{ax_stats}</p>
      <div class="widget-area">{ax_html}</div>
      <details><summary>raw AXtree JSON (source)</summary><pre>{axtree_json}</pre></details>
      <details><summary>widget after conversion</summary><pre>{converted_json}</pre></details>
    </div>

  </div>
</section>
"""


def main():
    examples = [json.loads(l) for l in open(FS_PATH, encoding="utf-8") if l.strip()]
    toc = " &middot; ".join(
        f'<a href="#{ex["id"]}">{ex["id"]}: {escape(ex["question"][:55])}{"..." if len(ex["question"]) > 55 else ""}</a>'
        for ex in examples
    )
    sections = "\n".join(render_example(ex) for ex in examples)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Few-shot examples — visual inspection</title>
<style>{_CSS}{EXTRA_CSS}</style>
</head><body>
<div class="header">
  <h1>Few-shot examples — visual inspection</h1>
  <div class="sub">{len(examples)} examples from <code>eval_set/few_shot_examples.jsonl</code> — widget and AXtree-converted forms shown side-by-side</div>
</div>
<div class="toc">{toc}</div>
{sections}
</body></html>"""
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_PATH}  ({len(html):,} chars, {len(examples)} examples)")


if __name__ == "__main__":
    main()
