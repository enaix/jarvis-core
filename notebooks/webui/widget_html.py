"""
Minimal HTML renderer for the internal Widget tree.

Walks a `Widget` produced by `ax_to_widget` and emits a self-contained HTML
document useful for eyeballing the result of the AXTree -> Widget reduction.
Widget type decides the tag, hints decide CSS classes (`vbox`, `hbox`, `main`).
"""

from __future__ import annotations

from html import escape

from widget_spec import Widget, WidgetType


_CSS = """
body { font-family: sans-serif; margin: 16px; line-height: 1.35; background: #fff; }
.page { max-width: 100%; margin: 0 auto; display: flex; justify-content: center; }
.widget { border: 1px solid #ccc; border-radius: 4px; padding: 6px; margin: 4px;
          box-sizing: border-box; vertical-align: top; }
.widget.block { display: block; background: #fafafa; width: fit-content; max-width: 100%; }
.widget.text  { display: block; background: #fff; border-color: #e0e0e0;
                white-space: pre-wrap; overflow-wrap: anywhere; }
.widget.link  { display: block; background: #eef5ff; border-color: #99c;
                color: #225; text-decoration: none; }
.widget.image { display: block; background: #f3fff3; border-color: #9c9; }
.widget.main  { border-color: #d33; border-width: 2px; }
.widget.vbox  { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.widget.hbox  { display: flex; flex-direction: row; flex-wrap: wrap;
                align-items: flex-start; justify-content: center; gap: 4px; }
.widget.hbox > .widget { flex: 0 1 auto; }
.widget .role { color: #888; font-size: 10px; margin-left: 6px; }
.widget .name { font-weight: 600; margin-bottom: 4px; }
"""


def _classes(w: Widget) -> str:
    return " ".join(["widget", str(w.type), *(str(h) for h in sorted(w.hints))])


def render_widget(w: Widget) -> str:
    """Render a single widget subtree as an HTML fragment."""
    cls = _classes(w)
    name = escape(w.name) if w.name else ""
    role = f'<span class="role">{escape(w.role)}</span>' if w.role else ""

    if w.type == WidgetType.TEXT:
        return f'<div class="{cls}">{name}{role}</div>'

    if w.type == WidgetType.IMAGE:
        alt = escape(w.name or "")
        return f'<div class="{cls}" title="{alt}">[image: {alt}]{role}</div>'

    if w.type == WidgetType.LINK:
        inner = name + "".join(render_widget(c) for c in w.children)
        return f'<a class="{cls}" href="#">{inner}{role}</a>'

    # BLOCK
    header = f'<div class="name">{name}</div>' if name else ""
    inner = "".join(render_widget(c) for c in w.children)
    return f'<div class="{cls}">{header}{inner}{role}</div>'


def render_document(w: Widget, title: str = "Widget tree") -> str:
    """Render a full standalone HTML document for a widget tree."""
    body = render_widget(w)
    return (
        "<!doctype html>\n"
        f"<html><head><meta charset=\"utf-8\"><title>{escape(title)}</title>"
        f"<style>{_CSS}</style></head><body><main class=\"page\">{body}</main></body></html>"
    )


def save_document(w: Widget, path: str, title: str = "Widget tree") -> str:
    """Render and write the document; returns the path written."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_document(w, title))
    return path
