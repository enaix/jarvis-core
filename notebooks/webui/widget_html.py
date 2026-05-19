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

/* --- Semantic text hints (promoted from text-wrapping AX roles) --- */
.widget.heading    { font-size: 1.6em; font-weight: 700; line-height: 1.2; }
.widget.subheading { font-size: 1.25em; font-weight: 600; line-height: 1.25; }
.widget.bold       { font-weight: 700; }
.widget.italic     { font-style: italic; }
.widget.quote      { border-left: 4px solid #aaa; padding-left: 12px;
                     color: #555; font-style: italic; }
.widget.caption    { font-size: 0.85em; color: #555; }
.widget.time       { font-family: ui-monospace, "SFMono-Regular", Menlo,
                                  Consolas, monospace;
                     color: #444; }
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
        # Convention from apply_link_hrefs_to_tree: for LINK widgets, `name`
        # carries the resolved URL (or the "[LINK]" placeholder when the
        # match was unsafe). Visible content lives in the link's children.
        href = escape(w.name) if w.name else "#"

        # Empty link: only accessible name available, show it as the label.
        if not w.children:
            label = escape(w.name) if w.name else "[link]"
            return f'<a class="{cls}" href="{href}">{label}{role}</a>'

        # Presentation collapse: if every child is a leaf (TEXT or IMAGE),
        # flatten the link contents into a single <a> with text and image
        # placeholders concatenated by spaces. Covers:
        #   - one or many TEXT chunks ("Download for" + "Linux")
        #   - one or many IMAGE leaves (social-icon rows)
        #   - mixed icon + label patterns (<a><img> Label</a>)
        # Card-style links with non-leaf children (heading, paragraph, block)
        # stay rendered nested so visual structure is preserved.
        if all(c.type in (WidgetType.TEXT, WidgetType.IMAGE) for c in w.children):
            parts: list[str] = []
            for c in w.children:
                if c.type == WidgetType.TEXT:
                    if c.name:
                        parts.append(escape(c.name))
                else:  # IMAGE
                    alt = (c.name or "").strip()
                    parts.append(f'[image: {escape(alt)}]' if alt else '[image]')
            inner = " ".join(parts) if parts else (escape(w.name) if w.name else "[link]")
            return f'<a class="{cls}" href="{href}">{inner}{role}</a>'

        # Mixed / complex children (heading, block, list inside a card-link)
        # -> keep nested rendering so structure is visible.
        inner = "".join(render_widget(c) for c in w.children)
        return f'<a class="{cls}" href="{href}">{inner}{role}</a>'

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
