"""
Widget specification (v0) for the WebUI dataset.

A Widget is a node in an internal representation tree derived from an AXTree.
Each widget has exactly one `type` (text / block / link / image) and an
unordered set of `hints` (vbox / hbox / main / ...).

The module also defines the role-to-widget reduction rules used by the
AXTree -> Widget conversion in loader.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional


class WidgetType(StrEnum):
    TEXT = "text"
    BLOCK = "block"
    LINK = "link"
    IMAGE = "image"


class WidgetHint(StrEnum):
    VBOX = "vbox"
    HBOX = "hbox"
    MAIN = "main"
    # Semantic hints promoted from text-wrapping AX roles by ax_to_widget.
    # These preserve the "kind" of text (heading / emphasis / quote / etc.)
    # that would otherwise be lost when a heading -> StaticText wrapper
    # collapses into its child.
    HEADING = "heading"
    SUBHEADING = "subheading"
    BOLD = "bold"
    ITALIC = "italic"
    QUOTE = "quote"
    CAPTION = "caption"
    TIME = "time"


@dataclass
class Widget:
    type: WidgetType
    hints: set[WidgetHint] = field(default_factory=set)
    name: str = ""
    children: list["Widget"] = field(default_factory=list)
    node_id: Optional[str] = None
    back_id: Optional[int] = None
    role: str = ""  # originating AXTree role, kept for debugging

    def add_hint(self, h: WidgetHint) -> "Widget":
        self.hints.add(h)
        return self

    def pretty(self, indent: int = 0) -> str:
        pad = "  " * indent
        hints = ",".join(sorted(self.hints)) if self.hints else "-"
        head = f"{pad}[{self.type}]({hints}) role={self.role!r} name={self.name!r}"
        lines = [head] + [c.pretty(indent + 1) for c in self.children]
        return "\n".join(lines)


# --- Reduction rules (AXTree role -> Widget) ------------------------------
#
# Rule 1 (leaf):    `... -> <role in TEXT/LINK/IMAGE_ROLES>` collapses to the
#                   corresponding leaf widget, carrying `name` as its text.
# Rule 2 (wrapper): `... -> <role in CONTAINER_ROLES> -> [w1, ..., wn]`
#                   collapses to a `block` widget wrapping its widget children.
#                   If it has exactly one child, the wrapper is transparent and
#                   the child bubbles up (optionally gaining hints).
# Rule 3 (hints):   roles in MAIN_ROLES add the `main` hint to the produced
#                   widget.
#
# Rules are applied bottom-up during a single post-order traversal; iteration
# "until there is a single root" (step 3 in the spec) is implicit in the
# recursion.

TEXT_ROLES: set[str] = {
    "StaticText", "heading", "paragraph", "LineBreak",
    "emphasis", "strong", "LabelText", "Abbr", "time",
    "ListMarker", "superscript", "insertion", "blockquote",
}

LINK_ROLES: set[str] = {"link"}

IMAGE_ROLES: set[str] = {
    "img", "image", "figure", "Video",
    "Iframe", "IframePresentational", "PluginObject",
}

CONTAINER_ROLES: set[str] = {
    "none", "generic", "Section", "group", "region",
    "article", "complementary", "navigation", "banner",
    "contentinfo", "form", "search",
    "list", "listitem", "main", "RootWebArea",
    "table", "row", "gridcell", "columnheader", "rowheader", "grid",
    "LayoutTable", "LayoutTableRow", "LayoutTableCell",
    "tablist", "tab", "tabpanel",
    "menu", "menubar", "menuitem", "MenuListPopup",
    "dialog", "alertdialog",
    "DescriptionList", "DescriptionListTerm", "DescriptionListDetail",
    "Figcaption", "Legend",
    "HeaderAsNonLandmark", "FooterAsNonLandmark",
    "Details", "DisclosureTriangle",
    "separator", "status", "Pre",
    "button", "checkbox", "switch", "textbox", "searchbox",
    "combobox", "listbox", "option", "spinbutton",
}

MAIN_ROLES: set[str] = {"main", "RootWebArea"}


def role_to_leaf_type(role: str) -> Optional[WidgetType]:
    """Return the leaf WidgetType for `role`, or None if `role` is not a leaf."""
    if role in TEXT_ROLES:
        return WidgetType.TEXT
    if role in LINK_ROLES:
        return WidgetType.LINK
    if role in IMAGE_ROLES:
        return WidgetType.IMAGE
    return None


def role_is_container(role: str) -> bool:
    return role in CONTAINER_ROLES or role_to_leaf_type(role) is None


# --- Text-role -> hint promotion ------------------------------------------
#
# When ax_to_widget collapses a text-wrapping role (heading -> StaticText,
# strong -> StaticText, ...) into its child TEXT widget, the parent role
# would otherwise be lost. We promote a curated subset of these roles to
# semantic hints on the resulting widget. The mapping is intentionally
# small to keep the hint vocabulary stable for downstream LLM training.
#
# Roles deliberately NOT mapped (kept as-is, lossy collapse):
#   * paragraph, LineBreak  -- structural-only, no useful semantic
#   * code, Pre, kbd, samp, var, mark, cite, q, dfn, insertion, deletion,
#     subscript, superscript, Abbr  -- usage in real data not yet verified
#
# heading uses aria-level: level 1 -> HEADING, level >= 2 -> SUBHEADING.
# Missing level defaults to HEADING.

def text_role_to_hints(role: str, level: Optional[int] = None) -> list[WidgetHint]:
    """Return the WidgetHints to apply when collapsing a text-wrapper role
    into its child TEXT widget."""
    if role == "heading":
        if level is not None and level >= 2:
            return [WidgetHint.SUBHEADING]
        return [WidgetHint.HEADING]
    if role == "strong":
        return [WidgetHint.BOLD]
    if role == "emphasis":
        return [WidgetHint.ITALIC]
    if role == "blockquote":
        return [WidgetHint.QUOTE]
    if role in ("caption", "Figcaption", "LabelText"):
        return [WidgetHint.CAPTION]
    if role == "time":
        return [WidgetHint.TIME]
    return []
