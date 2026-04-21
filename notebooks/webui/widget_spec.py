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
