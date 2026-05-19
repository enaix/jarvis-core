"""
axtree_to_widget.py - deterministic converters from AXtree-style JSON to
widget-tree generation format.

Two entry points:

1. `axtree_to_widget(node)` - converts a **nested** AXtree-style dict (a
   single root node carrying its own children inline). This is the
   simplified form used by t3 / t3_full / t3_lean variants in the ZS/FS
   experiment.

2. `chrome_axtree_to_widget(obj)` - converts the **real Chrome DevTools
   Protocol AXTree dump** (flat array of nodes with parentId/childIds
   references, typed role/name objects, properties array, ignored flag).
   Used by the t3_real variant. Internally normalizes the flat structure
   into a nested form and then delegates to `axtree_to_widget`.

Both never raise on garbage input - they fall back to an empty `block`
widget so downstream validation can categorize the failure.

Reduction rules mirror `widget_spec.py` (TEXT_ROLES / LINK_ROLES /
IMAGE_ROLES / CONTAINER_ROLES / MAIN_ROLES and `text_role_to_hints`).
"""
from __future__ import annotations

import sys
from pathlib import Path

# import role taxonomy from canonical widget_spec.py one level up
sys.path.insert(0, str(Path(__file__).parent.parent))
from widget_spec import (  # noqa: E402
    TEXT_ROLES, LINK_ROLES, IMAGE_ROLES, CONTAINER_ROLES, MAIN_ROLES,
    text_role_to_hints,
)

# combined known-role set, lowercased, for AXtree-level validation
_ALL_KNOWN_ROLES_LC = {r.lower() for r in
                      (TEXT_ROLES | LINK_ROLES | IMAGE_ROLES
                       | CONTAINER_ROLES | MAIN_ROLES)}

# LLMs tend to emit lowercase ARIA-style role names ("section", "heading"),
# while widget_spec.py uses the mixed-case shapes Chrome AXTree emits
# ("Section", "RootWebArea", "StaticText"). Lower-case all canonical role
# sets once so membership checks tolerate either style.
_TEXT_ROLES = {r.lower() for r in TEXT_ROLES}
_LINK_ROLES = {r.lower() for r in LINK_ROLES}
_IMAGE_ROLES = {r.lower() for r in IMAGE_ROLES}
_MAIN_ROLES = {r.lower() for r in MAIN_ROLES}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _hint_strs(role, level):
    """Run widget_spec.text_role_to_hints and return string values (not enums).
    Pass role lowercased - text_role_to_hints uses lowercase comparisons."""
    out = []
    for h in text_role_to_hints(role.lower(), level=level):
        v = h.value if hasattr(h, "value") else h
        out.append(str(v))
    return out


def _collect_text(node):
    """Walk a subtree and return the first non-empty `name` encountered, in
    document order. Used when a text-role node has no own `name` but has
    text-bearing children (e.g. `heading > StaticText "Title"`)."""
    if not isinstance(node, dict):
        return ""
    name = node.get("name") or ""
    if isinstance(name, str) and name.strip():
        return name
    for c in node.get("children") or []:
        t = _collect_text(c)
        if t:
            return t
    return ""


def _empty_block():
    return {"type": "block", "hints": [], "name": "", "children": []}


# --------------------------------------------------------------------------
# 1. Nested AXtree -> widget (the simplified form)
# --------------------------------------------------------------------------
def axtree_to_widget(node):
    """Convert one nested AXtree-style dict node to one widget dict node.

    Input shape: {"role": str, "name": str?, "level": int?, "children": list?}

    Robust to:
      - missing fields (defaults applied)
      - unknown roles (treated as generic container -> block)
      - text-role wrappers with StaticText children (idiomatic AXtree)
      - text-role leaves carrying text in `name` directly (simpler LLM style)
    """
    if not isinstance(node, dict):
        return _empty_block()

    role = str(node.get("role") or "").strip()
    role_lc = role.lower()
    name = node.get("name") if isinstance(node.get("name"), str) else ""
    level = node.get("level") if isinstance(node.get("level"), int) else None
    raw_children = [c for c in (node.get("children") or []) if isinstance(c, dict)]

    # ---- text roles (StaticText, heading, paragraph, emphasis, ...) -----
    if role_lc in _TEXT_ROLES:
        hints = sorted(set(_hint_strs(role, level)))
        text = name or _collect_text(node)
        return {"type": "text", "hints": hints, "name": text, "children": []}

    # ---- link -----------------------------------------------------------
    if role_lc in _LINK_ROLES:
        return {
            "type": "link",
            "hints": [],
            "name": name,   # convention: name holds the URL
            "children": [axtree_to_widget(c) for c in raw_children],
        }

    # ---- image ----------------------------------------------------------
    if role_lc in _IMAGE_ROLES:
        return {"type": "image", "hints": [], "name": name, "children": []}

    # ---- container or unknown ------------------------------------------
    hints = ["main"] if role_lc in _MAIN_ROLES else []
    return {
        "type": "block",
        "hints": sorted(set(hints)),
        "name": name,
        "children": [axtree_to_widget(c) for c in raw_children],
    }


# --------------------------------------------------------------------------
# 2. Real Chrome DevTools Protocol AXTree dump -> widget
# --------------------------------------------------------------------------
def _extract_role(n):
    """role can be either a plain string OR Chrome's typed object
    {"type": "internalRole"|"role", "value": <name>}."""
    r = n.get("role")
    if isinstance(r, dict):
        return str(r.get("value", "")).strip()
    return str(r or "").strip()


def _extract_name(n):
    """name can be either a plain string OR Chrome's typed object
    {"type": "computedString", "value": <str>, "sources": [...]}"""
    nm = n.get("name")
    if isinstance(nm, dict):
        v = nm.get("value")
        return str(v) if v is not None else ""
    return str(nm or "")


def _extract_level(n):
    """Pull 'level' out of Chrome's properties array (used on headings)."""
    for prop in (n.get("properties") or []):
        if isinstance(prop, dict) and prop.get("name") == "level":
            v = prop.get("value")
            if isinstance(v, dict):
                inner = v.get("value")
                if isinstance(inner, int):
                    return inner
            elif isinstance(v, int):
                return v
    return None


def chrome_axtree_to_widget(obj):
    """Convert a Chrome DevTools Protocol AXTree dump to a widget dict.

    Accepts:
      - {"nodes": [<node>, ...]}              standard Chrome wrapper
      - [<node>, ...]                         bare flat list of nodes
      - {"role": ..., "children": ...}        already nested (delegates to axtree_to_widget)

    Each node should look like:
      {"nodeId": "<id>",
       "role": <str | {type, value}>,
       "name": <str | {type, value, sources?}>,
       "properties": [{name, value: {type, value}}, ...],   # optional
       "childIds": ["<id>", ...],
       "parentId": "<id>",                                  # omitted for root
       "ignored": true|false,                               # optional
       "ignoredReasons": [...],                             # optional
       "backendDOMNodeId": <int>, "frameId": "<str>"}       # optional

    The flat structure is resolved into a nested form via parentId/childIds,
    then handed off to axtree_to_widget. Nodes with ignored=true (and their
    entire subtrees) are skipped. Robust to missing fields, cycles, and
    unrecognized roles.
    """
    # already-nested form -> hand off
    if isinstance(obj, dict) and "role" in obj and "nodes" not in obj:
        return axtree_to_widget(obj)

    # extract the flat list of nodes
    if isinstance(obj, dict) and "nodes" in obj:
        nodes = obj["nodes"]
    elif isinstance(obj, list):
        nodes = obj
    else:
        return _empty_block()

    if not isinstance(nodes, list) or not nodes:
        return _empty_block()

    by_id = {str(n["nodeId"]): n
             for n in nodes if isinstance(n, dict) and "nodeId" in n}
    if not by_id:
        return _empty_block()

    visited = set()

    def build_nested(nid):
        if nid in visited:
            return None  # cycle guard
        visited.add(nid)
        n = by_id.get(str(nid))
        if not isinstance(n, dict):
            return None
        if n.get("ignored") is True:
            return None  # drop entire subtree
        kids = []
        for cid in (n.get("childIds") or []):
            built = build_nested(str(cid))
            if built is not None:
                kids.append(built)
        nested = {
            "role": _extract_role(n),
            "name": _extract_name(n),
            "children": kids,
        }
        lvl = _extract_level(n)
        if lvl is not None:
            nested["level"] = lvl
        return nested

    # find the root: a node whose parentId is absent or points outside
    candidate_roots = [n for n in nodes
                       if isinstance(n, dict)
                       and (n.get("parentId") is None
                            or str(n.get("parentId")) not in by_id)]
    non_ignored = [r for r in candidate_roots if not r.get("ignored")]
    if non_ignored:
        root = non_ignored[0]
    elif candidate_roots:
        root = candidate_roots[0]
    else:
        root = next((n for n in nodes
                     if isinstance(n, dict) and not n.get("ignored")
                     and "nodeId" in n), None)
        if root is None:
            return _empty_block()

    nested = build_nested(str(root["nodeId"]))
    if nested is None:
        return _empty_block()
    return axtree_to_widget(nested)


# --------------------------------------------------------------------------
# 3. Nested AXtree -> Chrome DevTools Protocol flat dump (inverse direction)
# --------------------------------------------------------------------------
# Mostly Chrome-internal roles that use role.type = "internalRole" rather
# than "role" (the ARIA-standard roles). We treat any role NOT in this set
# as type="role". This matches what we observed in real Chrome AXTree dumps
# from WebUI (RootWebArea is internalRole; heading/paragraph/list/etc. are
# role).
_INTERNAL_ROLES = {
    "RootWebArea", "none", "LayoutTable", "LayoutTableRow", "LayoutTableCell",
    "Section", "MenuListPopup", "HeaderAsNonLandmark", "FooterAsNonLandmark",
}


def nested_to_chrome(node):
    """Convert a nested AXtree-style dict into a flat Chrome DevTools Protocol
    AXTree dump. The inverse of chrome_axtree_to_widget's flat-resolution step.

    Used by pipeline.py to convert the nested few-shot examples stored in
    eval_set/few_shot_examples.jsonl into the Chrome flat format expected
    by the t3_real prompt.

    Input  : {"role": ..., "name": ..., "level": ?, "children": [...]}
    Output : {"nodes": [<flat node with nodeId/parentId/childIds/typed
                         role/typed name/properties>, ...]}
    """
    nodes = []
    counter = [0]

    def visit(n, parent_id=None):
        if not isinstance(n, dict):
            return None
        counter[0] += 1
        my_id = str(counter[0])
        role = str(n.get("role") or "").strip()
        role_type = "internalRole" if role in _INTERNAL_ROLES else "role"

        chrome_node = {
            "nodeId": my_id,
            "role": {"type": role_type, "value": role},
            "name": {"type": "computedString", "value": (n.get("name") or "")},
            "properties": [],
            "childIds": [],
        }
        if parent_id is not None:
            chrome_node["parentId"] = parent_id
        # heading level -> properties array entry
        lvl = n.get("level")
        if isinstance(lvl, int):
            chrome_node["properties"].append({
                "name": "level",
                "value": {"type": "integer", "value": lvl},
            })

        nodes.append(chrome_node)

        for c in (n.get("children") or []):
            cid = visit(c, parent_id=my_id)
            if cid is not None:
                chrome_node["childIds"].append(cid)
        return my_id

    visit(node)
    return {"nodes": nodes}


# --------------------------------------------------------------------------
# 4. AXtree-level validation (independent of widget conversion)
# --------------------------------------------------------------------------
# Permissive `chrome_axtree_to_widget` and `axtree_to_widget` will return a
# schema-valid widget even when the source AXtree has integrity problems
# (broken refs, unknown roles, multiple parents, cycles, orphans). That hides
# real model failures behind 100% widget validity.
#
# These validators inspect the AXtree BEFORE conversion and classify:
#   'valid' — clean
#   'soft'  — issues the converter papers over (unknown role, orphan,
#             multiple parents, etc.)
#   'hard'  — structural break (missing root, cycle, broken refs, malformed
#             node)
#
# `to_widget_from_*` in metrics.py records BOTH this `axtree_status` and the
# post-conversion `widget_status`, so we can compare and surface hidden
# failures.


def _validate_chrome_flat(nodes):
    issues_hard = []
    issues_soft = []

    if not isinstance(nodes, list):
        return "hard", ["nodes is not a list"]
    if not nodes:
        return "hard", ["nodes list is empty"]

    by_id = {}
    for n in nodes:
        if not isinstance(n, dict):
            issues_hard.append("non-dict node in nodes list")
            continue
        nid = n.get("nodeId")
        if nid is None:
            issues_hard.append(f"node without nodeId (role={n.get('role')!r})")
            continue
        nid_s = str(nid)
        if nid_s in by_id:
            issues_hard.append(f"duplicate nodeId {nid_s!r}")
            continue
        by_id[nid_s] = n

    if not by_id:
        return "hard", ["no usable nodes (all missing/duplicate nodeId)"]

    # 1. role validity
    for nid, n in by_id.items():
        role = _extract_role(n)
        if not role:
            issues_hard.append(f"node {nid}: missing role")
        elif role.lower() not in _ALL_KNOWN_ROLES_LC:
            issues_soft.append(f"node {nid}: unknown role {role!r}")

    # 2. reference integrity (childIds / parentId point inside dump)
    for nid, n in by_id.items():
        for cid in (n.get("childIds") or []):
            if str(cid) not in by_id:
                issues_hard.append(f"node {nid}: childId {str(cid)!r} not in dump")
        pid = n.get("parentId")
        if pid is not None and str(pid) not in by_id:
            issues_hard.append(f"node {nid}: parentId {str(pid)!r} not in dump")

    # 3. multiple parents (node claimed as child by two different parents)
    parent_of = {}
    for nid, n in by_id.items():
        for cid in (n.get("childIds") or []):
            cid_s = str(cid)
            if cid_s in parent_of and parent_of[cid_s] != nid:
                issues_soft.append(
                    f"node {cid_s}: claimed by both {parent_of[cid_s]} and {nid}")
            else:
                parent_of[cid_s] = nid

    # 4. parentId/childIds bidirectional consistency
    for nid, n in by_id.items():
        pid = n.get("parentId")
        if pid is not None and str(pid) in by_id:
            parent = by_id[str(pid)]
            if nid not in [str(c) for c in (parent.get("childIds") or [])]:
                issues_soft.append(
                    f"node {nid}: parentId={pid} but parent doesn't list it as child")

    # 5. find roots (parentId missing or pointing outside dump), excluding ignored
    roots = [nid for nid, n in by_id.items()
             if (n.get("parentId") is None or str(n.get("parentId")) not in by_id)
             and not n.get("ignored")]
    if not roots:
        issues_hard.append("no root (every non-ignored node has parentId inside dump)")
    elif len(roots) > 1:
        issues_soft.append(f"multiple roots: {roots[:5]}")

    # 6. cycle detection via DFS coloring on childIds graph
    color = {}
    has_cycle = [False]

    def dfs(rid):
        if has_cycle[0]:
            return
        color[rid] = "gray"
        for cid in (by_id.get(rid, {}).get("childIds") or []):
            cs = str(cid)
            if cs not in by_id:
                continue
            if color.get(cs) == "gray":
                has_cycle[0] = True
                return
            if color.get(cs) != "black":
                dfs(cs)
        color[rid] = "black"

    for nid in by_id:
        if nid not in color:
            dfs(nid)
    if has_cycle[0]:
        issues_hard.append("cycle in childIds graph")

    # 7. orphans (unreachable from any root, excluding ignored)
    reachable = set()
    visited = set()

    def walk_reach(rid):
        if rid in visited:
            return
        visited.add(rid)
        reachable.add(rid)
        for cid in (by_id.get(rid, {}).get("childIds") or []):
            walk_reach(str(cid))

    for r in roots:
        walk_reach(r)
    orphans = [nid for nid in by_id
               if nid not in reachable and not by_id[nid].get("ignored")]
    if orphans:
        issues_soft.append(f"{len(orphans)} orphan non-ignored nodes")

    if issues_hard:
        return "hard", issues_hard + issues_soft
    if issues_soft:
        return "soft", issues_soft
    return "valid", []


def _validate_nested(root):
    issues_hard = []
    issues_soft = []

    def visit(n, path="$"):
        if not isinstance(n, dict):
            issues_hard.append(f"{path}: non-dict node")
            return
        role = _extract_role(n)
        if not role:
            issues_hard.append(f"{path}: missing role")
        elif role.lower() not in _ALL_KNOWN_ROLES_LC:
            issues_soft.append(f"{path}: unknown role {role!r}")
        for i, c in enumerate(n.get("children") or []):
            visit(c, f"{path}.children[{i}]")

    visit(root)
    if issues_hard:
        return "hard", issues_hard + issues_soft
    if issues_soft:
        return "soft", issues_soft
    return "valid", []


def validate_axtree(obj):
    """Source-level AXtree validation, independent of conversion to widget.

    Handles both Chrome flat format ({nodes: [...]} or [list of nodes]) and
    nested form ({role, name, children}).

    Returns (status, issues) where status is one of {'valid', 'soft', 'hard'}.
    """
    if isinstance(obj, dict) and "nodes" in obj:
        return _validate_chrome_flat(obj["nodes"])
    if isinstance(obj, list):
        return _validate_chrome_flat(obj)
    if isinstance(obj, dict) and "role" in obj:
        return _validate_nested(obj)
    return "hard", ["not a recognizable AXtree shape (no 'nodes' array, no 'role' field)"]


__all__ = [
    "axtree_to_widget", "chrome_axtree_to_widget",
    "nested_to_chrome", "validate_axtree",
]
