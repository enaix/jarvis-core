# axtree_html_mapper.py
# Map AXTree link nodes (role=link) -> HTML <a href=...> by matching AX name.value to anchor text/aria-label.

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from html import unescape
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup


AXTreeInput = Union[List[Dict[str, Any]], Dict[str, Any]]


def _norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def _get_axtree_nodes(axtree: AXTreeInput) -> List[Dict[str, Any]]:
    """
    Accepts:
      - list of AX nodes
      - dict with key "nodes"
    Returns: list of node dicts
    """
    if isinstance(axtree, list):
        return axtree

    if isinstance(axtree, dict):
        nodes = axtree.get("nodes")
        if isinstance(nodes, list):
            return nodes

    raise TypeError("Unsupported AXTree format. Expected list or dict-with-'nodes'.")


def _build_id_text_map(soup: BeautifulSoup) -> Dict[str, str]:
    """Map element id -> visible text (best-effort) for aria-labelledby."""
    out: Dict[str, str] = {}
    for el in soup.find_all(attrs={"id": True}):
        _id = el.get("id")
        if not _id:
            continue
        txt = el.get_text(" ", strip=True)
        if txt:
            out[_id] = txt
    return out


def _accessible_name_for_anchor(a_tag, id_text: Dict[str, str]) -> str:
    aria_label = a_tag.get("aria-label")
    if aria_label:
        return str(aria_label)

    aria_labelledby = a_tag.get("aria-labelledby")
    if aria_labelledby:
        parts: List[str] = []
        for _id in str(aria_labelledby).split():
            if _id in id_text:
                parts.append(id_text[_id])
        if parts:
            return " ".join(parts)

    # title on <a>
    title = a_tag.get("title")
    if title:
        return str(title)

    # visible text inside <a>
    txt = a_tag.get_text(" ", strip=True)
    if txt:
        return txt

    # ---- NEW: fallback for icon/image links ----
    img = a_tag.find("img")
    if img:
        alt = img.get("alt")
        if alt:
            return str(alt)
        img_title = img.get("title")
        if img_title:
            return str(img_title)

    return ""


@dataclass
class AnchorInfo:
    name: str
    name_norm: str
    href_raw: str
    href_resolved: str

def extract_html_anchors_from_text(
    html_text: str,
    base_url: Optional[str] = None,
    parser: str = "html.parser",
) -> List[AnchorInfo]:
    """
    Extract all <a> anchors from HTML *string* in DOM order.
    """
    soup = BeautifulSoup(html_text, parser)
    id_text = _build_id_text_map(soup)

    anchors: List[AnchorInfo] = []
    for a in soup.find_all("a"):
        href_attr = a.get("href")
        href_raw = "" if href_attr is None else str(href_attr)
        href_resolved = urljoin(base_url, href_raw) if base_url else href_raw

        name = _accessible_name_for_anchor(a, id_text)
        anchors.append(
            AnchorInfo(
                name=name,
                name_norm=_norm(name),
                href_raw=href_raw,
                href_resolved=href_resolved,
            )
        )
    return anchors

@dataclass
class AXLinkNode:
    backendDOMNodeId: int
    nodeId: str
    name: str
    name_norm: str


def extract_axtree_link_nodes(axtree: AXTreeInput) -> List[AXLinkNode]:
    """
    Extract AX nodes where role.value == 'link' (case-insensitive)
    and backendDOMNodeId exists.
    """
    nodes = _get_axtree_nodes(axtree)

    out: List[AXLinkNode] = []
    for n in nodes:
        role_val = (n.get("role") or {}).get("value", "")
        if _norm(str(role_val)) != "link":
            continue

        backend_id = n.get("backendDOMNodeId")
        if backend_id is None:
            continue

        name = ((n.get("name") or {}).get("value")) or ""
        out.append(
            AXLinkNode(
                backendDOMNodeId=int(backend_id),
                nodeId=str(n.get("nodeId")),
                name=name,
                name_norm=_norm(name),
            )
        )
    return out


def map_axtree_links_to_html_hrefs(
    axtree: AXTreeInput,
    html: str,
    base_url: Optional[str] = None,
    resolve_urls: bool = True,
    parser: str = "html.parser",
) -> Dict[int, Dict[str, str]]:
    ax_links = extract_axtree_link_nodes(axtree)
    anchors = extract_html_anchors_from_text(html, base_url=base_url, parser=parser)

    # Build exact index: normalized anchor name -> list of indices in DOM order
    name_to_indices: Dict[str, List[int]] = {}
    for idx, a in enumerate(anchors):
        name_to_indices.setdefault(a.name_norm, []).append(idx)

    used = [False] * len(anchors)
    out: Dict[int, Dict[str, str]] = {}

    for ax in ax_links:
        href = ""
        matched_html_name: Optional[str] = None
        best_k: Optional[int] = None

        # Exact match by name (global)
        cand_indices = name_to_indices.get(ax.name_norm, [])
        for k in cand_indices:
            if not used[k]:
                best_k = k
                break

        # Commit match (exact only)
        if best_k is not None:
            used[best_k] = True
            href = anchors[best_k].href_resolved if resolve_urls else anchors[best_k].href_raw
            matched_html_name = anchors[best_k].name
            score = 100
        else:
            score = 0

        out[ax.backendDOMNodeId] = {
            "nodeId": ax.nodeId,
            "href": href,
            "ax_name": ax.name,
            "matched_html_name": matched_html_name,
            "score": str(score),
        }

    return out


def extract_all_anchors(html_text: str, parser: str = "html.parser") -> List[Dict[str, Any]]:
    """
    Parse HTML *string* and return all <a> tags in DOM order.

    Returns a list of dicts:
      [{"i": 0, "href": "...", "text": "...", "aria_label": "...", "title": "..."}, ...]
    """
    soup = BeautifulSoup(html_text, parser)
    anchors = soup.find_all("a")

    out: List[Dict[str, Any]] = []
    for i, a in enumerate(anchors):
        out.append({
            "i": i,
            "href": a.get("href", ""),
            "text": a.get_text(" ", strip=True),
            "aria_label": a.get("aria-label", ""),
            "title": a.get("title", ""),
        })
    return out


def assert_mapping_quality(
    mapping: dict,
    *,
    require_href: bool = True,
    require_matched_html_name: bool = True,
    max_bad: int = 0,
    dump_path: str | None = None,
    show_examples: int = 10,
) -> None:
    """
    Validates mapping produced by map_axtree_links_to_html_hrefs.

    Checks (configurable):
      - matched_html_name is not None
      - href is not empty

    Raises AssertionError if number of bad entries > max_bad.
    Optionally dumps bad entries to JSON.
    """
    bad = {}

    for backend_id, rec in mapping.items():
        href = rec.get("href", "")
        matched_name = rec.get("matched_html_name", None)

        fails = []
        if require_matched_html_name and matched_name is None:
            fails.append("matched_html_name=None")
        if require_href and not href:
            fails.append("href=''")

        if fails:
            bad[backend_id] = {**rec, "_fails": fails}

    if dump_path and bad:
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(bad, f, ensure_ascii=False, indent=2)

    if len(bad) > max_bad:
        # show a few examples in the error
        examples = list(bad.items())[:show_examples]
        raise AssertionError(
            f"Mapping quality check failed: bad={len(bad)}/{len(mapping)}.\n"
            f"First {min(show_examples, len(examples))} bad examples:\n"
            + "\n".join([f"- backendDOMNodeId={k}: {v.get('ax_name')} -> {v.get('href')} | {v.get('_fails')}"
                         for k, v in examples])
        )


# ---------------------------------------------------------------------------
# v2 mapping (ported from notebooks/webui/loader.ipynb).
#
# Differences vs map_axtree_links_to_html_hrefs:
#   * Detects a link node via role.value OR properties[*].name in
#     {"role", "aria-role", "aria_role"} with value "link".
#   * Pulls anchor text from visible text, aria-label, title, aria-labelledby,
#     then <img> alt/aria-label/title.
#   * Resolves and normalizes hrefs (urljoin + small fixup table).
#   * Instead of a blind greedy first-unused assignment, classifies each match
#     into a category and only fills `href` when the match is unambiguous.
#
# Operates on a SINGLE page. Returns the mapping itself (not statistics).
# ---------------------------------------------------------------------------


def _unwrap_stringish(x: Any) -> str:
    """Pull a string out of nested AX-style values: str / {"value": ...} / list."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        if "value" in x:
            return _unwrap_stringish(x["value"])
        for k in ("name", "text", "label"):
            if k in x:
                return _unwrap_stringish(x[k])
        return ""
    if isinstance(x, list):
        for it in x:
            s = _unwrap_stringish(it)
            if s:
                return s
        return ""
    return str(x)


# Zero-width / invisible characters frequently leak through HTML copy-paste
# (zero-width space, ZWNJ, ZWJ, word joiner, BOM). They survive whitespace
# collapsing because Python's \s does not include them.
_ZERO_WIDTH_RE = re.compile(r"[​‌‍⁠﻿]")


def _normalize_ax_text(x: Any) -> str:
    """Unwrap + HTML-unescape + Unicode-normalize + collapse whitespace.

    Pipeline:
      1. Unwrap nested AX-style values to a string.
      2. HTML-unescape (`&amp;` -> `&`, `&nbsp;` -> NBSP, etc.).
      3. NFKC: fold compatibility variants — full-width digits/latin, ligatures,
         some non-breaking spaces (e.g. NARROW NO-BREAK SPACE U+202F),
         superscripts, etc. — to their canonical ASCII counterparts where one
         exists. AX text and visible anchor text often differ only in such
         variants; without this they match as different strings.
      4. Strip zero-width / invisible markers that survive \\s collapsing.
      5. NBSP (U+00A0) -> regular space. NFKC does NOT decompose plain NBSP,
         so this has to be explicit.
      6. Strip + collapse whitespace runs.

    NOTE: case-sensitive on purpose, matching loader.ipynb. AX names and visible
    anchor text are usually authored together, so casefold mostly hides bugs.
    """
    s = _unwrap_stringish(x)
    s = unescape(s or "")
    s = unicodedata.normalize("NFKC", s)
    s = _ZERO_WIDTH_RE.sub("", s)
    s = s.replace("\xa0", " ")
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


# Schemes that are not real navigational destinations. These are dropped from
# anchor_items so they can't accidentally match an AX link node — there is no
# "page" on the other side to point at, only a side-effect (script execution,
# inline blob, raw data). `mailto:`, `tel:`, `sms:` are intentionally NOT in
# this list — they are valid destinations for activation, just not pages.
_NON_NAV_SCHEMES = ("javascript:", "data:", "blob:", "about:")


def _normalize_url_v2(u: str) -> str:
    """Bring URL to a stable form so equal destinations match as equal strings.

    Steps:
      * Drop empty / `javascript:` / `data:` / `blob:` / `about:` (no page).
      * Apply small site-specific fixup table (extend as needed).
      * Lowercase scheme and host, leave path / query / fragment untouched
        (case-significant per RFC 3986 §6.2.2.1).
      * Drop default port (`:80` for http, `:443` for https).

    Returns "" for URLs that should not participate in matching at all.
    """
    u = (u or "").strip()
    if not u:
        return ""
    low = u.lower()
    if low.startswith(_NON_NAV_SCHEMES):
        return ""

    # Known site-specific fixups
    u = u.replace("https://libera.chat#", "https://libera.chat/#")

    # Lowercase scheme + host; drop default port
    p = urlsplit(u)
    if p.scheme:
        scheme = p.scheme.lower()
        host = (p.hostname or "").lower()
        port = p.port
        if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            port = None
        userinfo = ""
        if p.username is not None:
            userinfo = p.username
            if p.password is not None:
                userinfo += ":" + p.password
            userinfo += "@"
        netloc = userinfo + host + (f":{port}" if port is not None else "")
        u = urlunsplit((scheme, netloc, p.path, p.query, p.fragment))
    return u


def _anchor_name_v2(a_tag, id_text: Dict[str, str]) -> str:
    """Pick the first non-empty name from: visible text, aria-label, title,
    aria-labelledby, <img> alt/aria-label/title."""
    for c in (
        a_tag.get_text(" ", strip=True),
        a_tag.get("aria-label"),
        a_tag.get("title"),
    ):
        t = _normalize_ax_text(c)
        if t:
            return t

    aria_labelledby = a_tag.get("aria-labelledby")
    if aria_labelledby:
        parts: List[str] = []
        for _id in str(aria_labelledby).split():
            if _id in id_text:
                parts.append(id_text[_id])
        if parts:
            t = _normalize_ax_text(" ".join(parts))
            if t:
                return t

    img = a_tag.find("img")
    if img:
        for c in (img.get("alt"), img.get("aria-label"), img.get("title")):
            t = _normalize_ax_text(c)
            if t:
                return t

    return ""


def _node_is_link_v2(n: Dict[str, Any]) -> bool:
    """role.value == 'link' OR properties[*] with name in
    {role, aria-role, aria_role} and value 'link'."""
    role = n.get("role")
    role_val = (
        _normalize_ax_text(role.get("value")) if isinstance(role, dict)
        else _normalize_ax_text(role)
    )
    if role_val.lower() == "link":
        return True

    props = n.get("properties")
    if isinstance(props, list):
        for p in props:
            if not isinstance(p, dict):
                continue
            if p.get("name") in ("role", "aria-role", "aria_role"):
                if _normalize_ax_text(p.get("value")).lower() == "link":
                    return True
    return False


def _node_text_v2(n: Dict[str, Any]) -> str:
    for c in (n.get("name"), n.get("accessibleName"), n.get("text"), n.get("value")):
        t = _normalize_ax_text(c)
        if t:
            return t
    return ""


def map_axtree_links_to_html_hrefs_v2(
    axtree: AXTreeInput,
    html: str,
    base_url: Optional[str] = None,
    parser: str = "html.parser",
) -> Dict[int, Dict[str, Any]]:
    """
    Map AXTree link nodes to HTML <a href=...> for a SINGLE page, using the
    richer logic ported from notebooks/webui/loader.ipynb.

    Returns: backendDOMNodeId -> {
        nodeId:            str,
        ax_name:           str,            # raw AX name (unwrapped, not normalized)
        ax_text:           str,            # normalized text used as match key
        href:              str,            # filled only when the match is safe
        candidate_hrefs:   List[str],      # all hrefs whose anchors share this text
        matched_html_name: Optional[str],  # the anchor text actually matched on
        category:          str,
            # safe_1to1         - 1 node, 1 unique href             (href filled)
            # safe_multiple     - K>1 nodes, 1 href, anchors >= K   (href filled)
            # indistinguishable - K nodes, K anchors, multiple hrefs (href empty)
            # all_other         - same text -> multiple hrefs, counts not aligned
            # only_in_nodes     - text not present in any usable anchor
            # no_text           - link node had no text at all
        node_count:        int,            # AX link nodes sharing this text
        anchor_count:      int,            # HTML anchors sharing this text
    }

    Anchors without text or href are dropped from consideration. AX link nodes
    without backendDOMNodeId are also dropped (they cannot be keyed).
    """
    soup = BeautifulSoup(html, parser)
    id_text = _build_id_text_map(soup)

    # --- HTML side: anchors with text AND href ---
    anchor_items: List[Dict[str, Any]] = []
    for i, a in enumerate(soup.find_all("a")):
        text = _anchor_name_v2(a, id_text)
        if not text:
            continue
        href_attr = a.get("href")
        href_raw = "" if href_attr is None else str(href_attr).strip()
        if not href_raw:
            continue
        href = _normalize_url_v2(urljoin(base_url, href_raw) if base_url else href_raw)
        if not href:
            continue
        anchor_items.append({"i": i, "text": text, "href": href})

    # --- AX side: link nodes (split into "with text" and "no text") ---
    nodes_raw = _get_axtree_nodes(axtree)
    link_node_items: List[Dict[str, Any]] = []
    no_text_nodes: List[Dict[str, Any]] = []

    for j, n in enumerate(nodes_raw):
        if not _node_is_link_v2(n):
            continue
        backend_id = n.get("backendDOMNodeId")
        if backend_id is None:
            continue
        rec = {
            "j": j,
            "backend_id": int(backend_id),
            "node_id": str(n.get("nodeId")),
            "ax_name": _unwrap_stringish(n.get("name")),
            "text": _node_text_v2(n),
        }
        (link_node_items if rec["text"] else no_text_nodes).append(rec)

    # --- Indices ---
    anchors_by_text_href: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for it in anchor_items:
        anchors_by_text_href[it["text"]][it["href"]].append(it)

    nodes_by_text: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in link_node_items:
        nodes_by_text[it["text"]].append(it)

    # --- Build mapping ---
    out: Dict[int, Dict[str, Any]] = {}

    for text, n_list in nodes_by_text.items():
        href_map = anchors_by_text_href.get(text, {})
        nodes_count = len(n_list)
        unique_hrefs_count = len(href_map)
        anchors_count = sum(len(lst) for lst in href_map.values())
        candidate_hrefs = sorted(href_map.keys())

        if unique_hrefs_count == 0:
            category = "only_in_nodes"
            href = ""
            matched_html_name: Optional[str] = None
        elif nodes_count == 1 and unique_hrefs_count == 1:
            category = "safe_1to1"
            href = candidate_hrefs[0]
            matched_html_name = text
        elif nodes_count > 1 and unique_hrefs_count == 1 and anchors_count >= nodes_count:
            category = "safe_multiple"
            href = candidate_hrefs[0]
            matched_html_name = text
        elif nodes_count > 1 and anchors_count == nodes_count and unique_hrefs_count > 1:
            category = "indistinguishable"
            href = ""
            matched_html_name = text
        else:
            category = "all_other"
            href = ""
            matched_html_name = text

        for n in n_list:
            out[n["backend_id"]] = {
                "nodeId": n["node_id"],
                "ax_name": n["ax_name"],
                "ax_text": text,
                "href": href,
                "candidate_hrefs": list(candidate_hrefs),
                "matched_html_name": matched_html_name,
                "category": category,
                "node_count": nodes_count,
                "anchor_count": anchors_count,
            }

    for n in no_text_nodes:
        out[n["backend_id"]] = {
            "nodeId": n["node_id"],
            "ax_name": n["ax_name"],
            "ax_text": "",
            "href": "",
            "candidate_hrefs": [],
            "matched_html_name": None,
            "category": "no_text",
            "node_count": 1,
            "anchor_count": 0,
        }

    return out


# ---------------------------------------------------------------------------
# Apply v2 mapping back into a cleaned AXTree, replacing the visible name of
# each link-role node with the resolved href (or a fallback marker when the
# match is not safe). Used by the AXTree -> Widget conversion pipeline so
# that LINK widgets carry the URL instead of the visible anchor text.
# ---------------------------------------------------------------------------


def apply_link_hrefs_to_tree(
    tree: Dict[str, Any],
    mapping: Dict[int, Dict[str, Any]],
    fallback: str = "[LINK]",
    save_original_as: Optional[str] = "orig_name",
    link_roles: Optional[set] = None,
) -> Dict[str, int]:
    """Mutate `tree` in place, rewriting `name` of every link-role node so it
    carries the resolved href URL. Visible text stays preserved as the link's
    tree children (StaticText etc.) and -- if `save_original_as` is set -- a
    backup of the pre-substitution name.

    Expected node shape (as produced by build_tree / transform_tree in the
    notebook): {"id", "back_id", "ignored", "name": str, "role": str,
    "modified": bool, "children": [...]}.

    Substitution rule: if `mapping[node["back_id"]]` exists with category
    in {"safe_1to1", "safe_multiple"} AND a non-empty href, the node's
    `name` becomes that URL. Otherwise (no entry, ambiguous category, or
    empty href) the `name` becomes `fallback` (default "[LINK]").

    The original name is preserved under `save_original_as` (default
    "orig_name") if that key isn't already present on the node. Pass
    `save_original_as=None` to skip preservation.

    `link_roles` defaults to {"link"}. Override if your AXTree uses extra
    link-like role names (e.g., "MenuItemLink").

    Returns a counters dict useful for logging:
      {"with_href": N, "with_fallback": M, "total_link_nodes": N+M}.
    """
    if link_roles is None:
        link_roles = {"link"}
    safe_categories = {"safe_1to1", "safe_multiple"}
    counters = {"with_href": 0, "with_fallback": 0, "total_link_nodes": 0}

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("role") in link_roles:
            counters["total_link_nodes"] += 1
            back_id = node.get("back_id")
            entry = mapping.get(back_id) if back_id is not None else None
            if (
                entry
                and entry.get("category") in safe_categories
                and entry.get("href")
            ):
                new_name = entry["href"]
                counters["with_href"] += 1
            else:
                new_name = fallback
                counters["with_fallback"] += 1
            if save_original_as and save_original_as not in node:
                node[save_original_as] = node.get("name", "")
            node["name"] = new_name
        for child in node.get("children", []):
            walk(child)

    walk(tree)
    return counters
