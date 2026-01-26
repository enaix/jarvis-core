# axtree_html_mapper.py
# Map AXTree link nodes (role=link) -> HTML <a href=...> by matching AX name.value to anchor text/aria-label.

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

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
