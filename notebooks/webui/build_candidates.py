import json
import re
import os
from pathlib import Path
from html import unescape
from typing import Any, Dict, List, Optional
from unittest import loader
from page_loader import PageLoader, FileType


_WS_RE = re.compile(r"\s+")


def _unwrap_stringish(x: Any) -> str:
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
        parts = []
        for item in x:
            s = _unwrap_stringish(item)
            if s:
                parts.append(s)
        return " ".join(parts)
    return str(x)


def normalize_text(x: Any) -> str:
    s = _unwrap_stringish(x)
    s = unescape((s or "").strip())
    s = _WS_RE.sub(" ", s)
    return s


def get_role(node: dict) -> str:
    role = node.get("role")
    if isinstance(role, dict):
        return normalize_text(role.get("value", ""))
    return normalize_text(role)


def get_name(node: dict) -> str:
    return normalize_text(node.get("name", ""))


def get_value(node: dict) -> str:
    return normalize_text(node.get("value", ""))


def get_description(node: dict) -> str:
    return normalize_text(node.get("description", ""))


def get_own_text(node: dict) -> str:
    parts = [
        get_name(node),
        get_value(node),
        get_description(node),
    ]
    parts = [p for p in parts if p]
    return normalize_text(" ".join(parts))


def get_bbox(node: dict, bb_map: dict) -> Optional[dict]:
    backend_id = node.get("backendDOMNodeId")
    if backend_id is None:
        return None

    raw_bbox = bb_map.get(str(backend_id))
    if raw_bbox is None or not isinstance(raw_bbox, dict):
        return None

    try:
        x = float(raw_bbox.get("x", 0.0))
        y = float(raw_bbox.get("y", 0.0))
        width = float(raw_bbox.get("width", 0.0))
        height = float(raw_bbox.get("height", 0.0))
    except (TypeError, ValueError):
        return None

    if width <= 0 or height <= 0:
        return None

    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }


def bbox_area(bbox: Optional[dict]) -> float:
    if not bbox:
        return 0.0
    w = max(float(bbox.get("width", 0.0)), 0.0)
    h = max(float(bbox.get("height", 0.0)), 0.0)
    return w * h


def is_link_like(node: dict) -> bool:
    role = get_role(node).lower()
    return role in {
        "link",
        "menuitem",
        "tab",
        "button",
    }


def to_relative_posix_path(path: str | Path, root: str | Path) -> str:
    path = Path(path).resolve()
    root = Path(root).resolve()

    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(os.path.relpath(path, root))

    return rel.as_posix()

def build_indexes(nodes: List[dict]) -> tuple[Dict[str, dict], Dict[str, List[str]], Dict[str, Optional[str]], str]:
    nodes_by_id = {}
    children_by_id = {}
    parent_by_id = {}
    root_id = None

    for node in nodes:
        node_id = str(node["nodeId"])
        parent_id = node.get("parentId")
        parent_id = str(parent_id) if parent_id is not None else None
        child_ids = [str(x) for x in node.get("childIds", [])]

        nodes_by_id[node_id] = node
        children_by_id[node_id] = child_ids
        parent_by_id[node_id] = parent_id

        if parent_id is None:
            root_id = node_id

    if root_id is None:
        raise ValueError("Root node not found")

    return nodes_by_id, children_by_id, parent_by_id, root_id


def compute_depths(root_id: str, children_by_id: Dict[str, List[str]]) -> Dict[str, int]:
    depths = {}

    def dfs(node_id: str, depth: int) -> None:
        depths[node_id] = depth
        for child_id in children_by_id.get(node_id, []):
            dfs(child_id, depth + 1)

    dfs(root_id, 0)
    return depths


def compute_subtree_stats(
    root_id: str,
    nodes_by_id: Dict[str, dict],
    children_by_id: Dict[str, List[str]],
) -> Dict[str, dict]:
    stats = {}

    def dfs(node_id: str) -> dict:
        node = nodes_by_id[node_id]
        own_text = get_own_text(node)
        own_text_len = len(own_text)

        parts = [own_text] if own_text else []
        num_descendants = 0
        link_text_len = own_text_len if is_link_like(node) else 0

        for child_id in children_by_id.get(node_id, []):
            child_stats = dfs(child_id)
            if child_stats["text_subtree"]:
                parts.append(child_stats["text_subtree"])
            num_descendants += 1 + child_stats["num_descendants"]
            link_text_len += child_stats["link_text_len"]

        text_subtree = normalize_text(" ".join(p for p in parts if p))
        text_len = len(text_subtree)

        stats[node_id] = {
            "own_text": own_text,
            "own_text_len": own_text_len,
            "text_subtree": text_subtree,
            "text_len": text_len,
            "num_descendants": num_descendants,
            "link_text_len": link_text_len,
        }
        return stats[node_id]

    dfs(root_id)
    return stats


def is_wrapper_like(node: dict) -> bool:
    role = get_role(node).lower()
    ignored = bool(node.get("ignored", False))
    own_text = get_own_text(node)

    # wrapper-like считаем только ignored-пустышки с ролью none
    return ignored and role == "none" and not own_text


def estimate_page_area(bb_map: dict, fallback_width: float = 1920.0, fallback_height: float = 1080.0) -> float:
    max_x2 = 0.0
    max_y2 = 0.0

    for raw_bbox in bb_map.values():
        if raw_bbox is None or not isinstance(raw_bbox, dict):
            continue

        try:
            x = float(raw_bbox.get("x", 0.0))
            y = float(raw_bbox.get("y", 0.0))
            w = float(raw_bbox.get("width", 0.0))
            h = float(raw_bbox.get("height", 0.0))
        except (TypeError, ValueError):
            continue

        if w <= 0 or h <= 0:
            continue

        max_x2 = max(max_x2, x + w)
        max_y2 = max(max_y2, y + h)

    width = max(max_x2, fallback_width)
    height = max(max_y2, fallback_height)
    return width * height

def intersection_area(a: dict | None, b: dict | None) -> float:
    if not a or not b:
        return 0.0

    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]

    bx1, by1 = b["x"], b["y"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]

    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)


def bbox_contains_ratio(inner: dict | None, outer: dict | None) -> float:
    """
    Какая доля inner лежит внутри outer.
    1.0 = inner полностью внутри outer.
    """
    if not inner or not outer:
        return 0.0

    inner_area = bbox_area(inner)
    if inner_area <= 0:
        return 0.0

    inter = intersection_area(inner, outer)
    return inter / inner_area


def bbox_area_ratio(child_bbox: dict | None, parent_bbox: dict | None) -> float:
    """
    Отношение площади child к площади parent.
    """
    if not child_bbox or not parent_bbox:
        return 0.0

    p_area = bbox_area(parent_bbox)
    c_area = bbox_area(child_bbox)

    if p_area <= 0:
        return 0.0

    return c_area / p_area

def collect_candidate_descendants(node_id: str, children_by_id: dict, candidate_set: set[str]) -> list[str]:
    out = []

    def dfs(cur_id: str):
        for child_id in children_by_id.get(cur_id, []):
            if child_id in candidate_set:
                out.append(child_id)
            dfs(child_id)

    dfs(node_id)
    return out


BAD_ROLES = {
    "link", "button", "checkbox", "radio", "textbox",
    "searchbox", "heading", "labeltext", "image",
    "listitem", "row", "gridcell", "cell", "columnheader", "rowheader",
    "paragraph", "statictext", "text"
}


def basic_candidate_filter(
    node_id,
    nodes_by_id,
    children_by_id,
    parent_by_id,
    depths,
    subtree_stats,
    bb_map,
    page_area,
):
    node = nodes_by_id[node_id]
    role = get_role(node).lower()
    bbox = get_bbox(node, bb_map)
    stats = subtree_stats[node_id]
    child_ids = children_by_id.get(node_id, [])

    # ===== hard blockers =====
    if not child_ids:
        return False

    if bbox is None:
        return False

    if parent_by_id.get(node_id) is None:
        return False

    if role in {"rootwebarea", "webarea", "document"}:
        return False

    if depths[node_id] < 2 or depths[node_id] > 9:
        return False

    area = bbox_area(bbox)
    area_ratio = area / page_area if page_area > 0 else 0.0
    link_density = stats["link_text_len"] / max(stats["text_len"], 1)

    # ===== exceptions =====
    large_content_exception = (
        stats["text_len"] >= 200
        and link_density <= 0.25
        and bbox["height"] >= 200
    )

    sparse_content_exception = (
        stats["text_len"] >= 20
        and stats["num_descendants"] >= 1
        and bbox["height"] >= 80
        and area_ratio >= 0.003
        and link_density <= 0.35
        and role not in BAD_ROLES
    )

    # ВАЖНО:
    # разрешаем большие содержательные wrapper-like контейнеры,
    # а не режем их безусловно
    ignored_content_exception = (
        is_wrapper_like(node)
        and stats["text_len"] >= 80
        and stats["num_descendants"] >= 4
        and bbox["height"] >= 80
        and area_ratio >= 0.005
        and area_ratio < 0.95
        and link_density <= 0.60
    )

    if is_wrapper_like(node) and not ignored_content_exception:
        return False

    # ===== normal rules =====
    normal_candidate = True

    if stats["text_len"] < 30:
        normal_candidate = False

    if stats["num_descendants"] < 2:
        normal_candidate = False

    if area_ratio < 0.002:
        normal_candidate = False

    if bbox["height"] < 40:
        normal_candidate = False

    if role in BAD_ROLES:
        normal_candidate = False

    if area_ratio > 0.30 and not large_content_exception:
        normal_candidate = False

    if not normal_candidate and not sparse_content_exception:
        return False

    return True


def suppress_hierarchy_duplicates(
    candidate_ids: List[str],
    children_by_id: Dict[str, List[str]],
    subtree_stats: Dict[str, dict],
    ratio_threshold: float = 0.85,
) -> List[str]:
    candidate_set = set(candidate_ids)
    suppressed = set()

    def dfs(node_id: str) -> None:
        if node_id in suppressed:
            return

        if node_id in candidate_set:
            parent_text_len = max(subtree_stats[node_id]["text_len"], 1)
            for child_id in children_by_id.get(node_id, []):
                if child_id in candidate_set:
                    child_text_len = subtree_stats[child_id]["text_len"]
                    if child_text_len / parent_text_len >= ratio_threshold:
                        suppressed.add(node_id)
                        break

        for child_id in children_by_id.get(node_id, []):
            dfs(child_id)

    for node_id in candidate_ids:
        dfs(node_id)

    return [node_id for node_id in candidate_ids if node_id not in suppressed]


ROLE_PRIORITY = {
    "main": 6,
    "banner": 6,
    "contentinfo": 6,
    "navigation": 6,
    "section": 5,
    "region": 5,
    "article": 5,
    "complementary": 5,
    "form": 5,
    "table": 4,
    "list": 4,
    "heading": 4,
    "generic": 2,
    "none": 1,
}


def get_role_priority(node: dict) -> int:
    role = get_role(node).lower()
    return ROLE_PRIORITY.get(role, 3)


def suppress_hierarchy_duplicates_v2(
    candidate_ids: list[str],
    nodes_by_id: dict,
    children_by_id: dict,
    subtree_stats: dict,
    bb_map: dict,
    *,
    text_ratio_threshold: float = 0.85,
    area_ratio_threshold: float = 0.70,
    contain_threshold: float = 0.90,
    only_child_text_ratio_threshold: float = 0.85,
) -> list[str]:
    candidate_set = set(candidate_ids)
    suppressed = set()

    for parent_id in candidate_ids:
        if parent_id in suppressed:
            continue

        parent_node = nodes_by_id[parent_id]
        parent_bbox = get_bbox(parent_node, bb_map)
        parent_text_len = max(subtree_stats[parent_id]["text_len"], 1)
        parent_role = get_role(parent_node).lower()
        parent_role_priority = get_role_priority(parent_node)

        if parent_bbox is None:
            continue

        descendant_candidates = collect_candidate_descendants(
            parent_id,
            children_by_id,
            candidate_set,
        )

        if not descendant_candidates:
            continue

        significant_children = [
            cid
            for cid in children_by_id.get(parent_id, [])
            if subtree_stats[cid]["text_len"] >= 20
        ]

        for child_id in descendant_candidates:
            if child_id == parent_id:
                continue

            child_node = nodes_by_id[child_id]
            child_bbox = get_bbox(child_node, bb_map)
            child_text_len = subtree_stats[child_id]["text_len"]
            child_role = get_role(child_node).lower()
            child_role_priority = get_role_priority(child_node)

            if child_bbox is None:
                continue

            text_ratio = child_text_len / parent_text_len
            area_ratio = bbox_area_ratio(child_bbox, parent_bbox)
            contain_ratio = bbox_contains_ratio(child_bbox, parent_bbox)

            text_near_duplicate = (
                text_ratio >= 0.95
                and contain_ratio >= contain_threshold
            )

            strong_duplicate = (
                text_ratio >= text_ratio_threshold
                and area_ratio >= area_ratio_threshold
                and contain_ratio >= contain_threshold
            )

            only_child_dominates = (
                len(significant_children) == 1
                and child_id == significant_children[0]
                and text_ratio >= only_child_text_ratio_threshold
                and contain_ratio >= contain_threshold
            )

            # Если parent и child почти одинаковые, но parent семантически сильнее,
            # НЕ подавляем parent в пользу child.
            role_prefers_parent = (
                parent_role_priority > child_role_priority
                and contain_ratio >= contain_threshold
                and text_ratio >= 0.80
            )

            if role_prefers_parent:
                continue

            # Если child семантически сильнее parent и почти эквивалентен по смыслу,
            # тогда parent можно подавить осторожно.
            role_prefers_child = (
                child_role_priority > parent_role_priority
                and contain_ratio >= contain_threshold
                and text_ratio >= 0.80
            )

            if strong_duplicate or text_near_duplicate:
                suppressed.add(parent_id)
                break

            if only_child_dominates and role_prefers_child:
                suppressed.add(parent_id)
                break

    return [node_id for node_id in candidate_ids if node_id not in suppressed]


def build_candidate_record(
    page_id: str,
    screenshot_path: str,
    node_id: str,
    nodes_by_id: Dict[str, dict],
    children_by_id: Dict[str, List[str]],
    parent_by_id: Dict[str, Optional[str]],
    depths: Dict[str, int],
    subtree_stats: Dict[str, dict],
    bb_map: dict,
    page_area: float,
) -> dict:
    node = nodes_by_id[node_id]
    bbox = get_bbox(node, bb_map)
    stats = subtree_stats[node_id]

    area = bbox_area(bbox)
    area_ratio = area / page_area if page_area > 0 else 0.0
    link_density = stats["link_text_len"] / stats["text_len"] if stats["text_len"] > 0 else 0.0

    return {
        "page_id": page_id,
        "node_id": node_id,
        "backend_dom_node_id": node.get("backendDOMNodeId"),
        "parent_id": parent_by_id.get(node_id),
        "role": get_role(node),
        "ignored": bool(node.get("ignored", False)),
        "bbox": bbox,
        "depth": depths[node_id],
        "num_children": len(children_by_id.get(node_id, [])),
        "num_descendants": stats["num_descendants"],
        "own_text": stats["own_text"],
        "own_text_len": stats["own_text_len"],
        "text_subtree": stats["text_subtree"],
        "text_len": stats["text_len"],
        "link_text_len": stats["link_text_len"],
        "link_density": link_density,
        "bbox_area": area,
        "bbox_ratio": area_ratio,
        "screenshot_path": screenshot_path,
    }


SUPPRESSION_CONFIG = {
    "text_ratio_threshold": 0.85,
    "area_ratio_threshold": 0.70,
    "contain_threshold": 0.90,
    "only_child_text_ratio_threshold": 0.85,
}


def build_candidates_for_loader(loader: PageLoader, base_data_path: str | Path) -> list[dict]:
    if loader.best is None:
        raise ValueError(f"No valid best page found for loader.path={loader.path}")

    page = loader.best

    axtree = page.files.get(FileType.AXTree)
    bb_map = page.files.get(FileType.BB)

    if axtree is None:
        raise ValueError(f"AXTree not loaded for page: {loader.path}")
    if bb_map is None:
        raise ValueError(f"BB not loaded for page: {loader.path}")

    screenshot_fname = page.fnames.get(FileType.ScreenFull)

    if screenshot_fname:
        screenshot_abs_path = Path(page.path) / screenshot_fname
        screenshot_path = to_relative_posix_path(screenshot_abs_path, base_data_path)
    else:
        screenshot_path = ""

    page_id = str(loader.page_id)

    nodes = axtree["nodes"]
    nodes_by_id, children_by_id, parent_by_id, root_id = build_indexes(nodes)
    depths = compute_depths(root_id, children_by_id)
    subtree_stats = compute_subtree_stats(root_id, nodes_by_id, children_by_id)
    page_area = estimate_page_area(bb_map)

    raw_candidate_ids = []
    for node_id in nodes_by_id:
        ok = basic_candidate_filter(
            node_id=node_id,
            nodes_by_id=nodes_by_id,
            children_by_id=children_by_id,
            parent_by_id=parent_by_id,
            depths=depths,
            subtree_stats=subtree_stats,
            bb_map=bb_map,
            page_area=page_area,
        )
        if ok:
            raw_candidate_ids.append(node_id)

    final_candidate_ids = suppress_hierarchy_duplicates_v2(
        candidate_ids=raw_candidate_ids,
        nodes_by_id=nodes_by_id,
        children_by_id=children_by_id,
        subtree_stats=subtree_stats,
        bb_map=bb_map,
        **SUPPRESSION_CONFIG,
    )

    candidates = [
        build_candidate_record(
            page_id=page_id,
            screenshot_path=screenshot_path,
            node_id=node_id,
            nodes_by_id=nodes_by_id,
            children_by_id=children_by_id,
            parent_by_id=parent_by_id,
            depths=depths,
            subtree_stats=subtree_stats,
            bb_map=bb_map,
            page_area=page_area,
        )
        for node_id in final_candidate_ids
    ]

    return candidates


def append_jsonl(records: List[dict], out_path: str) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")





def basic_candidate_filter_debug(
    node_id,
    nodes_by_id,
    children_by_id,
    parent_by_id,
    depths,
    subtree_stats,
    bb_map,
    page_area,
):
    node = nodes_by_id[node_id]
    role = get_role(node).lower()
    bbox = get_bbox(node, bb_map)
    stats = subtree_stats[node_id]
    child_ids = children_by_id.get(node_id, [])

    reasons = []

    if not child_ids:
        reasons.append("no_child_ids")

    if bbox is None:
        reasons.append("bbox_none")

    if parent_by_id.get(node_id) is None:
        reasons.append("no_parent")

    if role in {"rootwebarea", "webarea", "document"}:
        reasons.append(f"hard_block_role:{role}")

    if depths[node_id] < 2 or depths[node_id] > 9:
        reasons.append("bad_depth")

    if bbox is not None:
        area = bbox_area(bbox)
        area_ratio = area / page_area if page_area > 0 else 0.0
        link_density = stats["link_text_len"] / max(stats["text_len"], 1)

        if stats["text_len"] < 30:
            reasons.append("text_len_lt_30")

        if stats["num_descendants"] < 2:
            reasons.append("num_descendants_lt_2")

        if area_ratio < 0.002:
            reasons.append("area_ratio_lt_0.002")

        if bbox["height"] < 40:
            reasons.append("bbox_height_lt_40")

        if role in BAD_ROLES:
            reasons.append(f"bad_role:{role}")

        if is_wrapper_like(node):
            reasons.append("wrapper_like")

        large_content_exception = (
            stats["text_len"] >= 200
            and link_density <= 0.25
            and bbox["height"] >= 200
        )
        if area_ratio > 0.30 and not large_content_exception:
            reasons.append("too_large_without_exception")
    else:
        area_ratio = None
        link_density = None

    passed = basic_candidate_filter(
        node_id=node_id,
        nodes_by_id=nodes_by_id,
        children_by_id=children_by_id,
        parent_by_id=parent_by_id,
        depths=depths,
        subtree_stats=subtree_stats,
        bb_map=bb_map,
        page_area=page_area,
    )

    return {
        "passed": passed,
        "reasons": reasons,
        "role": role,
        "text_len": stats["text_len"],
        "num_descendants": stats["num_descendants"],
        "depth": depths.get(node_id),
        "bbox": bbox,
        "area_ratio": area_ratio,
        "link_density": link_density,
    }


def inspect_page_nodes(loader):
    page = loader.best
    axtree = page.files.get(FileType.AXTree)
    bb_map = page.files.get(FileType.BB)

    nodes = axtree["nodes"]
    nodes_by_id, children_by_id, parent_by_id, root_id = build_indexes(nodes)
    depths = compute_depths(root_id, children_by_id)
    subtree_stats = compute_subtree_stats(root_id, nodes_by_id, children_by_id)
    page_area = estimate_page_area(bb_map)

    rows = []
    for node_id, node in nodes_by_id.items():
        dbg = basic_candidate_filter_debug(
            node_id=node_id,
            nodes_by_id=nodes_by_id,
            children_by_id=children_by_id,
            parent_by_id=parent_by_id,
            depths=depths,
            subtree_stats=subtree_stats,
            bb_map=bb_map,
            page_area=page_area,
        )

        rows.append({
            "node_id": node_id,
            "passed": dbg["passed"],
            "role": dbg["role"],
            "depth": dbg["depth"],
            "text_len": dbg["text_len"],
            "num_descendants": dbg["num_descendants"],
            "area_ratio": dbg["area_ratio"],
            "link_density": dbg["link_density"],
            "reasons": dbg["reasons"],
            "text_preview": subtree_stats[node_id]["text_subtree"][:120],
        })

    return rows




def debug_candidates_for_loader(loader, base_data_path):
    page = loader.best
    if page is None:
        raise ValueError("loader.best is None")

    axtree = page.files.get(FileType.AXTree)
    bb_map = page.files.get(FileType.BB)

    nodes = axtree["nodes"]
    nodes_by_id, children_by_id, parent_by_id, root_id = build_indexes(nodes)
    depths = compute_depths(root_id, children_by_id)
    subtree_stats = compute_subtree_stats(root_id, nodes_by_id, children_by_id)
    page_area = estimate_page_area(bb_map)

    raw_candidate_ids = []
    for node_id in nodes_by_id:
        ok = basic_candidate_filter(
            node_id=node_id,
            nodes_by_id=nodes_by_id,
            children_by_id=children_by_id,
            parent_by_id=parent_by_id,
            depths=depths,
            subtree_stats=subtree_stats,
            bb_map=bb_map,
            page_area=page_area,
        )
        if ok:
            raw_candidate_ids.append(node_id)


    final_candidate_ids = suppress_hierarchy_duplicates_v2(
        candidate_ids=raw_candidate_ids,
        nodes_by_id=nodes_by_id,
        children_by_id=children_by_id,
        subtree_stats=subtree_stats,
        bb_map=bb_map,
        text_ratio_threshold=0.75,
        area_ratio_threshold=0.55,
        contain_threshold=0.90,
        only_child_text_ratio_threshold=0.60,
    )

    suppressed_ids = [x for x in raw_candidate_ids if x not in set(final_candidate_ids)]

    return {
        "raw_ids": raw_candidate_ids,
        "final_ids": final_candidate_ids,
        "suppressed_ids": suppressed_ids,
        "nodes_by_id": nodes_by_id,
        "children_by_id": children_by_id,
        "parent_by_id": parent_by_id,
        "subtree_stats": subtree_stats,
        "bb_map": bb_map,
        "depths": depths,
        "page_area": page_area,
    }