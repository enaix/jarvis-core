"""
Pre-flight #6 — sanity check AX→widget mapping.

Цель: убедиться, что AX-кандидаты (на которых мы размечаем junk/not_junk)
ясно отображаются в widget tree (на котором применяется фильтрация).

Делает две проверки:

1. Аналитическая (быстро, без загрузки страниц):
   - Распределение candidate ролей по категориям ax_to_widget:
     * leaf (text/link/image) — кандидат станет leaf widget
     * container — кандидат может стать BLOCK widget или collapse-нуться
     * unknown — кандидат с ролью вне известных категорий
   - Это даёт первичную оценку: 1:1 mapping ожидается?

2. Реальный pipeline на 10 sample кандидатах:
   - Для каждого: load AXTree → build_tree → transform_tree → ax_to_widget
   - Считаем: появился ли node_id кандидата в widget tree
   - Считаем: collapsed away (не попал в widgets), present (1+ widgets)

Запуск:
    .venv\Scripts\python.exe notebooks\webui\check_ax_to_widget_mapping.py
"""

import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # notebooks/webui/

from widget_spec import (
    Widget, WidgetType, WidgetHint,
    LINK_ROLES, MAIN_ROLES,
    TEXT_ROLES, IMAGE_ROLES, CONTAINER_ROLES,
    role_to_leaf_type, role_is_container,
)


# === Конфигурация ===
DATASET_ROOT = Path(
    r"C:\Users\70133\.cache\huggingface\hub\datasets--biglab--webui-7k\snapshots\60f7b3c4b9409f75551664adc1564625dfc33c2e\dataset1"
)
CANDIDATES_FILE = Path(r"notebooks\webui\annotator\candidates_topic_filtered.jsonl")
N_PIPELINE_CHECK = 10
RANDOM_SEED = 42


def categorize_role(role):
    """Возвращает в какой категории widget'а окажется роль."""
    if role in TEXT_ROLES:
        return "text_leaf"
    if role in LINK_ROLES:
        return "link_leaf"
    if role in IMAGE_ROLES:
        return "image_leaf"
    if role in CONTAINER_ROLES:
        return "container"
    return "unknown"


# === Pipeline функции (выпарованы из loader_convert.ipynb для standalone скрипта) ===

def build_tree(node, nodes_by_id):
    children = []
    for child_id in node.get("childIds", []):
        child_id = str(child_id)
        child_node = nodes_by_id.get(child_id)
        if child_node:
            children.append(build_tree(child_node, nodes_by_id))

    role_field = node.get("role", {})
    role_value = role_field.get("value", "") if isinstance(role_field, dict) else (role_field or "")
    name_field = node.get("name", {})
    name_value = name_field.get("value", "") if isinstance(name_field, dict) else (name_field or "")

    return {
        "id": str(node["nodeId"]),
        "back_id": node.get("backendDOMNodeId"),
        "ignored": node.get("ignored"),
        "name": name_value,
        "role": role_value,
        "modified": False,
        "children": children,
    }


def is_useless_branch(node):
    if not node.get("ignored", False):
        return False
    if node.get("role") != "none":
        return False
    if not node.get("children"):
        return True
    return all(is_useless_branch(c) for c in node["children"])


def transform_tree(node):
    cleaned_children = []
    for child in node.get("children", []):
        if not is_useless_branch(child):
            cleaned_children.append(child)
    node["children"] = cleaned_children

    new_children = []
    skip = False
    if len(node["children"]) == 1:
        child = node["children"][0]
        if child["role"] in ["StaticText", "link", "heading", "paragraph", "gridcell"] and child["name"]:
            node["name"] = child["name"]
            node["modified"] = True
            skip = True

    if not skip:
        for child in node["children"]:
            if is_useless_branch(child):
                continue
            new_children.append(transform_tree(child))
    node["children"] = new_children
    return node


def ax_to_widget(node, bb=None):
    role = node.get("role", "") or ""
    name = (node.get("name", "") or "").strip()

    child_widgets = []
    for child in node.get("children", []):
        w = ax_to_widget(child, bb)
        if w is not None:
            child_widgets.append(w)

    leaf_type = role_to_leaf_type(role)

    if role in LINK_ROLES:
        if not child_widgets and not name:
            return None
        return Widget(
            type=WidgetType.LINK, name=name, children=child_widgets,
            node_id=node.get("id"), back_id=node.get("back_id"), role=role,
        )

    if leaf_type is not None and not child_widgets:
        if leaf_type == WidgetType.TEXT and not name:
            return None
        return Widget(
            type=leaf_type, name=name,
            node_id=node.get("id"), back_id=node.get("back_id"), role=role,
        )

    if not child_widgets:
        return None

    if len(child_widgets) == 1 and role_is_container(role):
        w = child_widgets[0]
        if role in MAIN_ROLES:
            w.add_hint(WidgetHint.MAIN)
        return w

    w = Widget(
        type=WidgetType.BLOCK, name=name, children=child_widgets,
        node_id=node.get("id"), back_id=node.get("back_id"), role=role,
    )
    if role in MAIN_ROLES:
        w.add_hint(WidgetHint.MAIN)
    return w


# === Пайплайн загрузки страницы ===

def find_axtree_for_page(page_id):
    for split_dir in DATASET_ROOT.iterdir():
        if not split_dir.is_dir():
            continue
        page_dir = split_dir / page_id
        if not page_dir.is_dir():
            continue
        for f in page_dir.iterdir():
            if f.is_file() and f.name.endswith("axtree.json.gz") and "1920" in f.name:
                return f
        # fallback to any
        for f in page_dir.iterdir():
            if f.is_file() and f.name.endswith("axtree.json.gz"):
                return f
    return None


def collect_widget_node_ids(widget):
    """Walks widget tree, returns set of node_ids."""
    if widget is None:
        return set()
    ids = {str(widget.node_id)} if widget.node_id is not None else set()
    for c in widget.children:
        ids |= collect_widget_node_ids(c)
    return ids


# === Main ===

def main():
    if not CANDIDATES_FILE.exists():
        print(f"ERROR: {CANDIDATES_FILE} not found")
        sys.exit(1)

    with CANDIDATES_FILE.open(encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(candidates)} candidates")

    # === 1. Analytical: распределение ролей по категориям ===
    print("\n=== 1. Analytical: candidate roles vs widget categories ===")
    cats = Counter()
    role_to_cat = {}
    for c in candidates:
        role = c.get("role", "")
        cat = categorize_role(role)
        cats[cat] += 1
        role_to_cat[role] = cat

    for cat, count in cats.most_common():
        pct = 100.0 * count / len(candidates)
        print(f"  {cat:15s}: {count:5d} ({pct:.1f}%)")

    # show roles in unknown
    unknown_roles = [r for r, c in role_to_cat.items() if c == "unknown"]
    if unknown_roles:
        print(f"\n  Unknown roles (not in TEXT/LINK/IMAGE/CONTAINER sets):")
        unknown_role_counts = Counter(c.get("role", "") for c in candidates if categorize_role(c.get("role", "")) == "unknown")
        for r, n in unknown_role_counts.most_common():
            print(f"    {r}: {n}")
    else:
        print("  All candidate roles fall into known widget categories.")

    # === 2. Pipeline: 10 random samples ===
    print(f"\n=== 2. Pipeline check on {N_PIPELINE_CHECK} sample candidates ===")
    random.seed(RANDOM_SEED)

    # Stratified sample: pick from different roles
    by_role = defaultdict(list)
    for c in candidates:
        by_role[c.get("role", "")].append(c)
    # Pick top roles
    top_roles = [r for r, _ in Counter(c.get("role", "") for c in candidates).most_common(10)]

    samples = []
    for r in top_roles:
        if by_role[r] and len(samples) < N_PIPELINE_CHECK:
            samples.append(random.choice(by_role[r]))
    while len(samples) < N_PIPELINE_CHECK:
        samples.append(random.choice(candidates))

    pipeline_results = Counter()
    for i, c in enumerate(samples, 1):
        page_id = str(c["page_id"])
        node_id = str(c["node_id"])
        role = c.get("role", "")

        ax_path = find_axtree_for_page(page_id)
        if ax_path is None:
            print(f"[{i}/{N_PIPELINE_CHECK}] {page_id}/{node_id} (role={role}): AXTree not found")
            pipeline_results["axtree_missing"] += 1
            continue

        try:
            with gzip.open(ax_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            nodes = data.get("nodes", [])
            nodes_by_id = {str(n["nodeId"]): n for n in nodes}

            root = next(
                (n for n in nodes if n.get("parentId") in (None, "")),
                None,
            )
            if root is None:
                print(f"[{i}/{N_PIPELINE_CHECK}] {page_id}/{node_id}: no root node")
                pipeline_results["no_root"] += 1
                continue

            tree = build_tree(root, nodes_by_id)
            tree = transform_tree(tree)
            widget = ax_to_widget(tree)

            widget_ids = collect_widget_node_ids(widget) if widget else set()

            in_widgets = node_id in widget_ids
            text_preview = (c.get("text_subtree", "") or "")[:80].replace("\n", " ")

            if in_widgets:
                pipeline_results["mapped_1to1"] += 1
                status = "✓ MAPPED"
            else:
                pipeline_results["collapsed_or_missing"] += 1
                status = "✗ NOT IN WIDGETS"
            print(f"[{i}/{N_PIPELINE_CHECK}] {page_id}/{node_id} role={role:<15s} {status} | {text_preview!r}")

        except Exception as e:
            print(f"[{i}/{N_PIPELINE_CHECK}] {page_id}/{node_id}: ERROR {type(e).__name__}: {e}")
            pipeline_results["error"] += 1

    print("\n=== Pipeline summary ===")
    for k, v in pipeline_results.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
