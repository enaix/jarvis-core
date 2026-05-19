import base64
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image


st.set_page_config(page_title="AXTree Annotator", layout="wide")


DEFAULT_CANDIDATES_PATH = r"notebooks\webui\annotator\candidates_topic_filtered.jsonl"
DEFAULT_LABELS_PATH = r"notebooks\webui\annotator\output\labels.jsonl"
DEFAULT_PAGE_FLAGS_PATH = r"notebooks\webui\annotator\output\page_flags.jsonl"
DEFAULT_DATASET_ROOT = r"C:\Users\70133\.cache\huggingface\hub\datasets--biglab--webui-7k\snapshots\60f7b3c4b9409f75551664adc1564625dfc33c2e\dataset1"
GENERATOR_SOURCE_PATH = Path(__file__).resolve().parent.parent / "build_candidates.py"

SUB_LABELS = ["cookie", "footer", "sidebar", "nav", "social", "legal", "ad", "recurring", "other"]
DEFAULT_SUB_LABEL = "other"


def compute_generator_version() -> str:
    """md5 от build_candidates.py — меняется когда меняется логика генератора."""
    try:
        with GENERATOR_SOURCE_PATH.open("rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:12]
    except FileNotFoundError:
        return "unknown"


def compute_content_hash(role, text_subtree) -> str:
    """md5 от role + text_subtree — стабильный ID кандидата при смене генератора."""
    payload = f"{role or ''}::{text_subtree or ''}".encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:16]


GENERATOR_VERSION = compute_generator_version()


def load_jsonl(path: str):
    file_path = Path(path)
    if not file_path.exists():
        return []

    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at line {line_num} in {file_path}: {e}") from e
    return rows


def write_jsonl(path: str, rows: list[dict]):
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str, row: dict):
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_candidate_key(page_id, node_id):
    return f"{str(page_id)}::{str(node_id)}"


def build_labeled_lookup(label_rows):
    lookup = {}
    for row in label_rows:
        key = make_candidate_key(row.get("page_id"), row.get("node_id"))
        lookup[key] = row
    return lookup


def build_bad_pages_set(page_rows):
    bad_pages = set()
    for row in page_rows:
        if row.get("page_action") == "bad_page":
            bad_pages.add(str(row.get("page_id")))
    return bad_pages


def compute_visible_indices(candidates, labeled_lookup, bad_pages):
    indices = []
    for i, row in enumerate(candidates):
        page_id = str(row.get("page_id"))
        node_id = row.get("node_id")
        key = make_candidate_key(page_id, node_id)

        if key in labeled_lookup:
            continue
        if page_id in bad_pages:
            continue

        indices.append(i)
    return indices


def resolve_image_path(image_path: str, dataset_root: str | None = None) -> Path:
    img_path = Path(image_path)
    if img_path.is_absolute():
        return img_path
    if dataset_root:
        return Path(dataset_root) / img_path
    return img_path


def load_image(image_path: str, dataset_root: str | None = None):
    img_path = resolve_image_path(image_path, dataset_root)
    if not img_path.exists():
        raise FileNotFoundError(f"Screenshot not found: {img_path}")
    return Image.open(img_path).convert("RGB")


def image_to_data_url(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def make_scrollable_bbox_html(
    img: Image.Image,
    bbox: dict,
    component_key: str,
    container_height: int = 780,
    top_margin: int = 20,
) -> str:
    img_w, img_h = img.size

    x = float(bbox.get("x", 0))
    y = float(bbox.get("y", 0))
    w = float(bbox.get("width", 0))
    h = float(bbox.get("height", 0))

    data_url = image_to_data_url(img)

    container_id = f"scroll-container-{component_key}"
    svg_id = f"page-svg-{component_key}"

    html = f"""
    <div id="{container_id}"
         style="
            height: {container_height}px;
            overflow-y: auto;
            overflow-x: hidden;
            border: 1px solid #444;
            border-radius: 10px;
            background: #111;
         ">
        <svg
            id="{svg_id}"
            viewBox="0 0 {img_w} {img_h}"
            xmlns="http://www.w3.org/2000/svg"
            style="
                display: block;
                width: 100%;
                height: auto;
                background: #111;
             "
        >
            <image
                href="{data_url}"
                x="0"
                y="0"
                width="{img_w}"
                height="{img_h}"
            />
            <rect
                x="{x}"
                y="{y}"
                width="{w}"
                height="{h}"
                fill="none"
                stroke="red"
                stroke-width="6"
            />
        </svg>
    </div>

    <script>
    (function() {{
        const container = document.getElementById("{container_id}");
        const svg = document.getElementById("{svg_id}");

        function scrollToBBox() {{
            if (!container || !svg) return;
            const displayedWidth = svg.clientWidth;
            if (!displayedWidth) return;

            const scale = displayedWidth / {img_w};
            const targetTop = Math.max(0, ({y} * scale) - {top_margin});
            container.scrollTop = targetTop;
        }}

        setTimeout(scrollToBBox, 80);
        setTimeout(scrollToBBox, 250);
        setTimeout(scrollToBBox, 500);
        window.addEventListener("load", scrollToBBox);
        window.addEventListener("resize", scrollToBBox);
    }})();
    </script>
    """
    return html


def get_row(data_index):
    data = st.session_state.get("data", [])
    if data_index is None:
        return None
    if data_index < 0 or data_index >= len(data):
        return None
    return data[data_index]


def refresh_state():
    data = st.session_state.get("data", [])
    label_rows = load_jsonl(st.session_state["labels_path"])
    page_rows = load_jsonl(st.session_state["page_flags_path"])

    st.session_state["label_rows"] = label_rows
    st.session_state["page_rows"] = page_rows
    st.session_state["labeled_lookup"] = build_labeled_lookup(label_rows)
    st.session_state["bad_pages"] = build_bad_pages_set(page_rows)
    st.session_state["visible_indices"] = compute_visible_indices(
        data,
        st.session_state["labeled_lookup"],
        st.session_state["bad_pages"],
    )


def find_first_visible():
    visible = st.session_state.get("visible_indices", [])
    return visible[0] if visible else None


def find_prev_index(current_index):
    if current_index is None:
        return None
    return current_index - 1 if current_index > 0 else 0


def find_next_index(current_index):
    data = st.session_state.get("data", [])
    if current_index is None:
        return 0 if data else None
    if current_index < len(data) - 1:
        return current_index + 1
    return len(data) - 1 if data else None


def find_next_visible_different_page(current_index):
    current_row = get_row(current_index)
    if current_row is None:
        return None

    current_page_id = str(current_row.get("page_id"))
    visible = st.session_state.get("visible_indices", [])
    data = st.session_state.get("data", [])

    for idx in visible:
        if idx <= current_index:
            continue
        row = data[idx]
        if str(row.get("page_id")) != current_page_id:
            return idx

    for idx in visible:
        row = data[idx]
        if str(row.get("page_id")) != current_page_id:
            return idx

    return None


def go_prev():
    prev_idx = find_prev_index(st.session_state.get("current_data_index"))
    if prev_idx is not None:
        st.session_state["current_data_index"] = prev_idx


def go_next():
    next_idx = find_next_index(st.session_state.get("current_data_index"))
    if next_idx is not None:
        st.session_state["current_data_index"] = next_idx


def current_candidate_label():
    row = get_row(st.session_state.get("current_data_index"))
    if row is None:
        return None
    key = make_candidate_key(row.get("page_id"), row.get("node_id"))
    return st.session_state.get("labeled_lookup", {}).get(key)


def current_page_is_bad():
    row = get_row(st.session_state.get("current_data_index"))
    if row is None:
        return False
    return str(row.get("page_id")) in st.session_state.get("bad_pages", set())


def save_candidate_label(label_value: str, sub_label: str = ""):
    current_index = st.session_state.get("current_data_index")
    row = get_row(current_index)
    if row is None:
        return

    key = make_candidate_key(row.get("page_id"), row.get("node_id"))
    if key in st.session_state.get("labeled_lookup", {}):
        return

    label_row = {
        "page_id": str(row.get("page_id")),
        "node_id": str(row.get("node_id")),
        "label": label_value,
        "sub_label": sub_label if label_value == "junk" else "",
        "content_hash": compute_content_hash(row.get("role"), row.get("text_subtree", "")),
        "generator_version": GENERATOR_VERSION,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "role": row.get("role"),
        "text_len": row.get("text_len"),
    }

    append_jsonl(st.session_state["labels_path"], label_row)
    refresh_state()

    next_idx = find_next_index(current_index)
    if next_idx is not None:
        st.session_state["current_data_index"] = next_idx


def undo_current_label():
    current_index = st.session_state.get("current_data_index")
    row = get_row(current_index)
    if row is None:
        return

    page_id = str(row.get("page_id"))
    node_id = str(row.get("node_id"))
    key = make_candidate_key(page_id, node_id)

    label_rows = load_jsonl(st.session_state["labels_path"])
    new_rows = []
    removed = False

    for item in label_rows:
        item_key = make_candidate_key(item.get("page_id"), item.get("node_id"))
        if item_key == key and not removed:
            removed = True
            continue
        new_rows.append(item)

    if removed:
        write_jsonl(st.session_state["labels_path"], new_rows)
        refresh_state()


def mark_current_page_bad():
    current_index = st.session_state.get("current_data_index")
    row = get_row(current_index)
    if row is None:
        return

    page_id = str(row.get("page_id"))
    if page_id in st.session_state.get("bad_pages", set()):
        return

    page_row = {
        "page_id": page_id,
        "page_action": "bad_page",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    append_jsonl(st.session_state["page_flags_path"], page_row)
    refresh_state()

    next_idx = find_next_visible_different_page(current_index)
    st.session_state["current_data_index"] = next_idx


def undo_current_bad_page():
    current_index = st.session_state.get("current_data_index")
    row = get_row(current_index)
    if row is None:
        return

    page_id = str(row.get("page_id"))
    page_rows = load_jsonl(st.session_state["page_flags_path"])

    new_rows = []
    removed = False
    for item in page_rows:
        if str(item.get("page_id")) == page_id and item.get("page_action") == "bad_page" and not removed:
            removed = True
            continue
        new_rows.append(item)

    if removed:
        write_jsonl(st.session_state["page_flags_path"], new_rows)
        refresh_state()


def load_all(candidates_path: str, labels_path: str, page_flags_path: str, dataset_root: str):
    candidates = load_jsonl(candidates_path)

    st.session_state["candidates_path"] = candidates_path
    st.session_state["labels_path"] = labels_path
    st.session_state["page_flags_path"] = page_flags_path
    st.session_state["dataset_root"] = dataset_root
    st.session_state["data"] = candidates

    refresh_state()

    first_visible = find_first_visible()
    st.session_state["current_data_index"] = first_visible if first_visible is not None else (0 if candidates else None)


def get_page_candidate_indices(page_id):
    data = st.session_state.get("data", [])
    page_id = str(page_id)
    return [i for i, row in enumerate(data) if str(row.get("page_id")) == page_id]


def get_page_labeled_indices(page_id):
    data = st.session_state.get("data", [])
    labeled_lookup = st.session_state.get("labeled_lookup", {})
    page_id = str(page_id)

    result = []
    for i, row in enumerate(data):
        if str(row.get("page_id")) != page_id:
            continue
        key = make_candidate_key(row.get("page_id"), row.get("node_id"))
        if key in labeled_lookup:
            result.append(i)
    return result


def get_page_visible_indices(page_id):
    data = st.session_state.get("data", [])
    visible_indices = st.session_state.get("visible_indices", [])
    page_id = str(page_id)

    result = []
    for i in visible_indices:
        if str(data[i].get("page_id")) == page_id:
            result.append(i)
    return result


def get_page_progress_info(current_data_index):
    row = get_row(current_data_index)
    if row is None:
        return None

    page_id = str(row.get("page_id"))
    all_indices = get_page_candidate_indices(page_id)
    labeled_indices = get_page_labeled_indices(page_id)
    visible_indices = get_page_visible_indices(page_id)

    current_pos_full = None
    if all_indices:
        current_pos_full = all_indices.index(current_data_index) + 1

    current_pos_visible = None
    if current_data_index in visible_indices:
        current_pos_visible = visible_indices.index(current_data_index) + 1

    return {
        "page_id": page_id,
        "total_on_page": len(all_indices),
        "labeled_on_page": len(labeled_indices),
        "remaining_on_page": len(visible_indices),
        "current_pos_full": current_pos_full,
        "current_pos_visible": current_pos_visible,
    }


st.title("AXTree Annotator")

with st.expander("Paths (click to expand)", expanded=("data" not in st.session_state)):
    candidates_path = st.text_input("Path to candidates.jsonl", value=DEFAULT_CANDIDATES_PATH)
    labels_path = st.text_input("Path to labels.jsonl", value=DEFAULT_LABELS_PATH)
    page_flags_path = st.text_input("Path to page_flags.jsonl", value=DEFAULT_PAGE_FLAGS_PATH)
    dataset_root = st.text_input("Dataset root", value=DEFAULT_DATASET_ROOT)

    if st.button("Load", width="stretch"):
        try:
            load_all(candidates_path, labels_path, page_flags_path, dataset_root)
            st.success("Loaded candidates, labels, and page flags")
        except Exception as e:
            st.error(str(e))

data = st.session_state.get("data", [])

if data:
    labeled_lookup = st.session_state.get("labeled_lookup", {})
    visible_indices = st.session_state.get("visible_indices", [])
    bad_pages = st.session_state.get("bad_pages", set())

    st.write(
        f"Total: **{len(data)}** | "
        f"Labeled candidates: **{len(labeled_lookup)}** | "
        f"Remaining candidates: **{len(visible_indices)}** | "
        f"Bad pages: **{len(bad_pages)}**"
    )
    st.caption(f"Generator version: `{GENERATOR_VERSION}`")

    current_data_index = st.session_state.get("current_data_index")
    row = get_row(current_data_index)

    if row is None:
        st.success("No current item.")
    else:
        page_info = get_page_progress_info(current_data_index)

        st.write(
            f"Current data index: **{current_data_index}** | "
            f"page_id: **{row.get('page_id')}** | "
            f"node_id: **{row.get('node_id')}**"
        )

        if page_info is not None:
            st.write(
                f"Page progress | "
                f"Total on page: **{page_info['total_on_page']}** | "
                f"Labeled on page: **{page_info['labeled_on_page']}** | "
                f"Remaining on page: **{page_info['remaining_on_page']}**"
            )
            if page_info["current_pos_full"] is not None:
                st.write(
                    f"Current candidate on page (full order): "
                    f"**{page_info['current_pos_full']} / {page_info['total_on_page']}**"
                )
            if page_info["current_pos_visible"] is not None:
                st.write(
                    f"Current candidate on page (remaining only): "
                    f"**{page_info['current_pos_visible']} / {page_info['remaining_on_page']}**"
                )

        screenshot_path = row.get("screenshot_path")
        bbox = row.get("bbox")

        try:
            if screenshot_path and bbox:
                base_img = load_image(screenshot_path, st.session_state.get("dataset_root"))
                component_key = f"{row.get('page_id')}_{row.get('node_id')}_{current_data_index}"
                html = make_scrollable_bbox_html(
                    base_img,
                    bbox,
                    component_key=component_key,
                    container_height=550,
                    top_margin=20,
                )
                st.components.v1.html(html, height=570, scrolling=False)
            else:
                st.warning("This row has no screenshot_path or bbox")
        except Exception as e:
            st.error(str(e))

        current_label = current_candidate_label()
        is_page_bad = current_page_is_bad()

        if current_label is not None:
            label_str = current_label.get("label", "")
            sub_str = current_label.get("sub_label", "")
            if label_str == "junk" and sub_str:
                st.info(f"Current label: **{label_str} / {sub_str}**")
            else:
                st.info(f"Current label: **{label_str}**")

        if is_page_bad:
            st.info("This page is marked as **bad_page**.")

        sub_label = st.radio(
            "Sub-label (для Junk)",
            options=SUB_LABELS,
            index=SUB_LABELS.index(DEFAULT_SUB_LABEL),
            horizontal=True,
            key="sub_label_radio",
            disabled=(current_label is not None or is_page_bad),
        )

        nav1, nav2, nav3, nav4, nav5, nav6 = st.columns(6)

        with nav1:
            if st.button("Prev", width="stretch"):
                go_prev()
                st.rerun()

        with nav2:
            if st.button("Next", width="stretch"):
                go_next()
                st.rerun()

        with nav3:
            if st.button("Junk", width="stretch", disabled=(current_label is not None or is_page_bad)):
                save_candidate_label("junk", sub_label=sub_label)
                st.rerun()

        with nav4:
            if st.button("Not junk", width="stretch", disabled=(current_label is not None or is_page_bad)):
                save_candidate_label("not_junk")
                st.rerun()

        with nav5:
            if st.button("Skip", width="stretch", disabled=(current_label is not None or is_page_bad)):
                save_candidate_label("skip")
                st.rerun()

        with nav6:
            if st.button("Bad page", width="stretch", disabled=is_page_bad):
                mark_current_page_bad()
                st.rerun()

        undo1, undo2, reload_col = st.columns(3)

        with undo1:
            if st.button("Undo current label", width="stretch", disabled=(current_label is None)):
                undo_current_label()
                st.rerun()

        with undo2:
            if st.button("Undo bad page", width="stretch", disabled=(not is_page_bad)):
                undo_current_bad_page()
                st.rerun()

        with reload_col:
            if st.button("Reload", width="stretch"):
                refresh_state()
                st.rerun()

        st.markdown("---")

        # Global recurrence (по всему WebUI 5420 pages) — основной сигнал
        grec = row.get("global_recurrence_count")
        gratio = row.get("global_recurrence_ratio")
        # Local recurrence (по нашим 201 corpus pages) — supplementary
        lrec = row.get("recurrence_count")

        if grec is not None:
            ratio_str = f"{gratio*100:.2f}%" if gratio is not None else ""
            if grec >= 10:
                st.warning(
                    f"**Global recurrence: {grec} pages ({ratio_str})** — almost certainly junk (cross-site boilerplate)"
                )
            elif grec >= 5:
                st.warning(
                    f"**Global recurrence: {grec} pages ({ratio_str})** — likely junk (recurring across sites)"
                )
            elif grec >= 2:
                st.info(f"Global recurrence: {grec} pages ({ratio_str})")
            else:
                st.caption(f"Global recurrence: {grec or 1} page (unique block in WebUI)")

        if lrec is not None and lrec >= 2:
            st.caption(f"Local recurrence (in our 201-page corpus): {lrec} pages")

        st.markdown("**Metadata**")
        st.write(f"page_id: {row.get('page_id')}")
        st.write(f"node_id: {row.get('node_id')}")
        st.write(f"role: {row.get('role')}")
        st.write(f"text_len: {row.get('text_len')}")
        st.write(f"num_descendants: {row.get('num_descendants')}")
        st.write(f"link_density: {row.get('link_density')}")

        st.markdown("**text_subtree**")
        st.text_area(
            "text_subtree",
            value=row.get("text_subtree", ""),
            height=260,
            key=f"text_subtree_{row.get('page_id')}_{row.get('node_id')}_{current_data_index}",
        )

else:
    st.info("Enter paths and click Load")