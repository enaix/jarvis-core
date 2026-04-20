import base64
import io
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image


st.set_page_config(page_title="AXTree Annotator", layout="wide")


DEFAULT_CANDIDATES_PATH = r"notebooks\webui\output\candidate\candidates.jsonl"
DEFAULT_LABELS_PATH = r"notebooks\webui\annotator\output\labels.jsonl"
DEFAULT_DATASET_ROOT = r"C:\Users\70133\.cache\huggingface\hub\datasets--biglab--webui-7k\snapshots\60f7b3c4b9409f75551664adc1564625dfc33c2e\dataset1"


def load_jsonl(path: str):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at line {line_num}: {e}") from e
    return rows


def load_labels(path: str):
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
                raise ValueError(f"Labels JSON decode error at line {line_num}: {e}") from e
    return rows


def append_jsonl(path: str, row: dict):
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_key(page_id, node_id):
    return f"{page_id}::{node_id}"


def labels_to_lookup(labels):
    lookup = {}
    for row in labels:
        key = make_key(row.get("page_id"), row.get("node_id"))
        lookup[key] = row
    return lookup


def compute_unlabeled_indices(candidates, label_lookup):
    indices = []
    for i, row in enumerate(candidates):
        key = make_key(row.get("page_id"), row.get("node_id"))
        if key not in label_lookup:
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
        window.addEventListener("load", scrollToBBox);
        window.addEventListener("resize", scrollToBBox);
    }})();
    </script>
    """
    return html


def get_current_unlabeled_candidate():
    unlabeled_indices = st.session_state.get("unlabeled_indices", [])
    cursor = st.session_state.get("cursor", 0)
    data = st.session_state.get("data", [])

    if not unlabeled_indices:
        return None, None

    cursor = max(0, min(cursor, len(unlabeled_indices) - 1))
    st.session_state["cursor"] = cursor

    data_index = unlabeled_indices[cursor]
    return data_index, data[data_index]


def refresh_unlabeled_indices():
    data = st.session_state.get("data", [])
    labels_lookup = st.session_state.get("labels_lookup", {})
    old_data_index, _ = get_current_unlabeled_candidate()

    unlabeled_indices = compute_unlabeled_indices(data, labels_lookup)
    st.session_state["unlabeled_indices"] = unlabeled_indices

    if not unlabeled_indices:
        st.session_state["cursor"] = 0
        return

    if old_data_index is None:
        st.session_state["cursor"] = 0
        return

    next_cursor = 0
    for pos, idx in enumerate(unlabeled_indices):
        if idx >= old_data_index:
            next_cursor = pos
            break
    else:
        next_cursor = len(unlabeled_indices) - 1

    st.session_state["cursor"] = next_cursor


def go_next():
    unlabeled_indices = st.session_state.get("unlabeled_indices", [])
    if unlabeled_indices and st.session_state["cursor"] < len(unlabeled_indices) - 1:
        st.session_state["cursor"] += 1


def go_prev():
    unlabeled_indices = st.session_state.get("unlabeled_indices", [])
    if unlabeled_indices and st.session_state["cursor"] > 0:
        st.session_state["cursor"] -= 1


def save_label(label_value: str):
    labels_path = st.session_state["labels_path"]
    data_index, row = get_current_unlabeled_candidate()
    if row is None:
        return

    label_row = {
        "page_id": row.get("page_id"),
        "node_id": row.get("node_id"),
        "label": label_value,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "role": row.get("role"),
        "text_len": row.get("text_len"),
    }

    append_jsonl(labels_path, label_row)

    key = make_key(row.get("page_id"), row.get("node_id"))
    st.session_state["labels_lookup"][key] = label_row
    refresh_unlabeled_indices()


def load_all(candidates_path: str, labels_path: str, dataset_root: str):
    data = load_jsonl(candidates_path)
    labels = load_labels(labels_path)
    labels_lookup = labels_to_lookup(labels)
    unlabeled_indices = compute_unlabeled_indices(data, labels_lookup)

    st.session_state["candidates_path"] = candidates_path
    st.session_state["labels_path"] = labels_path
    st.session_state["dataset_root"] = dataset_root
    st.session_state["data"] = data
    st.session_state["labels_lookup"] = labels_lookup
    st.session_state["unlabeled_indices"] = unlabeled_indices
    st.session_state["cursor"] = 0


st.title("AXTree Annotator")

candidates_path = st.text_input("Path to candidates.jsonl", value=DEFAULT_CANDIDATES_PATH)
labels_path = st.text_input("Path to labels.jsonl", value=DEFAULT_LABELS_PATH)
dataset_root = st.text_input("Dataset root", value=DEFAULT_DATASET_ROOT)

if st.button("Load", width="stretch"):
    try:
        load_all(candidates_path, labels_path, dataset_root)
        st.success("Loaded candidates and labels")
    except Exception as e:
        st.error(str(e))

data = st.session_state.get("data", [])
labels_lookup = st.session_state.get("labels_lookup", {})
unlabeled_indices = st.session_state.get("unlabeled_indices", [])

if data:
    total_count = len(data)
    labeled_count = len(labels_lookup)
    remaining_count = len(unlabeled_indices)

    st.write(
        f"Total: **{total_count}** | "
        f"Labeled: **{labeled_count}** | "
        f"Remaining: **{remaining_count}**"
    )

    if remaining_count == 0:
        st.success("All candidates in this file are labeled.")
    else:
        data_index, row = get_current_unlabeled_candidate()
        cursor = st.session_state.get("cursor", 0)

        st.write(
            f"Current unlabeled item: **{cursor + 1} / {remaining_count}** "
            f"(data index = {data_index})"
        )

        screenshot_path = row.get("screenshot_path")
        bbox = row.get("bbox")

        try:
            if screenshot_path and bbox:
                base_img = load_image(screenshot_path, st.session_state.get("dataset_root"))
                component_key = f"{row.get('page_id')}_{row.get('node_id')}"
                html = make_scrollable_bbox_html(
                    base_img,
                    bbox,
                    component_key=component_key,
                    container_height=780,
                    top_margin=20,
                )
                st.components.v1.html(html, height=800, scrolling=False)
            else:
                st.warning("This row has no screenshot_path or bbox")
        except Exception as e:
            st.error(str(e))

        nav1, nav2, nav3, nav4, nav5, nav6 = st.columns(6)

        with nav1:
            if st.button("Back", width="stretch"):
                go_prev()
                st.rerun()

        with nav2:
            if st.button("Junk", width="stretch"):
                save_label("junk")
                st.rerun()

        with nav3:
            if st.button("Not junk", width="stretch"):
                save_label("not_junk")
                st.rerun()

        with nav4:
            if st.button("Bad candidate", width="stretch"):
                save_label("bad_candidate")
                st.rerun()

        with nav5:
            if st.button("Skip", width="stretch"):
                save_label("skip")
                st.rerun()

        with nav6:
            if st.button("Reload labels", width="stretch"):
                try:
                    fresh_labels = load_labels(st.session_state["labels_path"])
                    st.session_state["labels_lookup"] = labels_to_lookup(fresh_labels)
                    refresh_unlabeled_indices()
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        st.markdown("---")
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
            key=f"text_subtree_{row.get('page_id')}_{row.get('node_id')}",
        )

else:
    st.info("Enter paths and click Load")