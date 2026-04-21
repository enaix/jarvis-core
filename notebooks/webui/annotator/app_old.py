import json
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw


st.set_page_config(page_title="AXTree Annotator", layout="wide")


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


def draw_bbox(image_path: str, bbox: dict):
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Screenshot not found: {img_path}")

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    x = bbox.get("x", 0)
    y = bbox.get("y", 0)
    w = bbox.get("width", 0)
    h = bbox.get("height", 0)

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    for offset in range(4):
        draw.rectangle(
            [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
            outline="red",
            width=2,
        )

    return img


def go_next():
    if "data" not in st.session_state:
        return
    if st.session_state["current_index"] < len(st.session_state["data"]) - 1:
        st.session_state["current_index"] += 1


st.title("AXTree Annotator")

default_path = r"notebooks\webui\output\candidate\1655885744591_candidates.jsonl"
jsonl_path = st.text_input("Path to candidates.jsonl", value=default_path)

col_load_1, col_load_2 = st.columns([1, 5])

with col_load_1:
    if st.button("Load file"):
        try:
            data = load_jsonl(jsonl_path)
            st.session_state["data"] = data
            st.session_state["jsonl_path"] = jsonl_path
            st.session_state["current_index"] = 0
            st.success(f"Loaded {len(data)} rows")
        except Exception as e:
            st.error(str(e))

data = st.session_state.get("data", [])

if data:
    current_index = st.session_state.get("current_index", 0)
    row = data[current_index]

    st.write(f"Item {current_index + 1} / {len(data)}")

    screenshot_path = row.get("screenshot_path")
    bbox = row.get("bbox")

    try:
        if screenshot_path and bbox:
            img_with_box = draw_bbox(screenshot_path, bbox)
            st.image(img_with_box, use_container_width=True)
        else:
            st.warning("This row has no screenshot_path or bbox")
    except Exception as e:
        st.error(str(e))

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Useful", use_container_width=True):
            go_next()
            st.rerun()

    with col2:
        if st.button("Useless", use_container_width=True):
            go_next()
            st.rerun()

    with col3:
        if st.button("Skip", use_container_width=True):
            go_next()
            st.rerun()

    st.caption(
        f"page_id={row.get('page_id')} | "
        f"node_id={row.get('node_id')} | "
        f"role={row.get('role')}"
    )

else:
    st.info("Enter path and click 'Load file'")