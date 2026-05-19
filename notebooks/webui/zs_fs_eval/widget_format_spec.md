# Widget Format Specification (generation v0)

This document defines the **widget tree** format as a *generation target* for
an LLM — i.e. the JSON an LLM is asked to produce in the zero-shot / few-shot
experiments. It is derived from `notebooks/webui/widget_spec.py` (the in-memory
`Widget` dataclass) and `widget_to_dict` in `loader_convert.ipynb` (the
pipeline serializer), but **trimmed** to the fields that make sense when
generating from scratch rather than converting from an AXTree.

## 1. What a widget tree is

A widget tree is a single JSON object describing a UI as a tree of nested
nodes. Every node has exactly one `type` and an unordered set of `hints`.
It is intentionally small: 4 node types, 10 hint values, 4 fields per node.
The format deliberately carries **no** geometry (bbox), **no** styling, and
**no** provenance — only structure, content, and coarse layout/semantic hints.

## 2. Node schema

Every node is an object with exactly these four fields, always present:

```json
{
  "type": "text | block | link | image",
  "hints": ["<hint>", ...],
  "name": "<string>",
  "children": [ <node>, ... ]
}
```

| Field      | Type            | Notes |
|------------|-----------------|-------|
| `type`     | string (enum)   | One of `text`, `block`, `link`, `image`. |
| `hints`    | array of string | Subset of the hint vocabulary (§4). Order-insensitive; serialize **sorted**. May be empty `[]`. |
| `name`     | string          | Meaning depends on `type` (§3). Use `""` when not applicable. |
| `children` | array of node   | Recursive. Empty `[]` for leaves. |

> The pipeline serializer also emits `node_id`, `back_id`, and `role`. Those are
> AXTree provenance and are **not** part of the generation format — an LLM has
> no AXTree to derive them from. The validator strips them if present.

## 3. Node types

### `text` — leaf
Plain text content. `name` holds the text. `children` **must** be `[]`.
`hints` may carry semantic text hints (`heading`, `bold`, …).

### `image` — leaf
An image. `name` holds the alt text / description. `children` **must** be `[]`.

### `link` — container
A hyperlink. `name` holds the **URL/href**. `children` hold the visible content
(usually `text`, sometimes `image`). A link with no children is allowed — then
`name` is shown as the label. (This mirrors the pipeline convention where link
enrichment writes the resolved URL into `name`.)

### `block` — container
A generic grouping node. `children` hold nested nodes. `name` is an **optional**
label/header for the group — usually `""`. Layout hints (`vbox`, `hbox`,
`main`) almost always live on `block` nodes.

## 4. Hint vocabulary

`hints` values come from a closed set of 10. Nothing else is valid.

**Layout hints** (mostly on `block`):
- `main` — the primary content region. Typically appears **once**, on or near
  the root.
- `vbox` — children are stacked vertically.
- `hbox` — children are laid out in a horizontal row.

**Semantic text hints** (mostly on `text`, sometimes on a `block` wrapping text):
- `heading` — top-level heading (≈ `<h1>`).
- `subheading` — lower-level heading (≈ `<h2>`+).
- `bold` — emphasized / strong.
- `italic` — emphasized / italic.
- `quote` — block quotation.
- `caption` — caption / label text (figure captions, form labels).
- `time` — a timestamp / date.

## 5. Structural rules

1. The tree has **exactly one root node** (the top-level JSON object).
2. `text` and `image` are **always leaves** (`children` is `[]`).
3. `block` and `link` may have children; a `link` may also be childless.
4. `hints` is a set: no duplicates, serialized as a sorted array.
5. `main` should appear at most once in a tree.
6. `vbox` and `hbox` are mutually exclusive on a single node in practice — a
   block lays its children out one way or the other.
7. Output is **only** the JSON object — no surrounding prose, no markdown
   code fences.

## 6. Complete example

A clean, hand-written tree answering *"what are good budget laptops?"*:

```json
{
  "type": "block",
  "hints": ["main", "vbox"],
  "name": "",
  "children": [
    {
      "type": "text",
      "hints": ["heading"],
      "name": "Best Budget Laptops in 2026",
      "children": []
    },
    {
      "type": "block",
      "hints": ["vbox"],
      "name": "",
      "children": [
        {
          "type": "text",
          "hints": ["subheading"],
          "name": "Apple MacBook Air (M3)",
          "children": []
        },
        {
          "type": "text",
          "hints": [],
          "name": "Lightweight, excellent battery life, strong performance for students.",
          "children": []
        },
        {
          "type": "link",
          "hints": [],
          "name": "https://www.apple.com/macbook-air/",
          "children": [
            { "type": "text", "hints": [], "name": "View details", "children": [] }
          ]
        }
      ]
    },
    {
      "type": "block",
      "hints": ["vbox"],
      "name": "",
      "children": [
        {
          "type": "text",
          "hints": ["subheading"],
          "name": "Lenovo IdeaPad Slim 5",
          "children": []
        },
        {
          "type": "text",
          "hints": [],
          "name": "Best value Windows option under $600.",
          "children": []
        }
      ]
    }
  ]
}
```

## 7. Compact prompt block

Drop this verbatim into a system prompt when the schema must be described
inline (zero-shot):

```
A widget tree is a JSON object. Each node has exactly these fields:
  { "type": <text|block|link|image>,
    "hints": [<hint>, ...],
    "name": <string>,
    "children": [<node>, ...] }

type:
  text  - leaf text. children MUST be []. name = the text.
  image - leaf image. children MUST be []. name = alt text.
  link  - hyperlink. name = URL. children = visible content (text/image).
  block - container. children = nested nodes. name = optional label ("" if none).

hints (array of strings, may be empty []). Each item must be EXACTLY one of
these 10 literal values, written as a bare word - never prefixed:
  main, vbox, hbox, heading, subheading, bold, italic, quote, caption, time
  (write "main" and "heading" - NOT "layout:main", NOT "text:heading")

  meaning: main = primary content region (~once per tree); vbox = stack
  children vertically; hbox = children in a row; heading / subheading =
  heading text; bold / italic / quote / caption / time = text styling.

Rules:
  - exactly one root node
  - text and image are always leaves (children = [])
  - hints is a set, serialized sorted; use [] when none
  - all four fields are always present
  - output ONLY the JSON object - no prose, no markdown fences
```

## 8. Validation

`widget_schema.json` (next to this file) is a JSON Schema (draft 2020-12)
encoding §2–§5. `metrics.py` should:

1. Strip unknown keys (`node_id`, `back_id`, `role`) before validating, so
   pipeline-format trees and generation-format trees both validate.
2. Optionally normalize: fill a missing `hints` with `[]` and a missing
   `name` with `""` before validating. Recording *whether* normalization
   was needed lets the thesis separate **hard** format errors (invalid
   `type`, `text` with children, unknown hint) from **soft** ones (an
   omitted empty field) — a useful breakdown for the results section.
