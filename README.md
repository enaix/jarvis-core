# jarvis-core

Pipeline for converting WebUI dataset pages into a **widget tree** — a typed,
hint-annotated parent–child tree with a bounded vocabulary (4 node types,
10 hints) — intended as a target format for multimodal LLM fine-tuning.

This is the code repository accompanying the BSc thesis
*"Graph-Based Representation of Web Interfaces for Multimodal Language Model
Fine-Tuning"* by Mikhail Nikolaev (HSE FCS, BSc DSBA, 2026).

## Pipeline

    WebUI raw pages
        │
        ▼
    corpus_build/      topic-filtered subset of WebUI
        │
        ▼
    preflight/         recurrence statistics, ax-tree → widget-tree validation
        │
        ▼
    labeling/          GPT-4o + human annotation, agreement reports
        │
        ▼
    training/          feature extraction, junk-classifier, ablations
        │
        ▼
    zs_fs_eval/        GPT-4o zero-/few-shot as widget-tree generator

## Repository layout

- `notebooks/webui/` — main pipeline (see `notebooks/webui/README.md`)
- `docs/filtering_annotation_guidelines.md` — annotation guidelines (Appendix E)
- `notebooks/webui/zs_fs_eval/widget_schema.json` — widget schema (Appendix C)
- `notebooks/webui/zs_fs_eval/widget_format_spec.md` — widget format specification
- `notebooks/webui/zs_fs_eval/prompts/` — system prompts used in eval (Appendix F)
- `notebooks/webui/training/results/` — ablation tables (§6)
- `notebooks/webui/zs_fs_eval/outputs/metrics.jsonl` — eval metrics (§6.3)

## Setup

    pip install -r notebooks/webui/requirements.txt

The WebUI dataset is not included in this repo — see the dataset's own
distribution for download instructions.

## License

GPL-3.0 — see [`LICENSE`](LICENSE).
