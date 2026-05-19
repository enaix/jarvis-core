# WebUI Pipeline

Code for converting WebUI dataset pages into widget-tree representations plus
supporting tooling for corpus construction, annotation, classifier training,
and LLM evaluation.

## Dataset

Raw WebUI data is downloaded automatically by `loader.ipynb`. Metadata
should be downloaded manually [from GDrive][webui-metadata]; remember to
update the metadata path inside the notebook.

[webui-metadata]: https://github.com/js0nwu/webui/blob/a66277390db23f9c8baaa21d8189ab499339db7c/downloads/downloader.py#L18

## Directories

### `corpus_build/`
Build a topic-filtered subset from raw WebUI pages.
- `build_topic_corpus.py` — top-level corpus builder
- `expand_corpus.py`, `expand_topic_corpus.py` — extending the corpus
- `sanity_check_random_pages.py` — sample verification

### `preflight/`
Pre-annotation checks on the corpus.
- `compute_recurrence.py`, `compute_global_recurrence.py` — content-hash recurrence statistics
- `check_ax_to_widget_mapping.py` — AXTree → widget-tree mapping validation

### `annotator/`
Streamlit annotation UI for marking AXTree candidates as junk/not_junk.
Outputs go to `annotator/output/` (gitignored).

### `labeling/`
LLM (GPT-4o) labeling pipeline plus human-LLM agreement analysis.
- `llm_label.py` — GPT-4o labeling
- `analyze_labels.py`, `analyze_llm_labels.py` — label statistics
- `compute_full_agreement.py` — agreement reports

### `training/`
Junk-classifier training (LogReg / GBM baselines on TF-IDF, structural,
and recurrence features).
- `prepare_data.py`, `extract_features.py` — data pipeline
- `train_lib.py` — training and evaluation library
- `run_ablations.py` — ablation experiments
- `filter_apply.py`, `filter_visualize.py` — applying and visualizing the trained classifier
- `data/` — labels and splits (large feature matrices gitignored)
- `results/` — ablation tables

### `zs_fs_eval/`
GPT-4o zero-/few-shot evaluation as a widget-tree generator.
- `pipeline.py` — main evaluation pipeline
- `metrics.py` — schema validity, type divergence, content density
- `analysis.py`, `make_results.py`, `analyze_manual_ratings.py` — aggregation
- `axtree_to_widget.py` — adapter from Chrome AXTree to widget tree
- `prompts/` — system prompts (T1–T4 configurations)
- `eval_set/` — 25 evaluation questions and few-shot examples
- `widget_schema.json` — widget format schema
- `widget_format_spec.md` — widget format specification
- `outputs/metrics.jsonl` — per-run metrics
- `outputs/manual_ratings.csv` — manual validation ratings

### Top-level files
- `axtree_html_mapper.py` — AXTree node ↔ HTML anchor mapping
- `build_candidates.py` — extract filtering candidates from AXTree
- `widget_spec.py`, `widget_html.py` — widget-tree data model and HTML round-trip
- `page_loader.py` — WebUI page loader
- `loader.ipynb`, `loader_convert.ipynb` — conversion pipeline notebooks
