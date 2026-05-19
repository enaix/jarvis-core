"""
Step 1 — Подготовка labels + train/val/test split.

- Загружает human labels (gold) + LLM labels (silver)
- Combined: human priority над LLM (на overlap)
- Исключает candidates из bad_pages
- Page-level split, stratified по topic (news/tutorial/terms/bare/other/list)
- Seed=2026 — фиксирован раз и навсегда (см. filtering_methodology.md)

Output:
- training/data/combined_labels.jsonl — финальный набор меток с label_source ('human' or 'llm')
- training/data/splits.json — page_id -> 'train'/'val'/'test' mapping
- training/data/data_summary.json — статистика

Запуск:
    .venv\\Scripts\\python.exe notebooks\\webui\\training\\prepare_data.py
"""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

CANDIDATES_FILE = Path(r"notebooks\webui\annotator\candidates_topic_filtered.jsonl")
HUMAN_LABELS = Path(r"notebooks\webui\annotator\output\labels.jsonl")
LLM_LABELS = Path(r"notebooks\webui\annotator\output\labels_llm.jsonl")
PAGE_FLAGS = Path(r"notebooks\webui\annotator\output\page_flags.jsonl")
TOPIC_CSV = Path(r"webui\metadata\screenclassification\silver_webui-multi_topic.csv")

OUTPUT_DIR = Path(r"notebooks\webui\training\data")
COMBINED_OUT = OUTPUT_DIR / "combined_labels.jsonl"
SPLITS_OUT = OUTPUT_DIR / "splits.json"
SUMMARY_OUT = OUTPUT_DIR / "data_summary.json"

LABELS = [
    "bare", "calculator", "camera", "chat", "editor", "form", "gallery", "list", "login", "maps",
    "mediaplayer", "menu", "modal", "news", "other", "profile", "search", "settings", "terms", "tutorial",
]

SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, val, test
RANDOM_SEED = 2026


def load_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def candidate_key(r):
    return f"{r['page_id']}::{r['node_id']}"


def load_topic_per_page():
    """Per-page dominant topic (max-confidence screenshot)."""
    page_topics = defaultdict(list)
    with TOPIC_CSV.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 21:
                continue
            path = row[0]
            sep = "\\" if "\\" in path else "/"
            page_id = path.split(sep)[0]
            if not page_id.isdigit():
                continue
            try:
                probs = [float(x) for x in row[1:21]]
            except ValueError:
                continue
            top_idx = max(range(len(probs)), key=lambda i: probs[i])
            page_topics[page_id].append((top_idx, probs[top_idx]))
    page_dominant = {}
    for pid, screenshots in page_topics.items():
        best = max(screenshots, key=lambda x: x[1])
        page_dominant[pid] = LABELS[best[0]]
    return page_dominant


def stratified_split(pages_with_topic, ratios, seed):
    """Page-level stratified split. Возвращает {page_id: 'train'/'val'/'test'}."""
    rnd = random.Random(seed)
    by_topic = defaultdict(list)
    for pid, topic in pages_with_topic.items():
        by_topic[topic].append(pid)

    splits = {}
    for topic, pids in by_topic.items():
        rnd.shuffle(pids)
        n = len(pids)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        # rest -> test (handles rounding)
        for i, pid in enumerate(pids):
            if i < n_train:
                splits[pid] = "train"
            elif i < n_train + n_val:
                splits[pid] = "val"
            else:
                splits[pid] = "test"
    return splits


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading sources ===")
    candidates = load_jsonl(CANDIDATES_FILE)
    human = load_jsonl(HUMAN_LABELS)
    llm = load_jsonl(LLM_LABELS)
    page_flags = load_jsonl(PAGE_FLAGS)
    print(f"  Candidates: {len(candidates)}")
    print(f"  Human labels: {len(human)}")
    print(f"  LLM labels: {len(llm)}")

    bad_pages = {str(r["page_id"]) for r in page_flags if r.get("page_action") == "bad_page"}
    print(f"  Bad pages: {len(bad_pages)}")

    # Index candidates by key
    cand_idx = {candidate_key(c): c for c in candidates}

    # Build combined labels with human priority
    print("\n=== Combining labels (human priority) ===")
    combined = {}
    n_human_added = 0
    n_llm_added = 0
    n_skipped_bad_page = 0

    # 1. Add human labels first (gold)
    for r in human:
        pid = str(r["page_id"])
        if pid in bad_pages:
            n_skipped_bad_page += 1
            continue
        k = candidate_key(r)
        if k not in cand_idx:
            continue  # candidate отсутствует в текущем corpus (старая запись)
        rec = dict(r)
        rec["label_source"] = "human"
        combined[k] = rec
        n_human_added += 1

    # 2. Add LLM labels where no human (silver)
    for r in llm:
        pid = str(r["page_id"])
        if pid in bad_pages:
            n_skipped_bad_page += 1
            continue
        k = candidate_key(r)
        if k not in cand_idx:
            continue
        if k in combined:
            continue  # human label already exists
        rec = dict(r)
        rec["label_source"] = "llm"
        combined[k] = rec
        n_llm_added += 1

    print(f"  Human added (gold): {n_human_added}")
    print(f"  LLM added (silver, where no human): {n_llm_added}")
    print(f"  Skipped because bad_page: {n_skipped_bad_page}")
    print(f"  Total combined unique: {len(combined)}")

    # Distribution
    label_dist = Counter(r["label"] for r in combined.values())
    print(f"\n=== Combined label distribution ===")
    for lbl, cnt in label_dist.most_common():
        pct = 100 * cnt / len(combined)
        print(f"  {lbl:10s}: {cnt:5d} ({pct:.1f}%)")

    # Write combined labels
    with COMBINED_OUT.open("w", encoding="utf-8") as f:
        for r in combined.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(combined)} combined labels to {COMBINED_OUT}")

    # === Topic per page ===
    print("\n=== Loading per-page topic ===")
    topic_map = load_topic_per_page()
    print(f"  {len(topic_map)} pages have topic data in CSV")

    # Pages in our combined dataset
    pages_in_data = sorted({str(r["page_id"]) for r in combined.values()})
    pages_with_topic = {pid: topic_map.get(pid, "unknown") for pid in pages_in_data}
    topic_distribution = Counter(pages_with_topic.values())
    print(f"  Pages in dataset: {len(pages_in_data)}")
    for t, n in topic_distribution.most_common():
        print(f"    {t}: {n}")

    # === Stratified page-level split ===
    print(f"\n=== Stratified split (page-level, seed={RANDOM_SEED}) ===")
    splits = stratified_split(pages_with_topic, SPLIT_RATIOS, RANDOM_SEED)
    split_counts = Counter(splits.values())
    print(f"  train pages: {split_counts['train']}")
    print(f"  val pages:   {split_counts['val']}")
    print(f"  test pages:  {split_counts['test']}")

    # Per-topic split breakdown
    print(f"\n  Per-topic split breakdown:")
    by_topic_split = defaultdict(lambda: Counter())
    for pid, sp in splits.items():
        by_topic_split[pages_with_topic[pid]][sp] += 1
    for t in topic_distribution:
        c = by_topic_split[t]
        print(f"    {t:10s}: train={c['train']:>3d} val={c['val']:>3d} test={c['test']:>3d}")

    # Per-split label counts
    print(f"\n  Per-split label counts:")
    by_split_label = defaultdict(lambda: Counter())
    by_split_source = defaultdict(lambda: Counter())
    for r in combined.values():
        sp = splits.get(str(r["page_id"]), "unknown")
        by_split_label[sp][r["label"]] += 1
        by_split_source[sp][r["label_source"]] += 1
    for sp in ("train", "val", "test"):
        ll = by_split_label[sp]
        ss = by_split_source[sp]
        total = sum(ll.values())
        print(f"    {sp:5s}: total={total:5d} | "
              f"junk={ll['junk']} not_junk={ll['not_junk']} skip={ll['skip']} | "
              f"human={ss['human']} llm={ss['llm']}")

    # Save splits
    with SPLITS_OUT.open("w", encoding="utf-8") as f:
        json.dump({"page_topic": pages_with_topic, "splits": splits, "seed": RANDOM_SEED}, f, ensure_ascii=False, indent=2)
    print(f"\n  Splits saved to {SPLITS_OUT}")

    # === Summary ===
    summary = {
        "total_candidates_in_corpus": len(candidates),
        "human_labels_total": len(human),
        "llm_labels_total": len(llm),
        "bad_pages": len(bad_pages),
        "combined_unique": len(combined),
        "combined_human_priority": n_human_added,
        "combined_llm_silver": n_llm_added,
        "label_distribution": dict(label_dist),
        "split_pages": dict(split_counts),
        "split_label_counts": {sp: dict(by_split_label[sp]) for sp in ("train", "val", "test")},
        "split_source_counts": {sp: dict(by_split_source[sp]) for sp in ("train", "val", "test")},
        "topic_distribution": dict(topic_distribution),
        "random_seed": RANDOM_SEED,
        "split_ratios": list(SPLIT_RATIOS),
    }
    with SUMMARY_OUT.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Summary saved to {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
