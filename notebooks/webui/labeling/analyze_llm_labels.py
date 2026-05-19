"""Анализ labels_llm.jsonl — distribution, confidence, sub-labels."""
import json
from collections import Counter, defaultdict
from pathlib import Path

LLM_LABELS_FILE = Path(r"notebooks\webui\annotator\output\labels_llm.jsonl")
HUMAN_LABELS_FILE = Path(r"notebooks\webui\annotator\output\labels.jsonl")


def main():
    with LLM_LABELS_FILE.open(encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    with HUMAN_LABELS_FILE.open(encoding="utf-8") as f:
        human_rows = [json.loads(l) for l in f if l.strip()]

    print(f"=== Total LLM labels: {len(rows)} ===")
    print(f"Human labels: {len(human_rows)}")

    # Detect double-labeling (LLM labeled candidate that human also labeled later via race)
    human_keys = {f"{r['page_id']}::{r['node_id']}" for r in human_rows}
    llm_keys = {f"{r['page_id']}::{r['node_id']}" for r in rows}
    overlap = human_keys & llm_keys
    print(f"Overlap (both human and LLM labeled): {len(overlap)}")
    print(f"  Of these, agreement run accounted for first 228 (intentional)")

    # Label distribution
    labels = Counter(r["label"] for r in rows)
    print("\n=== Label distribution (LLM) ===")
    for label, count in labels.most_common():
        pct = 100 * count / len(rows)
        print(f"  {label:10s}: {count:5d} ({pct:.1f}%)")

    # Sub-labels
    junk_rows = [r for r in rows if r["label"] == "junk"]
    sub_labels = Counter(r.get("sub_label", "") for r in junk_rows)
    print(f"\n=== Sub-labels (among {len(junk_rows)} junk) ===")
    for sl, count in sub_labels.most_common():
        pct = 100 * count / len(junk_rows) if junk_rows else 0
        print(f"  {sl or '(empty)':12s}: {count:5d} ({pct:.1f}%)")

    # Confidence distribution
    confs = [r.get("llm_confidence", 0) for r in rows]
    print(f"\n=== Confidence distribution ===")
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.0001)]
    for lo, hi in bins:
        n = sum(1 for c in confs if lo <= c < hi)
        pct = 100 * n / len(confs)
        print(f"  [{lo:.2f}, {hi:.2f}): {n:5d} ({pct:.1f}%)")
    avg = sum(confs) / len(confs) if confs else 0
    print(f"  Mean confidence: {avg:.3f}")
    print(f"  Median: {sorted(confs)[len(confs)//2]:.3f}")

    # Confidence by label
    print(f"\n=== Mean confidence by label ===")
    for label in ("junk", "not_junk", "skip"):
        sub = [r["llm_confidence"] for r in rows if r["label"] == label]
        if sub:
            print(f"  {label:10s}: {sum(sub)/len(sub):.3f} (n={len(sub)})")

    # Confidence by sub-label
    print(f"\n=== Mean confidence by junk sub-label ===")
    sub_confs = defaultdict(list)
    for r in rows:
        if r["label"] == "junk":
            sub_confs[r.get("sub_label", "")].append(r["llm_confidence"])
    for sl, vals in sorted(sub_confs.items(), key=lambda x: -len(x[1])):
        print(f"  {sl:12s}: {sum(vals)/len(vals):.3f} (n={len(vals)})")

    # Role distribution by class
    print(f"\n=== Top roles by label class ===")
    for label_type in ("junk", "not_junk"):
        roles = Counter(r["role"] for r in rows if r["label"] == label_type)
        print(f"  {label_type}: {roles.most_common(8)}")

    # Combined coverage
    print(f"\n=== Combined coverage (human + LLM) ===")
    all_keys = human_keys | llm_keys
    print(f"  Unique candidates labeled: {len(all_keys)}")
    print(f"  By human only: {len(human_keys - llm_keys)}")
    print(f"  By LLM only: {len(llm_keys - human_keys)}")
    print(f"  By both: {len(overlap)}")

    # Pages covered
    all_pages = {k.split("::")[0] for k in all_keys}
    print(f"  Total pages with at least one label: {len(all_pages)}")


if __name__ == "__main__":
    main()
