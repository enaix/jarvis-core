"""Анализ labels.jsonl — распределение, sub-labels, pace, integrity."""
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

LABELS_FILE = Path(r"notebooks\webui\annotator\output\labels.jsonl")
PAGE_FLAGS_FILE = Path(r"notebooks\webui\annotator\output\page_flags.jsonl")


def main():
    with LABELS_FILE.open(encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    bad_pages = []
    if PAGE_FLAGS_FILE.exists():
        with PAGE_FLAGS_FILE.open(encoding="utf-8") as f:
            bad_pages = [json.loads(l) for l in f if l.strip()]

    print(f"=== Total labels: {len(rows)} ===")
    print(f"Bad pages: {len(bad_pages)}")

    # Label distribution
    labels = Counter(r["label"] for r in rows)
    print("\n=== Label distribution ===")
    for label, count in labels.most_common():
        pct = 100 * count / len(rows)
        print(f"  {label:10s}: {count:4d} ({pct:.1f}%)")

    # Sub-labels
    junk_rows = [r for r in rows if r["label"] == "junk"]
    sub_labels = Counter(r.get("sub_label", "") for r in junk_rows)
    print(f"\n=== Sub-labels (among {len(junk_rows)} junk) ===")
    for sl, count in sub_labels.most_common():
        pct = 100 * count / len(junk_rows) if junk_rows else 0
        print(f"  {sl or '(empty)':12s}: {count:4d} ({pct:.1f}%)")

    # Pages
    pages = set(r["page_id"] for r in rows)
    per_page = defaultdict(lambda: Counter())
    for r in rows:
        per_page[r["page_id"]][r["label"]] += 1

    print(f"\n=== Pages: {len(pages)} ===")
    counts = sorted([sum(c.values()) for c in per_page.values()])
    print(f"  Median labels/page: {counts[len(counts)//2]}")
    print(f"  Min/Max: {min(counts)}/{max(counts)}")

    # Pace
    timestamps = sorted(datetime.fromisoformat(r["timestamp"]) for r in rows)
    duration = (timestamps[-1] - timestamps[0]).total_seconds()
    print(f"\n=== Pace ===")
    print(f"  First: {timestamps[0]}")
    print(f"  Last:  {timestamps[-1]}")
    print(f"  Duration: {duration/60:.0f} min ({duration/3600:.1f} hours)")
    print(f"  Avg sec/label: {duration/len(rows):.1f}")

    # Per-session pace (gaps > 10 min = new session)
    sessions = []
    cur_start = timestamps[0]
    cur_end = timestamps[0]
    cur_count = 1
    for t in timestamps[1:]:
        gap = (t - cur_end).total_seconds()
        if gap > 600:
            sessions.append((cur_start, cur_end, cur_count))
            cur_start = t
            cur_count = 1
        else:
            cur_count += 1
        cur_end = t
    sessions.append((cur_start, cur_end, cur_count))
    print(f"\n=== Sessions (gap >10min) ===")
    for s, e, n in sessions:
        sd = (e - s).total_seconds() / 60
        rate = n / max(sd, 0.01) * 60  # per hour
        print(f"  {s.strftime('%m-%d %H:%M')} - {e.strftime('%H:%M')}: "
              f"{n} labels in {sd:.0f} min ({rate:.0f}/hour)")

    # Integrity
    print(f"\n=== Integrity ===")
    junk_no_sub = sum(1 for r in rows if r["label"] == "junk" and not r.get("sub_label"))
    not_junk_with_sub = sum(1 for r in rows if r["label"] == "not_junk" and r.get("sub_label"))
    no_hash = sum(1 for r in rows if not r.get("content_hash"))
    print(f"  junk without sub_label: {junk_no_sub} (should be 0)")
    print(f"  not_junk with sub_label: {not_junk_with_sub} (should be 0)")
    print(f"  missing content_hash: {no_hash} (should be 0)")

    # Top roles per label
    print(f"\n=== Top roles by label class ===")
    for label_type in ("junk", "not_junk"):
        roles = Counter(r["role"] for r in rows if r["label"] == label_type)
        print(f"  {label_type}: {roles.most_common(8)}")


if __name__ == "__main__":
    main()
