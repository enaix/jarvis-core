"""
Pre-flight #5 — recurring fragment hashing.

- Читает candidates_topic_filtered.jsonl
- Для каждого кандидата вычисляет content_hash (md5 от role + normalized text_subtree, 16 hex)
- Группирует кандидатов по hash, считает количество уникальных страниц на hash
- Дописывает в каждый кандидат: content_hash, recurrence_count, recurrence_ratio
- Перезаписывает candidates_topic_filtered.jsonl с новыми полями
- В конце — статистика и топ-10 повторяющихся блоков

Recurrence_count = на скольких разных страницах встречается этот content_hash.
Recurrence_ratio = recurrence_count / total_pages.

Запуск:
    .venv\Scripts\python.exe notebooks\webui\compute_recurrence.py
"""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

CANDIDATES_FILE = Path(r"notebooks\webui\annotator\candidates_topic_filtered.jsonl")


def normalize_text(text):
    if not text:
        return ""
    # lowercase + collapse whitespace
    return " ".join(text.lower().split())


def compute_hash(role, text_subtree):
    norm = normalize_text(text_subtree)
    payload = f"{role or ''}::{norm}".encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:16]


def main():
    if not CANDIDATES_FILE.exists():
        print(f"ERROR: {CANDIDATES_FILE} not found")
        sys.exit(1)

    print(f"Reading {CANDIDATES_FILE}")
    with CANDIDATES_FILE.open(encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(candidates)} candidates")

    # Compute content_hash + group by hash → set of pages
    hash_to_pages = defaultdict(set)
    for c in candidates:
        h = compute_hash(c.get("role"), c.get("text_subtree"))
        c["content_hash"] = h
        hash_to_pages[h].add(str(c["page_id"]))

    total_pages = len({str(c["page_id"]) for c in candidates})
    print(f"Total unique pages: {total_pages}")
    print(f"Total unique content_hashes: {len(hash_to_pages)}")

    # Add recurrence to each candidate
    for c in candidates:
        h = c["content_hash"]
        count = len(hash_to_pages[h])
        c["recurrence_count"] = count
        c["recurrence_ratio"] = round(count / total_pages, 4)

    # Write back
    with CANDIDATES_FILE.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(candidates)} candidates with recurrence fields back to {CANDIDATES_FILE}")

    # === Statistics ===
    print("\n=== Recurrence distribution ===")
    rec_counts = Counter(c["recurrence_count"] for c in candidates)
    print("Recurrence count : # candidates")
    for k in sorted(rec_counts):
        if k <= 5 or k % 5 == 0 or k == max(rec_counts):
            print(f"  {k:3d} : {rec_counts[k]:5d}")

    thresholds = [1, 2, 3, 5, 10, 20, 50]
    print("\n=== Cumulative ===")
    for t in thresholds:
        n = sum(1 for c in candidates if c["recurrence_count"] >= t)
        pct = 100.0 * n / len(candidates)
        print(f"  recurrence >= {t:3d}: {n:5d} candidates ({pct:.1f}%)")

    # Per-role recurrence
    print("\n=== Avg recurrence by role ===")
    role_to_recurrence = defaultdict(list)
    for c in candidates:
        role_to_recurrence[c.get("role", "")].append(c["recurrence_count"])
    role_summary = []
    for role, recs in role_to_recurrence.items():
        if len(recs) < 5:
            continue
        avg = sum(recs) / len(recs)
        max_rec = max(recs)
        role_summary.append((role, len(recs), avg, max_rec))
    role_summary.sort(key=lambda x: -x[2])
    print(f"  {'role':<25s} {'n':>5s} {'avg_rec':>10s} {'max_rec':>10s}")
    for role, n, avg, mx in role_summary[:15]:
        print(f"  {role:<25s} {n:>5d} {avg:>10.2f} {mx:>10d}")

    # Top recurring content_hashes
    print("\n=== Top 15 recurring content_hashes ===")
    hash_examples = {}
    for c in candidates:
        h = c["content_hash"]
        if h not in hash_examples:
            hash_examples[h] = c
    top = sorted(hash_examples.items(), key=lambda kv: -kv[1]["recurrence_count"])[:15]
    for h, c in top:
        text = (c.get("text_subtree") or "")[:120].replace("\n", " ")
        role = c.get("role", "")
        print(f"  rec={c['recurrence_count']:>3d} | role={role:<18s} | hash={h} | {text!r}")


if __name__ == "__main__":
    main()
