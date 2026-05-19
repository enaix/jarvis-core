"""
Расширить candidates_topic_filtered.jsonl до целевого количества candidates.

- Читает существующий candidates_topic_filtered.jsonl (~3148 candidates, 201 pages)
- Сэмплирует дополнительные topic-filtered pages из WebUI (вне уже обработанных)
- Прогоняет build_candidates_for_loader на новых
- Объединяет в candidates_topic_filtered.jsonl
- Пересчитывает global_recurrence (читает из global_hash_recurrence.json — уже built на всех 5420)
- Пересчитывает local recurrence_count для нового combined set

Запуск:
    .venv\\Scripts\\python.exe notebooks\\webui\\expand_topic_corpus.py
"""

import csv
import gzip
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # notebooks/webui/

from page_loader import PageLoader  # noqa: E402
import build_candidates  # noqa: E402


# === Конфигурация ===
DATASET_ROOT = Path(
    r"C:\Users\70133\.cache\huggingface\hub\datasets--biglab--webui-7k\snapshots\60f7b3c4b9409f75551664adc1564625dfc33c2e\dataset1"
)
TOPIC_CSV = Path(r"webui\metadata\screenclassification\silver_webui-multi_topic.csv")
OUTPUT_DIR = Path(r"notebooks\webui\output\candidate")  # per-page jsonl кэш остаётся здесь
CORPUS_FILE = Path(r"notebooks\webui\annotator\candidates_topic_filtered.jsonl")
GLOBAL_HASH_FILE = OUTPUT_DIR / "global_hash_recurrence.json"

LABELS = [
    "bare", "calculator", "camera", "chat", "editor", "form", "gallery", "list", "login", "maps",
    "mediaplayer", "menu", "modal", "news", "other", "profile", "search", "settings", "terms", "tutorial",
]

INCLUDE_TOPICS = {"news", "tutorial", "terms", "bare", "other", "list"}
MIN_CONFIDENCE = 0.5
TARGET_CANDIDATES = 6000
RANDOM_SEED = 2027  # отличается от исходного 2026, чтобы добавить разнообразия


def load_corpus(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_corpus(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_topic_dominant():
    """Per-page dominant topic + confidence."""
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
        page_dominant[pid] = (LABELS[best[0]], best[1])
    return page_dominant


def list_local_pages():
    page_dirs = {}
    for split_dir in DATASET_ROOT.iterdir():
        if not split_dir.is_dir():
            continue
        for page_dir in split_dir.iterdir():
            if page_dir.is_dir() and page_dir.name.isdigit():
                page_dirs[page_dir.name] = page_dir
    return page_dirs


def per_page_jsonl_path(page_id):
    return OUTPUT_DIR / f"{page_id}_candidates.jsonl"


def write_jsonl_path(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_for_hash(text):
    return " ".join((text or "").lower().split())


def compute_hash(role, text_subtree):
    payload = f"{role or ''}::{normalize_for_hash(text_subtree)}".encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:16]


def main():
    random.seed(RANDOM_SEED)

    print("=== 1. Loading existing corpus ===")
    existing = load_corpus(CORPUS_FILE)
    existing_pages = {str(r["page_id"]) for r in existing}
    print(f"  {len(existing)} candidates from {len(existing_pages)} pages")

    if len(existing) >= TARGET_CANDIDATES:
        print(f"  Already at {len(existing)} >= target {TARGET_CANDIDATES}, nothing to do")
        return

    needed = TARGET_CANDIDATES - len(existing)
    avg_per_page = len(existing) / len(existing_pages) if existing_pages else 15
    pages_to_add_estimate = int(needed / avg_per_page * 1.15)  # +15% buffer for failures
    print(f"  Need {needed} more candidates, estimated {pages_to_add_estimate} more pages "
          f"(avg {avg_per_page:.1f}/page)")

    print("\n=== 2. Topic filter + local availability ===")
    page_dominant = load_topic_dominant()
    eligible = {pid: (t, c) for pid, (t, c) in page_dominant.items()
                if t in INCLUDE_TOPICS and c >= MIN_CONFIDENCE}
    local_pages = list_local_pages()
    sampleable = {pid: eligible[pid] for pid in eligible
                  if pid in local_pages and pid not in existing_pages}
    print(f"  Eligible AND local AND not in existing corpus: {len(sampleable)}")

    n_sample = min(pages_to_add_estimate, len(sampleable))
    selected = random.sample(sorted(sampleable.keys()), n_sample)
    print(f"  Sampled {n_sample} new pages")
    new_topics = Counter(sampleable[pid][0] for pid in selected)
    for t, c in new_topics.most_common():
        print(f"    {t}: {c}")

    print("\n=== 3. Building candidates for new pages ===")
    new_records = []
    skipped = []
    for i, pid in enumerate(selected, 1):
        per_page_file = per_page_jsonl_path(pid)
        topic, conf = sampleable[pid]
        if per_page_file.exists():
            with per_page_file.open(encoding="utf-8") as f:
                cands = [json.loads(line) for line in f if line.strip()]
            new_records.extend(cands)
            if i <= 5 or i % 25 == 0 or i == n_sample:
                print(f"  [{i}/{n_sample}] {pid} ({topic}, {conf:.2f}): {len(cands)} cached")
            continue

        page_dir = local_pages[pid]
        try:
            loader = PageLoader(str(page_dir), debug=False)
            cands = build_candidates.build_candidates_for_loader(loader, DATASET_ROOT)
        except Exception as e:
            print(f"  [{i}/{n_sample}] {pid}: ERROR {type(e).__name__}: {e}")
            skipped.append((pid, str(e)))
            continue

        if not cands:
            skipped.append((pid, "0 candidates"))
            continue

        write_jsonl_path(per_page_file, cands)
        new_records.extend(cands)
        if i <= 5 or i % 25 == 0 or i == n_sample:
            print(f"  [{i}/{n_sample}] {pid} ({topic}, {conf:.2f}): {len(cands)} candidates")

    print(f"\n  Successfully processed: {n_sample - len(skipped)} pages, skipped: {len(skipped)}")

    # Add content_hash to new records (existing already have it)
    for r in new_records:
        if "content_hash" not in r:
            r["content_hash"] = compute_hash(r.get("role"), r.get("text_subtree", ""))

    print("\n=== 4. Combining and reapplying global recurrence ===")
    combined = existing + new_records

    # Read global hash db
    if not GLOBAL_HASH_FILE.exists():
        print(f"  WARNING: {GLOBAL_HASH_FILE} not found, global_recurrence will not be set on new records")
        global_counts = {}
        global_total = 5420
    else:
        with GLOBAL_HASH_FILE.open(encoding="utf-8") as f:
            data = json.load(f)
        global_counts = data.get("hash_counts", {})
        global_total = data.get("total_pages", 5420)
        print(f"  Loaded global hash db: {len(global_counts)} hashes from {global_total} pages")

    # Apply global recurrence to all (including new ones)
    for r in combined:
        h = r.get("content_hash")
        if h is None:
            h = compute_hash(r.get("role"), r.get("text_subtree", ""))
            r["content_hash"] = h
        gc = global_counts.get(h, 0)
        r["global_recurrence_count"] = gc
        r["global_recurrence_ratio"] = round(gc / max(global_total, 1), 5)

    # Recompute local recurrence
    local_hash_pages = defaultdict(set)
    for r in combined:
        local_hash_pages[r["content_hash"]].add(str(r["page_id"]))
    for r in combined:
        cnt = len(local_hash_pages[r["content_hash"]])
        r["recurrence_count"] = cnt
        r["recurrence_ratio"] = round(cnt / len({str(x["page_id"]) for x in combined}), 5)

    print(f"  Combined: {len(combined)} candidates from {len({str(r['page_id']) for r in combined})} pages")

    write_corpus(CORPUS_FILE, combined)
    print(f"  Wrote to {CORPUS_FILE}")

    # Final stats
    print("\n=== Final stats ===")
    print(f"  Candidates: {len(combined)}")
    print(f"  Pages: {len({str(r['page_id']) for r in combined})}")
    grec_2 = sum(1 for r in combined if r.get("global_recurrence_count", 0) >= 2)
    grec_5 = sum(1 for r in combined if r.get("global_recurrence_count", 0) >= 5)
    print(f"  global_rec >= 2: {grec_2} ({100*grec_2/len(combined):.1f}%)")
    print(f"  global_rec >= 5: {grec_5} ({100*grec_5/len(combined):.1f}%)")
    roles = Counter(r.get("role", "") for r in combined)
    print(f"  Top roles: {roles.most_common(8)}")


if __name__ == "__main__":
    main()
