"""
Пересоздать corpus кандидатов с topic-filter.

- Читает webui/metadata/screenclassification/silver_webui-multi_topic.csv
- Per-page: берёт screenshot с max confidence как dominant topic
- Фильтрует страницы где dominant topic ∈ INCLUDE_TOPICS И confidence ≥ MIN_CONFIDENCE
- Кросс-проверяет с локально доступными страницами в WebUI кэше
- Сэмплирует N_TARGET страниц (с запасом на page-failure rate генератора)
- Прогоняет build_candidates_for_loader на каждой
- Использует кэш existing per-page *_candidates.jsonl (если есть)
- Пишет combined output/candidate/candidates_topic_filtered.jsonl

Запуск из корня репо:
    .venv\Scripts\python.exe notebooks\webui\build_topic_corpus.py
"""

import csv
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
OUTPUT_FILE = Path(r"notebooks\webui\annotator\candidates_topic_filtered.jsonl")

LABELS = [
    "bare", "calculator", "camera", "chat", "editor", "form", "gallery", "list", "login", "maps",
    "mediaplayer", "menu", "modal", "news", "other", "profile", "search", "settings", "terms", "tutorial",
]

INCLUDE_TOPICS = {"news", "tutorial", "terms", "bare", "other", "list"}
MIN_CONFIDENCE = 0.5
N_TARGET = 220        # с запасом на ~6% page-failure
RANDOM_SEED = 2026


def load_per_page_dominant_topic():
    """Per-page: dominant topic (label, confidence) — берём screenshot с max confidence."""
    page_topics = defaultdict(list)  # page_id -> list of (label_idx, prob)

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

    # Per page: best screenshot
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


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    random.seed(RANDOM_SEED)

    print("=== Loading topic data ===")
    page_dominant = load_per_page_dominant_topic()
    print(f"Pages with topic data: {len(page_dominant)}")

    print(f"\n=== Filtering to INCLUDE_TOPICS = {sorted(INCLUDE_TOPICS)} ===")
    print(f"MIN_CONFIDENCE = {MIN_CONFIDENCE}")
    eligible_pages = {
        pid: (topic, conf)
        for pid, (topic, conf) in page_dominant.items()
        if topic in INCLUDE_TOPICS and conf >= MIN_CONFIDENCE
    }
    print(f"Eligible by topic+confidence: {len(eligible_pages)}")

    print("\n=== Cross-checking with local WebUI cache ===")
    local_pages = list_local_pages()
    print(f"Local pages: {len(local_pages)}")

    sampleable = {pid: eligible_pages[pid] for pid in eligible_pages if pid in local_pages}
    print(f"Eligible AND local: {len(sampleable)}")

    if len(sampleable) < N_TARGET:
        print(f"WARNING: only {len(sampleable)} sampleable, requested {N_TARGET}")

    sampleable_ids = sorted(sampleable.keys())
    n = min(N_TARGET, len(sampleable_ids))
    selected = random.sample(sampleable_ids, n)

    print(f"\n=== Sampled {n} pages ===")
    sampled_topics = Counter(sampleable[pid][0] for pid in selected)
    for t, c in sampled_topics.most_common():
        print(f"  {t}: {c}")

    print("\n=== Building candidates ===")
    all_records = []
    skipped = []
    for i, pid in enumerate(selected, 1):
        per_page_file = per_page_jsonl_path(pid)
        topic, conf = sampleable[pid]

        if per_page_file.exists():
            with per_page_file.open(encoding="utf-8") as f:
                cands = [json.loads(line) for line in f if line.strip()]
            all_records.extend(cands)
            if i <= 5 or i % 25 == 0 or i == n:
                print(f"[{i}/{n}] {pid} ({topic}, {conf:.2f}): {len(cands)} cached")
            continue

        page_dir = local_pages[pid]
        try:
            loader = PageLoader(str(page_dir), debug=False)
            cands = build_candidates.build_candidates_for_loader(loader, DATASET_ROOT)
        except Exception as e:
            print(f"[{i}/{n}] {pid} ({topic}, {conf:.2f}): ERROR {type(e).__name__}: {e}")
            skipped.append((pid, str(e)))
            continue

        if not cands:
            skipped.append((pid, "0 candidates"))
            if i <= 5 or i % 25 == 0:
                print(f"[{i}/{n}] {pid} ({topic}, {conf:.2f}): 0 candidates, skip")
            continue

        write_jsonl(per_page_file, cands)
        all_records.extend(cands)
        if i <= 5 or i % 25 == 0 or i == n:
            print(f"[{i}/{n}] {pid} ({topic}, {conf:.2f}): {len(cands)} candidates")

    print(f"\nProcessed: {n - len(skipped)} pages, skipped: {len(skipped)}")

    write_jsonl(OUTPUT_FILE, all_records)
    n_pages = len({r["page_id"] for r in all_records})
    print(f"Wrote {len(all_records)} candidates from {n_pages} pages to {OUTPUT_FILE}")

    print("\n=== Stats by topic ===")
    page_to_topic = {pid: sampleable[pid][0] for pid in selected if pid in sampleable}
    candidates_by_topic = defaultdict(int)
    pages_by_topic = defaultdict(set)
    for r in all_records:
        pid = str(r["page_id"])
        t = page_to_topic.get(pid, "?")
        candidates_by_topic[t] += 1
        pages_by_topic[t].add(pid)
    for t in sorted(candidates_by_topic, key=lambda k: -candidates_by_topic[k]):
        c = candidates_by_topic[t]
        p = len(pages_by_topic[t])
        print(f"  {t:10s}: {p:3d} pages, {c:4d} candidates, avg {c/p:.1f}/page")

    roles = Counter(r.get("role", "") for r in all_records)
    print(f"\nTop roles: {roles.most_common(10)}")


if __name__ == "__main__":
    main()
