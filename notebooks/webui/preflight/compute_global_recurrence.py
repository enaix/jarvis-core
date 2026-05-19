"""
Pre-flight #5 (расширенная версия) — global recurrence на всех 5420 страницах WebUI.

Шаг 1: проходит по всем axtree.json.gz страницам в WebUI кэше, вычисляет
       (role, text_subtree) → hash для каждого узла, считает per-hash число страниц.
       Сохраняет в global_hash_recurrence.json: { hash: page_count }.

Шаг 2: дописывает в candidates_topic_filtered.jsonl поля
       global_recurrence_count и global_recurrence_ratio (на базе всех 5420 страниц).

Семантически: «сколько уникальных страниц во всём WebUI содержат блок
с таким же role + текстом». Для footer/nav/cookie/etc — десятки страниц.
Для уникального article body — 1 страница.

Использует только AXTree (без image loading) — должно быть быстро.

Запуск из корня репо:
    .venv\Scripts\python.exe notebooks\webui\compute_global_recurrence.py
"""

import gzip
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # notebooks/webui/

import build_candidates as bc  # noqa: E402


# === Конфигурация ===
DATASET_ROOT = Path(
    r"C:\Users\70133\.cache\huggingface\hub\datasets--biglab--webui-7k\snapshots\60f7b3c4b9409f75551664adc1564625dfc33c2e\dataset1"
)
CANDIDATES_FILE = Path(r"notebooks\webui\annotator\candidates_topic_filtered.jsonl")
GLOBAL_HASH_FILE = Path(r"notebooks\webui\output\candidate\global_hash_recurrence.json")

# Минимальная длина текста для попадания в hash database — те же правила что в build_candidates
MIN_TEXT_LEN = 30
# Минимальная длина роли — отбрасывает пустые/none узлы (но none-роли с реальным текстом всё равно ловим)
PROGRESS_EVERY = 100


def normalize_for_hash(text):
    return " ".join((text or "").lower().split())


def compute_hash(role, text_subtree):
    payload = f"{role or ''}::{normalize_for_hash(text_subtree)}".encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:16]


def find_desktop_axtree(page_dir):
    """Найти <screen_type>-axtree.json.gz из 1920-широкого viewport. Если нет — любой.
    Файлы лежат плоско с префиксом screen_type, например default_1920-1080-axtree.json.gz."""
    candidates = []
    for f in page_dir.iterdir():
        if f.is_file() and f.name.endswith("axtree.json.gz"):
            candidates.append(f)
    # Prefer desktop 1920
    for c in candidates:
        if "1920" in c.name:
            return c
    return candidates[0] if candidates else None


def list_pages():
    page_dirs = []
    for split_dir in DATASET_ROOT.iterdir():
        if not split_dir.is_dir():
            continue
        for page_dir in split_dir.iterdir():
            if page_dir.is_dir() and page_dir.name.isdigit():
                page_dirs.append(page_dir)
    return page_dirs


def hashes_from_axtree(ax_path):
    """Возвращает множество (role, text_subtree) hash для одной страницы."""
    try:
        with gzip.open(ax_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()

    nodes = data.get("nodes", [])
    if not nodes:
        return set()

    try:
        nodes_by_id, children_by_id, parent_by_id, root_id = bc.build_indexes(nodes)
        stats = bc.compute_subtree_stats(root_id, nodes_by_id, children_by_id)
    except Exception:
        return set()

    page_hashes = set()
    for node_id, st in stats.items():
        text_subtree = st.get("text_subtree", "")
        if len(text_subtree) < MIN_TEXT_LEN:
            continue
        node = nodes_by_id.get(node_id, {})
        role = bc.get_role(node)
        h = compute_hash(role, text_subtree)
        page_hashes.add(h)
    return page_hashes


def main():
    print("=== Step 1: building global hash recurrence database ===")
    pages = list_pages()
    print(f"Pages to process: {len(pages)}")

    hash_counter = Counter()
    skipped = 0
    t0 = time.time()
    last_t = t0

    for i, page_dir in enumerate(pages, 1):
        ax_path = find_desktop_axtree(page_dir)
        if ax_path is None:
            skipped += 1
            continue
        page_hashes = hashes_from_axtree(ax_path)
        for h in page_hashes:
            hash_counter[h] += 1

        if i % PROGRESS_EVERY == 0 or i == len(pages):
            now = time.time()
            elapsed = now - t0
            rate = i / max(elapsed, 0.001)
            eta = (len(pages) - i) / max(rate, 0.001)
            print(f"  [{i}/{len(pages)}] {rate:.1f} pages/sec, elapsed {elapsed:.0f}s, eta {eta:.0f}s, "
                  f"unique hashes so far: {len(hash_counter)}, skipped: {skipped}")
            last_t = now

    total_pages = len(pages) - skipped
    print(f"\nProcessed: {total_pages} pages, skipped: {skipped}")
    print(f"Unique hashes: {len(hash_counter)}")

    # Save global hash database
    GLOBAL_HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "total_pages": total_pages,
        "unique_hashes": len(hash_counter),
        "hash_counts": dict(hash_counter),
    }
    with GLOBAL_HASH_FILE.open("w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False)
    print(f"Wrote global hash db to {GLOBAL_HASH_FILE}")

    # Distribution
    print("\n=== Global recurrence distribution ===")
    rec_dist = Counter(hash_counter.values())
    cumulative = [(t, sum(c for v, c in rec_dist.items() if v >= t))
                  for t in [1, 2, 3, 5, 10, 20, 50, 100, 500, 1000]]
    print(f"  Total unique hashes: {len(hash_counter)}")
    print("  Cumulative:")
    for t, n in cumulative:
        pct = 100.0 * n / len(hash_counter) if hash_counter else 0
        print(f"    recurrence >= {t:5d}: {n:7d} hashes ({pct:.2f}%)")

    # === Step 2: update candidates_topic_filtered.jsonl ===
    print(f"\n=== Step 2: updating {CANDIDATES_FILE} with global_recurrence ===")
    if not CANDIDATES_FILE.exists():
        print(f"ERROR: {CANDIDATES_FILE} not found, skipping step 2")
        return

    with CANDIDATES_FILE.open(encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(candidates)} candidates")

    matched = 0
    for c in candidates:
        h = c.get("content_hash")
        if h is None:
            # Не было прежней recurrence-обработки, считаем сейчас
            h = compute_hash(c.get("role"), c.get("text_subtree", ""))
            c["content_hash"] = h
        global_count = hash_counter.get(h, 0)
        c["global_recurrence_count"] = global_count
        c["global_recurrence_ratio"] = round(global_count / max(total_pages, 1), 5)
        if global_count > 0:
            matched += 1

    with CANDIDATES_FILE.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(candidates)} candidates ({matched} matched in global db) back to {CANDIDATES_FILE}")

    # Distribution среди НАШИХ кандидатов
    print("\n=== Global recurrence distribution among our candidates ===")
    cand_dist = Counter(c["global_recurrence_count"] for c in candidates)
    thresholds = [1, 2, 3, 5, 10, 20, 50, 100, 500, 1000]
    for t in thresholds:
        n = sum(1 for c in candidates if c["global_recurrence_count"] >= t)
        pct = 100.0 * n / len(candidates)
        print(f"  global_rec >= {t:5d}: {n:5d} candidates ({pct:.1f}%)")

    # Top recurring among our candidates
    print("\n=== Top 20 globally recurring blocks among our candidates ===")
    by_grec = sorted(candidates, key=lambda c: -c["global_recurrence_count"])
    seen_hashes = set()
    shown = 0
    for c in by_grec:
        h = c["content_hash"]
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        text = (c.get("text_subtree") or "")[:120].replace("\n", " ")
        role = c.get("role", "")
        print(f"  grec={c['global_recurrence_count']:>5d} | role={role:<18s} | hash={h} | {text!r}")
        shown += 1
        if shown >= 20:
            break


if __name__ == "__main__":
    main()
