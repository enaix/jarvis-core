"""
Расширить корпус кандидатов до ~100 страниц.

- Берёт существующие 25 пилотных страниц как seed (из candidates.jsonl)
- Сэмплирует N_NEW рандомных страниц вне пилота (seed=2026 для воспроизводимости)
- Прогоняет build_candidates_for_loader на каждой
- Пишет per-page *_candidates.jsonl в output/candidate/ (если уже есть — пропускает)
- Создаёт combined output/candidate/candidates_expanded.jsonl (pilot + new)
- В конце — diversity stats: сколько страниц с complementary/sidebar/banner ролями

Запускать из notebooks/webui/:
    .venv\Scripts\python.exe notebooks\webui\expand_corpus.py
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # notebooks/webui/

from page_loader import PageLoader  # noqa: E402
import build_candidates  # noqa: E402


# === Конфигурация ===
DATASET_ROOT = Path(
    r"C:\Users\70133\.cache\huggingface\hub\datasets--biglab--webui-7k\snapshots\60f7b3c4b9409f75551664adc1564625dfc33c2e\dataset1"
)
OUTPUT_DIR = Path(r"notebooks\webui\output\candidate")
PILOT_CANDIDATES = OUTPUT_DIR / "candidates.jsonl"
EXPANDED_CANDIDATES = OUTPUT_DIR / "candidates_expanded.jsonl"

N_NEW = 80               # сколько новых страниц добавить
RANDOM_SEED = 2026       # для воспроизводимости


def load_pilot_page_ids():
    if not PILOT_CANDIDATES.exists():
        return set()
    page_ids = set()
    with PILOT_CANDIDATES.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            page_ids.add(str(row["page_id"]))
    return page_ids


def list_dataset_pages():
    page_dirs = []
    for split_dir in DATASET_ROOT.iterdir():
        if not split_dir.is_dir():
            continue
        for page_dir in split_dir.iterdir():
            if page_dir.is_dir() and page_dir.name.isdigit():
                page_dirs.append(page_dir)
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

    pilot_ids = load_pilot_page_ids()
    print(f"Pilot pages: {len(pilot_ids)}")

    all_pages = list_dataset_pages()
    print(f"Dataset pages: {len(all_pages)}")

    available = [p for p in all_pages if p.name not in pilot_ids]
    print(f"Available (not in pilot): {len(available)}")

    n_new = min(N_NEW, len(available))
    new_pages = random.sample(available, n_new)
    print(f"Sampling {n_new} new pages")

    print()
    print("=== Building candidates ===")
    new_records = []
    skipped = []
    for i, page_dir in enumerate(new_pages, 1):
        page_id = page_dir.name
        per_page_file = per_page_jsonl_path(page_id)
        if per_page_file.exists():
            print(f"[{i}/{n_new}] {page_id}: per-page file exists, loading")
            with per_page_file.open(encoding="utf-8") as f:
                cands = [json.loads(line) for line in f if line.strip()]
            new_records.extend(cands)
            continue

        try:
            loader = PageLoader(str(page_dir), debug=False)
            cands = build_candidates.build_candidates_for_loader(loader, DATASET_ROOT)
        except Exception as e:
            print(f"[{i}/{n_new}] {page_id}: ERROR {type(e).__name__}: {e}")
            skipped.append((page_id, str(e)))
            continue

        if not cands:
            print(f"[{i}/{n_new}] {page_id}: 0 candidates, skipping write")
            skipped.append((page_id, "0 candidates"))
            continue

        write_jsonl(per_page_file, cands)
        new_records.extend(cands)
        if i <= 5 or i % 10 == 0 or i == n_new:
            top_role = Counter(c.get("role", "") for c in cands).most_common(1)
            print(f"[{i}/{n_new}] {page_id}: {len(cands)} candidates, top role: {top_role[0] if top_role else None}")

    print()
    print(f"New pages successfully processed: {n_new - len(skipped)}")
    print(f"Skipped: {len(skipped)}")
    if skipped:
        for pid, reason in skipped[:10]:
            print(f"  {pid}: {reason}")

    # === Combined file ===
    print()
    print("=== Building combined candidates_expanded.jsonl ===")
    pilot_records = []
    if PILOT_CANDIDATES.exists():
        with PILOT_CANDIDATES.open(encoding="utf-8") as f:
            pilot_records = [json.loads(line) for line in f if line.strip()]

    combined = pilot_records + new_records
    write_jsonl(EXPANDED_CANDIDATES, combined)
    print(f"Wrote {len(combined)} candidates ({len(pilot_records)} pilot + {len(new_records)} new) to {EXPANDED_CANDIDATES}")

    # === Diversity stats ===
    print()
    print("=== Diversity stats (pilot vs new vs combined) ===")
    sets = [
        ("pilot", pilot_records),
        ("new", new_records),
        ("combined", combined),
    ]
    for name, rows in sets:
        if not rows:
            print(f"{name}: empty")
            continue
        n_pages = len({r["page_id"] for r in rows})
        n_cands = len(rows)
        roles = Counter(r.get("role", "") for r in rows)
        # Сколько страниц содержат хотя бы один из ключевых ролей junk-кандидатов
        pages_with_role = {}
        for role_target in ("complementary", "navigation", "banner", "contentinfo", "Section"):
            pages_with = {r["page_id"] for r in rows if r.get("role", "") == role_target}
            pages_with_role[role_target] = len(pages_with)

        print(f"{name}: {n_pages} pages, {n_cands} candidates, avg {n_cands/n_pages:.1f}/page")
        print(f"  top roles: {roles.most_common(8)}")
        print(f"  pages containing role:")
        for role, count in pages_with_role.items():
            pct = 100.0 * count / n_pages if n_pages else 0
            print(f"    {role}: {count}/{n_pages} ({pct:.0f}%)")
        print()


if __name__ == "__main__":
    main()
