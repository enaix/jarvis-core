"""
Pre-flight #2 — sanity check: репрезентативен ли пилот из 25 страниц?

Цель: взять 5 случайных страниц вне пилота, прогнать build_candidates,
сравнить статистику (кандидатов на страницу, role distribution, area coverage)
с пилотом. Если сильно разные — пилот не репрезентативен.

Запускать из notebooks/webui/, чтобы относительные импорты сработали.
"""

import json
import os
import random
import sys
from pathlib import Path
from collections import Counter

# Если запускаешь из notebooks/webui/ — должно работать
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # notebooks/webui/

from page_loader import Page, PageLoader, FileType  # noqa: E402
import build_candidates  # noqa: E402


# === Конфигурация ===
DATASET_ROOT = r"C:\Users\70133\.cache\huggingface\hub\datasets--biglab--webui-7k\snapshots\60f7b3c4b9409f75551664adc1564625dfc33c2e\dataset1"
PILOT_CANDIDATES = r"notebooks\webui\output\candidate\candidates.jsonl"
N_RANDOM_PAGES = 5
RANDOM_SEED = 42  # для воспроизводимости


def load_pilot_stats():
    """Считаем статистику пилота из существующего candidates.jsonl."""
    with open(PILOT_CANDIDATES, encoding="utf-8") as f:
        cands = [json.loads(line) for line in f]

    pages = sorted(set(c["page_id"] for c in cands))
    per_page = Counter(c["page_id"] for c in cands)
    roles = Counter(c.get("role", "") for c in cands)
    bbox_ratios = [c.get("bbox_ratio", 0.0) for c in cands]
    text_lens = [c.get("text_len", 0) for c in cands]

    return {
        "n_pages": len(pages),
        "n_candidates": len(cands),
        "page_ids": set(pages),
        "per_page_counts": list(per_page.values()),
        "roles": roles,
        "bbox_ratios": bbox_ratios,
        "text_lens": text_lens,
    }


def list_dataset_pages(dataset_root):
    """Возвращает список всех page_id в WebUI-кэше."""
    root = Path(dataset_root)
    # WebUI structure: dataset_root/<split>/<page_id>/<screen_type>/...
    # Adjust if structure differs.
    page_dirs = []
    for split_dir in root.iterdir():
        if not split_dir.is_dir():
            continue
        for page_dir in split_dir.iterdir():
            if page_dir.is_dir() and page_dir.name.isdigit():
                page_dirs.append(page_dir)
    return page_dirs


def build_candidates_for_page(page_dir, dataset_root):
    """Прогоняет build_candidates для одной страницы. Возвращает кандидатов или None при ошибке."""
    try:
        loader = PageLoader(str(page_dir), debug=False)
        candidates = build_candidates.build_candidates_for_loader(loader, dataset_root)
        return candidates
    except Exception as e:
        print(f"  ERROR на {page_dir.name}: {type(e).__name__}: {e}")
        return None


def page_stats(candidates):
    """Считаем per-page-stats для одного результата."""
    if not candidates:
        return None
    return {
        "n_candidates": len(candidates),
        "roles": Counter(c.get("role", "") for c in candidates),
        "bbox_ratios": [c.get("bbox_ratio", 0.0) for c in candidates],
        "text_lens": [c.get("text_len", 0) for c in candidates],
    }


def summarize_distribution(values, label):
    if not values:
        print(f"  {label}: empty")
        return
    sorted_v = sorted(values)
    n = len(sorted_v)
    print(
        f"  {label}: n={n}, min={min(sorted_v):.3f}, "
        f"median={sorted_v[n//2]:.3f}, max={max(sorted_v):.3f}, "
        f"mean={sum(sorted_v)/n:.3f}"
    )


def main():
    random.seed(RANDOM_SEED)

    print("=== Pilot statistics ===")
    pilot = load_pilot_stats()
    print(f"Pilot pages: {pilot['n_pages']}")
    print(f"Pilot candidates: {pilot['n_candidates']}")
    print(f"Pilot avg candidates/page: {pilot['n_candidates']/pilot['n_pages']:.1f}")
    summarize_distribution(pilot["per_page_counts"], "candidates_per_page")
    summarize_distribution(pilot["bbox_ratios"], "bbox_ratio")
    summarize_distribution(pilot["text_lens"], "text_len")
    print(f"Top 10 roles: {pilot['roles'].most_common(10)}")

    print()
    print("=== Listing dataset pages ===")
    all_pages = list_dataset_pages(DATASET_ROOT)
    print(f"Found {len(all_pages)} pages in dataset")

    available = [p for p in all_pages if p.name not in pilot["page_ids"]]
    print(f"Available (not in pilot): {len(available)}")

    if len(available) < N_RANDOM_PAGES:
        print(f"WARNING: only {len(available)} pages available, requested {N_RANDOM_PAGES}")
        N = len(available)
    else:
        N = N_RANDOM_PAGES

    random_pages = random.sample(available, N)
    print(f"Sampled pages: {[p.name for p in random_pages]}")

    print()
    print("=== Building candidates for random pages ===")
    random_results = []
    for i, page_dir in enumerate(random_pages, 1):
        print(f"[{i}/{N}] {page_dir.name}")
        cands = build_candidates_for_page(page_dir, DATASET_ROOT)
        if cands is None:
            continue
        stats = page_stats(cands)
        random_results.append((page_dir.name, stats))
        print(f"  -> {stats['n_candidates']} candidates")
        print(f"  -> top roles: {stats['roles'].most_common(5)}")

    print()
    print("=== Aggregate random vs pilot ===")
    if not random_results:
        print("No results, aborting")
        return

    rand_total = sum(r[1]["n_candidates"] for r in random_results)
    rand_per_page = [r[1]["n_candidates"] for r in random_results]
    rand_bbox_ratios = [v for _, s in random_results for v in s["bbox_ratios"]]
    rand_text_lens = [v for _, s in random_results for v in s["text_lens"]]
    rand_roles = Counter()
    for _, s in random_results:
        rand_roles.update(s["roles"])

    print(f"Random: {len(random_results)} pages, {rand_total} candidates, avg {rand_total/len(random_results):.1f}/page")
    print(f"Pilot:  {pilot['n_pages']} pages, {pilot['n_candidates']} candidates, avg {pilot['n_candidates']/pilot['n_pages']:.1f}/page")

    print()
    print("Per-page candidate counts:")
    summarize_distribution(rand_per_page, "  random")
    summarize_distribution(pilot["per_page_counts"], "  pilot ")

    print()
    print("bbox_ratio:")
    summarize_distribution(rand_bbox_ratios, "  random")
    summarize_distribution(pilot["bbox_ratios"], "  pilot ")

    print()
    print("text_len:")
    summarize_distribution(rand_text_lens, "  random")
    summarize_distribution(pilot["text_lens"], "  pilot ")

    print()
    print("Top roles:")
    print(f"  random: {rand_roles.most_common(10)}")
    print(f"  pilot : {pilot['roles'].most_common(10)}")

    print()
    print("=== Interpretation ===")
    rand_avg = rand_total / len(random_results)
    pilot_avg = pilot["n_candidates"] / pilot["n_pages"]
    if abs(rand_avg - pilot_avg) / pilot_avg > 0.5:
        print(f"WARNING: avg candidates/page differs >50% (random {rand_avg:.1f} vs pilot {pilot_avg:.1f})")
        print("Pilot may not be representative — consider expanding before annotation.")
    else:
        print(f"OK: avg candidates/page is in similar range (random {rand_avg:.1f} vs pilot {pilot_avg:.1f})")

    rand_top_role = rand_roles.most_common(1)[0][0] if rand_roles else None
    pilot_top_role = pilot["roles"].most_common(1)[0][0] if pilot["roles"] else None
    if rand_top_role != pilot_top_role:
        print(f"NOTE: top role differs: random={rand_top_role}, pilot={pilot_top_role}")


if __name__ == "__main__":
    main()
