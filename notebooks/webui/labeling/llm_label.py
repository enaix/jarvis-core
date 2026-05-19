"""
LLM-based labeling для variant A pipeline.

Прогоняет GPT-4o на candidates с теми же правилами что у разметчика
(см. claude-output/filtering_annotation_guidelines.md).

Два режима:

1. AGREEMENT mode (--agreement) — для проверки качества LLM:
   читает существующий labels.jsonl (human labels), берёт те же candidates,
   запускает LLM, считает agreement.

2. BULK mode (default) — для массовой разметки:
   читает все candidates из candidates_topic_filtered.jsonl, исключает
   уже размеченные human, размечает оставшиеся через LLM.

Результаты:
- labels_llm.jsonl — формат как labels.jsonl плюс llm_confidence, llm_reason
- agreement_report.json — статистика согласованности (только в agreement mode)

Зависимости:
    pip install openai

Переменная окружения:
    OPENAI_API_KEY — твой ключ от OpenAI

Запуск:
    # Сначала agreement check на разметанных 200:
    .venv\\Scripts\\python.exe notebooks\\webui\\llm_label.py --agreement

    # Потом bulk на оставшихся:
    .venv\\Scripts\\python.exe notebooks\\webui\\llm_label.py --bulk --max 2500
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)


# === Конфигурация ===
CANDIDATES_FILE = Path(r"notebooks\webui\annotator\candidates_topic_filtered.jsonl")
HUMAN_LABELS_FILE = Path(r"notebooks\webui\annotator\output\labels.jsonl")
LLM_LABELS_FILE = Path(r"notebooks\webui\annotator\output\labels_llm.jsonl")
AGREEMENT_REPORT_FILE = Path(r"notebooks\webui\annotator\output\agreement_report.json")

MODEL = "gpt-4o-2024-11-20"  # фиксированная версия для воспроизводимости
TEMPERATURE = 0.0
CONCURRENCY = 3  # параллельных запросов (низкий из-за TPM лимита 30k/min на free tier)
MAX_TEXT_CHARS = 2000  # обрезка text_subtree
MAX_RETRIES = 6  # для 429 / network errors

SYSTEM_PROMPT = """\
You are a data annotator for a web page filtering task. The goal is to clean an AXTree-based dataset of \
article-style web pages to train a multimodal language model that generates web-style article content. \
You classify candidate AXTree blocks as junk (should be removed) or not_junk (should be kept).

Definitions:
- "junk" — boilerplate or non-article UI: cookie banners, footers (copyright, legal links), \
  site navigation menus, sidebars, social share widgets, ads, newsletter signups, recurring \
  promotional blocks. These should be removed.
- "not_junk" — article content, hero sections with meaningful text, product descriptions, \
  forms (contact, search), comments under articles, headlines, body text. These should be kept.
- "skip" — only when the candidate is technically not labelable: empty content, screenshot \
  broken, bbox doesn't match visible content. NOT for "I am uncertain" — pick junk or not_junk \
  with the dominant label using priority rules below.

Priority rules for ambiguous cases:
1. If candidate is recognizable cross-page UI construction (header, footer, nav, cookie banner, \
   share buttons), it is junk regardless of unique words inside.
2. If candidate is the page's main content (article body, product description, form, main \
   heading), it is not_junk even if small.
3. If candidate is a coarse container mixing junk and content: assign by what dominates >70%. \
   If ~50/50, output skip.
4. Empty layout-wrapper (no text, no descendants) is junk with sub_label "other".

For junk, also output sub_label from this fixed set:
- "cookie" — cookie/GDPR consent
- "footer" — copyright, terms/privacy links, made-with
- "nav" — navigation menu, breadcrumbs
- "sidebar" — sidebar navigation, related-links column when no unique content
- "social" — social share buttons, follow-us, embed social widgets
- "legal" — standalone legal blocks
- "ad" — advertisement, promo, newsletter signup
- "recurring" — block that recurs across many pages (use when global_recurrence ≥ 5)
- "other" — junk that doesn't fit above categories

For not_junk and skip, sub_label must be empty string.

Strong signal: if global_recurrence_count ≥ 5, the block recurs across the WebUI dataset \
on that many pages — this is a very strong signal for junk (especially nav/footer/ad).

Output JSON only. Be decisive — pick junk or not_junk unless the candidate is truly unlabelable.
"""

USER_PROMPT_TEMPLATE = """\
Candidate metadata:
- AXTree role: {role}
- text_len: {text_len}
- num_descendants: {num_descendants}
- link_density: {link_density}
- bbox normalized: x={x_norm}, y={y_norm}, w={w_norm}, h={h_norm}
- bbox_area_ratio: {bbox_ratio}
- global_recurrence_count: {grec} (out of 5420 WebUI pages)
- global_recurrence_ratio: {gratio}

Candidate text content (truncated):
{text}

Classify this candidate. Output JSON: {{"label": "junk|not_junk|skip", "sub_label": "<one of the sub_labels or empty>", "confidence": 0.0-1.0, "reason": "short explanation under 30 words"}}\
"""


JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["junk", "not_junk", "skip"],
        },
        "sub_label": {
            "type": "string",
            "enum": ["", "cookie", "footer", "sidebar", "nav", "social", "legal", "ad", "recurring", "other"],
        },
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["label", "sub_label", "confidence", "reason"],
    "additionalProperties": False,
}


def load_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def candidate_key(c):
    return f"{c['page_id']}::{c['node_id']}"


def build_user_prompt(cand):
    bbox = cand.get("bbox", {})
    # Normalize bbox if width/height info available; otherwise use raw
    x = bbox.get("x", 0)
    y = bbox.get("y", 0)
    w = bbox.get("width", 0)
    h = bbox.get("height", 0)
    # We don't know page W/H from candidate alone, but bbox_ratio is the area ratio
    # so we report raw bbox + ratios
    text = (cand.get("text_subtree") or "")[:MAX_TEXT_CHARS]
    return USER_PROMPT_TEMPLATE.format(
        role=cand.get("role", ""),
        text_len=cand.get("text_len", 0),
        num_descendants=cand.get("num_descendants", 0),
        link_density=round(cand.get("link_density", 0), 3),
        x_norm=int(x),
        y_norm=int(y),
        w_norm=int(w),
        h_norm=int(h),
        bbox_ratio=round(cand.get("bbox_ratio", 0), 4),
        grec=cand.get("global_recurrence_count", 0),
        gratio=round(cand.get("global_recurrence_ratio", 0), 5),
        text=text,
    )


async def label_one(client, cand, sem):
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(cand)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "candidate_label",
                        "strict": True,
                        "schema": JSON_SCHEMA,
                    },
                },
            )
            content = resp.choices[0].message.content
            parsed = json.loads(content)
            return cand, parsed, None
        except Exception as e:
            return cand, None, str(e)


def make_label_row(cand, parsed):
    return {
        "page_id": str(cand["page_id"]),
        "node_id": str(cand["node_id"]),
        "label": parsed["label"],
        "sub_label": parsed["sub_label"] if parsed["label"] == "junk" else "",
        "content_hash": cand.get("content_hash", ""),
        "generator_version": cand.get("generator_version", "from_corpus"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "role": cand.get("role"),
        "text_len": cand.get("text_len"),
        "llm_model": MODEL,
        "llm_confidence": parsed.get("confidence", 0),
        "llm_reason": parsed.get("reason", ""),
    }


async def run_labeling(candidates, max_count=None, write_incrementally=True):
    """Запускает LLM labeling. С write_incrementally=True пишет каждый результат
    в LLM_LABELS_FILE сразу как получает — interrupt-safe."""
    if max_count is not None:
        candidates = candidates[:max_count]

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY env var not set")
        sys.exit(1)
    # max_retries — встроенный exponential backoff на 429 / network errors
    client = AsyncOpenAI(api_key=api_key, max_retries=MAX_RETRIES, timeout=60.0)

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [label_one(client, c, sem) for c in candidates]

    print(f"Submitting {len(tasks)} requests with concurrency={CONCURRENCY}, model={MODEL}")
    if write_incrementally:
        LLM_LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
        out_file = LLM_LABELS_FILE.open("a", encoding="utf-8")
        print(f"Writing incrementally (append mode) to {LLM_LABELS_FILE}")
    else:
        out_file = None

    t0 = time.time()
    results = []
    try:
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            cand, parsed, err = await coro
            if err:
                print(f"  [{i}/{len(tasks)}] {candidate_key(cand)} ERROR: {err}")
                continue
            results.append((cand, parsed))
            if out_file is not None:
                row = make_label_row(cand, parsed)
                out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_file.flush()  # сразу на диск, без буфера
            if i <= 5 or i % 25 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 0.001)
                print(f"  [{i}/{len(tasks)}] {rate:.1f} req/sec, elapsed {elapsed:.0f}s, "
                      f"last: {parsed['label']}/{parsed['sub_label']} conf={parsed['confidence']:.2f}")
    finally:
        if out_file is not None:
            out_file.close()

    print(f"\nProcessed {len(results)}/{len(tasks)} successfully in {time.time()-t0:.0f}s")
    return results


def write_llm_labels(results, append=False):
    """Legacy bulk-write. Используется только если run_labeling запущена с write_incrementally=False."""
    LLM_LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    rows = [make_label_row(c, p) for c, p in results]
    with LLM_LABELS_FILE.open(mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} LLM labels to {LLM_LABELS_FILE} (mode={mode})")


def compute_agreement(human_labels, llm_labels):
    """Сравнить человека и LLM на пересечении candidate'ов."""
    human_idx = {f"{r['page_id']}::{r['node_id']}": r for r in human_labels}
    llm_idx = {f"{r['page_id']}::{r['node_id']}": r for r in llm_labels}

    common = set(human_idx) & set(llm_idx)
    print(f"Human labels: {len(human_idx)}, LLM labels: {len(llm_idx)}, common: {len(common)}")

    if not common:
        print("No common candidates to compare")
        return {}

    # Confusion matrix
    cm = defaultdict(int)  # (human_label, llm_label) -> count
    junk_cm = defaultdict(int)  # for junk: (human_sub, llm_sub) -> count
    for k in common:
        h = human_idx[k]
        l = llm_idx[k]
        cm[(h["label"], l["label"])] += 1
        if h["label"] == "junk" and l["label"] == "junk":
            junk_cm[(h.get("sub_label", "other"), l.get("sub_label", "other"))] += 1

    # Overall agreement (label only)
    total = len(common)
    label_match = sum(c for (hl, ll), c in cm.items() if hl == ll)
    label_agreement = label_match / total if total else 0

    # Binary junk vs not_junk (excluding skip)
    bin_total = sum(c for (hl, ll), c in cm.items() if hl in ("junk", "not_junk") and ll in ("junk", "not_junk"))
    bin_match = sum(c for (hl, ll), c in cm.items() if hl == ll and hl in ("junk", "not_junk"))
    bin_agreement = bin_match / bin_total if bin_total else 0

    # Sub-label agreement (only among junk-junk)
    junk_total = sum(junk_cm.values())
    junk_sub_match = sum(c for (hs, ls), c in junk_cm.items() if hs == ls)
    sub_agreement = junk_sub_match / junk_total if junk_total else 0

    print(f"\n=== Confusion matrix (human -> LLM) ===")
    labels_set = sorted({l for (h, l) in cm.keys()} | {h for (h, l) in cm.keys()})
    header_label = "human \\ LLM"
    print(f"  {header_label:<15s} " + " ".join(f"{l:>10s}" for l in labels_set))
    for hl in labels_set:
        row = [f"{hl:<15s} "]
        for ll in labels_set:
            row.append(f"{cm.get((hl, ll), 0):>10d}")
        print(" ".join(row))

    print(f"\n=== Agreement summary ===")
    print(f"  Label agreement (3-way junk/not_junk/skip): {label_agreement:.3f} ({label_match}/{total})")
    print(f"  Binary agreement (junk vs not_junk only):   {bin_agreement:.3f} ({bin_match}/{bin_total})")
    print(f"  Sub-label agreement (among junk):           {sub_agreement:.3f} ({junk_sub_match}/{junk_total})")

    print(f"\n=== Disagreement examples (up to 10) ===")
    disagreements = [(k, human_idx[k], llm_idx[k]) for k in common
                     if human_idx[k]["label"] != llm_idx[k]["label"]]
    for k, h, l in disagreements[:10]:
        print(f"  {k}: human={h['label']}/{h.get('sub_label','')}, "
              f"LLM={l['label']}/{l.get('sub_label','')} | "
              f"reason: {l.get('llm_reason','')[:100]}")

    report = {
        "model": MODEL,
        "n_common": total,
        "label_agreement": label_agreement,
        "binary_agreement": bin_agreement,
        "sub_label_agreement": sub_agreement,
        "confusion_matrix": {f"{h}|{l}": c for (h, l), c in cm.items()},
        "junk_subclass_matrix": {f"{h}|{l}": c for (h, l), c in junk_cm.items()},
    }

    AGREEMENT_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with AGREEMENT_REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to {AGREEMENT_REPORT_FILE}")
    return report


async def main_agreement():
    """Прогнать LLM на тех же candidate'ах что и человек, считать agreement.
    Resume-friendly: пропускает уже размеченные LLM ранее."""
    human_labels = load_jsonl(HUMAN_LABELS_FILE)
    if not human_labels:
        print(f"ERROR: {HUMAN_LABELS_FILE} is empty or missing")
        sys.exit(1)

    candidates = load_jsonl(CANDIDATES_FILE)
    cand_idx = {candidate_key(c): c for c in candidates}

    existing_llm = load_jsonl(LLM_LABELS_FILE)
    already_done = {f"{r['page_id']}::{r['node_id']}" for r in existing_llm}

    targets = []
    for hl in human_labels:
        k = f"{hl['page_id']}::{hl['node_id']}"
        if k in cand_idx and k not in already_done:
            targets.append(cand_idx[k])

    print(f"Human labels: {len(human_labels)}, already LLM-labeled: {len(already_done)}, "
          f"need LLM on: {len(targets)}")

    if targets:
        await run_labeling(targets)  # пишет incrementally в LLM_LABELS_FILE

    llm_labels = load_jsonl(LLM_LABELS_FILE)
    compute_agreement(human_labels, llm_labels)


PAGE_FLAGS_FILE = Path(r"notebooks\webui\annotator\output\page_flags.jsonl")
MIN_CANDIDATES_PER_PAGE = 3  # pages с меньшим числом candidates пропускаются (likely generator failure)


async def main_bulk(max_count=None):
    """Разметить все ещё не размеченные (ни человеком, ни LLM) candidate'ы.

    Фильтры:
    - Skip candidates чьих pages нет в page_flags.jsonl с меткой bad_page
    - Skip candidates чьих pages генератор дал < MIN_CANDIDATES_PER_PAGE
    - Skip уже размеченные (human или LLM)
    """
    human_labels = load_jsonl(HUMAN_LABELS_FILE)
    existing_llm = load_jsonl(LLM_LABELS_FILE)
    candidates = load_jsonl(CANDIDATES_FILE)
    page_flags = load_jsonl(PAGE_FLAGS_FILE)

    already = set()
    for r in human_labels:
        already.add(f"{r['page_id']}::{r['node_id']}")
    for r in existing_llm:
        already.add(f"{r['page_id']}::{r['node_id']}")

    # Bad pages from human flags
    bad_pages = {str(r["page_id"]) for r in page_flags if r.get("page_action") == "bad_page"}
    print(f"Bad pages flagged by human: {len(bad_pages)}")

    # Pages with too few candidates (likely generator failure)
    page_candidate_counts = {}
    for c in candidates:
        page_candidate_counts[str(c["page_id"])] = page_candidate_counts.get(str(c["page_id"]), 0) + 1
    low_candidate_pages = {pid for pid, cnt in page_candidate_counts.items() if cnt < MIN_CANDIDATES_PER_PAGE}
    print(f"Pages with <{MIN_CANDIDATES_PER_PAGE} candidates: {len(low_candidate_pages)}")

    skip_pages = bad_pages | low_candidate_pages
    print(f"Total pages to skip: {len(skip_pages)}")

    targets = []
    skipped_already = 0
    skipped_bad_page = 0
    skipped_low_count = 0
    for c in candidates:
        k = candidate_key(c)
        pid = str(c["page_id"])
        if k in already:
            skipped_already += 1
            continue
        if pid in bad_pages:
            skipped_bad_page += 1
            continue
        if pid in low_candidate_pages:
            skipped_low_count += 1
            continue
        targets.append(c)

    print(f"Total candidates: {len(candidates)}")
    print(f"  Skipped already-labeled: {skipped_already}")
    print(f"  Skipped bad_page: {skipped_bad_page}")
    print(f"  Skipped low-candidate page: {skipped_low_count}")
    print(f"  To label by LLM: {len(targets)}")

    if max_count and max_count < len(targets):
        targets = targets[:max_count]
        print(f"Limited to {max_count} for this run")

    if not targets:
        print("Nothing to do")
        return

    await run_labeling(targets)  # пишет incrementally в LLM_LABELS_FILE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agreement", action="store_true", help="Run LLM on human-labeled subset and compute agreement")
    parser.add_argument("--bulk", action="store_true", help="Bulk-label all unlabeled candidates with LLM")
    parser.add_argument("--max", type=int, default=None, help="Max candidates to process in bulk mode")
    args = parser.parse_args()

    if args.agreement:
        asyncio.run(main_agreement())
    elif args.bulk:
        asyncio.run(main_bulk(args.max))
    else:
        parser.print_help()
        print("\nProvide --agreement OR --bulk")
