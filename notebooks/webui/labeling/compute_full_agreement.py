"""Считает agreement между человеком и LLM на полном overlap (без API calls)."""
import json
from collections import Counter, defaultdict
from pathlib import Path

HUMAN = Path(r"notebooks\webui\annotator\output\labels.jsonl")
LLM = Path(r"notebooks\webui\annotator\output\labels_llm.jsonl")
REPORT = Path(r"notebooks\webui\annotator\output\agreement_report_full.json")


def load(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    h_rows = load(HUMAN)
    l_rows = load(LLM)

    h_idx = {f"{r['page_id']}::{r['node_id']}": r for r in h_rows}
    l_idx = {f"{r['page_id']}::{r['node_id']}": r for r in l_rows}

    common = set(h_idx) & set(l_idx)
    print(f"Human labels: {len(h_idx)}")
    print(f"LLM labels: {len(l_idx)}")
    print(f"Common candidates (both labeled): {len(common)}")

    if not common:
        return

    cm = defaultdict(int)
    junk_sub_cm = defaultdict(int)
    for k in common:
        h = h_idx[k]
        l = l_idx[k]
        cm[(h["label"], l["label"])] += 1
        if h["label"] == "junk" and l["label"] == "junk":
            hs = h.get("sub_label", "other")
            ls = l.get("sub_label", "other")
            junk_sub_cm[(hs, ls)] += 1

    total = len(common)
    label_match = sum(c for (hl, ll), c in cm.items() if hl == ll)
    label_agreement = label_match / total

    bin_total = sum(c for (hl, ll), c in cm.items()
                    if hl in ("junk", "not_junk") and ll in ("junk", "not_junk"))
    bin_match = sum(c for (hl, ll), c in cm.items()
                    if hl == ll and hl in ("junk", "not_junk"))
    bin_agreement = bin_match / bin_total if bin_total else 0

    junk_total = sum(junk_sub_cm.values())
    junk_sub_match = sum(c for (hs, ls), c in junk_sub_cm.items() if hs == ls)
    sub_agreement = junk_sub_match / junk_total if junk_total else 0

    print(f"\n=== Confusion matrix (rows=human, cols=LLM) ===")
    labels_set = sorted({l for (h, l) in cm.keys()} | {h for (h, l) in cm.keys()})
    header = "human \\ LLM"
    print(f"  {header:<15s} " + " ".join(f"{l:>10s}" for l in labels_set))
    for hl in labels_set:
        row = [f"{hl:<15s} "]
        for ll in labels_set:
            row.append(f"{cm.get((hl, ll), 0):>10d}")
        print(" ".join(row))

    # FN / FP rates by class (вы LLM ошибается на junk vs not_junk)
    h_junk = sum(c for (hl, ll), c in cm.items() if hl == "junk")
    h_not_junk = sum(c for (hl, ll), c in cm.items() if hl == "not_junk")
    llm_missed_junk = sum(c for (hl, ll), c in cm.items() if hl == "junk" and ll == "not_junk")
    llm_over_flagged = sum(c for (hl, ll), c in cm.items() if hl == "not_junk" and ll == "junk")

    print(f"\n=== Error rates (human as ground truth) ===")
    print(f"  LLM missed junk (FN by LLM): {llm_missed_junk}/{h_junk} = {100*llm_missed_junk/max(h_junk,1):.1f}%")
    print(f"  LLM over-flagged (FP by LLM): {llm_over_flagged}/{h_not_junk} = {100*llm_over_flagged/max(h_not_junk,1):.1f}%")

    print(f"\n=== Agreement summary ===")
    print(f"  3-way label agreement (junk/not_junk/skip): {label_agreement:.3f} ({label_match}/{total})")
    print(f"  Binary agreement (junk vs not_junk):        {bin_agreement:.3f} ({bin_match}/{bin_total})")
    print(f"  Sub-label agreement (among junk-junk):       {sub_agreement:.3f} ({junk_sub_match}/{junk_total})")

    # Sub-label confusion (top mismatches)
    print(f"\n=== Sub-label confusion matrix (among junk-junk) ===")
    sub_labels_set = sorted({s for (h, l) in junk_sub_cm.keys() for s in (h, l)})
    sub_header = "human \\ LLM"
    print(f"  {sub_header:<14s} " + " ".join(f"{s:>10s}" for s in sub_labels_set))
    for hs in sub_labels_set:
        row = [f"{hs:<14s} "]
        for ls in sub_labels_set:
            row.append(f"{junk_sub_cm.get((hs, ls), 0):>10d}")
        print(" ".join(row))

    # Agreement by class
    print(f"\n=== Agreement by class ===")
    for cls in ("junk", "not_junk", "skip"):
        total_cls = sum(c for (hl, _), c in cm.items() if hl == cls)
        match_cls = sum(c for (hl, ll), c in cm.items() if hl == cls and ll == cls)
        if total_cls:
            print(f"  {cls:10s}: LLM agrees on {match_cls}/{total_cls} = {100*match_cls/total_cls:.1f}%")

    report = {
        "n_common": total,
        "label_agreement": label_agreement,
        "binary_agreement": bin_agreement,
        "sub_label_agreement": sub_agreement,
        "llm_fn_junk_rate": llm_missed_junk / max(h_junk, 1),
        "llm_fp_rate": llm_over_flagged / max(h_not_junk, 1),
        "confusion_matrix": {f"{h}|{l}": c for (h, l), c in cm.items()},
        "junk_subclass_matrix": {f"{h}|{l}": c for (h, l), c in junk_sub_cm.items()},
    }
    with REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to {REPORT}")


if __name__ == "__main__":
    main()
