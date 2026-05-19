"""
analysis.py - pairwise comparison layer for the ZS/FS experiment.

Consumes outputs/raw_outputs.jsonl, re-parses each output into a widget tree
via metrics.to_widget, and computes:

  1. Variance within a condition - for each (question, type, shots, seed_mode)
     cell, the mean pairwise TED and embedding distance between its runs.
     Aggregated to one structural + one semantic variance per
     (type, shots, seed_mode).

  2. Cross-condition similarity - how far apart outputs are when exactly one
     axis changes:
       type divergence : t1 vs t2  (holding shots, seed_mode, question)
       shot divergence : zs vs fs  (holding type, seed_mode, question)

  3. Explicit answers to the three guiding questions:
       - does few-shot reduce variance?
       - do Type 1 and Type 2 differ?
       - how does seed_mode (temperature) affect variance?

Writes outputs/analysis.json and prints summary tables.

Requires metrics.py (and through it: zss, sentence-transformers, numpy).
"""
from __future__ import annotations

import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path

import metrics as M

HERE = Path(__file__).parent
RAW_PATH = HERE / "outputs" / "raw_outputs.jsonl"
ANALYSIS_PATH = HERE / "outputs" / "analysis.json"

# expected ascending-variance order; used only for the seed_mode sanity check
SEED_ORDER = ["fixed", "semi", "random"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_records():
    """Re-parse raw_outputs.jsonl into
        records[(qid, type, shots, seed_mode)] = [run dict, ...]
    where each run dict is {run_idx, widget|None, status, text}."""
    rows = M.read_jsonl(RAW_PATH)
    if not rows:
        raise SystemExit(f"no rows at {RAW_PATH} - run pipeline.py first")
    records = defaultdict(list)
    for r in rows:
        widget, info = M.parse_output(r)
        records[(r["question_id"], r["type"], r["shots"], r["seed_mode"])].append({
            "run_idx": r.get("run_idx"),
            "widget": widget,
            "status": info["status"],
            "text": M.extract_text(widget) if widget is not None else "",
        })
    return records


def build_embedding_cache(records):
    """Batch-encode every distinct non-empty extracted text exactly once, so
    pairwise semantic distance is a cheap dot product afterwards."""
    texts = sorted({run["text"] for runs in records.values()
                    for run in runs if run["text"].strip()})
    if not texts:
        return {}
    vecs = M.get_embedder().encode(texts, normalize_embeddings=True)
    return dict(zip(texts, vecs))


# --------------------------------------------------------------------------
# Distance helpers
# --------------------------------------------------------------------------
def make_emb_distance(cache):
    """Return a (text_a, text_b) -> distance function backed by the cache."""
    import numpy as np

    def dist(ta, tb):
        a, b = ta.strip(), tb.strip()
        if not a and not b:
            return 0.0
        if not a or not b:
            return 1.0
        return 1.0 - float(np.dot(cache[ta], cache[tb]))

    return dist


def mean_pairwise(items, dist_fn):
    """Mean distance over all unordered pairs within `items`. None if < 2."""
    pairs = list(itertools.combinations(items, 2))
    if not pairs:
        return None
    return statistics.fmean(dist_fn(a, b) for a, b in pairs)


def mean_cross(items_a, items_b, dist_fn):
    """Mean distance over all cross pairs items_a x items_b. None if either
    side is empty."""
    if not items_a or not items_b:
        return None
    return statistics.fmean(dist_fn(a, b) for a in items_a for b in items_b)


def _agg(values):
    """Mean of the non-None values, plus how many contributed."""
    vals = [v for v in values if v is not None]
    return {
        "mean": statistics.fmean(vals) if vals else None,
        "n": len(vals),
    }


# --------------------------------------------------------------------------
# 1. Variance within a condition
# --------------------------------------------------------------------------
def compute_variance(records, edist):
    """Per-cell variance, then aggregated per (type, shots, seed_mode)."""
    cell_rows = []
    for (qid, typ, shots, sm), runs in records.items():
        valid = [r for r in runs if r["widget"] is not None]
        widgets = [r["widget"] for r in valid]
        texts = [r["text"] for r in valid]
        cell_rows.append({
            "question_id": qid, "type": typ, "shots": shots, "seed_mode": sm,
            "n_runs": len(runs), "n_valid": len(valid),
            "structural_variance": mean_pairwise(widgets, M.ted),
            "semantic_variance": mean_pairwise(texts, edist),
        })

    per_condition = []
    by_cond = defaultdict(list)
    for row in cell_rows:
        by_cond[(row["type"], row["shots"], row["seed_mode"])].append(row)
    for (typ, shots, sm), rows in sorted(by_cond.items()):
        s = _agg([r["structural_variance"] for r in rows])
        m = _agg([r["semantic_variance"] for r in rows])
        per_condition.append({
            "type": typ, "shots": shots, "seed_mode": sm,
            "n_cells": len(rows),
            "n_cells_with_variance": s["n"],
            "mean_structural_variance": s["mean"],
            "mean_semantic_variance": m["mean"],
        })
    return cell_rows, per_condition


# --------------------------------------------------------------------------
# 2. Cross-condition similarity
# --------------------------------------------------------------------------
def compute_cross_condition(records, edist):
    """Type divergence for ALL pairs of types (t1_t2, t1_t3, t2_t3 when
    present); shot divergence (zs vs fs) per type.

    Returns:
      type_pair_rows: dict {pair_name -> [per-cell rows]} for every unordered
                     pair of types present in the data
      shot_rows: flat list of per-cell rows (one per (qid, type, sm))
    """
    def valid_widgets(key):
        return [r["widget"] for r in records.get(key, []) if r["widget"] is not None]

    def valid_texts(key):
        return [r["text"] for r in records.get(key, []) if r["widget"] is not None]

    qids = sorted({k[0] for k in records})
    types = sorted({k[1] for k in records})
    shots_levels = sorted({k[2] for k in records})
    seed_modes = sorted({k[3] for k in records})

    type_pair_rows = {}
    for i, ta in enumerate(types):
        for tb in types[i + 1:]:
            rows = []
            for qid in qids:
                for shots in shots_levels:
                    for sm in seed_modes:
                        ka = (qid, ta, shots, sm)
                        kb = (qid, tb, shots, sm)
                        rows.append({
                            "question_id": qid, "shots": shots, "seed_mode": sm,
                            "structural_divergence": mean_cross(
                                valid_widgets(ka), valid_widgets(kb), M.ted),
                            "semantic_divergence": mean_cross(
                                valid_texts(ka), valid_texts(kb), edist),
                        })
            type_pair_rows[f"{ta}_{tb}"] = rows

    shot_rows = []
    if "zs" in shots_levels and "fs" in shots_levels:
        for qid in qids:
            for typ in types:
                for sm in seed_modes:
                    ka = (qid, typ, "zs", sm)
                    kb = (qid, typ, "fs", sm)
                    shot_rows.append({
                        "question_id": qid, "type": typ, "seed_mode": sm,
                        "structural_divergence": mean_cross(
                            valid_widgets(ka), valid_widgets(kb), M.ted),
                        "semantic_divergence": mean_cross(
                            valid_texts(ka), valid_texts(kb), edist),
                    })

    return type_pair_rows, shot_rows


# --------------------------------------------------------------------------
# 3. Guiding questions
# --------------------------------------------------------------------------
def guiding_answers(per_condition, type_pair_rows, shot_rows):
    def cond_mean(filter_fn, key):
        return _agg([c[key] for c in per_condition if filter_fn(c)])

    # Q1: does few-shot reduce variance?
    zs_s = cond_mean(lambda c: c["shots"] == "zs", "mean_structural_variance")
    fs_s = cond_mean(lambda c: c["shots"] == "fs", "mean_structural_variance")
    zs_m = cond_mean(lambda c: c["shots"] == "zs", "mean_semantic_variance")
    fs_m = cond_mean(lambda c: c["shots"] == "fs", "mean_semantic_variance")
    q1 = {
        "zs_structural_variance": zs_s["mean"],
        "fs_structural_variance": fs_s["mean"],
        "zs_semantic_variance": zs_m["mean"],
        "fs_semantic_variance": fs_m["mean"],
        "few_shot_reduces_structural_variance":
            (fs_s["mean"] < zs_s["mean"]) if None not in (fs_s["mean"], zs_s["mean"]) else None,
        "few_shot_reduces_semantic_variance":
            (fs_m["mean"] < zs_m["mean"]) if None not in (fs_m["mean"], zs_m["mean"]) else None,
    }

    # Q2: do the architectural types differ? Per-pair structural/semantic
    # divergence + within-condition variance for each participating type.
    by_type_var = {}
    for typ in sorted({c["type"] for c in per_condition}):
        by_type_var[typ] = cond_mean(
            lambda c, t=typ: c["type"] == t, "mean_structural_variance")["mean"]
    pair_divergences = {}
    for pair_name, rows in type_pair_rows.items():
        pair_divergences[pair_name] = {
            "mean_structural_divergence": _agg([r["structural_divergence"] for r in rows])["mean"],
            "mean_semantic_divergence": _agg([r["semantic_divergence"] for r in rows])["mean"],
        }
    q2 = {
        "within_type_structural_variance": by_type_var,
        "type_pair_divergence": pair_divergences,
        "note": ("compare each pair's divergence against the within-type "
                 "variances of its members; divergence > variance means the "
                 "two pipelines produce systematically different widgets"),
    }

    # Q3: how does seed_mode affect variance?
    by_sm = {}
    for sm in SEED_ORDER:
        s = cond_mean(lambda c, sm=sm: c["seed_mode"] == sm, "mean_structural_variance")
        m = cond_mean(lambda c, sm=sm: c["seed_mode"] == sm, "mean_semantic_variance")
        if s["n"] or m["n"]:
            by_sm[sm] = {"structural": s["mean"], "semantic": m["mean"]}
    present = [sm for sm in SEED_ORDER if sm in by_sm]
    s_seq = [by_sm[sm]["structural"] for sm in present]
    monotonic = (len(s_seq) >= 2 and all(x is not None for x in s_seq)
                 and all(a <= b for a, b in zip(s_seq, s_seq[1:])))
    q3 = {
        "by_seed_mode": by_sm,
        "structural_variance_monotonic_in_temperature": monotonic,
        "note": ("expected ordering fixed <= semi <= random; "
                 "fixed is the near-deterministic control"),
    }

    # shot divergence summary (companion to Q1)
    shot_struct = _agg([r["structural_divergence"] for r in shot_rows])
    shot_seman = _agg([r["semantic_divergence"] for r in shot_rows])

    return {
        "few_shot_reduces_variance": q1,
        "type1_vs_type2_differ": q2,
        "seed_mode_effect": q3,
        "shot_divergence_summary": {
            "mean_structural_divergence": shot_struct["mean"],
            "mean_semantic_divergence": shot_seman["mean"],
        },
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _fmt(x):
    return "  n/a" if x is None else f"{x:.3f}"


def print_summary(per_condition, type_pair_rows, shot_rows, guiding):
    print("\n=== variance within condition ===")
    print(f"{'type':<5}{'shots':<6}{'seed':<8}{'struct':>9}{'seman':>9}"
          f"{'cells':>7}{'cov':>6}")
    print("-" * 50)
    for c in per_condition:
        cov = f"{c['n_cells_with_variance']}/{c['n_cells']}"
        print(f"{c['type']:<5}{c['shots']:<6}{c['seed_mode']:<8}"
              f"{_fmt(c['mean_structural_variance']):>9}"
              f"{_fmt(c['mean_semantic_variance']):>9}"
              f"{c['n_cells']:>7}{cov:>6}")

    print("\n=== cross-type divergence ===")
    for pair_name, rows in type_pair_rows.items():
        ts = _agg([r["structural_divergence"] for r in rows])
        tm = _agg([r["semantic_divergence"] for r in rows])
        print(f"  {pair_name}: struct={_fmt(ts['mean'])} "
              f"seman={_fmt(tm['mean'])}  (n={ts['n']})")
    ss = _agg([r["structural_divergence"] for r in shot_rows])
    sm = _agg([r["semantic_divergence"] for r in shot_rows])
    print(f"  shot divergence (zs vs fs): struct={_fmt(ss['mean'])} "
          f"seman={_fmt(sm['mean'])}  (n={ss['n']})")

    print("\n=== guiding questions ===")
    q1 = guiding["few_shot_reduces_variance"]
    print(f"  Q1 few-shot reduces variance?")
    print(f"     structural: zs={_fmt(q1['zs_structural_variance'])} "
          f"fs={_fmt(q1['fs_structural_variance'])} "
          f"-> {q1['few_shot_reduces_structural_variance']}")
    print(f"     semantic:   zs={_fmt(q1['zs_semantic_variance'])} "
          f"fs={_fmt(q1['fs_semantic_variance'])} "
          f"-> {q1['few_shot_reduces_semantic_variance']}")
    q2 = guiding["type1_vs_type2_differ"]
    print(f"  Q2 architectural types differ?")
    print(f"     within-type variance: " + ", ".join(
        f"{t}={_fmt(v)}" for t, v in q2["within_type_structural_variance"].items()))
    for pair_name, d in q2["type_pair_divergence"].items():
        print(f"     {pair_name} divergence: struct={_fmt(d['mean_structural_divergence'])} "
              f"seman={_fmt(d['mean_semantic_divergence'])}")
    q3 = guiding["seed_mode_effect"]
    print(f"  Q3 seed_mode effect (struct variance):")
    for sm_name, vals in q3["by_seed_mode"].items():
        print(f"     {sm_name:<8} struct={_fmt(vals['structural'])} "
              f"seman={_fmt(vals['semantic'])}")
    print(f"     monotonic in temperature: "
          f"{q3['structural_variance_monotonic_in_temperature']}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    records = load_records()
    print(f"loaded {sum(len(v) for v in records.values())} outputs "
          f"in {len(records)} cells")

    emb_cache = build_embedding_cache(records)
    edist = make_emb_distance(emb_cache)

    cell_rows, per_condition = compute_variance(records, edist)
    type_pair_rows, shot_rows = compute_cross_condition(records, edist)
    guiding = guiding_answers(per_condition, type_pair_rows, shot_rows)

    ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "variance": {
                "per_cell": cell_rows,
                "per_condition": per_condition,
            },
            "cross_condition": {
                "type_divergence_by_pair": type_pair_rows,
                "shot_divergence_per_cell": shot_rows,
            },
            "guiding_questions": guiding,
        }, f, ensure_ascii=False, indent=2)

    print_summary(per_condition, type_pair_rows, shot_rows, guiding)
    print(f"\nwrote {ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
