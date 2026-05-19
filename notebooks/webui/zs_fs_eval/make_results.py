"""
make_results.py - turn metrics.jsonl + analysis.json into charts and a
results.md skeleton for the thesis ZS/FS section.

Reads:
  outputs/metrics.jsonl   (from metrics.py)
  outputs/analysis.json   (from analysis.py)
Writes:
  outputs/charts/*.png
  outputs/results.md

The skeleton has every number and table filled in automatically; prose
interpretation is left as clearly-marked **[INTERPRET]** placeholders for
you to write into the thesis text.

Requires matplotlib.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
METRICS_PATH = HERE / "outputs" / "metrics.jsonl"
ANALYSIS_PATH = HERE / "outputs" / "analysis.json"
CHARTS_DIR = HERE / "outputs" / "charts"
RESULTS_PATH = HERE / "outputs" / "results.md"

SEED_ORDER = ["fixed", "semi", "random"]
STATUS_COLORS = {"valid": "#4c9f70", "soft_error": "#e0a458", "hard_error": "#c14953"}
COMBO_COLORS = {
    "t1/zs": "#3d5a80", "t1/fs": "#98c1d9",
    "t2/zs": "#ee6c4d", "t2/fs": "#f4a261",
    "t3/zs": "#6a4c93", "t3/fs": "#c8b6e2",
}


def conditions_present(metrics_rows):
    """Return [(type, shots), ...] for type/shots combos actually present."""
    types = sorted({r["type"] for r in metrics_rows})
    shots = sorted({r["shots"] for r in metrics_rows}, key=lambda s: (s != "zs", s))
    return [(t, s) for t in types for s in shots]


# --------------------------------------------------------------------------
# Loading / small helpers
# --------------------------------------------------------------------------
def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def cond_value(per_condition, typ, shots, seed_mode, key):
    for c in per_condition:
        if c["type"] == typ and c["shots"] == shots and c["seed_mode"] == seed_mode:
            return c.get(key)
    return None


def split_pair_name(pair_name, types_present):
    """Split an analysis pair name like 't1_t3_full' into ('t1', 't3_full').

    Type names themselves can contain underscores (t3_full, t3_lean, t3_real),
    so naive .split('_') is wrong. We try every known type as the left half
    and pick the one whose remainder is also a known type."""
    for ta in sorted(types_present, key=len, reverse=True):
        prefix = ta + "_"
        if pair_name.startswith(prefix):
            tb = pair_name[len(prefix):]
            if tb in types_present:
                return ta, tb
    # fallback: best-effort split
    parts = pair_name.split("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (pair_name, pair_name)


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def chart_validity(metrics_rows, conditions, path):
    """Stacked bar: valid / soft / hard share per (type, shots) condition."""
    counts = defaultdict(Counter)
    for r in metrics_rows:
        counts[(r["type"], r["shots"])][r["status"]] += 1
    labels = [f"{t}/{s}" for t, s in conditions]
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(conditions)), 4.2))
    bottom = np.zeros(len(conditions))
    for status in ["valid", "soft_error", "hard_error"]:
        vals = []
        for t, s in conditions:
            c = counts[(t, s)]
            n = sum(c.values()) or 1
            vals.append(100.0 * c[status] / n)
        vals = np.array(vals)
        ax.bar(labels, vals, bottom=bottom, label=status,
               color=STATUS_COLORS[status])
        bottom += vals
    ax.set_ylabel("share of outputs (%)")
    ax.set_title("Schema validity by condition")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def chart_variance_by_condition(per_condition, conditions, path):
    """Grouped bars: structural variance, x = seed_mode, one bar per
    type/shots combo. Shows seed, shot and type effects in one view."""
    n_cond = len(conditions)
    x = np.arange(len(SEED_ORDER))
    width = min(0.22, 0.85 / max(n_cond, 1))
    offset0 = -(n_cond - 1) / 2 * width
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * n_cond + 5), 4.5))
    for i, (t, s) in enumerate(conditions):
        vals = [cond_value(per_condition, t, s, sm, "mean_structural_variance") or 0.0
                for sm in SEED_ORDER]
        color = COMBO_COLORS.get(f"{t}/{s}", "#999")
        ax.bar(x + offset0 + i * width, vals, width, label=f"{t}/{s}", color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(SEED_ORDER)
    ax.set_xlabel("seed mode (temperature)")
    ax.set_ylabel("mean structural variance (normalized TED)")
    ax.set_title("Output variance by condition")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def chart_variance_zs_vs_fs(per_condition, conditions, path):
    """Structural variance, zs vs fs averaged over types present, x = seed_mode.
    Directly addresses 'does few-shot reduce variance?'."""
    types_present = sorted({t for t, _ in conditions})
    x = np.arange(len(SEED_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, shots in enumerate(["zs", "fs"]):
        vals = []
        for sm in SEED_ORDER:
            xs = [cond_value(per_condition, t, shots, sm, "mean_structural_variance")
                  for t in types_present]
            xs = [v for v in xs if v is not None]
            vals.append(sum(xs) / len(xs) if xs else 0.0)
        ax.bar(x + (i - 0.5) * width, vals, width, label=shots,
               color=("#3d5a80" if shots == "zs" else "#ee6c4d"))
    ax.set_xticks(x)
    ax.set_xticklabels(SEED_ORDER)
    ax.set_xlabel("seed mode (temperature)")
    ax.set_ylabel("mean structural variance")
    ax.set_title("Zero-shot vs few-shot variance")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def chart_type_divergence(per_condition, type_pair_rows, path):
    """For each type pair present, plot divergence vs within-type variances.
    Each pair becomes one chart group on the x-axis."""
    pair_colors = {"t1_t2": "#6a4c93", "t1_t3": "#3a8e8e", "t2_t3": "#c47a1f"}
    types_present = sorted({c["type"] for c in per_condition})
    pairs = list(type_pair_rows.keys())
    n_pairs = len(pairs)
    if n_pairs == 0:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "no type pairs", ha="center", va="center")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, n_pairs, figsize=(4.5 * n_pairs, 4.2),
                             sharey=True, squeeze=False)
    x = np.arange(len(SEED_ORDER))
    width = 0.27
    for idx, pair in enumerate(pairs):
        ax = axes[0][idx]
        ta, tb = split_pair_name(pair, types_present)
        rows = type_pair_rows[pair]
        div_by_sm = defaultdict(list)
        for r in rows:
            if r["structural_divergence"] is not None:
                div_by_sm[r["seed_mode"]].append(r["structural_divergence"])

        def type_var(typ, sm):
            xs = [cond_value(per_condition, typ, s, sm, "mean_structural_variance")
                  for s in ("zs", "fs")]
            xs = [v for v in xs if v is not None]
            return sum(xs) / len(xs) if xs else 0.0

        a_var = [type_var(ta, sm) for sm in SEED_ORDER]
        b_var = [type_var(tb, sm) for sm in SEED_ORDER]
        div = [(sum(div_by_sm[sm]) / len(div_by_sm[sm])) if div_by_sm[sm] else 0.0
               for sm in SEED_ORDER]
        ax.bar(x - width, a_var, width, label=f"{ta} within-var",
               color=COMBO_COLORS.get(f"{ta}/zs", "#888"))
        ax.bar(x, b_var, width, label=f"{tb} within-var",
               color=COMBO_COLORS.get(f"{tb}/zs", "#888"))
        ax.bar(x + width, div, width, label="divergence",
               color=pair_colors.get(pair, "#444"))
        ax.set_xticks(x)
        ax.set_xticklabels(SEED_ORDER)
        ax.set_xlabel("seed mode")
        ax.set_title(f"{ta} vs {tb}")
        ax.legend(fontsize=7)
        if idx == 0:
            ax.set_ylabel("mean structural distance")
    fig.suptitle("Cross-type divergence vs within-type variance", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def chart_parse_strategy(metrics_rows, path):
    """How the raw model text had to be parsed."""
    strat = Counter(r["parse_strategy"] for r in metrics_rows)
    order = ["direct", "fenced", "substring", "unparseable", "empty"]
    labels = [s for s in order if strat.get(s)]
    vals = [strat[s] for s in labels]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(labels, vals, color="#5b8c5a")
    ax.set_ylabel("number of outputs")
    ax.set_title("Parse strategy used to recover JSON")
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# results.md
# --------------------------------------------------------------------------
def build_results_md(metrics_rows, analysis, conditions):
    per_condition = analysis["variance"]["per_condition"]
    guiding = analysis["guiding_questions"]
    type_pair_rows = analysis["cross_condition"]["type_divergence_by_pair"]

    n_outputs = len(metrics_rows)
    questions = sorted({r["question_id"] for r in metrics_rows})
    seed_modes = sorted({r["seed_mode"] for r in metrics_rows})
    runs_per_cell = max((r["run_idx"] for r in metrics_rows), default=-1) + 1
    types_present = sorted({r["type"] for r in metrics_rows})
    model = "TODO (fill in)"

    # validity table
    vc = defaultdict(Counter)
    for r in metrics_rows:
        vc[(r["type"], r["shots"])][r["status"]] += 1
    val_rows = []
    for t, s in conditions:
        c = vc[(t, s)]
        n = sum(c.values())
        val_rows.append([
            f"{t}/{s}",
            f"{c['valid']} ({100*c['valid']//max(n,1)}%)",
            f"{c['soft_error']} ({100*c['soft_error']//max(n,1)}%)",
            f"{c['hard_error']} ({100*c['hard_error']//max(n,1)}%)",
            n,
        ])

    # variance table
    var_rows = []
    for c in per_condition:
        var_rows.append([
            c["type"], c["shots"], c["seed_mode"],
            fmt(c["mean_structural_variance"]),
            fmt(c["mean_semantic_variance"]),
            f"{c['n_cells_with_variance']}/{c['n_cells']}",
        ])

    q1 = guiding["few_shot_reduces_variance"]
    q2 = guiding["type1_vs_type2_differ"]
    q3 = guiding["seed_mode_effect"]

    q3_rows = []
    for sm, vals in q3["by_seed_mode"].items():
        q3_rows.append([sm, fmt(vals["structural"]), fmt(vals["semantic"])])

    # type-pair divergence table (covers t1_t2, t1_t3, t2_t3 when present)
    pair_rows = []
    for pair, d in q2["type_pair_divergence"].items():
        ta, tb = split_pair_name(pair, types_present)
        a_var = q2["within_type_structural_variance"].get(ta)
        b_var = q2["within_type_structural_variance"].get(tb)
        max_var = max(v for v in (a_var, b_var) if v is not None) if (a_var, b_var) != (None, None) else None
        verdict = (
            "above noise" if max_var is not None and d["mean_structural_divergence"] is not None
            and d["mean_structural_divergence"] > max_var else "within noise"
        )
        pair_rows.append([
            f"{ta} vs {tb}",
            fmt(d["mean_structural_divergence"]),
            fmt(d["mean_semantic_divergence"]),
            fmt(a_var),
            fmt(b_var),
            verdict,
        ])

    parse = Counter(r["parse_strategy"] for r in metrics_rows)

    types_desc = {
        "t1": "Type 1 (prompt → widget)",
        "t2": "Type 2 (prompt → text → widget)",
        "t3": "Type 3 (prompt → AXtree → script → widget)",
    }
    types_listed = "; ".join(types_desc[t] for t in types_present if t in types_desc)

    md = f"""# Zero-Shot / Few-Shot Widget Generation — Results

> Auto-generated skeleton (`make_results.py`). Tables, numbers and charts are
> filled in from the experiment outputs. Text marked **[INTERPRET]** is for you
> to write into the thesis.

## Setup

- Model: {model}
- Eval questions: {len(questions)}
- Architectures: {types_listed}; each in zero-shot and few-shot
- Seed modes: {", ".join(seed_modes)} — {runs_per_cell} runs per cell
- Total outputs scored: {n_outputs}

## 1. Schema validity

{md_table(["Condition", "Valid", "Soft error", "Hard error", "n"], val_rows)}

![validity by condition](charts/validity_by_condition.png)

**[INTERPRET]** How well does the model produce schema-valid widget JSON at
all? Compare zero-shot vs few-shot and across architectures. Note the
soft-vs-hard split — soft errors are recoverable omissions, hard errors are
genuine format failures.

## 2. Output stability (variance across runs)

Variance = mean pairwise distance between the runs of one cell. Structural =
normalized tree edit distance; semantic = 1 − cosine similarity of text
embeddings. Lower = the model produces more consistent widgets.

{md_table(["Type", "Shots", "Seed", "Struct. var", "Seman. var", "Coverage"], var_rows)}

![variance by condition](charts/variance_by_condition.png)

### 2.1 Does few-shot reduce variance?

- Structural variance: zero-shot {fmt(q1['zs_structural_variance'])}, few-shot {fmt(q1['fs_structural_variance'])} → few-shot lower: **{q1['few_shot_reduces_structural_variance']}**
- Semantic variance: zero-shot {fmt(q1['zs_semantic_variance'])}, few-shot {fmt(q1['fs_semantic_variance'])} → few-shot lower: **{q1['few_shot_reduces_semantic_variance']}**

![zero-shot vs few-shot variance](charts/variance_zs_vs_fs.png)

**[INTERPRET]** ...

### 2.2 Effect of seed mode (temperature)

{md_table(["Seed mode", "Struct. var", "Seman. var"], q3_rows)}

Structural variance monotonic in temperature (fixed ≤ semi ≤ random):
**{q3['structural_variance_monotonic_in_temperature']}**

**[INTERPRET]** The `fixed` mode (temperature 0) is the near-deterministic
control — variance there should be close to zero. Comment on whether the
expected ordering holds and what the gap between modes implies.

## 3. Architectural comparison

Each pair compares two architectures: if the cross-pair divergence exceeds
both members' within-type variance, the two pipelines produce systematically
different widgets, not just run-to-run noise.

{md_table(["Pair", "Struct. div", "Seman. div", "Within-var A", "Within-var B", "Verdict"], pair_rows)}

![type divergence](charts/type_divergence.png)

**[INTERPRET]** Compare the divergence numbers to the within-variance numbers.
A "above noise" verdict means the architectural choice has a real systematic
effect; "within noise" means the architectures produce statistically
indistinguishable outputs at this resolution.

## 4. Parse robustness

Strategy needed to recover JSON from raw model text:
{", ".join(f"{k}={parse[k]}" for k in ["direct","fenced","substring","unparseable","empty"] if parse.get(k))}

![parse strategy](charts/parse_strategy.png)

**[INTERPRET]** A high `direct` share means the model followed the
"JSON only, no fences" instruction; `unparseable`/`empty` are outright failures.
For Type 3, "parse strategy" refers to parsing the AXtree JSON; the subsequent
AXtree→widget conversion is deterministic and not counted as a parse step.

## Limitations

- Preliminary evaluation: {len(questions)} questions, {runs_per_cell} runs per
  cell. **[INTERPRET]** — state planned extension (more questions, more runs,
  human eval) for the defense.
- No gold widgets: evaluation measures stability and cross-condition agreement,
  not correctness against a reference.
- Seed control is temperature-based (the API offers no strict seed); "fixed"
  is temperature 0, not a bit-exact seed.
- **[INTERPRET]** — add any model/prompt-specific caveats.
"""
    return md


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    metrics_rows = read_jsonl(METRICS_PATH)
    if not metrics_rows:
        raise SystemExit(f"no metrics at {METRICS_PATH} - run metrics.py first")
    if not ANALYSIS_PATH.exists():
        raise SystemExit(f"no analysis at {ANALYSIS_PATH} - run analysis.py first")
    analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    per_condition = analysis["variance"]["per_condition"]
    type_pair_rows = analysis["cross_condition"]["type_divergence_by_pair"]
    conditions = conditions_present(metrics_rows)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_validity(metrics_rows, conditions, CHARTS_DIR / "validity_by_condition.png")
    chart_variance_by_condition(per_condition, conditions, CHARTS_DIR / "variance_by_condition.png")
    chart_variance_zs_vs_fs(per_condition, conditions, CHARTS_DIR / "variance_zs_vs_fs.png")
    chart_type_divergence(per_condition, type_pair_rows, CHARTS_DIR / "type_divergence.png")
    chart_parse_strategy(metrics_rows, CHARTS_DIR / "parse_strategy.png")

    RESULTS_PATH.write_text(build_results_md(metrics_rows, analysis, conditions), encoding="utf-8")
    print(f"wrote 5 charts -> {CHARTS_DIR}")
    print(f"wrote results skeleton -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
