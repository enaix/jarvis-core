"""analyze_manual_ratings.py — aggregate manual rating CSV + correlate with auto metrics.

Reads outputs/manual_ratings.csv (exported from the rating page) and produces:
  1. Mean rating per (type, shots) condition, with std and distribution
  2. Bar chart -> outputs/charts/manual_ratings_by_condition.png
  3. Join with metrics.jsonl to compute correlations:
       rating vs auto-validity (per-row)
       rating vs axtree-validity (per-row, t3/t4 only)
       rating vs output_chars / node_count / max_depth
  4. Print "surprising" rows where manual and auto disagree most

Run:  python analyze_manual_ratings.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE / "outputs"
CHARTS = OUT / "charts"
CHARTS.mkdir(exist_ok=True)

CSV_PATH = OUT / "manual_ratings.csv"
METRICS_PATH = OUT / "metrics.jsonl"

# ---- load + curate ------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
# rename t3_real -> t3 for display consistency with notebook
df["type"] = df["type"].replace({"t3_real": "t3"})
df["cond"] = df["type"] + "/" + df["shots"]

print(f"loaded {len(df)} ratings  |  {df['rating'].notna().sum()} non-null")
print(f"rating distribution: {dict(df['rating'].value_counts().sort_index())}")
print()

TYPE_ORDER = ["t1", "t2", "t3", "t4"]
COND_ORDER = [f"{t}/{s}" for t in TYPE_ORDER for s in ["zs", "fs"]]

# ---- 1. mean per condition ----------------------------------------------
agg = (
    df.groupby("cond")
    .agg(
        n=("rating", "count"),
        mean=("rating", "mean"),
        std=("rating", "std"),
        med=("rating", "median"),
        min=("rating", "min"),
        max=("rating", "max"),
    )
    .reindex(COND_ORDER)
    .round(2)
)
print("Mean manual rating by condition:")
print(agg.to_string())
print()

# ---- 2. bar chart -------------------------------------------------------
COLORS = {
    "t1/zs": "#3d5a80", "t1/fs": "#98c1d9",
    "t2/zs": "#ee6c4d", "t2/fs": "#f4a261",
    "t3/zs": "#6a4c93", "t3/fs": "#c8b6e2",
    "t4/zs": "#2d8659", "t4/fs": "#94c9a9",
}

fig, ax = plt.subplots(figsize=(10, 5))
means = agg["mean"].values
stds = agg["std"].values
ax.bar(COND_ORDER, means, yerr=stds, capsize=4,
       color=[COLORS[c] for c in COND_ORDER], alpha=0.9)
ax.set_ylabel("Mean rating (1-5)")
ax.set_title(f"Manual rating by condition (N={int(agg['n'].iloc[0])} per condition, ±1 std)")
ax.set_ylim(0, 5.4)
ax.axhline(3, ls="--", color="#888", lw=0.7, alpha=0.5)
for i, (m, n) in enumerate(zip(means, agg["n"].values)):
    if not np.isnan(m):
        ax.text(i, m + 0.12, f"{m:.1f}", ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
out_png = CHARTS / "manual_ratings_by_condition.png"
plt.savefig(out_png, dpi=140)
print(f"saved chart -> {out_png}")
print()

# ---- 3. join with metrics.jsonl for correlations ------------------------
metrics_rows = []
with open(METRICS_PATH, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            try:
                metrics_rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
m = pd.DataFrame(metrics_rows)
# build strict per-row status (widget for t1/t2, axtree for t3_real/t4)
STATUS_MAP = {"valid": "valid", "soft_error": "soft", "hard_error": "hard"}
direct = m["type"].isin(["t1", "t2"])
m.loc[direct, "strict_status"] = m.loc[direct, "status"].map(STATUS_MAP)
m.loc[~direct, "strict_status"] = m.loc[~direct, "axtree_status"]

# build join key (keep t3_real form to match CSV card_ids)
m["card_id"] = (m["question_id"] + "_"
                + m["type"] + "_"
                + m["shots"] + "_" + m["seed_mode"] + "_" + m["run_idx"].astype(str))

# Dedup: pipeline resume-logic can append duplicates. Keep last (most recent).
n_before = len(m)
m = m.drop_duplicates(subset=["card_id"], keep="last")
print(f"dedup metrics: {n_before} -> {len(m)} rows")

# Now rename for display
m["type"] = m["type"].replace({"t3_real": "t3"})

joined = df.merge(
    m[["card_id", "status", "axtree_status", "strict_status",
       "output_chars", "node_count", "max_depth"]],
    on="card_id", how="left",
)
print(f"joined: {joined['status'].notna().sum()} / {len(joined)} rows have metrics")
print()

# ---- 3a. rating vs widget-level status ----------------------------------
status_map = {"valid": "valid", "soft_error": "soft", "hard_error": "hard"}
joined["auto_status"] = joined["status"].map(status_map)
print("Mean rating by WIDGET-level status (post-adapter):")
print(joined.groupby("auto_status")["rating"].agg(["count", "mean", "std"]).round(2))
print()

# ---- 3b. rating vs STRICT status (pre-adapter for t3/t4) ----------------
print("Mean rating by STRICT status (pre-adapter for t3/t4):")
print(joined.groupby("strict_status")["rating"].agg(["count", "mean", "std"]).round(2))
print()

# ---- 3c. continuous correlations (overall + per-architecture) -----------
print("Pearson correlation of rating with continuous metrics:")
print(f"  {'metric':<14}{'overall':>14}{'t1':>10}{'t2':>10}{'t3':>10}{'t4':>10}")
for col in ["output_chars", "node_count", "max_depth"]:
    line = f"  {col:<14}"
    for label, subset in [
        ("overall", joined),
        ("t1", joined[joined["type"] == "t1"]),
        ("t2", joined[joined["type"] == "t2"]),
        ("t3", joined[joined["type"] == "t3"]),
        ("t4", joined[joined["type"] == "t4"]),
    ]:
        valid = subset[[col, "rating"]].dropna()
        if len(valid) >= 3 and valid[col].std() > 0:
            r = valid[col].corr(valid["rating"])
            line += f"{r:+.2f} (n={len(valid)})".rjust(14 if label == "overall" else 10)
        else:
            line += "n/a".rjust(14 if label == "overall" else 10)
    print(line)
print()

# ---- 4. surprising rows -------------------------------------------------
# - high rating but hard auto error
# - low rating but auto valid
print("=" * 64)
print("SURPRISING rows (manual ↔ auto disagreement):")
print("=" * 64)
hi_low = joined[(joined["rating"] >= 4) & (joined["auto_status"] == "hard")]
lo_hi = joined[(joined["rating"] <= 2) & (joined["auto_status"] == "valid")]
if len(hi_low):
    print(f"\n  Manual high (≥4) but auto HARD ({len(hi_low)} rows):")
    for _, r in hi_low.iterrows():
        print(f"    {r['card_id']:<35}  rating={int(r['rating'])}  auto=hard")
if len(lo_hi):
    print(f"\n  Manual low (≤2) but auto VALID ({len(lo_hi)} rows):")
    for _, r in lo_hi.iterrows():
        print(f"    {r['card_id']:<35}  rating={int(r['rating'])}  auto=valid")
if not len(hi_low) and not len(lo_hi):
    print("  (none)")
print()

# ---- 5. concordance with automatic ranking ------------------------------
# Note: strict_status for t3/t4 = axtree_status, which can be NaN for
# rows where AXtree validation didn't run. Use sum/len explicitly (vs .mean()
# which would ignore NaN and inflate the percentage).
def pct_valid_strict(g):
    return (g["strict_status"].fillna("missing") == "valid").sum() / len(g) * 100

widget_valid = m.groupby(["type", "shots"]).apply(
    lambda g: (g["status"] == "valid").sum() / len(g) * 100
).reindex([(t, s) for t in TYPE_ORDER for s in ["zs", "fs"]])
strict_valid = m.groupby(["type", "shots"]).apply(pct_valid_strict
).reindex([(t, s) for t in TYPE_ORDER for s in ["zs", "fs"]])
manual_mean = agg["mean"].values

print("Auto validity vs manual mean rating per condition:")
print(f"  {'cond':<8}{'widget %valid':>16}{'strict %valid':>16}{'manual mean':>14}")
for i, cond in enumerate(COND_ORDER):
    t, s = cond.split("/")
    wp = widget_valid.loc[(t, s)] if (t, s) in widget_valid.index else float("nan")
    sp = strict_valid.loc[(t, s)] if (t, s) in strict_valid.index else float("nan")
    print(f"  {cond:<8}{wp:>15.1f}%{sp:>15.1f}%{manual_mean[i]:>13.2f}")

# Spearman correlations
from scipy.stats import spearmanr  # noqa: E402
widget_arr = widget_valid.values
strict_arr = strict_valid.values

mask = ~np.isnan(widget_arr) & ~np.isnan(manual_mean)
rho_w, p_w = spearmanr(widget_arr[mask], manual_mean[mask])
rho_s, p_s = spearmanr(strict_arr[mask], manual_mean[mask])

print(f"\nSpearman rank correlation (n={mask.sum()} conditions):")
print(f"  widget-level vs manual:  ρ = {rho_w:+.3f}  (p = {p_w:.3f})  "
      f"-- post-adapter, recovery hides errors")
print(f"  strict-level  vs manual: ρ = {rho_s:+.3f}  (p = {p_s:.3f})  "
      f"-- pre-adapter, fair across architectures")
print()
print("Interpretation: a higher strict-vs-manual correlation than widget-vs-manual")
print("indicates manual rating sees through the permissive adapter, confirming the")
print("decision to report strict (pre-adapter) validity in the main results.")
