"""
Step 3+4 — Train models + run all ablations + save results.

Запускает:
- Baselines (random, majority)
- Ablation 1 — Features (incremental: text → +geo → +struct → +rec)
- Ablation 2 — Training data source (human / llm / combined)
- Ablation 3 — Sub-class breakdown
- Saves результаты в training/results/

На выходе — CSV таблицы готовые для thesis.

Запуск:
    .venv\\Scripts\\python.exe notebooks\\webui\\training\\run_ablations.py
"""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from train_lib import (
    load_split, assemble_features, binary_y, filter_labels,
    train_logreg, train_lightgbm, evaluate, f1_bootstrap_ci, subclass_breakdown,
)

DATA_DIR = Path(r"notebooks\webui\training\data")
RESULTS_DIR = Path(r"notebooks\webui\training\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# Feature group definitions для ablation 1 (incremental)
FEATURE_PROGRESSION = [
    ("text",                 ["text"]),
    ("text+geo",             ["text", "geometric"]),
    ("text+geo+struct",      ["text", "geometric", "structural", "role"]),
    ("text+geo+struct+rec",  ["text", "geometric", "structural", "role", "recurrence"]),
]


def load_all():
    print("=== Loading splits ===")
    train_data = load_split("train")
    val_data = load_split("val")
    test_data = load_split("test")
    for name, d in [("train", train_data), ("val", val_data), ("test", test_data)]:
        print(f"  {name}: {len(d['labels'])} samples")
    return train_data, val_data, test_data


def restrict_split(split_data, indices):
    """Returns new split_data dict with only rows at given indices."""
    return {
        "geometric": split_data["geometric"][indices],
        "structural": split_data["structural"][indices],
        "recurrence": split_data["recurrence"][indices],
        "role_onehot": split_data["role_onehot"][indices],
        "word_tfidf": split_data["word_tfidf"][indices],
        "char_tfidf": split_data["char_tfidf"][indices],
        "labels": [split_data["labels"][i] for i in indices],
    }


def run_one_config(train_data, test_data, feature_groups, model_type, train_indices=None):
    """Train + evaluate one configuration. Returns dict с metrics."""
    if train_indices is not None:
        train_use = restrict_split(train_data, train_indices)
    else:
        train_use = train_data

    train_keep = filter_labels(train_use, source_filter=None, exclude_skip=True)
    test_keep = filter_labels(test_data, source_filter=None, exclude_skip=True)

    train_used = restrict_split(train_use, train_keep)
    test_used = restrict_split(test_data, test_keep)

    X_train = assemble_features(train_used, feature_groups)
    y_train = binary_y(train_used["labels"])
    X_test = assemble_features(test_used, feature_groups)
    y_test = binary_y(test_used["labels"])

    if model_type == "logreg":
        model = train_logreg(X_train, y_train, random_state=SEED)
    elif model_type == "lgbm":
        model = train_lightgbm(X_train, y_train, random_state=SEED)
    else:
        raise ValueError(f"Unknown model {model_type}")

    # Evaluate on full test (mixed) and gold-only test
    eval_mixed = evaluate(model, X_test, y_test, test_used["labels"], gold_only=False)
    eval_gold = evaluate(model, X_test, y_test, test_used["labels"], gold_only=True)

    # Bootstrap CIs (page-level)
    test_pages = [l["page_id"] for l in test_used["labels"]]
    ci_mixed = f1_bootstrap_ci(eval_mixed, test_pages, seed=SEED)
    if eval_gold is not None:
        gold_indices = [i for i, l in enumerate(test_used["labels"]) if l.get("label_source") == "human"]
        gold_pages = [test_pages[i] for i in gold_indices]
        ci_gold = f1_bootstrap_ci(eval_gold, gold_pages, seed=SEED)
    else:
        ci_gold = None

    return {
        "n_train": int(len(y_train)),
        "n_train_junk": int(y_train.sum()),
        "n_train_not_junk": int((y_train == 0).sum()),
        "eval_mixed": eval_mixed,
        "eval_gold": eval_gold,
        "ci_mixed": ci_mixed,
        "ci_gold": ci_gold,
        "model": model,
    }


def baselines(test_data):
    """Random + Majority baselines on test set."""
    test_keep = filter_labels(test_data, source_filter=None, exclude_skip=True)
    labels_kept = [test_data["labels"][i] for i in test_keep]
    y_test = binary_y(labels_kept)
    test_pages = [l["page_id"] for l in labels_kept]

    rng = np.random.default_rng(SEED)
    junk_rate = float(y_test.mean())

    results = {}
    # Majority — predict most common class
    majority_pred = np.zeros_like(y_test) if junk_rate < 0.5 else np.ones_like(y_test)
    results["majority"] = {
        "f1": float(f1_score(y_test, majority_pred, zero_division=0)),
        "precision": float(precision_score(y_test, majority_pred, zero_division=0)),
        "recall": float(recall_score(y_test, majority_pred, zero_division=0)),
    }

    # Random — proportional to class rate
    random_pred = (rng.random(len(y_test)) < junk_rate).astype(int)
    results["random"] = {
        "f1": float(f1_score(y_test, random_pred, zero_division=0)),
        "precision": float(precision_score(y_test, random_pred, zero_division=0)),
        "recall": float(recall_score(y_test, random_pred, zero_division=0)),
    }

    # Gold-only versions
    gold_idx = [i for i, l in enumerate(labels_kept) if l.get("label_source") == "human"]
    if gold_idx:
        y_gold = y_test[gold_idx]
        gold_majority_pred = majority_pred[gold_idx]
        gold_random_pred = random_pred[gold_idx]
        results["majority_gold"] = {
            "f1": float(f1_score(y_gold, gold_majority_pred, zero_division=0)),
            "precision": float(precision_score(y_gold, gold_majority_pred, zero_division=0)),
            "recall": float(recall_score(y_gold, gold_majority_pred, zero_division=0)),
        }
        results["random_gold"] = {
            "f1": float(f1_score(y_gold, gold_random_pred, zero_division=0)),
            "precision": float(precision_score(y_gold, gold_random_pred, zero_division=0)),
            "recall": float(recall_score(y_gold, gold_random_pred, zero_division=0)),
        }

    return results


def ablation_1_features(train_data, test_data):
    """Incremental feature ablation × 2 models."""
    print("\n=== Ablation 1: features (incremental) ===")
    rows = []
    for feat_name, groups in FEATURE_PROGRESSION:
        for model_type in ("logreg", "lgbm"):
            print(f"  Training {model_type} | features={feat_name}")
            try:
                res = run_one_config(train_data, test_data, groups, model_type)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
            row = {
                "features": feat_name,
                "model": model_type,
                "n_train": res["n_train"],
                "f1_mixed": res["eval_mixed"]["f1"],
                "f1_mixed_low": res["ci_mixed"]["low"],
                "f1_mixed_high": res["ci_mixed"]["high"],
                "f1_gold": res["eval_gold"]["f1"] if res["eval_gold"] else None,
                "f1_gold_low": res["ci_gold"]["low"] if res["ci_gold"] else None,
                "f1_gold_high": res["ci_gold"]["high"] if res["ci_gold"] else None,
                "precision_mixed": res["eval_mixed"]["precision"],
                "recall_mixed": res["eval_mixed"]["recall"],
                "roc_auc_mixed": res["eval_mixed"]["roc_auc"],
            }
            rows.append(row)
            gold_str = f"{row['f1_gold']:.3f}" if row['f1_gold'] is not None else "N/A"
            print(f"    F1 (mixed): {row['f1_mixed']:.3f} [{row['f1_mixed_low']:.3f}, {row['f1_mixed_high']:.3f}] | "
                  f"F1 (gold): {gold_str}")
    return rows


def ablation_2_data_source(train_data, test_data):
    """Train data source ablation: human / llm / combined × 2 models, full features."""
    print("\n=== Ablation 2: training data source ===")
    rows = []
    full_features = ["text", "geometric", "structural", "role", "recurrence"]
    for source_name, source_filter in [("human", "human"), ("llm", "llm"), ("combined", None)]:
        train_idx = filter_labels(train_data, source_filter=source_filter, exclude_skip=True)
        if not train_idx:
            continue
        for model_type in ("logreg", "lgbm"):
            print(f"  Training {model_type} | data={source_name} ({len(train_idx)} samples)")
            try:
                res = run_one_config(train_data, test_data, full_features, model_type, train_indices=train_idx)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
            row = {
                "data_source": source_name,
                "model": model_type,
                "n_train": res["n_train"],
                "n_train_junk": res["n_train_junk"],
                "f1_mixed": res["eval_mixed"]["f1"],
                "f1_mixed_low": res["ci_mixed"]["low"],
                "f1_mixed_high": res["ci_mixed"]["high"],
                "f1_gold": res["eval_gold"]["f1"] if res["eval_gold"] else None,
                "f1_gold_low": res["ci_gold"]["low"] if res["ci_gold"] else None,
                "f1_gold_high": res["ci_gold"]["high"] if res["ci_gold"] else None,
                "precision_mixed": res["eval_mixed"]["precision"],
                "recall_mixed": res["eval_mixed"]["recall"],
            }
            rows.append(row)
            gold_str = f"{row['f1_gold']:.3f}" if row['f1_gold'] is not None else "N/A"
            print(f"    F1 (mixed): {row['f1_mixed']:.3f} | F1 (gold): {gold_str}")
    return rows


def ablation_3_subclass(train_data, test_data):
    """Sub-class breakdown — per junk sub_label recall на test."""
    print("\n=== Ablation 3: sub-class breakdown ===")
    full_features = ["text", "geometric", "structural", "role", "recurrence"]
    res = run_one_config(train_data, test_data, full_features, "lgbm")
    breakdown = subclass_breakdown(res["eval_mixed"])
    rows = []
    for sub, m in sorted(breakdown.items(), key=lambda x: -x[1]["n"]):
        rows.append({
            "sub_label": sub,
            "n_test": m["n"],
            "recall": m["recall"],
            "accuracy": m["accuracy"],
        })
        print(f"  {sub:12s}: n={m['n']:>4d}, recall={m['recall']:.3f}, acc={m['accuracy']:.3f}")
    return rows


def write_csv(rows, path, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main():
    train_data, val_data, test_data = load_all()

    print("\n=== Baselines ===")
    bl = baselines(test_data)
    for k, v in bl.items():
        print(f"  {k}: F1={v['f1']:.3f}, P={v['precision']:.3f}, R={v['recall']:.3f}")
    with (RESULTS_DIR / "baselines.json").open("w", encoding="utf-8") as f:
        json.dump(bl, f, ensure_ascii=False, indent=2)

    rows1 = ablation_1_features(train_data, test_data)
    write_csv(rows1, RESULTS_DIR / "ablation_1_features.csv",
              ["features", "model", "n_train", "f1_mixed", "f1_mixed_low", "f1_mixed_high",
               "f1_gold", "f1_gold_low", "f1_gold_high", "precision_mixed", "recall_mixed", "roc_auc_mixed"])

    rows2 = ablation_2_data_source(train_data, test_data)
    write_csv(rows2, RESULTS_DIR / "ablation_2_data_source.csv",
              ["data_source", "model", "n_train", "n_train_junk", "f1_mixed", "f1_mixed_low", "f1_mixed_high",
               "f1_gold", "f1_gold_low", "f1_gold_high", "precision_mixed", "recall_mixed"])

    rows3 = ablation_3_subclass(train_data, test_data)
    write_csv(rows3, RESULTS_DIR / "ablation_3_subclass.csv",
              ["sub_label", "n_test", "recall", "accuracy"])

    print(f"\n=== All results saved to {RESULTS_DIR} ===")


if __name__ == "__main__":
    main()
