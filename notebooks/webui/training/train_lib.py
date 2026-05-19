"""
Reusable library: load features, build train sets, train LogReg/LightGBM, evaluate.

Используется run_ablations.py — каждая ablation table вызывает эти функции с разными аргументами.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse import load_npz, hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score

DATA_DIR = Path(__file__).resolve().parent / "data"  # notebooks/webui/training/data


# ============== LOADING ==============

def load_split(split):
    """Загружает все feature matrices и labels для split'а ('train'/'val'/'test')."""
    features_npz = np.load(DATA_DIR / f"features_{split}.npz")
    word_tfidf = load_npz(DATA_DIR / f"word_tfidf_{split}.npz")
    char_tfidf = load_npz(DATA_DIR / f"char_tfidf_{split}.npz")
    with (DATA_DIR / f"labels_{split}.json").open(encoding="utf-8") as f:
        labels = json.load(f)
    return {
        "geometric": features_npz["geometric"],
        "structural": features_npz["structural"],
        "recurrence": features_npz["recurrence"],
        "role_onehot": features_npz["role_onehot"],
        "word_tfidf": word_tfidf,
        "char_tfidf": char_tfidf,
        "labels": labels,
    }


# ============== FEATURE COMBINATION ==============

def assemble_features(split_data, feature_groups):
    """Combine feature matrices в одну sparse matrix.

    feature_groups: subset of {'text', 'geometric', 'structural', 'recurrence', 'role'}
    """
    parts = []
    if "text" in feature_groups:
        parts.append(split_data["word_tfidf"])
        parts.append(split_data["char_tfidf"])
    dense_parts = []
    if "geometric" in feature_groups:
        dense_parts.append(split_data["geometric"])
    if "structural" in feature_groups:
        dense_parts.append(split_data["structural"])
    if "recurrence" in feature_groups:
        dense_parts.append(split_data["recurrence"])
    if "role" in feature_groups:
        dense_parts.append(split_data["role_onehot"])
    if dense_parts:
        dense = np.hstack(dense_parts).astype(np.float32)
        parts.append(csr_matrix(dense))
    if not parts:
        raise ValueError("No feature groups selected")
    if len(parts) == 1:
        return parts[0].tocsr()
    return hstack(parts).tocsr()


# ============== LABELS ==============

def binary_y(labels, target_class="junk"):
    """Возвращает np array of 0/1 для junk-класса. skip → исключается отдельно."""
    return np.array([1 if l["label"] == target_class else 0 for l in labels], dtype=np.int32)


def filter_labels(split_data, source_filter=None, exclude_skip=True):
    """Returns indices of labels matching filters.

    source_filter: 'human' / 'llm' / None (no filter)
    exclude_skip: drop label='skip' rows (binary task only)
    """
    keep = []
    for i, l in enumerate(split_data["labels"]):
        if exclude_skip and l["label"] == "skip":
            continue
        if source_filter is not None and l.get("label_source") != source_filter:
            continue
        keep.append(i)
    return keep


# ============== TRAINING ==============

def train_logreg(X_train, y_train, class_weight="balanced", C=1.0, random_state=42):
    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=2000,
        solver="liblinear",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, random_state=42):
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("pip install lightgbm")
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        is_unbalance=True,
        random_state=random_state,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


# ============== EVALUATION ==============

def evaluate(model, X_test, y_test, labels_meta, gold_only=False):
    """Compute F1/precision/recall/AUC. Returns dict.

    labels_meta — list of {label, sub_label, label_source} per X_test row.
    gold_only — if True, restrict to label_source='human' rows.
    """
    if gold_only:
        idx = [i for i, l in enumerate(labels_meta) if l.get("label_source") == "human"]
        if not idx:
            return None
        X_eval = X_test[idx]
        y_eval = y_test[idx]
        meta_eval = [labels_meta[i] for i in idx]
    else:
        X_eval = X_test
        y_eval = y_test
        meta_eval = labels_meta

    y_pred = model.predict(X_eval)

    # ROC-AUC требует proba
    try:
        y_proba = model.predict_proba(X_eval)[:, 1]
        auc = float(roc_auc_score(y_eval, y_proba))
    except Exception:
        auc = None

    f1 = float(f1_score(y_eval, y_pred, zero_division=0))
    precision = float(precision_score(y_eval, y_pred, zero_division=0))
    recall = float(recall_score(y_eval, y_pred, zero_division=0))
    cm = confusion_matrix(y_eval, y_pred, labels=[0, 1])

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": auc,
        "n_samples": len(y_eval),
        "n_junk": int(y_eval.sum()),
        "n_not_junk": int((y_eval == 0).sum()),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
        "meta": meta_eval,
        "y_pred": y_pred.tolist(),
        "y_true": y_eval.tolist(),
    }


# ============== BOOTSTRAP CI (PAGE-LEVEL) ==============

def page_level_bootstrap(metric_fn, y_true, y_pred, page_ids, n_iter=1000, seed=42):
    """Bootstrap CI by resampling pages (not individual candidates).

    metric_fn: f(y_true_subset, y_pred_subset) → scalar
    Returns dict with mean, low (2.5%), high (97.5%).
    """
    rnd = random.Random(seed)
    page_to_idx = defaultdict(list)
    for i, pid in enumerate(page_ids):
        page_to_idx[pid].append(i)
    pages = list(page_to_idx.keys())

    samples = []
    for _ in range(n_iter):
        chosen = [rnd.choice(pages) for _ in range(len(pages))]
        idx_subset = []
        for p in chosen:
            idx_subset.extend(page_to_idx[p])
        if not idx_subset:
            continue
        try:
            v = metric_fn([y_true[i] for i in idx_subset], [y_pred[i] for i in idx_subset])
            samples.append(v)
        except Exception:
            continue
    if not samples:
        return {"mean": None, "low": None, "high": None}
    samples.sort()
    n = len(samples)
    return {
        "mean": float(sum(samples) / n),
        "low": float(samples[int(0.025 * n)]),
        "high": float(samples[int(0.975 * n)]),
    }


def f1_bootstrap_ci(eval_result, page_ids, seed=42):
    """Wrapper для F1 bootstrap CI."""
    fn = lambda yt, yp: f1_score(yt, yp, zero_division=0)
    return page_level_bootstrap(fn, eval_result["y_true"], eval_result["y_pred"], page_ids, seed=seed)


# ============== SUB-CLASS BREAKDOWN ==============

def subclass_breakdown(eval_result):
    """Per-sub_label F1 (только для junk predictions). Возвращает dict sub_label → metrics."""
    by_sub = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for i, m in enumerate(eval_result["meta"]):
        if m["label"] == "junk":
            sub = m.get("sub_label", "other") or "other"
            by_sub[sub]["y_true"].append(eval_result["y_true"][i])
            by_sub[sub]["y_pred"].append(eval_result["y_pred"][i])
    result = {}
    for sub, data in by_sub.items():
        if not data["y_true"]:
            continue
        # Recall on this sub-class (как часто junk-кандидаты этого типа правильно классифицированы)
        recall = float(recall_score(data["y_true"], data["y_pred"], zero_division=0))
        n = len(data["y_true"])
        n_correct = sum(1 for t, p in zip(data["y_true"], data["y_pred"]) if t == p)
        result[sub] = {
            "n": n,
            "recall": recall,
            "accuracy": n_correct / n if n else 0,
        }
    return result
