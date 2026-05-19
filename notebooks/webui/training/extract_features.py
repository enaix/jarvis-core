"""
Step 2 — Feature extraction.

Builds feature matrix для каждого candidate из combined_labels:
- Text: TF-IDF на text_subtree (word 1-2 gram + char 3-5 gram)
- Geometric: normalized bbox координаты, area_ratio
- Structural: role one-hot (top-15), text_len(log), num_descendants(log), link_density, depth
- Recurrence: global_recurrence_count(log), global_recurrence_ratio

Train TF-IDF vocab ТОЛЬКО на train fold (no leakage).

Output:
- training/data/features_train.npz / features_val.npz / features_test.npz
- training/data/feature_meta.json (feature names, vocab size, и т.д.)
- training/data/labels_train.json / val / test (binary y, sub_label, label_source)

Запуск:
    .venv\\Scripts\\python.exe notebooks\\webui\\training\\extract_features.py
"""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

CANDIDATES_FILE = Path(r"notebooks\webui\annotator\candidates_topic_filtered.jsonl")
LABELS_FILE = Path(r"notebooks\webui\training\data\combined_labels.jsonl")
SPLITS_FILE = Path(r"notebooks\webui\training\data\splits.json")
DATA_DIR = Path(r"notebooks\webui\training\data")

TFIDF_WORD_MAX = 10000
TFIDF_CHAR_MAX = 10000
TOP_K_ROLES = 15


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def cand_key(r):
    return f"{r['page_id']}::{r['node_id']}"


def safe_log(x):
    return math.log1p(max(0, float(x or 0)))


def build_numeric_features(cand, role_to_idx, page_dim_default=(1920, 1080)):
    """Возвращает numpy array фичей для одного candidate (БЕЗ TF-IDF)."""
    bbox = cand.get("bbox") or {}
    W, H = page_dim_default  # все наши screenshots 1920x1080
    x = float(bbox.get("x", 0)) / W
    y = float(bbox.get("y", 0)) / H
    w = float(bbox.get("width", 0)) / W
    h = float(bbox.get("height", 0)) / H

    role = cand.get("role", "")
    role_one_hot = [0] * (len(role_to_idx) + 1)  # +1 for "other"
    if role in role_to_idx:
        role_one_hot[role_to_idx[role]] = 1
    else:
        role_one_hot[-1] = 1

    text_len = safe_log(cand.get("text_len"))
    own_text_len = safe_log(cand.get("own_text_len"))
    num_descendants = safe_log(cand.get("num_descendants"))
    num_children = safe_log(cand.get("num_children"))
    link_density = float(cand.get("link_density") or 0)
    depth = float(cand.get("depth") or 0)
    bbox_area = safe_log(cand.get("bbox_area"))
    bbox_ratio = float(cand.get("bbox_ratio") or 0)

    grec_count = safe_log(cand.get("global_recurrence_count"))
    grec_ratio = float(cand.get("global_recurrence_ratio") or 0)
    rec_count = safe_log(cand.get("recurrence_count"))

    geometric = [x, y, w, h, bbox_area, bbox_ratio]
    structural = [text_len, own_text_len, num_descendants, num_children, link_density, depth]
    recurrence = [grec_count, grec_ratio, rec_count]

    return geometric, structural, recurrence, role_one_hot


def feature_names(role_to_idx):
    geo = ["bbox_x_norm", "bbox_y_norm", "bbox_w_norm", "bbox_h_norm", "bbox_area_log", "bbox_ratio"]
    struct = ["text_len_log", "own_text_len_log", "num_descendants_log", "num_children_log",
              "link_density", "depth"]
    rec = ["grec_count_log", "grec_ratio", "rec_count_log"]
    role_names = sorted(role_to_idx, key=lambda r: role_to_idx[r])
    role_cols = [f"role={r}" for r in role_names] + ["role=other"]
    return {
        "geometric": geo,
        "structural": struct,
        "recurrence": rec,
        "role_one_hot": role_cols,
    }


def main():
    print("=== Loading data ===")
    candidates = load_jsonl(CANDIDATES_FILE)
    labels = load_jsonl(LABELS_FILE)
    with SPLITS_FILE.open(encoding="utf-8") as f:
        splits_data = json.load(f)
    splits = splits_data["splits"]

    cand_idx = {cand_key(c): c for c in candidates}
    print(f"  Candidates: {len(candidates)}")
    print(f"  Combined labels: {len(labels)}")

    # Determine top-K roles from TRAIN ONLY (no leakage)
    train_keys = [cand_key(r) for r in labels if splits.get(r["page_id"]) == "train"]
    train_roles = Counter(cand_idx[k].get("role", "") for k in train_keys if k in cand_idx)
    top_roles = [r for r, _ in train_roles.most_common(TOP_K_ROLES)]
    role_to_idx = {r: i for i, r in enumerate(top_roles)}
    print(f"\n=== Top {TOP_K_ROLES} roles (from train) ===")
    for r in top_roles:
        print(f"  {r}: {train_roles[r]}")

    # Build per-split data
    splits_data_out = {"train": [], "val": [], "test": []}
    for r in labels:
        sp = splits.get(r["page_id"])
        if sp not in ("train", "val", "test"):
            continue
        k = cand_key(r)
        if k not in cand_idx:
            continue
        cand = cand_idx[k]
        splits_data_out[sp].append((r, cand))

    for sp in ("train", "val", "test"):
        print(f"\n=== {sp}: {len(splits_data_out[sp])} samples ===")

    # === TF-IDF (word + char), fit ONLY on train ===
    print("\n=== Fitting TF-IDF on train only ===")
    train_texts = [c.get("text_subtree") or "" for _, c in splits_data_out["train"]]

    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=TFIDF_WORD_MAX,
        sublinear_tf=True,
        min_df=2,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=TFIDF_CHAR_MAX,
        sublinear_tf=True,
        min_df=2,
    )
    train_word_x = word_vec.fit_transform(train_texts)
    train_char_x = char_vec.fit_transform(train_texts)
    print(f"  word vocab: {len(word_vec.vocabulary_)}")
    print(f"  char vocab: {len(char_vec.vocabulary_)}")

    # Build numeric matrices for all splits
    def build_split_features(sp_name):
        items = splits_data_out[sp_name]
        texts = [c.get("text_subtree") or "" for _, c in items]
        word_x = word_vec.transform(texts) if sp_name != "train" else train_word_x
        char_x = char_vec.transform(texts) if sp_name != "train" else train_char_x

        geo_mat, struct_mat, rec_mat, role_mat = [], [], [], []
        for _, cand in items:
            g, s, rec, role = build_numeric_features(cand, role_to_idx)
            geo_mat.append(g)
            struct_mat.append(s)
            rec_mat.append(rec)
            role_mat.append(role)
        geo_mat = np.array(geo_mat, dtype=np.float32)
        struct_mat = np.array(struct_mat, dtype=np.float32)
        rec_mat = np.array(rec_mat, dtype=np.float32)
        role_mat = np.array(role_mat, dtype=np.float32)
        return word_x, char_x, geo_mat, struct_mat, rec_mat, role_mat, items

    print("\n=== Building feature matrices per split ===")
    splits_features = {}
    for sp in ("train", "val", "test"):
        wx, cx, g, s, rec, ro, items = build_split_features(sp)
        splits_features[sp] = {
            "word_tfidf": wx,
            "char_tfidf": cx,
            "geometric": g,
            "structural": s,
            "recurrence": rec,
            "role_onehot": ro,
            "items": items,
        }
        print(f"  {sp}: word_tfidf={wx.shape}, char_tfidf={cx.shape}, "
              f"geo={g.shape}, struct={s.shape}, rec={rec.shape}, role={ro.shape}")

    # Save (sparse matrices via scipy.sparse.save_npz, dense via np.savez, items as JSON)
    print("\n=== Saving features ===")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    from scipy.sparse import save_npz
    for sp in ("train", "val", "test"):
        f = splits_features[sp]
        np.savez(
            DATA_DIR / f"features_{sp}.npz",
            geometric=f["geometric"],
            structural=f["structural"],
            recurrence=f["recurrence"],
            role_onehot=f["role_onehot"],
        )
        save_npz(DATA_DIR / f"word_tfidf_{sp}.npz", f["word_tfidf"])
        save_npz(DATA_DIR / f"char_tfidf_{sp}.npz", f["char_tfidf"])

        labels_out = []
        for label_row, _ in f["items"]:
            labels_out.append({
                "page_id": str(label_row["page_id"]),
                "node_id": str(label_row["node_id"]),
                "label": label_row["label"],
                "sub_label": label_row.get("sub_label", ""),
                "label_source": label_row["label_source"],
                "role": label_row.get("role", ""),
            })
        with (DATA_DIR / f"labels_{sp}.json").open("w", encoding="utf-8") as fp:
            json.dump(labels_out, fp, ensure_ascii=False)
        print(f"  Saved {sp}: features + labels ({len(labels_out)} samples)")

    # Save metadata
    fnames = feature_names(role_to_idx)
    fnames["word_vocab_size"] = len(word_vec.vocabulary_)
    fnames["char_vocab_size"] = len(char_vec.vocabulary_)
    fnames["top_roles"] = top_roles
    fnames["tfidf_word_max"] = TFIDF_WORD_MAX
    fnames["tfidf_char_max"] = TFIDF_CHAR_MAX

    with (DATA_DIR / "feature_meta.json").open("w", encoding="utf-8") as fp:
        json.dump(fnames, fp, ensure_ascii=False, indent=2)
    print(f"  Saved feature_meta.json")

    # Save vocab artifacts (for inference / reproducibility)
    import pickle
    with (DATA_DIR / "tfidf_vectorizers.pkl").open("wb") as fp:
        pickle.dump({"word": word_vec, "char": char_vec, "role_to_idx": role_to_idx}, fp)
    print(f"  Saved tfidf_vectorizers.pkl")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
