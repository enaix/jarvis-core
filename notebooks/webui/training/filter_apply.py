"""
Применение обученного классификатора к AXTree pages.

Содержит:
- train_best_model() — переобучает LogReg+full_features на combined train (default best config)
- featurize_candidate(cand) — строит feature vector для одного кандидата (как в extract_features.py)
- predict_for_candidates(candidates, model) — predict junk/not_junk + probabilities
- collect_descendants(page_dir, candidate_node_ids) — для каждого кандидата находит все ID descendants в AXTree
- compute_compression(page_dir, junk_candidate_ids) — считает кол-во AX-узлов всего vs после удаления junk subtrees
"""

import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, hstack, vstack, load_npz
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # notebooks/webui/

import build_candidates as bc  # noqa: E402

# Paths anchored to script location (work whether run from project root or notebook in training/)
_SCRIPT_DIR = Path(__file__).resolve().parent  # notebooks/webui/training/
_WEBUI_DIR = _SCRIPT_DIR.parent  # notebooks/webui/

DATA_DIR = _SCRIPT_DIR / "data"
CANDIDATES_FILE = _WEBUI_DIR / "annotator" / "candidates_topic_filtered.jsonl"
DATASET_ROOT = Path(
    r"C:\Users\70133\.cache\huggingface\hub\datasets--biglab--webui-7k\snapshots\60f7b3c4b9409f75551664adc1564625dfc33c2e\dataset1"
)


def safe_log(x):
    return math.log1p(max(0, float(x or 0)))


# ============== ARTIFACTS ==============

def load_artifacts():
    """TF-IDF vectorizers + role mapping (saved by extract_features.py)."""
    with (DATA_DIR / "tfidf_vectorizers.pkl").open("rb") as f:
        d = pickle.load(f)
    return d  # {'word': TfidfVectorizer, 'char': TfidfVectorizer, 'role_to_idx': {...}}


def load_candidates_index():
    """Read candidates_topic_filtered.jsonl, index by page_id."""
    by_page = defaultdict(list)
    with CANDIDATES_FILE.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by_page[str(r["page_id"])].append(r)
    return by_page


# ============== FEATURIZATION ==============

def featurize_candidate(cand, vec, role_to_idx):
    """Build feature vector for one candidate (sparse, same shape как train).
    Returns 1×D sparse matrix."""
    text = cand.get("text_subtree") or ""
    word_x = vec["word"].transform([text])
    char_x = vec["char"].transform([text])

    bbox = cand.get("bbox") or {}
    W, H = 1920, 1080
    geo = [
        float(bbox.get("x", 0)) / W,
        float(bbox.get("y", 0)) / H,
        float(bbox.get("width", 0)) / W,
        float(bbox.get("height", 0)) / H,
        safe_log(cand.get("bbox_area")),
        float(cand.get("bbox_ratio") or 0),
    ]
    struct = [
        safe_log(cand.get("text_len")),
        safe_log(cand.get("own_text_len")),
        safe_log(cand.get("num_descendants")),
        safe_log(cand.get("num_children")),
        float(cand.get("link_density") or 0),
        float(cand.get("depth") or 0),
    ]
    rec = [
        safe_log(cand.get("global_recurrence_count")),
        float(cand.get("global_recurrence_ratio") or 0),
        safe_log(cand.get("recurrence_count")),
    ]
    role = cand.get("role", "")
    role_one_hot = [0] * (len(role_to_idx) + 1)
    if role in role_to_idx:
        role_one_hot[role_to_idx[role]] = 1
    else:
        role_one_hot[-1] = 1

    numeric = np.array([geo + struct + rec + role_one_hot], dtype=np.float32)
    return hstack([word_x, char_x, csr_matrix(numeric)]).tocsr()


def featurize_many(cands, vec, role_to_idx):
    if not cands:
        return None
    parts = [featurize_candidate(c, vec, role_to_idx) for c in cands]
    return vstack(parts).tocsr()


# ============== TRAINING (best config) ==============

def train_best_model():
    """Re-train best config (LogReg + full features + combined train data) and return model + artifacts."""
    print("Loading train features...")
    feat_train = np.load(DATA_DIR / "features_train.npz")
    word_train = load_npz(DATA_DIR / "word_tfidf_train.npz")
    char_train = load_npz(DATA_DIR / "char_tfidf_train.npz")
    with (DATA_DIR / "labels_train.json").open(encoding="utf-8") as f:
        labels_train = json.load(f)

    # Filter out skip
    keep = [i for i, l in enumerate(labels_train) if l["label"] != "skip"]
    y_train = np.array([1 if labels_train[i]["label"] == "junk" else 0 for i in keep], dtype=np.int32)

    # Combine all features
    dense = np.hstack([
        feat_train["geometric"][keep],
        feat_train["structural"][keep],
        feat_train["recurrence"][keep],
        feat_train["role_onehot"][keep],
    ]).astype(np.float32)
    X_train = hstack([word_train[keep], char_train[keep], csr_matrix(dense)]).tocsr()

    print(f"X_train shape: {X_train.shape}, y_train: junk={int(y_train.sum())}/{len(y_train)}")
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                               solver="liblinear", random_state=42)
    model.fit(X_train, y_train)
    print("Model trained.")

    artifacts = load_artifacts()
    return model, artifacts


# ============== PREDICTION FOR PAGE ==============

def predict_for_page(page_candidates, model, artifacts, threshold=0.5):
    """For a list of candidates, predict junk_prob and binary label."""
    if not page_candidates:
        return []
    X = featurize_many(page_candidates, artifacts, artifacts["role_to_idx"])
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    out = []
    for i, c in enumerate(page_candidates):
        out.append({
            "candidate": c,
            "predicted_junk": bool(pred[i]),
            "junk_prob": float(proba[i]),
        })
    return out


# ============== AXTREE WALKING (for compression) ==============

def load_page_axtree(page_id):
    """Find page directory in dataset and load 1920-viewport AXTree.
    Returns (nodes list, nodes_by_id, children_by_id, root_id, bb_map)."""
    page_dir = None
    for split in DATASET_ROOT.iterdir():
        if not split.is_dir():
            continue
        cand_dir = split / page_id
        if cand_dir.is_dir():
            page_dir = cand_dir
            break
    if page_dir is None:
        return None

    ax_path = bb_path = None
    for f in page_dir.iterdir():
        if not f.is_file():
            continue
        if f.name.endswith("axtree.json.gz") and "1920" in f.name:
            ax_path = f
        if f.name.endswith("bb.json.gz") and "1920" in f.name:
            bb_path = f

    if ax_path is None:
        return None

    import gzip
    with gzip.open(ax_path, "rt", encoding="utf-8") as f:
        ax_data = json.load(f)
    nodes = ax_data.get("nodes", [])
    nodes_by_id, children_by_id, parent_by_id, root_id = bc.build_indexes(nodes)

    bb_map = {}
    if bb_path is not None:
        try:
            with gzip.open(bb_path, "rt", encoding="utf-8") as f:
                bb_data = json.load(f)
            # bb_data: list of {nodeId/backendDOMNodeId/...} or dict — depends
            if isinstance(bb_data, list):
                for it in bb_data:
                    bid = it.get("backendDOMNodeId") or it.get("backendNodeId")
                    if bid is not None:
                        bb_map[int(bid)] = it
            elif isinstance(bb_data, dict):
                for k, v in bb_data.items():
                    try:
                        bb_map[int(k)] = v
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass

    return {
        "page_dir": page_dir,
        "nodes": nodes,
        "nodes_by_id": nodes_by_id,
        "children_by_id": children_by_id,
        "root_id": root_id,
        "bb_map": bb_map,
        "axtree_path": ax_path,
    }


def collect_descendants(node_id, children_by_id):
    """Returns set of node_id including the given node and all its descendants."""
    out = {str(node_id)}
    stack = [str(node_id)]
    while stack:
        nid = stack.pop()
        for c in children_by_id.get(nid, []):
            out.add(str(c))
            stack.append(str(c))
    return out


def find_screenshot(page_dir):
    """Find 1920-viewport full screenshot path."""
    for f in page_dir.iterdir():
        if not f.is_file():
            continue
        if f.name.endswith("screenshot-full.webp") and "1920" in f.name:
            return f
    # fallback: any screenshot
    for f in page_dir.iterdir():
        if f.is_file() and f.name.endswith("screenshot-full.webp"):
            return f
    return None


# ============== COMPRESSION METRIC ==============

def compute_compression(predictions, ax_data):
    """Given predictions for a page's candidates and its AXTree data,
    compute how many nodes get removed via top-down junk-subtree pruning.

    Returns dict с counts.
    """
    # Walk top-down: a candidate counted as junk → mark its subtree
    junk_node_ids = [str(p["candidate"]["node_id"]) for p in predictions if p["predicted_junk"]]
    children_by_id = ax_data["children_by_id"]

    removed = set()
    for jid in junk_node_ids:
        removed |= collect_descendants(jid, children_by_id)

    total_nodes = len(ax_data["nodes_by_id"])
    return {
        "n_total_nodes": total_nodes,
        "n_removed_nodes": len(removed),
        "n_kept_nodes": total_nodes - len(removed),
        "compression_ratio": len(removed) / max(total_nodes, 1),
        "n_junk_candidates": len(junk_node_ids),
        "n_total_candidates": len(predictions),
        "removed_node_ids": removed,
    }
