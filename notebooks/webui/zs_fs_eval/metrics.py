"""
metrics.py - per-output scoring and distance-function library for the
ZS/FS widget-generation experiment.

Two roles:

1. CLI:  `python metrics.py`
   Reads outputs/raw_outputs.jsonl, scores every model output, writes
   outputs/metrics.jsonl (one row per output) plus a stdout summary.
   Per-output scoring covers what can be measured on a single tree in
   isolation: parse strategy, schema validity (hard vs soft errors), and
   tree statistics (node count, depth, type/hint histograms).

2. Library:  helpers imported by analysis.py for pairwise work:
       to_widget(raw)            -> (widget|None, info)
       tree_stats(widget)        -> dict
       ted(a, b)                 -> normalized structural tree edit distance
       extract_text(widget)      -> concatenated text content
       embedding_distance(a, b)  -> 1 - cosine similarity of text embeddings

No API calls. `jsonschema` is required for validation. `zss` is required
for ted(); `sentence-transformers` for embedding_distance() - both are
imported lazily, only when those functions are first used.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RAW_PATH = HERE / "outputs" / "raw_outputs.jsonl"
METRICS_PATH = HERE / "outputs" / "metrics.jsonl"
SCHEMA_PATH = HERE / "widget_schema.json"

PROVENANCE_KEYS = {"node_id", "back_id", "role"}


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def strip_fences(text: str) -> str:
    """Drop a leading ```json / ``` line and a trailing ``` line, if present."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def parse_raw(raw: str):
    """Return (obj_or_None, strategy). Tries, in order: direct json.loads,
    fence-stripped, and first-brace-to-last-brace substring."""
    if not raw or not raw.strip():
        return None, "empty"
    try:
        return json.loads(raw), "direct"
    except json.JSONDecodeError:
        pass
    stripped = strip_fences(raw)
    if stripped != raw.strip():
        try:
            return json.loads(stripped), "fenced"
        except json.JSONDecodeError:
            pass
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(raw[s:e + 1]), "substring"
        except json.JSONDecodeError:
            pass
    return None, "unparseable"


def strip_provenance(node):
    """Recursively drop AXTree provenance keys (node_id/back_id/role) that the
    generation format does not use but a model might emit anyway."""
    if not isinstance(node, dict):
        return node
    clean = {k: v for k, v in node.items() if k not in PROVENANCE_KEYS}
    if isinstance(clean.get("children"), list):
        clean["children"] = [strip_provenance(c) for c in clean["children"]]
    return clean


def normalize(node):
    """Safe best-effort fill of missing fields. Never changes `type`, never
    drops children. Returns (new_node, changes_list). Used to tell soft
    errors (a fillable omission) from hard ones."""
    changes = []

    def _norm(n, path="$"):
        if not isinstance(n, dict):
            return n  # not a node - schema validation will hard-fail later
        out = {"type": n.get("type")}

        if "hints" not in n:
            changes.append(f"{path}: filled missing hints")
            hints = []
        else:
            hints = n["hints"]
        if isinstance(hints, list):
            deduped = []
            for h in hints:
                if h not in deduped:
                    deduped.append(h)
            if len(deduped) != len(hints):
                changes.append(f"{path}: deduped hints")
            if all(isinstance(h, str) for h in deduped) and deduped != sorted(deduped):
                changes.append(f"{path}: sorted hints")
                deduped = sorted(deduped)
            out["hints"] = deduped
        else:
            out["hints"] = hints

        if "name" not in n:
            changes.append(f"{path}: filled missing name")
            out["name"] = ""
        else:
            out["name"] = n["name"]

        if "children" not in n:
            changes.append(f"{path}: filled missing children")
            out["children"] = []
        elif isinstance(n["children"], list):
            out["children"] = [_norm(c, f"{path}.children[{i}]")
                               for i, c in enumerate(n["children"])]
        else:
            out["children"] = n["children"]
        return out

    return _norm(node), changes


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
_validator = None


def get_validator():
    global _validator
    if _validator is None:
        from jsonschema import Draft202012Validator
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        _validator = Draft202012Validator(schema)
    return _validator


def schema_errors(widget):
    """List of human-readable schema violation strings ([] if valid)."""
    v = get_validator()
    return [f"{list(e.path)}: {e.message}"
            for e in sorted(v.iter_errors(widget), key=lambda e: list(e.path))]


def to_widget(raw: str):
    """Full pipeline: parse -> strip provenance -> validate -> maybe normalize.

    Returns (usable_widget_or_None, info) where
      info = {parse_strategy, status, hard_error_msgs, norm_changes}
      status in {"valid", "soft_error", "hard_error"}

    A usable widget (the cleaned / normalized tree) is returned for "valid"
    and "soft_error"; None for "hard_error".
    """
    info = {"parse_strategy": None, "status": None,
            "hard_error_msgs": [], "norm_changes": []}
    obj, strategy = parse_raw(raw)
    info["parse_strategy"] = strategy
    if obj is None:
        info["status"] = "hard_error"
        info["hard_error_msgs"] = ["could not parse JSON"]
        return None, info

    obj = strip_provenance(obj)
    if not schema_errors(obj):
        info["status"] = "valid"
        return obj, info

    norm, changes = normalize(obj)
    norm_errs = schema_errors(norm)
    if not norm_errs:
        info["status"] = "soft_error"
        info["norm_changes"] = changes
        return norm, info

    info["status"] = "hard_error"
    info["hard_error_msgs"] = norm_errs
    return None, info


def _to_widget_via(raw, converter, run_axtree_validation=False):
    """Generic pipeline: parse raw JSON, optionally validate the parsed source
    as an AXtree, run it through `converter` (which must return a widget dict),
    then validate. Status semantics identical to to_widget.

    When run_axtree_validation=True, the parsed source is independently checked
    against AXtree-level integrity rules (broken refs, cycles, unknown roles,
    multiple parents, orphans) and the result is stored in info['axtree_status']
    + info['axtree_issues']. This surfaces failures the permissive converter
    would otherwise paper over.

    `converter` is expected never to raise on garbage input.
    """
    info = {"parse_strategy": None, "status": None,
            "hard_error_msgs": [], "norm_changes": [],
            "axtree_status": None, "axtree_issues": []}
    obj, strategy = parse_raw(raw)
    info["parse_strategy"] = strategy
    if obj is None:
        info["status"] = "hard_error"
        info["hard_error_msgs"] = ["could not parse JSON"]
        if run_axtree_validation:
            info["axtree_status"] = "hard"
            info["axtree_issues"] = ["unparseable JSON"]
        return None, info

    if run_axtree_validation:
        from axtree_to_widget import validate_axtree
        ax_status, ax_issues = validate_axtree(obj)
        info["axtree_status"] = ax_status
        # truncate the issues list so metrics.jsonl rows stay compact
        info["axtree_issues"] = ax_issues[:5]

    widget = converter(obj)

    if not schema_errors(widget):
        info["status"] = "valid"
        return widget, info

    norm, changes = normalize(widget)
    norm_errs = schema_errors(norm)
    if not norm_errs:
        info["status"] = "soft_error"
        info["norm_changes"] = changes
        return norm, info

    info["status"] = "hard_error"
    info["hard_error_msgs"] = norm_errs
    return None, info


def to_widget_from_axtree(raw: str):
    """T3 (simplified-nested AXtree) pathway: parse -> axtree_to_widget -> validate.
    Also runs AXtree-level integrity validation on the source."""
    from axtree_to_widget import axtree_to_widget
    return _to_widget_via(raw, axtree_to_widget, run_axtree_validation=True)


def to_widget_from_chrome_axtree(raw: str):
    """T3 real-format pathway: parse Chrome DevTools Protocol AXTree (flat
    list with parentId/childIds references) -> chrome_axtree_to_widget
    (resolves flat structure into nested, then delegates to axtree_to_widget)
    -> validate. Also runs AXtree-level integrity validation on the source.

    This is also used by t4 (text -> AXtree -> widget) since the model output
    is the same Chrome flat format."""
    from axtree_to_widget import chrome_axtree_to_widget
    return _to_widget_via(raw, chrome_axtree_to_widget, run_axtree_validation=True)


def parse_output(row):
    """Dispatch by row['type']:
        t3_real, t4 -> Chrome DevTools Protocol flat-format AXtree pathway
        t3*         -> simplified nested AXtree pathway (covers t3, t3_full,
                       t3_lean and any future simplified t3 variant)
        others      -> parse model output as widget JSON directly (t1, t2)
    Returns the same (widget|None, info) shape regardless of pathway.
    For AXtree pathways, info also contains 'axtree_status' and
    'axtree_issues' fields surfacing source-level integrity problems
    that the permissive converter would otherwise hide."""
    raw = row.get("output", "") or ""
    t = str(row.get("type", ""))
    if t in ("t3_real", "t4"):
        return to_widget_from_chrome_axtree(raw)
    if t.startswith("t3"):
        return to_widget_from_axtree(raw)
    return to_widget(raw)


# --------------------------------------------------------------------------
# Per-tree statistics
# --------------------------------------------------------------------------
def tree_stats(widget):
    """Node count, max leaf depth, and type/hint histograms for one tree."""
    types, hints = Counter(), Counter()
    leaf_depths = []

    def walk(n, d):
        if not isinstance(n, dict):
            return
        types[str(n.get("type", "?"))] += 1
        for h in (n.get("hints") or []):
            hints[str(h)] += 1
        kids = n.get("children") or []
        if not kids:
            leaf_depths.append(d)
        for c in kids:
            walk(c, d + 1)

    walk(widget, 0)
    return {
        "node_count": sum(types.values()),
        "max_depth": max(leaf_depths) if leaf_depths else 0,
        "type_hist": dict(types),
        "hint_hist": dict(hints),
    }


# --------------------------------------------------------------------------
# Distance functions (used by analysis.py for pairwise comparisons)
# --------------------------------------------------------------------------
def _to_zss(node):
    """Widget dict -> zss.Node. Label encodes type + sorted hints, so TED
    measures STRUCTURAL difference; text content is compared separately via
    embedding_distance()."""
    import zss
    if not isinstance(node, dict):
        return zss.Node("INVALID")
    typ = str(node.get("type", "?"))
    hints = sorted(str(h) for h in (node.get("hints") or []) if isinstance(h, str))
    z = zss.Node(typ + "|" + ",".join(hints))
    for c in (node.get("children") or []):
        z.addkid(_to_zss(c))
    return z


def ted(a, b, normalize_by_size=True):
    """Structural tree edit distance between two widget dicts. When
    normalize_by_size, the raw edit distance is divided by the larger tree's
    node count, giving a roughly [0, 1] score (0 = identical structure)."""
    import zss
    raw = zss.simple_distance(_to_zss(a), _to_zss(b))
    if not normalize_by_size:
        return float(raw)
    denom = max(tree_stats(a)["node_count"], tree_stats(b)["node_count"], 1)
    return float(raw) / denom


def extract_text(widget):
    """Concatenate all human-readable text in document order: `name` of text,
    image and block nodes. Link `name` (a URL) is intentionally skipped - a
    link's visible content lives in its children."""
    parts = []

    def walk(n):
        if not isinstance(n, dict):
            return
        name = n.get("name") or ""
        if name and n.get("type") in ("text", "image", "block"):
            parts.append(str(name))
        for c in (n.get("children") or []):
            walk(c)

    walk(widget)
    return " ".join(parts)


_embedder = None


def get_embedder(model_name="all-MiniLM-L6-v2"):
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(model_name)
    return _embedder


def embedding_distance(text_a, text_b):
    """1 - cosine similarity of sentence embeddings. 0 = identical content,
    larger = more semantically different. Empty-text edge cases are explicit:
    both empty -> 0.0, exactly one empty -> 1.0."""
    a = (text_a or "").strip()
    b = (text_b or "").strip()
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    import numpy as np
    emb = get_embedder().encode([a, b], normalize_embeddings=True)
    return 1.0 - float(np.dot(emb[0], emb[1]))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_rows(raw_rows):
    """raw_outputs.jsonl rows -> metrics rows (per-output scalar metrics)."""
    out = []
    for r in raw_rows:
        widget, info = parse_output(r)
        stats = (tree_stats(widget) if widget is not None
                 else {"node_count": 0, "max_depth": 0,
                       "type_hist": {}, "hint_hist": {}})
        out.append({
            "question_id": r.get("question_id"),
            "type": r.get("type"),
            "shots": r.get("shots"),
            "seed_mode": r.get("seed_mode"),
            "run_idx": r.get("run_idx"),
            "api_error": bool(r.get("error")),
            "parse_strategy": info["parse_strategy"],
            "status": info["status"],
            "axtree_status": info.get("axtree_status"),
            "axtree_issues": info.get("axtree_issues", []),
            "hard_error_msgs": info["hard_error_msgs"],
            "norm_changes": info["norm_changes"],
            "node_count": stats["node_count"],
            "max_depth": stats["max_depth"],
            "type_hist": stats["type_hist"],
            "hint_hist": stats["hint_hist"],
            "output_chars": len(r.get("output", "") or ""),
        })
    return out


def print_summary(metric_rows):
    print(f"scored {len(metric_rows)} outputs -> {METRICS_PATH}\n")
    groups = defaultdict(Counter)
    for row in metric_rows:
        groups[(row["type"], row["shots"])][row["status"]] += 1
    print(f"{'condition':<14}{'valid':>8}{'soft':>8}{'hard':>8}{'n':>7}")
    print("-" * 45)
    for key in sorted(groups):
        c = groups[key]
        n = sum(c.values())
        print(f"{key[0] + '/' + key[1]:<14}"
              f"{c['valid']:>8}{c['soft_error']:>8}{c['hard_error']:>8}{n:>7}")
    total = Counter(row["status"] for row in metric_rows)
    n = sum(total.values())
    print("-" * 45)
    print(f"{'ALL':<14}{total['valid']:>8}{total['soft_error']:>8}"
          f"{total['hard_error']:>8}{n:>7}")
    print("\nparse strategy:", dict(Counter(r["parse_strategy"] for r in metric_rows)))

    # AXtree-level vs widget-level status comparison for AXtree-based types
    ax_rows = [r for r in metric_rows if r.get("axtree_status") is not None]
    if ax_rows:
        print(f"\n=== AXtree-level validation (for {len(ax_rows)} t3*/t4 rows) ===")
        print(f"{'condition':<14}{'ax_valid':>10}{'ax_soft':>10}{'ax_hard':>10}{'n':>7}")
        print("-" * 51)
        ax_groups = defaultdict(Counter)
        for row in ax_rows:
            ax_groups[(row["type"], row["shots"])][row["axtree_status"]] += 1
        for key in sorted(ax_groups):
            c = ax_groups[key]
            n = sum(c.values())
            print(f"{key[0] + '/' + key[1]:<14}"
                  f"{c['valid']:>10}{c['soft']:>10}{c['hard']:>10}{n:>7}")
        # Surface the hidden-failure rate: widget-valid but axtree-hard/soft
        hidden_hard = sum(1 for r in ax_rows
                          if r["status"] == "valid" and r["axtree_status"] == "hard")
        hidden_soft = sum(1 for r in ax_rows
                          if r["status"] == "valid" and r["axtree_status"] == "soft")
        n_widget_valid = sum(1 for r in ax_rows if r["status"] == "valid")
        if n_widget_valid:
            print(f"\nHidden failures (widget=valid BUT axtree≠valid):")
            print(f"  axtree=hard, widget=valid : {hidden_hard:>4}/{n_widget_valid} "
                  f"({100*hidden_hard/n_widget_valid:.1f}% of widget-valid)")
            print(f"  axtree=soft, widget=valid : {hidden_soft:>4}/{n_widget_valid} "
                  f"({100*hidden_soft/n_widget_valid:.1f}% of widget-valid)")
        # Top AXtree-level issues
        ax_issue_counter = Counter()
        for r in ax_rows:
            for issue in (r.get("axtree_issues") or []):
                # normalize issue messages (strip node ids that vary)
                import re
                normalized = re.sub(r"\bnode \w+", "node X", issue)
                normalized = re.sub(r"\bnodeId \S+", "nodeId X", normalized)
                normalized = re.sub(r"\d+", "N", normalized)
                ax_issue_counter[normalized[:80]] += 1
        if ax_issue_counter:
            print(f"\nTop AXtree-level issues (normalized; over all t3*/t4 rows):")
            for issue, cnt in ax_issue_counter.most_common(10):
                print(f"  {cnt:>4}  {issue}")

    if total["hard_error"]:
        msgs = Counter()
        for row in metric_rows:
            for m in row["hard_error_msgs"]:
                msgs[m] += 1
        print("\ntop hard-error messages (widget-level):")
        for m, cnt in msgs.most_common(8):
            print(f"  {cnt:>4}  {m}")


def main():
    raw_rows = read_jsonl(RAW_PATH)
    if not raw_rows:
        raise SystemExit(f"no rows at {RAW_PATH} - run pipeline.py first")
    metric_rows = score_rows(raw_rows)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        for row in metric_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print_summary(metric_rows)


if __name__ == "__main__":
    main()
