"""
pipeline.py - ZS/FS widget-generation experiment runner.

Runs the cross product:
    type (t1/t2) x shots (zs/fs) x seed_mode (fixed/semi/random) x N runs x questions

  - Type 1 (t1): user question -> widget tree (single pass)
  - Type 2 (t2): user question -> plain text -> widget tree (two pass)
        The intermediate plain text is generated ONCE per question
        (temperature 0) and cached, so stage-2 conversion variance is
        isolated from content-generation variance.
  - shots: zero-shot (schema only) vs few-shot (schema + K example messages)
  - seed_mode: maps to a temperature; runs N times each to measure variance.

Writes one JSONL row per generation call to outputs/raw_outputs.jsonl.
RESUMABLE: re-running skips combinations already present in raw_outputs.jsonl.

No parsing / scoring happens here - raw model text is stored verbatim.
metrics.py and analysis.py consume raw_outputs.jsonl downstream.

Set MODEL_BACKEND = "mock" to exercise the full pipeline with no API calls.

Env vars: ANTHROPIC_API_KEY (anthropic) or OPENAI_API_KEY (openai).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
HERE = Path(__file__).parent
QUESTIONS_PATH = HERE / "eval_set" / "questions.jsonl"
FS_EXAMPLES_PATH = HERE / "eval_set" / "few_shot_examples.jsonl"
PROMPTS_DIR = HERE / "prompts"
OUT_DIR = HERE / "outputs"
OUT_PATH = OUT_DIR / "raw_outputs.jsonl"
INTERMEDIATE_TEXT_PATH = OUT_DIR / "intermediate_text.jsonl"

# "anthropic" | "openai" | "mock"
MODEL_BACKEND = "openai"
MODEL_NAME = "gpt-4o"

# seed_mode -> temperature + how many runs to do for variance estimation.
# At temperature 0 the runs act as a near-deterministic control; you can
# drop "fixed" runs to 1 to save budget once you've confirmed that.
SEED_MODES = {
    "fixed":  {"temperature": 0.0, "runs": 3},
    "semi":   {"temperature": 0.4, "runs": 3},
    "random": {"temperature": 0.8, "runs": 3},
}
TYPES = ["t1", "t2", "t3", "t3_full", "t3_lean", "t3_real", "t4"]
SHOTS = ["zs", "fs"]
# Per-type override of SHOTS. Default (when a type is not listed here) is SHOTS.
# Used to keep the schema-bias check variants ZS-only — they exist to test
# whether t3's 100% validity depends on my prescribed role enumeration, and
# to test the model against the real Chrome DevTools Protocol AXTree format.
SHOTS_PER_TYPE = {
    "t3_full": ["zs"],
    "t3_lean": ["zs"],
    # t3_real now runs BOTH zs and fs — fs converts nested few-shot examples
    # to Chrome flat format on the fly (see fs_messages_chrome below).
    # t4 runs both zs and fs (t2-style architecture with AXtree output).
}
FS_K = 3                # number of few-shot examples to inject
MAX_TOKENS = 4096
SLEEP_BETWEEN_CALLS = 0.5  # seconds, gentle rate limiting

# Smoke-test knob: set to an int to run only the first N questions, then set
# back to None for the full run. The resume logic keeps the partial outputs,
# so a small smoke test is not wasted - the full run continues from there.
LIMIT_QUESTIONS = None


# --------------------------------------------------------------------------
# Model backend (swappable)
# --------------------------------------------------------------------------
def call_model(messages, system, temperature, seed=None):
    """Call the configured backend.

    messages: list of {"role": "user"|"assistant", "content": str}
    system:   system prompt string
    Returns (text, usage_dict).
    """
    if MODEL_BACKEND == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        resp = client.messages.create(
            model=MODEL_NAME,
            system=system,
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
        )
        text = resp.content[0].text
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
        return text, usage

    if MODEL_BACKEND == "openai":
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY
        oai_messages = [{"role": "system", "content": system}] + messages
        kwargs = dict(
            model=MODEL_NAME,
            messages=oai_messages,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
        )
        if seed is not None:
            kwargs["seed"] = seed  # OpenAI: best-effort reproducibility
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content
        usage = {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        }
        return text, usage

    if MODEL_BACKEND == "mock":
        # Canned outputs so the whole pipeline can be exercised offline.
        last = messages[-1]["content"] if messages else ""
        sys_lc = (system or "").lower()
        # Type 2 stage 1: plain-text answer (system asks for prose, not JSON)
        if "plain text" in sys_lc and "convert" not in sys_lc:
            return (f"Mock plain-text answer to: {last[:80]}",
                    {"input_tokens": 10, "output_tokens": 20})
        # Type 3 real (Chrome DevTools Protocol flat format)
        if "chrome" in sys_lc or "parentid" in sys_lc:
            chrome_dump = {
                "nodes": [
                    {"nodeId": "1",
                     "role": {"type": "internalRole", "value": "RootWebArea"},
                     "name": {"type": "computedString", "value": "Mock"},
                     "childIds": ["2"]},
                    {"nodeId": "2",
                     "role": {"type": "role", "value": "main"},
                     "childIds": ["3"], "parentId": "1"},
                    {"nodeId": "3",
                     "role": {"type": "role", "value": "heading"},
                     "name": {"type": "computedString",
                              "value": f"messages={len(messages)} temp={temperature}"},
                     "properties": [{"name": "level",
                                     "value": {"type": "integer", "value": 1}}],
                     "childIds": [], "parentId": "2"},
                ]
            }
            return (json.dumps(chrome_dump, ensure_ascii=False),
                    {"input_tokens": 50, "output_tokens": 100})
        # Type 3 (simplified nested AXtree)
        if "axtree" in sys_lc:
            axtree = {
                "role": "main",
                "children": [
                    {"role": "heading", "level": 1,
                     "name": "Mock AXtree answer", "children": []},
                    {"role": "paragraph",
                     "name": f"messages={len(messages)} temp={temperature}",
                     "children": []},
                ],
            }
            return (json.dumps(axtree, ensure_ascii=False),
                    {"input_tokens": 10, "output_tokens": 30})
        # Type 1 / Type 2 stage 2: widget JSON
        widget = {
            "type": "block", "hints": ["main", "vbox"], "name": "",
            "children": [
                {"type": "text", "hints": ["heading"],
                 "name": "Mock answer", "children": []},
                {"type": "text", "hints": [],
                 "name": f"messages={len(messages)} temp={temperature}",
                 "children": []},
            ],
        }
        return (json.dumps(widget, ensure_ascii=False),
                {"input_tokens": 10, "output_tokens": 30})

    raise ValueError(f"unknown MODEL_BACKEND: {MODEL_BACKEND}")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_prompts():
    widget_schema = (PROMPTS_DIR / "schema_block.txt").read_text(encoding="utf-8")
    axtree_schema = (PROMPTS_DIR / "axtree_schema_block.txt").read_text(encoding="utf-8")
    axtree_full = (PROMPTS_DIR / "axtree_schema_full.txt").read_text(encoding="utf-8")
    axtree_lean = (PROMPTS_DIR / "axtree_schema_lean.txt").read_text(encoding="utf-8")
    axtree_real = (PROMPTS_DIR / "axtree_schema_real.txt").read_text(encoding="utf-8")

    def load(name, schema_text):
        raw = (PROMPTS_DIR / name).read_text(encoding="utf-8")
        return raw.replace("{SCHEMA}", schema_text)

    return {
        "t1": load("system_t1.txt", widget_schema),
        "t2_stage1": load("system_t2_stage1.txt", widget_schema),
        "t2_stage2": load("system_t2_stage2.txt", widget_schema),
        "t3": load("system_t3.txt", axtree_schema),
        "t3_full": load("system_t3_full.txt", axtree_full),
        "t3_lean": load("system_t3_lean.txt", axtree_lean),
        "t3_real": load("system_t3_real.txt", axtree_real),
        # t4 = t2-style architecture (fixed plain-text intermediate) but the
        # second stage produces real Chrome AXTree. Uses the same schema_real
        # text as t3_real, just with a different system framing (text → AXtree
        # rather than question → AXtree).
        "t4": load("system_t4.txt", axtree_real),
    }


def fs_messages(fs_examples, k, input_field, output_field="widget"):
    """Build few-shot chat messages.

    input_field is "question" for Type 1 / Type 3, "text" for Type 2 stage 2.
    output_field is "widget" for T1 / T2, "axtree" for T3.
    Each example contributes a user message (the input) and an assistant
    message (the gold output JSON).
    """
    msgs = []
    for ex in fs_examples[:k]:
        msgs.append({"role": "user", "content": ex[input_field]})
        msgs.append({"role": "assistant",
                     "content": json.dumps(ex[output_field], ensure_ascii=False)})
    return msgs


def fs_messages_chrome(fs_examples, k, input_field):
    """Few-shot messages for t3_real: converts the nested `axtree` field of
    each example into Chrome DevTools Protocol flat format on the fly, so
    the model sees demonstrations in the exact format we ask it to produce.
    """
    from axtree_to_widget import nested_to_chrome
    msgs = []
    for ex in fs_examples[:k]:
        chrome_dump = nested_to_chrome(ex["axtree"])
        msgs.append({"role": "user", "content": ex[input_field]})
        msgs.append({"role": "assistant",
                     "content": json.dumps(chrome_dump, ensure_ascii=False, indent=2)})
    return msgs


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    questions = read_jsonl(QUESTIONS_PATH)
    fs_examples = read_jsonl(FS_EXAMPLES_PATH)
    if not questions:
        raise SystemExit(f"no questions found at {QUESTIONS_PATH}")
    if not fs_examples:
        raise SystemExit(f"no few-shot examples found at {FS_EXAMPLES_PATH}")
    if LIMIT_QUESTIONS:
        questions = questions[:LIMIT_QUESTIONS]
        print(f"LIMIT_QUESTIONS={LIMIT_QUESTIONS}: running first {len(questions)} questions only")
    prompts = load_prompts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- plan / cost estimate -------------------------------------------
    per_q = sum(sm["runs"] for sm in SEED_MODES.values()) * len(TYPES) * len(SHOTS)
    total_gen = per_q * len(questions)
    total_stage1 = len(questions)
    print(f"backend={MODEL_BACKEND} model={MODEL_NAME}")
    print(f"questions={len(questions)} fs_examples={len(fs_examples)} (using K={FS_K})")
    print(f"generation calls: {total_gen}  + stage-1 text calls: {total_stage1}"
          f"  = {total_gen + total_stage1} total")

    # ---- resume bookkeeping ---------------------------------------------
    done = set()
    for r in read_jsonl(OUT_PATH):
        done.add((r["question_id"], r["type"], r["shots"],
                  r["seed_mode"], r["run_idx"]))
    if done:
        print(f"resume: {len(done)} generation calls already done, skipping them")

    # ---- Type 2 stage 1: intermediate text, ONCE per question -----------
    intermediate = {r["question_id"]: r["text"] for r in read_jsonl(INTERMEDIATE_TEXT_PATH)}
    with open(INTERMEDIATE_TEXT_PATH, "a", encoding="utf-8") as f:
        for q in questions:
            if q["id"] in intermediate:
                continue
            text, usage = call_model(
                messages=[{"role": "user", "content": q["question"]}],
                system=prompts["t2_stage1"],
                temperature=0.0,
            )
            intermediate[q["id"]] = text
            f.write(json.dumps({"question_id": q["id"], "text": text,
                                "usage": usage}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[stage1] {q['id']}")
            time.sleep(SLEEP_BETWEEN_CALLS)

    # ---- main generation loop -------------------------------------------
    n_done = 0
    n_err = 0
    with open(OUT_PATH, "a", encoding="utf-8") as out:
        for q in questions:
            for typ in TYPES:
                for shots in SHOTS_PER_TYPE.get(typ, SHOTS):
                    for sm_name, sm in SEED_MODES.items():
                        for run_idx in range(sm["runs"]):
                            key = (q["id"], typ, shots, sm_name, run_idx)
                            if key in done:
                                continue

                            if typ == "t1":
                                system = prompts["t1"]
                                msgs = []
                                if shots == "fs":
                                    msgs += fs_messages(fs_examples, FS_K, "question")
                                msgs.append({"role": "user", "content": q["question"]})
                            elif typ == "t2":  # t2 stage 2
                                system = prompts["t2_stage2"]
                                msgs = []
                                if shots == "fs":
                                    msgs += fs_messages(fs_examples, FS_K, "text")
                                msgs.append({"role": "user",
                                             "content": intermediate[q["id"]]})
                            elif typ == "t4":
                                # Type 4: t2-style architecture (fixed plain-text
                                # intermediate, generated once per question at temp=0
                                # and cached in intermediate_text.jsonl) BUT second
                                # stage produces real Chrome AXTree (not widget).
                                # Isolates AXtree-conversion variance from content
                                # variance, parallel to t2's setup.
                                system = prompts["t4"]
                                msgs = []
                                if shots == "fs":
                                    msgs += fs_messages_chrome(fs_examples, FS_K, "text")
                                msgs.append({"role": "user",
                                             "content": intermediate[q["id"]]})
                            elif typ in ("t3", "t3_full", "t3_lean", "t3_real"):
                                # All t3 variants generate AXtree; conversion happens in metrics.py.
                                # t3       = original scaffolded prompt (~17 listed roles, nested)
                                # t3_full  = full ARIA / AXtree role enumeration (~70 roles, nested)
                                # t3_lean  = structural contract only, no role enumeration, nested
                                # t3_real  = real Chrome DevTools Protocol flat-format AXtree dump
                                system = prompts[typ]
                                msgs = []
                                if shots == "fs":
                                    if typ == "t3_real":
                                        # Convert nested few-shot examples to Chrome flat format
                                        msgs += fs_messages_chrome(fs_examples, FS_K, "question")
                                    else:
                                        msgs += fs_messages(fs_examples, FS_K,
                                                            "question", output_field="axtree")
                                msgs.append({"role": "user", "content": q["question"]})
                            else:
                                raise ValueError(f"unknown type: {typ}")

                            seed = run_idx if MODEL_BACKEND == "openai" else None
                            t0 = time.time()
                            try:
                                text, usage = call_model(
                                    msgs, system, sm["temperature"], seed=seed)
                                err = None
                            except Exception as exc:  # noqa: BLE001
                                text, usage, err = "", {}, repr(exc)
                                n_err += 1

                            row = {
                                "question_id": q["id"],
                                "type": typ,
                                "shots": shots,
                                "seed_mode": sm_name,
                                "temperature": sm["temperature"],
                                "run_idx": run_idx,
                                "seed": seed,
                                "model": MODEL_NAME,
                                "output": text,
                                "usage": usage,
                                "latency_s": round(time.time() - t0, 2),
                                "error": err,
                            }
                            out.write(json.dumps(row, ensure_ascii=False) + "\n")
                            out.flush()
                            n_done += 1
                            status = "ERR" if err else "ok"
                            print(f"[{status}] {q['id']} {typ} {shots} "
                                  f"{sm_name} run{run_idx}")
                            time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\ndone: {n_done} new calls this run, {n_err} errors")
    print(f"output -> {OUT_PATH}")


if __name__ == "__main__":
    main()
