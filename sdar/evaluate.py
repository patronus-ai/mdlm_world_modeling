"""
Score prediction JSONLs produced by `eval_generate.py`.

Reuses the same normalization as the llm_training / wedlm_training scorers:
  - strip <think>...</think> blocks (full content) — SDAR emits CoT here
  - strip <tool_response>...</tool_response> wrapper, returning inner content
  - strip residual <tool_response>/<think> tag tokens
  - .strip() outer whitespace
  - if the result parses as JSON, re-serialize with sort_keys=True and
    compact separators, so {"b":2,"a":1} == {"a":1,"b":2}

Optionally, for datasets like api_bank where the reference is a wrapper
({api_name, input, output, exception}) but the model only emits the response
payload, pass --extract-output-field <name> to reduce the reference to
{output, exception} before comparison.

By default rows whose generation starts with `ERROR:` are dropped (these are
shard-level inference failures), as are rows with empty references after
normalization.

Usage:
    python evaluate.py \
        --input-dir ./outputs/eval_results \
        --datasets test occubench api_bank \
        --output-dir ./outputs/eval_results \
        --extract-output-field api_bank
"""
import argparse
import json
import os
import re
import sys
from typing import Optional

try:
    from rouge import Rouge
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False


# ── normalization ────────────────────────────────────────────────────────────

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
WRAPPER_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
TAG_RE = re.compile(r"</?(?:tool_response|think)\s*>")


def strip_tags(text: str) -> str:
    """Remove <think>...</think> blocks, unwrap <tool_response>, drop tag tokens."""
    text = THINK_BLOCK_RE.sub("", text)
    m = WRAPPER_RE.search(text)
    if m:
        return m.group(1).strip()
    return TAG_RE.sub("", text).strip()


def normalize_json(text: str) -> str:
    """If JSON-parseable, return canonical compact form with sorted keys."""
    try:
        return json.dumps(
            json.loads(text), sort_keys=True, ensure_ascii=False,
            separators=(",", ":"),
        )
    except (json.JSONDecodeError, TypeError):
        return text.strip()


def normalize(text: str) -> str:
    return normalize_json(strip_tags(text))


def extract_output_field(text: str) -> str:
    """For api_bank-style refs: keep only {output, exception}."""
    s = strip_tags(text)
    try:
        obj = json.loads(s)
    except Exception:
        return s
    if not isinstance(obj, dict):
        return s
    keep = {k: obj[k] for k in ("output", "exception") if k in obj}
    if not keep:
        return s
    return json.dumps(keep, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def prepare_pair(pred_raw: str, ref_raw: str, *, ref_extract_output: bool) -> tuple[str, str]:
    pred = normalize(pred_raw)
    ref = extract_output_field(ref_raw) if ref_extract_output else normalize(ref_raw)
    return pred, ref


# ── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(pairs: list[tuple[str, str]]) -> dict:
    rouge = Rouge() if HAS_ROUGE else None
    exact = semantic = 0
    rouge_l: list[float] = []

    for pred, ref in pairs:
        if pred == ref:
            exact += 1
        try:
            if json.loads(pred) == json.loads(ref):
                semantic += 1
        except (json.JSONDecodeError, TypeError):
            if pred == ref:
                semantic += 1
        if rouge is not None:
            try:
                rouge_l.append(rouge.get_scores(pred or " ", ref or " ")[0]["rouge-l"]["f"])
            except Exception:
                rouge_l.append(0.0)

    n = len(pairs) or 1
    out = {
        "num_samples": len(pairs),
        "exact_match": exact / n,
        "semantic_match": semantic / n,
    }
    if rouge is not None:
        out["rouge_l_f1"] = sum(rouge_l) / n
    return out


# ── per-dataset driver ───────────────────────────────────────────────────────

def score_file(
    name: str, path: str, output_dir: str, *,
    drop_errors: bool, drop_empty_ref: bool, ref_extract_output: bool,
    dedupe_by_index: bool,
) -> Optional[dict]:
    if not os.path.exists(path):
        print(f"[skip] {path} not found", file=sys.stderr)
        return None

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if dedupe_by_index:
        seen, deduped = set(), []
        for r in rows:
            idx = r.get("index")
            if idx in seen:
                continue
            seen.add(idx)
            deduped.append(r)
        rows = deduped

    pairs: list[tuple[str, str]] = []
    n_err = n_empty = 0
    cleaned_rows = []
    for r in rows:
        pred = r.get("prediction_raw", r.get("predicted", r.get("generated", "")))
        ref = r.get("reference", r.get("expected", r.get("ground_truth", "")))
        if drop_errors and isinstance(pred, str) and pred.startswith("ERROR"):
            n_err += 1
            continue
        p_clean, r_clean = prepare_pair(pred, ref, ref_extract_output=ref_extract_output)
        if drop_empty_ref and not r_clean.strip():
            n_empty += 1
            continue
        pairs.append((p_clean, r_clean))
        cleaned_rows.append({
            "index": r.get("index"),
            "prediction_raw": pred,
            "prediction_clean": p_clean,
            "reference_raw": ref,
            "reference_clean": r_clean,
            "match": p_clean == r_clean,
        })

    metrics = compute_metrics(pairs)
    metrics["num_dropped_errors"] = n_err
    metrics["num_empty_ref_dropped"] = n_empty

    print(f"\n  {name}:")
    print(f"    samples kept       : {metrics['num_samples']}")
    print(f"    dropped (ERROR)    : {n_err}")
    print(f"    dropped (empty ref): {n_empty}")
    print(f"    exact_match        : {metrics['exact_match']:.4f}")
    print(f"    semantic_match     : {metrics['semantic_match']:.4f}")
    if "rouge_l_f1" in metrics:
        print(f"    rouge_l_f1         : {metrics['rouge_l_f1']:.4f}")

    out_path = os.path.join(output_dir, f"{name}_scored.jsonl")
    with open(out_path, "w") as f:
        for row in cleaned_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"    -> {out_path}")
    return metrics


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input-dir", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--file-pattern", default="{name}_outputs.jsonl")
    p.add_argument("--keep-errors", action="store_true",
                   help="Include rows whose prediction starts with 'ERROR:'.")
    p.add_argument("--keep-empty-ref", action="store_true",
                   help="Include rows whose normalized ref is empty.")
    p.add_argument("--no-dedupe", action="store_true",
                   help="Disable de-duplication by `index` (default: keep first).")
    p.add_argument("--extract-output-field", nargs="*", default=[],
                   help="Dataset names whose references should be reduced to "
                        "{output, exception} before comparison (e.g. api_bank).")
    args = p.parse_args()

    if not HAS_ROUGE:
        print("[warn] `rouge` not installed — ROUGE-L will be skipped. Install: pip install rouge",
              file=sys.stderr)

    os.makedirs(args.output_dir, exist_ok=True)
    extract = set(args.extract_output_field)
    all_results: dict[str, dict] = {}

    for name in args.datasets:
        path = os.path.join(args.input_dir, args.file_pattern.format(name=name))
        m = score_file(
            name, path, args.output_dir,
            drop_errors=not args.keep_errors,
            drop_empty_ref=not args.keep_empty_ref,
            ref_extract_output=name in extract,
            dedupe_by_index=not args.no_dedupe,
        )
        if m is not None:
            all_results[name] = m

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"  {'Dataset':20s}  {'EM':>8s}  {'Semantic':>8s}  {'ROUGE-L':>8s}  {'N':>6s}")
    for ds, m in all_results.items():
        rl = m.get("rouge_l_f1", float("nan"))
        print(f"  {ds:20s}  {m['exact_match']:8.4f}  {m['semantic_match']:8.4f}  "
              f"{rl:8.4f}  {m['num_samples']:6d}")

    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
