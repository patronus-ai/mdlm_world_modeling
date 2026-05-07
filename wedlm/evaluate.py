"""
Score prediction JSONLs produced by `eval_generate.py`.

Reuses the same normalization as the llm_training scorer:
  - strip <tool_response>...</tool_response> and <think>...</think> tags
  - .strip() outer whitespace
  - if the result parses as JSON, re-serialize with sort_keys=True and
    compact separators (so {"b":2,"a":1} == {"a":1,"b":2})

Input format (one row per line; written by eval_generate.py):
    {
      "index": int,
      "prediction_raw": str,
      "reference": str,
      "config": str,
      "skipped_overlong": bool
    }

Usage:
    python evaluate.py \
        --input-dir ./outputs/eval_results \
        --datasets occubench api_bank intercode \
        --output-dir ./outputs/eval_results

For datasets like api_bank where the reference contains a wrapper object
({api_name, input, output, exception}) but the model only predicts the
response payload, pass --extract-output-field <dataset_name>.
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


# ── normalization (same as llm_training/dist/evaluate.py) ────────────────────

TAG_RE = re.compile(r"</?(?:tool_response|think)\s*>")
WRAPPER_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)


def strip_tags(text: str) -> str:
    """Remove <tool_response> / <think> tags and trim whitespace.

    If the input is wrapped in a single <tool_response>...</tool_response>,
    return the inner content; otherwise just strip the tag tokens themselves.
    """
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
    """For api_bank-style refs: keep only the model-predicted subset
    ({output, exception}) so it's comparable to what the model emits.
    """
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
    drop_skipped: bool, drop_empty_ref: bool, ref_extract_output: bool,
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

    pairs: list[tuple[str, str]] = []
    n_skipped = n_empty = 0
    cleaned_rows = []
    for r in rows:
        if drop_skipped and r.get("skipped_overlong"):
            n_skipped += 1
            continue
        pred = r.get("prediction_raw", r.get("predicted", r.get("generated", "")))
        ref = r.get("reference", r.get("expected", r.get("ground_truth", "")))
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
    metrics["num_skipped_overlong"] = n_skipped
    metrics["num_empty_ref_dropped"] = n_empty

    print(f"\n  {name}:")
    print(f"    samples kept       : {metrics['num_samples']}")
    print(f"    skipped (overlong) : {n_skipped}")
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
    p.add_argument("--input-dir", required=True,
                   help="Directory containing <name>_outputs.jsonl files.")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--file-pattern", default="{name}_outputs.jsonl",
                   help='Filename template, with {name} placeholder.')
    p.add_argument("--keep-overlong", action="store_true",
                   help="Include rows flagged skipped_overlong (default: drop).")
    p.add_argument("--keep-empty-ref", action="store_true",
                   help="Include rows whose normalized ref is empty (default: drop).")
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
            drop_skipped=not args.keep_overlong,
            drop_empty_ref=not args.keep_empty_ref,
            ref_extract_output=name in extract,
        )
        if m is not None:
            all_results[name] = m

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
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
