"""
Generation-based evaluation against an OpenAI-compatible chat endpoint
(e.g. a vLLM server). Computes exact match, JSON-semantic match, and ROUGE-L
over one or more JSONL files of {"messages": [...]} samples — the last
message is treated as the ground-truth assistant response and stripped from
the prompt.

Usage:
    python evaluate.py \
        --api-url http://localhost:8000/v1/chat/completions \
        --model my-model \
        --data-dir ./data \
        --datasets test eval_a eval_b \
        --output-dir ./output/eval_results

Environment:
    No secrets are read from env. If your endpoint requires an API key, pass
    --api-key <key> on the command line (or set OPENAI_API_KEY in the
    surrounding shell — it's read only for the auth header, never logged).
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

try:
    from rouge import Rouge
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False


# ── output cleaning / comparison ─────────────────────────────────────────────

TAG_RE = re.compile(r"</?(?:tool_response|think)\s*>")


def strip_tags(text: str) -> str:
    return TAG_RE.sub("", text).strip()


def normalize_json(text: str) -> str:
    try:
        return json.dumps(json.loads(text), sort_keys=True, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return text.strip()


def prepare_pair(pred_raw: str, ref_raw: str) -> tuple[str, str]:
    """Strip surface tags + JSON-normalise both sides for fair comparison."""
    pred = strip_tags(pred_raw)
    return normalize_json(pred), normalize_json(ref_raw)


# ── API call ─────────────────────────────────────────────────────────────────

def call_api(
    api_url: str, model: str, messages: list, *,
    api_key: str | None = None, max_tokens: int = 2048,
    temperature: float = 0.0, timeout: int = 120,
) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {e}"


# ── data loading ─────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            msgs = row["messages"]
            if len(msgs) < 2:
                continue
            samples.append({
                "prompt": msgs[:-1],
                "reference": msgs[-1]["content"],
            })
    return samples


# ── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    rouge = Rouge() if HAS_ROUGE else None
    exact = semantic = 0
    rouge_l = []

    for pred_raw, ref_raw in zip(predictions, references):
        pred, ref = prepare_pair(pred_raw, ref_raw)
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

    n = len(predictions) or 1
    out = {
        "num_samples": len(predictions),
        "exact_match": exact / n,
        "semantic_match": semantic / n,
    }
    if rouge is not None:
        out["rouge_l_f1"] = sum(rouge_l) / n
    return out


# ── per-dataset driver ───────────────────────────────────────────────────────

def evaluate_dataset(
    name: str, path: str, output_dir: str, *,
    api_url: str, model: str, api_key: str | None,
    max_workers: int, max_tokens: int,
) -> dict:
    print(f"\n{'='*60}\nEvaluating: {name}\n{'='*60}")
    samples = load_jsonl(path)
    print(f"Loaded {len(samples)} samples from {path}")

    predictions: list[str | None] = [None] * len(samples)
    t0 = time.time()
    completed = 0

    def task(i):
        return i, call_api(api_url, model, samples[i]["prompt"],
                           api_key=api_key, max_tokens=max_tokens)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(task, i): i for i in range(len(samples))}
        for fut in as_completed(futs):
            i, pred = fut.result()
            predictions[i] = pred
            completed += 1
            if completed % 100 == 0 or completed == len(samples):
                el = time.time() - t0
                print(f"  [{completed}/{len(samples)}] {el:.0f}s ({completed/el:.1f} req/s)")

    references = [s["reference"] for s in samples]
    metrics = compute_metrics(predictions, references)  # type: ignore[arg-type]

    print(f"\n  Results for {name}:")
    for k, v in metrics.items():
        print(f"    {k:18s}: {v:.4f}" if isinstance(v, float) else f"    {k:18s}: {v}")

    out_path = os.path.join(output_dir, f"{name}_outputs.jsonl")
    with open(out_path, "w") as f:
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            pred_clean, ref_clean = prepare_pair(pred or "", ref)
            f.write(json.dumps({
                "index": i,
                "prediction_raw": pred,
                "prediction_clean": pred_clean,
                "reference_clean": ref_clean,
                "match": pred_clean == ref_clean,
            }, ensure_ascii=False) + "\n")
    print(f"  Saved per-sample outputs to {out_path}")
    return metrics


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--model", required=True, help="Model id served by the endpoint.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument(
        "--datasets", nargs="+", required=True,
        help='Dataset stems; loads "<data-dir>/<stem>.jsonl" for each.',
    )
    p.add_argument("--output-dir", default="./output/eval_results")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"),
                   help="Bearer token for the endpoint (default: $OPENAI_API_KEY).")
    p.add_argument("--max-workers", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=2048)
    args = p.parse_args()

    if not HAS_ROUGE:
        print("[warn] `rouge` not installed — ROUGE-L will be skipped. Install: pip install rouge", file=sys.stderr)

    os.makedirs(args.output_dir, exist_ok=True)
    all_results: dict[str, dict] = {}

    for name in args.datasets:
        path = os.path.join(args.data_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            print(f"[skip] {path} not found", file=sys.stderr)
            continue
        all_results[name] = evaluate_dataset(
            name, path, args.output_dir,
            api_url=args.api_url, model=args.model, api_key=args.api_key,
            max_workers=args.max_workers, max_tokens=args.max_tokens,
        )

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    header = f"  {'Dataset':20s}  {'EM':>8s}  {'Semantic':>8s}  {'ROUGE-L':>8s}  {'N':>6s}"
    print(header)
    for ds, m in all_results.items():
        rl = m.get("rouge_l_f1", float("nan"))
        print(f"  {ds:20s}  {m['exact_match']:8.4f}  {m['semantic_match']:8.4f}  {rl:8.4f}  {m['num_samples']:6d}")

    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
