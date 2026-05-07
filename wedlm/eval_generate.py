"""
Run the WeDLM native inference engine over one or more JSONL eval files and
write per-dataset prediction JSONLs.

Input format (per line):
    {"messages": [{"role": "system|user|assistant", "content": "..."}, ...]}

The last assistant message is treated as the ground truth and stripped from
the prompt (consistent with the SFT training format).

Output format (per line):
    {
      "index": int,
      "prediction_raw": str,
      "reference": str,
      "config": <dataset name>,
      "skipped_overlong": bool
    }

Run `evaluate.py` afterwards to score these JSONL files.
"""
import argparse
import json
import os
import time
import traceback

from transformers import AutoTokenizer
from wedlm import LLM, SamplingParams


def load_eval(path: str, tokenizer, prompt_budget: int):
    prompts, refs, skipped = [], [], set()
    with open(path) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            msgs = row["messages"]
            if len(msgs) < 2:
                continue
            ref = msgs[-1]["content"]
            prompt = tokenizer.apply_chat_template(
                msgs[:-1], tokenize=False, add_generation_prompt=True
            )
            tok_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            if tok_len > prompt_budget:
                skipped.add(len(prompts))
            prompts.append(prompt)
            refs.append(ref)
    return prompts, refs, skipped


def extract_pred(output) -> str:
    if isinstance(output, dict):
        return output.get("text", "")
    return str(output)


def run_dataset(
    name: str, path: str, output_dir: str, *,
    llm, sampling_params, tokenizer, prompt_budget: int, batch_size: int,
):
    print(f"\n{'=' * 60}\nEvaluating: {name}\n{'=' * 60}")

    prompts, refs, skipped = load_eval(path, tokenizer, prompt_budget)
    total = len(prompts)
    n_skip = len(skipped)
    print(f"Loaded {total} samples ({n_skip} overlong, will be skipped)")

    preds: list[str | None] = [None] * total
    for i in skipped:
        preds[i] = "SKIPPED_OVERLONG"

    active = [(i, prompts[i]) for i in range(total) if i not in skipped]
    n_active = len(active)
    t0 = time.time()

    for s in range(0, n_active, batch_size):
        e = min(s + batch_size, n_active)
        batch = active[s:e]
        idxs = [b[0] for b in batch]
        bp = [b[1] for b in batch]

        try:
            outs = llm.generate(bp, sampling_params)
            for orig_i, out in zip(idxs, outs):
                preds[orig_i] = extract_pred(out)
        except Exception as ex:
            tb = traceback.format_exc()
            print(f"  Batch {s}-{e} failed: {type(ex).__name__}: {ex!r}", flush=True)
            print(f"  Traceback (first 1500 chars):\n{tb[:1500]}", flush=True)
            for orig_i, p in zip(idxs, bp):
                try:
                    o = llm.generate([p], sampling_params)
                    preds[orig_i] = extract_pred(o[0])
                except Exception as ex2:
                    preds[orig_i] = f"ERROR: {type(ex2).__name__}: {ex2!r}"

        if e % (batch_size * 10) == 0 or e == n_active:
            el = time.time() - t0
            rate = e / el if el > 0 else 0
            print(f"  [{e}/{n_active}] {el:.0f}s ({rate:.1f} samples/s)", flush=True)

    el = time.time() - t0
    print(f"Inference done in {el:.1f}s")

    out_path = os.path.join(output_dir, f"{name}_outputs.jsonl")
    with open(out_path, "w") as f:
        for i, (pred, ref) in enumerate(zip(preds, refs)):
            f.write(json.dumps({
                "index": i,
                "prediction_raw": pred,
                "reference": ref,
                "config": name,
                "skipped_overlong": i in skipped,
            }, ensure_ascii=False) + "\n")
    print(f"Saved to {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model-path", required=True,
                   help="Path to a trained / merged WeDLM checkpoint, or a HF id.")
    p.add_argument("--data-dir", required=True,
                   help="Directory containing eval JSONL files.")
    p.add_argument("--datasets", nargs="+", required=True,
                   help='Dataset stems; loads "<data-dir>/eval_<stem>.jsonl" by default. '
                        'Override with --file-pattern.')
    p.add_argument("--file-pattern", default="eval_{name}.jsonl",
                   help='Filename template, with {name} placeholder. '
                        'Default: "eval_{name}.jsonl".')
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--max-output-tokens", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--wedlm-window-size", type=int, default=16)
    p.add_argument("--wedlm-entropy-threshold", type=float, default=0.6)
    p.add_argument("--wedlm-pos-penalty-factor", type=float, default=0.02)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []
    for tok in ["<|im_end|>", "<|endoftext|>"]:
        if tok in tokenizer.get_vocab():
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid not in stop_token_ids:
                stop_token_ids.append(tid)

    print(f"Loading WeDLM engine: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        wedlm_window_size=args.wedlm_window_size,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_output_tokens,
        stop_token_ids=stop_token_ids,
        wedlm_entropy_threshold=args.wedlm_entropy_threshold,
        wedlm_pos_penalty_factor=args.wedlm_pos_penalty_factor,
    )

    prompt_budget = args.max_model_len - args.max_output_tokens - 64

    for name in args.datasets:
        path = os.path.join(args.data_dir, args.file_pattern.format(name=name))
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        run_dataset(
            name, path, args.output_dir,
            llm=llm, sampling_params=sampling_params,
            tokenizer=tokenizer, prompt_budget=prompt_budget,
            batch_size=args.batch_size,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
