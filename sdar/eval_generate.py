"""
Multi-GPU batched generation for SDAR block-diffusion checkpoints.

Sharding: round-robin across GPUs via ProcessPoolExecutor.
Batching: each shard sorts prompts by length and forwards in groups of
PER_BATCH (left-padded to a multiple of block_length).

Input format (per line of <data-dir>/eval_<name>.jsonl):
    {"messages": [{"role": "system|user|assistant", "content": "..."}, ...]}

The last message's content is treated as the ground truth and stripped from
the prompt before generation.

Output format (per line, written to <output-dir>/<name>_outputs.jsonl):
    {"index": int, "config": str, "prediction_raw": str,
     "reference": str, "generated": str, "ground_truth": str}

Run `evaluate.py` afterwards to score these JSONL files.

Usage:
    python eval_generate.py \
        --sdar-dir ./SDAR \
        --model-path ./outputs/sdar_sft/checkpoint-1000 \
        --data-dir ./data \
        --datasets test occubench api_bank \
        --output-dir ./outputs/eval_results \
        --num-gpus 8
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor


def process_shard(args):
    (gpu_id, sdar_dir, model_path, indices, samples,
     prompt_truncate, gen_length, block_length, denoising_steps,
     per_batch, temperature, top_p, top_k, confidence_threshold,
     remasking_strategy) = args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.path.insert(0, sdar_dir)

    import torch
    from transformers.cache_utils import DynamicCache
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from generate import sample_with_temperature_topk_topp, get_num_transfer_tokens

    @torch.no_grad()
    def block_diffusion_generate_batched(
        model, prompt, mask_id, gen_length, block_length, denoising_steps,
        temperature, top_k, top_p, remasking_strategy, confidence_threshold,
        stopping_criteria_idx,
    ):
        """Batched copy of SDAR's block_diffusion_generate (upstream is BS=1)."""
        model.eval()
        input_ids = prompt["input_ids"]
        batch_size, prompt_length = input_ids.shape
        past_key_values = DynamicCache()

        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length

        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
        block_diffusion_attention_mask = (
            block_mask.repeat_interleave(block_length, dim=0)
                     .repeat_interleave(block_length, dim=1).unsqueeze(0)
        )
        position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

        x = torch.full((batch_size, total_length), mask_id,
                       dtype=torch.long, device=model.device)
        x[:, :prompt_length] = input_ids
        prefill_blocks = prompt_length // block_length
        prefill_length = prefill_blocks * block_length

        if prefill_length > 0:
            cur_attn_mask = block_diffusion_attention_mask[:, :prefill_length, :prefill_length]
            model(x[:, :prefill_length],
                  attention_mask=cur_attn_mask,
                  position_ids=position_ids[:, :prefill_length],
                  past_key_values=past_key_values, use_cache=True, store_kv=True)

        num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)

        for num_block in range(prefill_blocks, num_blocks):
            cur_x = x[:, num_block*block_length:(num_block+1)*block_length].clone()
            cur_attn_mask = block_diffusion_attention_mask[
                :, num_block*block_length:(num_block+1)*block_length,
                :(num_block+1)*block_length]
            cur_position_ids = position_ids[:, num_block*block_length:(num_block+1)*block_length]
            for step in range(denoising_steps + 1):
                mask_index = (cur_x == mask_id)
                if mask_index.sum() == 0:
                    model(cur_x, attention_mask=cur_attn_mask,
                          position_ids=cur_position_ids,
                          past_key_values=past_key_values, use_cache=True, store_kv=True)
                    break
                logits = model(cur_x, attention_mask=cur_attn_mask,
                               position_ids=cur_position_ids,
                               past_key_values=past_key_values,
                               use_cache=True, store_kv=False).logits
                x0, x0_p = sample_with_temperature_topk_topp(
                    logits, temperature=temperature, top_k=top_k, top_p=top_p)
                confidence = torch.where(mask_index, x0_p, -torch.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    high = confidence[j] > confidence_threshold
                    if high.sum() >= num_transfer_tokens[step]:
                        transfer_index[j] = high
                    else:
                        _, idx = torch.topk(confidence[j], num_transfer_tokens[step])
                        transfer_index[j, idx] = True
                cur_x[transfer_index] = x0[transfer_index]
            x[:, num_block*block_length:(num_block+1)*block_length] = cur_x

            if stopping_criteria_idx is not None:
                gen_so_far = x[:, prompt_length:]
                hit = torch.zeros(batch_size, dtype=torch.bool, device=model.device)
                for sid in stopping_criteria_idx:
                    hit |= (gen_so_far == sid).any(dim=1)
                if hit.all():
                    break
        return x

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    gen_cfg = GenerationConfig.from_pretrained(model_path)
    stop_ids = gen_cfg.eos_token_id
    if isinstance(stop_ids, int):
        stop_ids = [stop_ids]

    tokenized = []
    for idx, sample in zip(indices, samples):
        msgs = sample["messages"]
        gt = msgs[-1]["content"]
        prompt_text = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(
            prompt_text, return_tensors="pt",
            truncation=True, max_length=prompt_truncate,
            add_special_tokens=False,
        )["input_ids"][0]
        tokenized.append({
            "index": idx, "ids": ids, "len": ids.numel(),
            "gt": gt, "config": sample["dataset_config"],
        })

    tokenized.sort(key=lambda x: x["len"])
    batches = [tokenized[i:i + per_batch] for i in range(0, len(tokenized), per_batch)]

    results = []
    t0 = time.time()
    n_done = 0
    n_total = len(tokenized)

    for b_i, batch in enumerate(batches):
        max_len = max(t["len"] for t in batch)
        padded_len = math.ceil(max_len / block_length) * block_length
        bs = len(batch)
        input_ids = torch.full((bs, padded_len), pad_id, dtype=torch.long, device="cuda:0")
        for i, t in enumerate(batch):
            input_ids[i, padded_len - t["len"]:] = t["ids"].to("cuda:0")

        try:
            with torch.no_grad():
                out = block_diffusion_generate_batched(
                    model, {"input_ids": input_ids}, mask_id=mask_id,
                    gen_length=gen_length, block_length=block_length,
                    denoising_steps=denoising_steps,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    remasking_strategy=remasking_strategy,
                    confidence_threshold=confidence_threshold,
                    stopping_criteria_idx=stop_ids,
                )
            for i, t in enumerate(batch):
                gen_ids = out[i, padded_len:].tolist()
                trimmed = []
                for tid in gen_ids:
                    if tid == mask_id or tid in stop_ids:
                        break
                    trimmed.append(tid)
                prediction = tokenizer.decode(trimmed, skip_special_tokens=False)
                results.append({
                    "index": t["index"], "config": t["config"],
                    "prediction_raw": prediction, "reference": t["gt"],
                    "generated": prediction, "ground_truth": t["gt"],
                })
        except Exception as e:
            for t in batch:
                msg = f"ERROR: {type(e).__name__}: {e!r}"
                results.append({
                    "index": t["index"], "config": t["config"],
                    "prediction_raw": msg, "reference": t["gt"],
                    "generated": msg, "ground_truth": t["gt"],
                })

        n_done += bs
        if (b_i + 1) % 5 == 0 or b_i + 1 == len(batches):
            el = time.time() - t0
            rate = n_done / el if el > 0 else 0
            print(f"  [GPU {gpu_id}] {n_done}/{n_total} bs={bs} pad_len={padded_len} "
                  f"rate={rate:.2f}/s elapsed={el:.0f}s", flush=True)

    return results


def run_dataset(
    name, samples, output_dir, *,
    sdar_dir, model_path, num_gpus,
    prompt_truncate, gen_length, block_length, denoising_steps, per_batch,
    temperature, top_p, top_k, confidence_threshold, remasking_strategy,
):
    indices = list(range(len(samples)))
    shards = [[] for _ in range(num_gpus)]
    shard_indices = [[] for _ in range(num_gpus)]
    for i in indices:
        g = i % num_gpus
        shards[g].append(samples[i])
        shard_indices[g].append(i)

    args_list = [
        (gpu, sdar_dir, model_path, shard_indices[gpu], shards[gpu],
         prompt_truncate, gen_length, block_length, denoising_steps,
         per_batch, temperature, top_p, top_k, confidence_threshold,
         remasking_strategy)
        for gpu in range(num_gpus)
    ]
    all_results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=num_gpus) as pool:
        for shard_results in pool.map(process_shard, args_list):
            all_results.extend(shard_results)
    el = time.time() - t0
    print(f"[{name}] all shards done elapsed={el:.1f}s "
          f"rate={len(all_results)/el:.2f}/s")

    all_results.sort(key=lambda r: r["index"])
    out_path = os.path.join(output_dir, f"{name}_outputs.jsonl")
    with open(out_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[{name}] saved {len(all_results)} rows -> {out_path}")


def load_eval(path: str, name: str) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "messages" not in row:
                continue
            samples.append({"messages": row["messages"], "dataset_config": name})
    return samples


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sdar-dir", required=True,
                   help="Path to your SDAR checkout (provides generate.py).")
    p.add_argument("--model-path", required=True,
                   help="Path to a trained / merged SDAR checkpoint dir.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--datasets", nargs="+", required=True,
                   help="Dataset stems; loads <data-dir>/<file-pattern>.")
    p.add_argument("--file-pattern", default="eval_{name}.jsonl",
                   help='Filename template (default: "eval_{name}.jsonl").')
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-gpus", type=int, default=8)
    p.add_argument("--prompt-truncate", type=int, default=8000)
    p.add_argument("--gen-length", type=int, default=1024)
    p.add_argument("--block-length", type=int, default=4)
    p.add_argument("--denoising-steps", type=int, default=4)
    p.add_argument("--per-batch", type=int, default=8,
                   help="Samples per forward pass per GPU.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--confidence-threshold", type=float, default=0.85)
    p.add_argument("--remasking-strategy", default="low_confidence_dynamic")
    args = p.parse_args()

    if not os.path.isdir(args.sdar_dir):
        print(f"ERROR: --sdar-dir {args.sdar_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(os.path.join(args.sdar_dir, "generate.py")):
        print(f"ERROR: {args.sdar_dir}/generate.py not found", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    for name in args.datasets:
        path = os.path.join(args.data_dir, args.file_pattern.format(name=name))
        if not os.path.exists(path):
            print(f"[skip] {path} not found", file=sys.stderr)
            continue
        samples = load_eval(path, name)
        print(f"\n{'='*60}\nEvaluating: {name} (n={len(samples)})\n{'='*60}")
        run_dataset(
            name, samples, args.output_dir,
            sdar_dir=args.sdar_dir, model_path=args.model_path,
            num_gpus=args.num_gpus,
            prompt_truncate=args.prompt_truncate, gen_length=args.gen_length,
            block_length=args.block_length, denoising_steps=args.denoising_steps,
            per_batch=args.per_batch,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            confidence_threshold=args.confidence_threshold,
            remasking_strategy=args.remasking_strategy,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
