"""
Download chat-format data from a HuggingFace dataset and write JSONL files
ready for ms-swift SFT training.

Output format (one row per line):
    {"messages": [{"role": "system|user|assistant|tool", "content": "..."}]}

Usage:
    HF_TOKEN=$HF_TOKEN python prepare_data.py \
        --repo my-org/my-dataset \
        --train-configs cfg_a cfg_b \
        --test-configs cfg_test \
        --out-dir ./data

If your dataset has only a single config, pass --train-configs default
(or whatever the config name is). Set --no-token if the dataset is public.
"""
import argparse
import json
import os
import sys

from datasets import load_dataset


def extract_messages(row: dict) -> dict:
    return {
        "messages": [
            {"role": m["role"], "content": m["content"]}
            for m in row["messages"]
        ]
    }


def write_split(
    repo: str, configs: list[str], hf_split: str,
    out_path: str, token: str | None,
) -> int:
    total = 0
    with open(out_path, "w") as f:
        for cfg in configs:
            print(f"  Loading {repo} / {cfg} ({hf_split})...")
            try:
                ds = load_dataset(repo, cfg, split=hf_split, token=token)
            except Exception as e:
                print(f"    -> SKIPPED: {e}", file=sys.stderr)
                continue
            for row in ds:
                if "messages" not in row:
                    continue
                f.write(json.dumps(extract_messages(row), ensure_ascii=False) + "\n")
            print(f"    -> {len(ds)} samples")
            total += len(ds)
    print(f"  {os.path.basename(out_path)}: {total} total samples\n")
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo", required=True, help="HuggingFace dataset repo id.")
    p.add_argument("--train-configs", nargs="+", required=True,
                   help="Config name(s) to load for the train split.")
    p.add_argument("--test-configs", nargs="*", default=[],
                   help="Config name(s) to load for the test split (optional).")
    p.add_argument("--train-split", default="train")
    p.add_argument("--test-split", default="test")
    p.add_argument("--out-dir", default="./data")
    p.add_argument("--no-token", action="store_true",
                   help="Don't pass HF_TOKEN (use for public datasets).")
    args = p.parse_args()

    token = None if args.no_token else os.environ.get("HF_TOKEN")
    if not args.no_token and not token:
        print("Set HF_TOKEN or pass --no-token.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    print("=== Train ===")
    write_split(args.repo, args.train_configs, args.train_split,
                os.path.join(args.out_dir, "train.jsonl"), token)

    if args.test_configs:
        print("=== Test ===")
        write_split(args.repo, args.test_configs, args.test_split,
                    os.path.join(args.out_dir, "test.jsonl"), token)

    print("Done.")


if __name__ == "__main__":
    main()
