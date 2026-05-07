"""
Convert chat-format JSONL into the message-array JSONL that LlamaFactory's
SFT trainer expects when registered as a sharegpt-style dataset.

Input  (one row per line; either form):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

Output (one row per line):
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

After running this, register the file in
<SDAR>/training/llama_factory_sdar/data/dataset_info.json so that LlamaFactory
can find it via the `dataset:` key in your YAML config.

Usage:
    python prepare_data.py --input data/raw.jsonl --output data/train.jsonl
"""
import argparse
import json
import sys


def normalize_row(line: str) -> str | None:
    line = line.strip()
    if not line:
        return None
    obj = json.loads(line)
    if isinstance(obj, dict) and "messages" in obj:
        msgs = obj["messages"]
    elif isinstance(obj, list):
        msgs = obj
    else:
        return None
    if not msgs or not all(
        isinstance(m, dict) and "role" in m and "content" in m for m in msgs
    ):
        return None
    out = [{"role": m["role"], "content": m["content"]} for m in msgs]
    return json.dumps(out, ensure_ascii=False)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    n_in = n_out = n_skip = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            n_in += 1
            try:
                norm = normalize_row(line)
            except json.JSONDecodeError:
                norm = None
            if norm is None:
                n_skip += 1
                continue
            fout.write(norm + "\n")
            n_out += 1

    print(f"in : {n_in}", file=sys.stderr)
    print(f"out: {n_out}  ->  {args.output}", file=sys.stderr)
    print(f"skipped (malformed): {n_skip}", file=sys.stderr)


if __name__ == "__main__":
    main()
