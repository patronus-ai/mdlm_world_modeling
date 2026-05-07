"""
Convert chat-format JSONL into the message-array JSONL that WeDLM's SFT
trainer expects.

Input  (one row per line):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Output (one row per line):
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

Usage:
    python prepare_data.py --input data/raw.jsonl --output data/train.jsonl

If your raw rows are already JSON arrays (no "messages" wrapper), pass
--passthrough to copy them as-is after a sanity check.
"""
import argparse
import json
import sys


def normalize_row(line: str, passthrough: bool) -> str | None:
    line = line.strip()
    if not line:
        return None
    obj = json.loads(line)
    if passthrough:
        if not isinstance(obj, list):
            return None
        msgs = obj
    elif isinstance(obj, dict) and "messages" in obj:
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
    p.add_argument("--passthrough", action="store_true",
                   help="Treat each input row as the raw message array.")
    args = p.parse_args()

    n_in = n_out = n_skip = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            n_in += 1
            try:
                norm = normalize_row(line, args.passthrough)
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
