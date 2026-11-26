"""Can the answer-bearing number still reach the readout?

For each program and each mask, compute the minimum number of attention layers
needed for the *value token* of the definition (the last token of the first gold
span) to influence the readout position. 36 layers are available.

    python w3_reach.py --data data/needle_h1.jsonl --target 4000 \
        --modes full,window=64,random=64,gold_union=64 --run results/3b/vis_h1_final

"unreachable" means the mask itself cuts the path; "reachable but wrong" means
the information can arrive and the model still fails — the two are different
claims and must not be merged.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import zlib
from pathlib import Path

import w3_core as core
import w3_run as run


def value_token(offsets, prompt_code_start: int, spans) -> int:
    """Last token overlapping the first gold span: the literal that carries the value."""
    start, end = spans[0]
    s, e = prompt_code_start + int(start), prompt_code_start + int(end)
    hits = [i for i, (a, b) in enumerate(offsets) if b > a and a < e and b > s]
    if not hits:
        raise ValueError("cannot locate the value token")
    return hits[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/needle_h1.jsonl")
    ap.add_argument("--target", type=int, default=4000)
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--modes", default="full,window=64,random=64,gold_union=64")
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--run", default="", help="optional results dir with per_item.csv for accuracy")
    ap.add_argument("--group-filter", default="", help="substring of per_item group, e.g. d4000")
    a = ap.parse_args()

    tok = core.load_tokenizer()
    rows = [json.loads(line) for line in Path(a.data).open(encoding="utf-8")]
    rows = [r for r in rows if r.get("target_tokens") == a.target and r.get("hops") == a.hops]
    if not rows:
        print(f"no programs with hops={a.hops}, target={a.target} in {a.data}")
        return 1

    acc = collections.defaultdict(list)
    if a.run:
        with (Path(a.run) / "per_item.csv").open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if a.group_filter and a.group_filter not in row["group"]:
                    continue
                acc[row["mode"]].append(int(row["correct"]))

    print(f"{len(rows)} programs, hops={a.hops}, target={a.target}, {a.layers} layers")
    print(f"{'mode':16s} {'unreachable':>12s} {'min-hops (n)':>26s} {'accuracy':>10s}")
    for label in a.modes.split(","):
        recipe = run.parse_mode(label)
        hops = collections.Counter()
        for r in rows:
            prompt = core.build_prompt(tok, r["code"], "completion")
            ids, offsets = core.encode_prompt(tok, prompt)
            gold = core.span_token_positions(offsets, prompt.code_start, r["gold_static_spans"])
            val = value_token(offsets, prompt.code_start, r["gold_static_spans"])
            key = f"{r['seed']}|{recipe[1].get('w', 0)}"
            mask = run.build_mask(recipe, len(ids), [], sorted({t for g in gold for t in g}),
                                  None, seed=zlib.crc32(key.encode()))
            reached = core.reach_hops(mask, len(ids) - 1, a.layers)
            hops[reached.get(val, 0)] += 1
        unreachable = hops.pop(0, 0)
        spread = ", ".join(f"{k}:{v}" for k, v in sorted(hops.items())) or "-"
        acc_text = (f"{sum(acc[label])}/{len(acc[label])}" if label in acc else "-")
        print(f"{label:16s} {unreachable:5d}/{len(rows):<6d} {spread:>26s} {acc_text:>10s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
