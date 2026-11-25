"""Week-3 distance-controlled probe set.

The week-1 programs confound two things: how many dependency hops are needed, and
how far the evidence sits from the query. This generator separates them:

    v0000 = 7            <- distant definition
    d0001 = 3            )
    ...                  )  gap of a chosen number of Qwen tokens
    v0001 = v0000 + 2    <- second hop (optional)
    result = v0001       <- query

`hops` controls reasoning depth, `--distances` controls the gap in Qwen tokens.
Labels are 0..9 by construction, and the label is re-checked with the week-1
interpreter.

    python w3_needle.py --distances 20 100 400 1600 --hops 1 2 3 --per 20
    python w3_run.py --model ... --data-dir week3/data --split needle_test --lengths 0 \
        --modes full,window=64,gold_union=64
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w3_core as core  # noqa: E402


def build_needle(seed: int, hops: int, target_tokens: int, tok, prefix: str = "n") -> dict:
    rng = random.Random(seed)
    deltas = [rng.choice([-3, -2, -1, 1, 2, 3]) for _ in range(hops)]
    total = sum(deltas)
    lo, hi = max(0, -total), min(9, 9 - total)
    base = rng.randint(lo, hi)
    answer = base + total

    names = [f"{prefix}{seed % 100000:05d}{k}" for k in range(hops + 1)]
    head = [(f"{names[0]} = {base}\n", True)]
    body = []
    for h in range(hops):
        op = "+" if deltas[h] > 0 else "-"
        body.append((f"{names[h + 1]} = {names[h]} {op} {abs(deltas[h])}\n", True))
    final = f"result = {names[hops]}\n"

    def assemble(distractors: list[str]) -> str:
        return "".join(t for t, _ in head) + "".join(distractors) + "".join(t for t, _ in body) + final

    # calibrate the number of irrelevant lines against the real tokenizer
    # Measure one filler line instead of guessing: Qwen tokenises every digit
    # separately, so a line is ~13 tokens, not the ~7.5 that byte length suggests.
    def filler(k: int) -> str:
        return f"d{seed % 1000:03d}{k:05d} = {rng.randint(0, 9)}\n"

    base_tokens = len(tok.encode(assemble([]), add_special_tokens=False))
    per_line = max(1, len(tok.encode(assemble([filler(0)]), add_special_tokens=False)) - base_tokens)
    target_total = target_tokens + 40
    distractor_lines = [filler(k) for k in range(max(0, round((target_total - base_tokens) / per_line)))]
    for _ in range(4):  # converge in a couple of steps if the estimate is off
        n_tok = len(tok.encode(assemble(distractor_lines), add_special_tokens=False))
        delta = int(round((target_total - n_tok) / per_line))
        if abs(delta) < 1:
            break
        if delta > 0:
            start = len(distractor_lines)
            distractor_lines += [filler(start + i) for i in range(delta)]
        else:
            distractor_lines = distractor_lines[: max(0, len(distractor_lines) + delta)]
    code = assemble(distractor_lines)

    spans, cursor = [], 0
    for text, _ in head:
        spans.append([cursor, cursor + len(text.rstrip("\n").encode())])
        cursor += len(text.encode())
    cursor += sum(len(d.encode()) for d in distractor_lines)
    for text, _ in body:
        spans.append([cursor, cursor + len(text.rstrip("\n").encode())])
        cursor += len(text.encode())
    spans.append([cursor, cursor + len(final.rstrip("\n").encode())])
    cursor += len(final.encode())

    if len(code.encode()) != cursor:
        raise AssertionError(f"length mismatch {len(code.encode())} != {cursor}")
    sys.path.insert(0, str(core.PROJECT / "week1"))
    from dataset import interpret  # noqa: E402

    if interpret(code) != answer:
        raise AssertionError("label mismatch")
    return {
        "id": f"needle_h{hops}_d{target_tokens}_s{seed}",
        "family_id": f"needle_h{hops}_d{target_tokens}",
        "skeleton_id": f"needle_h{hops}", "family": "needle", "split": "needle",
        "seed": seed, "code": code, "target": "result", "answer": answer,
        "byte_length": len(code.encode()), "hops": hops, "target_tokens": target_tokens,
        "gold_static_spans": spans,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distances", type=int, nargs="+", default=[20, 100, 400, 1600])
    ap.add_argument("--hops", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--per", type=int, default=20)
    ap.add_argument("--split", default="needle_test")
    ap.add_argument("--out", type=Path, default=core.W3 / "data")
    a = ap.parse_args()

    tok = core.load_tokenizer()
    rows = []
    for hops in a.hops:
        for dist in a.distances:
            for i in range(a.per):
                rows.append(build_needle(1000 * hops + dist + i, hops, dist, tok))
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"{a.split}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    # convenience slices so w3_run can target one depth without filtering
    for hops in a.hops:
        sel = [r for r in rows if r["hops"] == hops]
        (a.out / f"needle_h{hops}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in sel), encoding="utf-8")

    for hops in a.hops:
        for dist in a.distances:
            sel = [r for r in rows if r["hops"] == hops and r["target_tokens"] == dist]
            toks = [len(tok.encode(r["code"], add_special_tokens=False)) for r in sel]
            print(f"hops={hops} target={dist:5d} n={len(sel):3d} tokens={min(toks)}-{max(toks)} "
                  f"bytes={min(r['byte_length'] for r in sel)}-{max(r['byte_length'] for r in sel)}")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
