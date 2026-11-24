"""Week-3 ceiling curve: how large a program can the frozen model still read?

Generates diagnostic programs at several byte lengths (more bytes = more
irrelevant assignments) with the week-1 generator, then measures full-attention
accuracy with the same prompt and readout as the main probe.

This is a *diagnostic*, not a benchmark: it exists to pick the length at which a
ceiling exists, because a mask cannot destroy accuracy that was never there.

    python w3_curve.py --model Qwen/Qwen2.5-Coder-7B-Instruct --lengths 96 160 256 512
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w3_core as core  # noqa: E402
import w3_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[96, 160, 256, 512])
    ap.add_argument("--per", type=int, default=30, help="items per family per length")
    ap.add_argument("--style", default="completion")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=777000)
    ap.add_argument("--out", type=Path, default=core.W3 / "results" / "curve.csv")
    a = ap.parse_args()

    sys.path.insert(0, str(core.PROJECT / "week1"))
    from dataset import make_sample  # noqa: E402
    from transformers import AutoModelForCausalLM  # noqa: E402

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[a.dtype]
    tok = core.load_tokenizer()
    dids = core.digit_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, attn_implementation="sdpa").to(device).eval()

    rows_out = []
    for length in a.lengths:
        for fi, family in enumerate(("chain", "call", "branch")):
            correct = 0
            n = 0
            for i in range(a.per):
                answer = i % 10
                template = 6 + (i % 2)
                seed = a.seed + length * 13 + fi * 1000 + i
                try:
                    s = make_sample(seed, family, template, answer, length, "diag")
                except ValueError:
                    continue
                prompt = core.build_prompt(tok, s["code"], a.style)
                ids, _ = core.encode_prompt(tok, prompt)
                mask = core.mask_full(len(ids), device)
                p, _, _ = w3_run.digit_scores(model, ids, mask, device, dids)
                correct += int(int(p.argmax()) == answer)
                n += 1
            acc = correct / n if n else float("nan")
            lo, hi = core.wilson(correct, n)
            rows_out.append({"bytes": length, "family": family, "n": n, "correct": correct,
                             "accuracy": round(acc, 4), "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4)})
            print(f"{length:5d} B  {family:7s} n={n:3d} acc={acc:.2f}  wilson=[{lo:.2f},{hi:.2f}]", flush=True)
        allc = [r for r in rows_out if r["bytes"] == length]
        k = sum(r["correct"] for r in allc)
        n = sum(r["n"] for r in allc)
        print(f"{length:5d} B  overall n={n:3d} acc={k / n:.2f}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
