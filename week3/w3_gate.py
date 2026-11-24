"""Week-3 gate: before comparing masks, prove the readout works.

Three levels, same prompt builder and same 10-way digit readout:

  trivial  one-line arithmetic that a competent model must get right
  leak     a real program with "# result = <gold>" appended; checks the readout
           itself (if the model cannot copy a visible digit, nothing else counts)
  real     the same programs with full attention

If `leak` or `trivial` is not near ceiling, the prompt or the scoring surface is
broken and the Full/Window comparison would be uninterpretable.

    python w3_gate.py --model Qwen/Qwen2.5-Coder-7B-Instruct --style chat
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w3_core as core  # noqa: E402
import w3_run  # noqa: E402


def build_levels(rows: list[dict], n: int, style: str, seed: int = 0) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    base = [r for r in rows if r["byte_length"] == 256][:n]
    trivial = []
    for i, r in enumerate(base):
        a = rng.randint(0, 4)
        b = rng.randint(0, 9 - a)
        trivial.append({"id": f"trivial{i}", "code": f"x = {a}\ny = {b}\nresult = x + y\n",
                        "answer": a + b, "gold_static_spans": [[len(f'x = {a}\ny = {b}\n'),
                                                                len(f'x = {a}\ny = {b}\nresult = x + y')]]})
    leak = []
    for r in base:
        body = r["code"].rstrip("\n")
        leak.append({"id": "leak_" + r["id"],
                     "code": body + f"\n# result = {r['answer']}\n",
                     "answer": int(r["answer"]), "gold_static_spans": []})
    return {"trivial": trivial, "leak": leak, "real": base}


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--style", default="chat", choices=["chat", "completion"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", type=Path, default=core.W3 / "results" / "gate.json")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[a.dtype]
    tok = core.load_tokenizer()
    dids = core.digit_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, attn_implementation="sdpa").to(device).eval()

    rows = core.load_split("test")
    levels = build_levels(rows, a.limit, a.style)
    report = {"model": a.model, "style": a.style, "levels": {}}
    for name, items in levels.items():
        preds, golds, probs = [], [], []
        for it in items:
            prompt = core.build_prompt(tok, it["code"], a.style)
            ids, _ = core.encode_prompt(tok, prompt)
            mask = core.mask_full(len(ids), device)
            p, _, _ = w3_run.digit_scores(model, ids, mask, device, dids)
            preds.append(int(p.argmax()))
            golds.append(int(it["answer"]))
            probs.append(float(p[it["answer"]]))
        k = sum(int(p == g) for p, g in zip(preds, golds))
        lo, hi = core.wilson(k, len(items))
        report["levels"][name] = {"n": len(items), "correct": k, "accuracy": k / len(items),
                                  "wilson": [round(lo, 4), round(hi, 4)],
                                  "mean_p_gold": sum(probs) / len(probs),
                                  "pred_hist": {str(d): preds.count(d) for d in range(10)},
                                  "pairs": list(zip(golds, preds))}
        print(f"{name:8s} n={len(items):3d} acc={k / len(items):.2f} "
              f"wilson=[{lo:.2f},{hi:.2f}] mean_p_gold={report['levels'][name]['mean_p_gold']:.3f}")

    g = report["levels"]
    ok = g["trivial"]["accuracy"] >= 0.8 and g["leak"]["accuracy"] >= 0.8
    report["gate_pass"] = bool(ok)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\ngate_pass={ok}  -> {a.out}")
    if not ok:
        print("STOP: fix the prompt style / readout before running any mask comparison.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
