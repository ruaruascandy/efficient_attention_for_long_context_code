"""Week-3 model probe: frozen Qwen, fixed prompt, 10-way digit readout,
and a family of attention-visibility masks applied to every layer.

    python w3_run.py --model Qwen/Qwen2.5-Coder-7B-Instruct --limit 8 --modes full,window=64
    python w3_run.py --model Qwen/Qwen2.5-Coder-7B-Instruct --split test --modes full,window=64

Masks are boolean 4D tensors (True = may attend), which this transformers version
passes straight to SDPA. A mask changes *visibility*, never the token sequence,
so every mode sees the same input ids.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w3_core as core  # noqa: E402


# ------------------------------------------------------------------ mask recipes
def parse_mode(spec: str):
    """full | window=W | local_global=W:first:K | local_global=W:linestart:S | block=L | gold_union=W"""
    if spec == "full":
        return ("full", {})
    name, _, rest = spec.partition("=")
    if name == "window":
        return ("window", {"w": int(rest)})
    if name == "block":
        return ("block", {"lines": int(rest)})
    if name == "gold_union":
        return ("gold_union", {"w": int(rest)})
    if name == "random":
        # same per-row budget as window=W, but the W keys are drawn at random:
        # separates "how many tokens are visible" from "which ones".
        return ("random", {"w": int(rest)})
    if name == "local_global":
        w, kind, val = rest.split(":")
        return ("local_global", {"w": int(w), "kind": kind, "val": int(val)})
    raise ValueError(f"unknown mode: {spec}")


def build_mask(recipe, n: int, tok_line_starts: list[int], gold_tokens: list[int], device,
               seed: int = 0):
    name, p = recipe
    if name == "full":
        return core.mask_full(n, device)
    if name == "window":
        return core.mask_window(n, p["w"], device)
    if name == "block":
        return core.mask_line_block(n, tok_line_starts, p["lines"], device=device)
    if name == "gold_union":
        # window plus every evidence token visible (dependency-oracle upper bound).
        return core.mask_local_global(n, p["w"], gold_tokens, device)
    if name == "random":
        return core.mask_random(n, p["w"], seed=seed, device=device)
    if name == "local_global":
        if p["kind"] == "first":
            globals_idx = list(range(min(p["val"], n)))
        elif p["kind"] == "linestart":
            globals_idx = tok_line_starts[:: max(1, p["val"])]
        else:
            raise ValueError(p["kind"])
        return core.mask_local_global(n, p["w"], globals_idx, device)
    raise ValueError(name)


def mode_label(recipe) -> str:
    name, p = recipe
    if name in ("full", "window", "block", "gold_union", "random"):
        return name if not p else f"{name}={p.get('w', p.get('lines'))}"
    return f"local_global={p['w']}:{p['kind']}:{p['val']}"


# ------------------------------------------------------------------------ scoring
@torch.no_grad()
def digit_scores(model, ids: list[int], mask, device, digit_ids: list[int]):
    """Score the ten digit tokens at the readout position.

    Only the last hidden state goes through the LM head: at 8k tokens the full
    logits tensor ([1, 8192, 151936]) would cost ~2.5 GB for no reason.
    """
    n = len(ids)
    input_ids = torch.tensor([ids], device=device)
    kwargs = {"use_cache": False}
    if mask is not None:
        m = mask[None, None, :, :] if mask.dim() == 2 else mask
        kwargs["attention_mask"] = m
    base = getattr(model, "model", None)
    if base is not None and hasattr(model, "lm_head"):
        out = base(input_ids=input_ids, **kwargs)
        logits = model.lm_head(out.last_hidden_state[:, -1])[0]
    else:
        logits = model(input_ids=input_ids, **kwargs).logits[0, -1]
    d = torch.tensor(digit_ids, device=logits.device)
    sub = logits[d].float()
    probs = torch.softmax(sub, dim=-1)
    top2 = torch.topk(probs, 2).values
    ent = float(-(probs * (probs + 1e-12).log()).sum())
    return probs.cpu(), float(top2[0] - top2[1]), ent


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--random-model", action="store_true", help="tiny random Qwen2, no weights needed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-dir", type=Path, default=core.DATA_DIR)
    ap.add_argument("--lengths", type=int, nargs="+", default=[256])
    ap.add_argument("--style", default="completion", choices=["completion", "chat"])
    ap.add_argument("--modes", default="full")
    ap.add_argument("--ablate", default="none", choices=["none", "gold-value", "gold-drop"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--no-reach", action="store_true", help="skip reachability analysis")
    ap.add_argument("--group-by", default="byte_length", help="summary grouping key, e.g. byte_length or family_id")
    a = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[a.dtype]

    tok = core.load_tokenizer()
    dids = core.digit_ids(tok)

    rows = [r for r in core.load_split(a.split, a.data_dir) if 0 in a.lengths or r["byte_length"] in a.lengths]
    if a.limit:
        rows = rows[: a.limit]

    if a.random_model:
        from transformers import Qwen2Config, Qwen2ForCausalLM

        cfg = Qwen2Config(
            vocab_size=len(tok), hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=32768, tie_word_embeddings=False,
        )
        model = Qwen2ForCausalLM(cfg).to(device)
        layers = int(cfg.num_hidden_layers)
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            a.model, dtype=dtype, attn_implementation=a.attn
        ).to(device)
        layers = int(model.config.num_hidden_layers)
    model.eval()

    recipes = [parse_mode(s) for s in a.modes.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)
    per_item = a.out / "per_item.csv"
    fields = ["id", "split", "byte_length", "family", "group", "style", "mode", "ablate",
              "n_tokens", "n_code_tokens", "connections", "conn_frac", "layers",
              "gold_n", "gold_gap", "gold_visible_frac", "gold_direct_frac", "gold_min_hops", "gold_max_hops",
              "pred", "gold", "correct", "p_gold", "margin", "entropy", "sec"]
    t0 = time.time()

    with per_item.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        skipped = 0
        for k, row in enumerate(rows):
            code = row["code"]
            gold_answer = int(row["answer"])
            spans = row["gold_static_spans"]
            if a.ablate == "gold-value":
                mut = core.mutate_gold_value(code, spans)
                if mut is None or not (0 <= mut[1] <= 9):
                    skipped += 1
                    continue
                code, gold_answer = mut
            elif a.ablate == "gold-drop":
                code = core.drop_spans(code, spans)

            prompt = core.build_prompt(tok, code, a.style)
            ids, offsets = core.encode_prompt(tok, prompt)
            n = len(ids)
            gold_spans = core.span_token_positions(offsets, prompt.code_start, spans)
            gold_tokens = sorted({t for ts in gold_spans for t in ts})
            if not gold_tokens and a.ablate != "gold-drop":
                continue
            line_chars = core.line_starts(prompt.text)
            tok_lines = core.char_to_token_starts(line_chars, offsets, 0)

            for recipe in recipes:
                # Seed from the program's own seed and the mask width, never from
                # the id string: ids carry the position suffix (start/end), so an
                # id-based seed silently changes the random mask when a variant is
                # added, which would make earlier random=W numbers irreproducible.
                rand_key = f"{row.get('seed', row['id'])}|{recipe[1].get('w', 0)}"
                mask = build_mask(recipe, n, tok_lines, gold_tokens, device,
                                  seed=zlib.crc32(rand_key.encode()))
                t1 = time.time()
                probs, margin, ent = digit_scores(model, ids, mask, device, dids)
                sec = time.time() - t1
                pred = int(probs.argmax())
                vis = {"gold_n": len(gold_tokens), "gold_visible_frac": float("nan"), "gold_direct_frac": float("nan"),
                       "gold_min_hops": "", "gold_max_hops": ""}
                if a.ablate == "gold-drop":
                    pass  # span offsets no longer refer to the evidence after deletion
                elif not a.no_reach:
                    gv = core.gold_visibility(mask, gold_tokens, n - 1, layers)
                    vis = {"gold_n": gv["gold_n"], "gold_visible_frac": gv["gold_visible_frac"], "gold_direct_frac": gv["gold_direct_frac"],
                           "gold_min_hops": gv["gold_min_hops"], "gold_max_hops": gv["gold_max_hops"]}
                writer.writerow({
                    "id": row["id"], "split": a.split, "byte_length": row["byte_length"],
                    "family": row["family"], "group": str(row.get(a.group_by, row["byte_length"])), "style": a.style, "mode": mode_label(recipe),
                    "ablate": a.ablate, "n_tokens": n,
                    "n_code_tokens": sum(1 for s, e in offsets if e > s and s >= prompt.code_start and e <= prompt.code_end),
                    "connections": core.connection_count(mask),
                    "conn_frac": core.connection_count(mask) / (n * (n + 1) / 2),
                    "layers": layers, "gold_gap": (n - 1 - min(gold_tokens)) if gold_tokens else "", **vis,
                    "pred": pred, "gold": gold_answer, "correct": int(pred == gold_answer),
                    "p_gold": float(probs[gold_answer]), "margin": margin, "entropy": ent, "sec": sec,
                })
            if (k + 1) % 20 == 0:
                print(f"  {k + 1}/{len(rows)} items, {time.time() - t0:.0f}s", flush=True)

    # summary
    import collections

    agg = collections.defaultdict(lambda: {"n": 0, "correct": 0, "conn": 0.0, "vis": 0.0, "direct": 0.0, "gap": 0.0, "tok": 0.0})
    with per_item.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (r["group"], r["mode"], r["ablate"])
            g = agg[key]
            g["n"] += 1
            g["correct"] += int(r["correct"])
            g["conn"] += float(r["connections"])
            if r["gold_visible_frac"]:
                g["vis"] += float(r["gold_visible_frac"])
            if r["gold_gap"]:
                g["gap"] += float(r["gold_gap"])
            if r["gold_direct_frac"]:
                g["direct"] += float(r["gold_direct_frac"])
            g["tok"] += float(r["n_tokens"])
    summary = a.out / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "mode", "ablate", "n", "accuracy", "wilson_lo", "wilson_hi",
                    "mean_connections", "mean_gold_visible_frac", "mean_gold_direct_frac", "mean_gold_gap", "mean_tokens"])
        for (group, mode, ablate), g in sorted(agg.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
            lo, hi = core.wilson(g["correct"], g["n"])
            w.writerow([group, mode, ablate, g["n"], f"{g['correct'] / g['n']:.4f}",
                        f"{lo:.4f}", f"{hi:.4f}", f"{g['conn'] / g['n']:.1f}",
                        f"{g['vis'] / g['n']:.3f}", f"{g['direct'] / g['n']:.3f}", f"{g['gap'] / g['n']:.1f}", f"{g['tok'] / g['n']:.1f}"])
    meta = {
        "model": a.model, "random_model": a.random_model, "split": a.split,
        "lengths": a.lengths, "style": a.style, "modes": [mode_label(r) for r in recipes],
        "ablate": a.ablate, "layers": layers, "dtype": a.dtype, "attn": a.attn,
        "device": device, "torch": torch.__version__, "items": len(rows),
        "skipped": skipped, "seconds": round(time.time() - t0, 1),
    }
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


