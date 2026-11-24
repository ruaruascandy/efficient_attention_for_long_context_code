"""Week-3 preflight: how long is the data in Qwen tokens, and does a mask even cut
the evidence off? Run this before spending GPU time on step 3.

    python w3_preflight.py --layers 28 --make-length 4096 --sample 10

Writes week3/preflight/preflight.md and two CSVs. No model weights needed.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w3_core as core  # noqa: E402
import w3_run  # noqa: E402


def make_length(rows: list[dict], length: int, out_dir: Path) -> list[dict]:
    """Paired longer variants of the same programs.

    The week-1 generator's name pool supports at most ~1.2 KB, so this inserts
    fresh irrelevant assignments and re-verifies the label instead.
    """
    made = []
    for r in rows:
        grown = core.lengthen_program(r["code"], r["gold_static_spans"], length,
                                      int(r["answer"]), seed=r["seed"])
        row = dict(r)
        row.update(grown)
        row["id"] = f"{r['id']}_preview{length}"
        made.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"preview_l{length}.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in made), encoding="utf-8")
    return made


def token_stats(tok, rows, prompt_style="completion"):
    out = []
    for r in rows:
        p = core.build_prompt(tok, r["code"], prompt_style)
        ids, offsets = core.encode_prompt(tok, p)
        spans = core.span_token_positions(offsets, p.code_start, r["gold_static_spans"])
        gold = sorted({t for s in spans for t in s})
        out.append({
            "n_tokens": len(ids),
            "n_code_tokens": sum(1 for a, b in offsets if b > a and a >= p.code_start and b <= p.code_end),
            "gold_n": len(gold),
            "gold_frac": len(gold) / len(ids),
            "far_gap": (len(ids) - 1) - min(gold) if gold else 0,
            "line_starts": core.char_to_token_starts(core.line_starts(p.text), offsets, 0),
            "gold_tokens": gold,
            "_prompt": p,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--make-length", type=int, default=0, help="generate paired preview variants at this byte length")
    ap.add_argument("--sample", type=int, default=10, help="items per length for reachability")
    ap.add_argument("--layers", type=int, nargs="+", default=[28, 36], help="Qwen2.5-Coder 7B / 3B depth")
    ap.add_argument("--windows", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--out", type=Path, default=core.W3 / "preflight")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    tok = core.load_tokenizer()
    rows = core.load_split(a.split)
    groups: dict[int, list[dict]] = {L: [r for r in rows if r["byte_length"] == L] for L in a.lengths}
    if a.make_length:
        base = groups[min(a.lengths)]
        groups[a.make_length] = make_length(base, a.make_length, a.out)

    stats = {L: token_stats(tok, rs) for L, rs in groups.items() if rs}

    lines = ["# Week-3 preflight", "",
             "Program length and evidence distance in Qwen tokens (Qwen2.5-Coder tokenizer).",
             "", "| source bytes | items | tokens mean | min | max | evidence tokens | far gap mean | far gap max |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for L, st in sorted(stats.items()):
        toks = [x["n_tokens"] for x in st]
        gaps = [x["far_gap"] for x in st]
        lines.append(f"| {L} | {len(st)} | {statistics.mean(toks):.1f} | {min(toks)} | {max(toks)} | "
                     f"{100 * statistics.mean(x['gold_frac'] for x in st):.1f}% | "
                     f"{statistics.mean(gaps):.1f} | {max(gaps)} |")

    lines += ["", "## Does a sliding window actually cut the evidence off?", "",
              "A window of W positions per layer has receptive field 1 + L*(W-1) after L layers,",
              "because information is relayed through depth. The table gives the fraction of items",
              "whose farthest evidence token is *unreachable* from the readout.", ""]
    header = "| source bytes | " + " | ".join(f"W={w}" for w in a.windows) + " |"
    for L in a.layers:
        lines += [f"Layers = {L}  (receptive field W=64 -> {core.receptive_field_window(64, L)} tokens)", "",
                  header, "|---:|" + "---:|" * len(a.windows)]
        for Lb, st in sorted(stats.items()):
            cells = []
            for w in a.windows:
                rf = core.receptive_field_window(w, L)
                cells.append(f"{100 * sum(1 for x in st if x['far_gap'] >= rf) / len(st):.0f}%")
            lines.append(f"| {Lb} | " + " | ".join(cells) + " |")
        lines.append("")

    # reachability + connection counts per mask recipe
    modes = ["full", "window=8", "window=16", "window=32", "window=64", "window=128",
             "local_global=32:first:4", "block=4", "gold_union=32"]
    lines += ["## Mask geometry (sampled items)", "",
              "`conn` = allowed (query, key) pairs; `visible` = gold evidence reachable within L layers.",
              ""]
    rec_rows = []
    for L, st in sorted(stats.items()):
        sample = st[: a.sample]
        for L_layer in a.layers:
            for spec in modes:
                recipe = w3_run.parse_mode(spec)
                conns, fracs, vis = [], [], []
                for x in sample:
                    n = x["n_tokens"]
                    mask = w3_run.build_mask(recipe, n, x["line_starts"], x["gold_tokens"], None)
                    conns.append(core.connection_count(mask))
                    fracs.append(core.connection_count(mask) / (n * (n + 1) / 2))
                    gv = core.gold_visibility(mask, x["gold_tokens"], n - 1, L_layer)
                    vis.append(gv["gold_visible_frac"])
                rec_rows.append({
                    "bytes": L, "layers": L_layer, "mode": w3_run.mode_label(recipe),
                    "conn": statistics.mean(conns), "conn_frac": statistics.mean(fracs),
                    "gold_visible_frac": statistics.mean(vis),
                })
                lines.append(f"| {L} | {L_layer} | {w3_run.mode_label(recipe)} | "
                             f"{statistics.mean(conns):,.0f} | {100 * statistics.mean(fracs):.1f}% | "
                             f"{100 * statistics.mean(vis):.0f}% |")

    lines = lines[: lines.index("## Mask geometry (sampled items)") + 3] + [
        "| source bytes | layers | mode | conn | conn frac | gold visible |",
        "|---:|---:|---|---:|---:|---:|",
    ] + [f"| {r['bytes']} | {r['layers']} | {r['mode']} | {r['conn']:,.0f} | "
         f"{100 * r['conn_frac']:.1f}% | {100 * r['gold_visible_frac']:.0f}% |" for r in rec_rows]

    (a.out / "preflight.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import csv as _csv

    with (a.out / "mask_geometry.csv").open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rec_rows[0].keys()))
        w.writeheader()
        w.writerows(rec_rows)
    print("\n".join(lines))
    print(f"\nwrote {a.out / 'preflight.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
