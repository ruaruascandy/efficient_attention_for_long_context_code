"""Paired comparisons between two mask modes on the same programs.

Conditions in this project are paired (the same program under different masks),
so a paired exact McNemar test is the right test — two independent Wilson
intervals can overlap while the paired difference is significant.

    python w3_paired.py --run results/3b/vis_h1_final --left window=64 --right random=64
    python w3_paired.py --run results/3b/vis_end_4k --left window=64 --right gold_union=64

Reads per_item.csv (columns: group, id, mode, correct) and prints, per group,
the discordant counts and the two-sided exact p-value.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(path: Path) -> dict[str, dict[str, int]]:
    """{(group, id): {mode: correct}}"""
    out: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[(row["group"], row["id"])][row["mode"]] = int(row["correct"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="directory containing per_item.csv")
    ap.add_argument("--left", required=True, help="mode label, e.g. window=64")
    ap.add_argument("--right", required=True, help="mode label, e.g. random=64")
    ap.add_argument("--merged", action="store_true", help="pool all groups instead of per group")
    a = ap.parse_args()

    table = load(Path(a.run) / "per_item.csv")
    acc = defaultdict(lambda: {"n": 0, "both": 0, "left": 0, "right": 0, "neither": 0})
    for (group, _sid), modes in table.items():
        if a.left not in modes or a.right not in modes:
            continue
        key = "all" if a.merged else group
        l, r = modes[a.left], modes[a.right]
        d = acc[key]
        d["n"] += 1
        d["both"] += int(l == 1 and r == 1)
        d["left"] += int(l == 1 and r == 0)
        d["right"] += int(l == 0 and r == 1)
        d["neither"] += int(l == 0 and r == 0)

    print(f"{a.left}  vs  {a.right}   ({a.run})")
    print(f"{'group':26s} {'n':>4s} {'L only':>7s} {'R only':>7s} {'p (exact)':>11s}")
    for group in sorted(acc):
        d = acc[group]
        p = mcnemar_exact(d["left"], d["right"])
        print(f"{group:26s} {d['n']:4d} {d['left']:7d} {d['right']:7d} {p:11.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
