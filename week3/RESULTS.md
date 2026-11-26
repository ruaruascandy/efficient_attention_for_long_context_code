# Week-3 run outputs

Index of `results/` — which run backs which table in
[Week3_Experiment_Report_EN.md](Week3_Experiment_Report_EN.md) /
[Week3_Experiment_Report_ZH.md](Week3_Experiment_Report_ZH.md).

All runs used Qwen2.5-Coder-3B-Instruct, frozen, bfloat16, RTX 4000 Ada (19.5 GB).
`summary.csv` is grouped; `per_item.csv` is one row per (program, mode);
`meta.json` records model, precision, device, item count, skipped items and wall-clock.

| Directory | Table | What it is | n / cell |
|---|---|---|---|
| `3b/gate.json` | Table 1 | trivial / leak / real gate | 40 each |
| `3b/curve.csv` | Table 2 | three-hop ceiling curve over source size | 20 per family |
| `3b/needle_full/` | Table 3 | depth × distance under full attention | 20 |
| `3b/vis_h1_final/` | **Table 4** | six visibility patterns × five distances — main table | 50 |
| `3b/vis_h1_long/` | — | same experiment, earlier generator (distances 216–6,976) and `--no-reach` off; replication only | 50 |
| `3b/vis_end_4k/` | Table 6 | same 4,051-token length, definition next to the query (gap ≈46) | 50 |
| `3b/vis_goldwin/` | Table 7 | window sweep with gold tokens global (64/128/256/512) | 50 |
| `3b/vis_localglobal/` | Table 8 | deployable fixed anchors (first 4 tokens, every line start, every 4th line start) vs the oracle | 50 |
| `3b/vis_h1/` | — | preliminary pass of the same comparison, n=20, superseded by `vis_h1_final` | 20 |
| `3b/cf/`, `3b/drop/` | §3.7 | counterfactual and deletion on the three-hop set — **uninterpretable**, the task itself is at 15% | 180 |
| `3b/vis_4k/` | §3.7 | 4 KB visibility on three-hop data — **uninterpretable**, full attention is already at chance | 60 |

## How to read them

- `mean_connections` is a logical count of allowed (query, key) pairs. It is not FLOPs and not a speed-up.
- `mean_gold_gap` / `gold_gap` is the token distance from the readout to the earliest evidence token.
- Empty cells (`nan`) in `gold_visible_frac` / `gold_direct_frac` mean `--no-reach` was passed; visibility is reported per length in [preflight/mask_geometry.csv](preflight/mask_geometry.csv) instead.
- `skipped` in `meta.json` counts programs whose counterfactual label fell outside 0–9 and was therefore dropped.

## Note on git

These files are **tracked** (added once with `git add -f`, which is needed only to
break the repository-wide `results/` ignore rule the first time). Because they are
tracked, later runs that rewrite `summary.csv` show up as ordinary modifications:
commit them with `git add week3/results` — no `-f` needed. A *new* run directory
(e.g. `results/3b/vis_goldwin2/`) is still covered by the ignore rule and does need
`git add -f` once.
