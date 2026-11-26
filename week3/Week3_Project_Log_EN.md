English version (Week 3 log v1)

Bridge note (placed before the Week 3 log)

Week 2 closed with two results: grouping did not beat per-block selection (a null result), and the baseline was still at chance, so the selection strategies had no usable model-side comparison. Week 3 therefore changes two things at once and says so up front: the model is a frozen pretrained Qwen2.5-Coder-3B-Instruct instead of the 4-layer byte model, and the task is a one-hop, distance-controlled program instead of the three-hop Week 1 programs. Three points:

1. The Week 1 three-hop task is re-measured on the new model before anything else and is at chance (15% overall, Wilson 7–29%, n = 40; 7–30% by source length). It is kept as a "hard slice" and is never used to compare attention patterns.
2. Depth and distance are separated by a generator that places a single definition at a controllable token distance from the query. At one hop, full attention answers correctly from 73 to 4,051 tokens (98–100%), so distance itself is not the limiting variable.
3. Every comparison uses one checkpoint, one prompt, one readout and one set of programs; only the 4-D boolean attention mask changes. Before any number is reported, the mask path is verified to reach SDPA (an explicit full mask reproduces the unmasked logits).

Experiment Log — Week 3
Topic: Attention-visibility experiments with a frozen Qwen — does keeping evidence visible preserve long-range retrieval?
Plan slot: Week 3 (35-hour allocation)
Environment: NVIDIA RTX 4000 Ada (19.5 GB), torch 2.6.0+cu124, transformers 4.57.1, bfloat16; model Qwen2.5-Coder-3B-Instruct (36 layers, hidden 2,048, 16 heads / 2 KV heads), weights frozen; Qwen tokenizer shipped with the repository; no training of any kind.

Objective

With weights, prompt, data and answer scoring held fixed, measure how much answering ability remains when the attention mask changes, whether the loss comes from seeing fewer tokens or from seeing the wrong ones, and how much an oracle that keeps every evidence token visible recovers.

What was done

1. Gate (n = 40): one-line arithmetic (trivial), a real program with the answer written into a comment (leak), and the actual 256-byte programs (real). Pass line: trivial and leak ≥ 80%.
2. Ceiling curve on the Week 1 three-hop task: 96/160/256/512 bytes, 20 programs per dependency family.
3. Distance-controlled generator: one dependency chain, a single definition at a target token distance, filler lines calibrated against the real tokenizer, labels re-derived by the Week 1 interpreter; a `--position end` variant keeps the same length but moves the definition next to the query.
4. Six visibility patterns applied to the same token sequence: full; window=W; window=256; random=W (the same per-row budget with random placement); block=4 (a true cut-off); gold_union=W (window ∪ all gold evidence tokens made global — an oracle that uses gold annotations).
5. Readout: program + `\nprint(result)\n# Output:` + one space token, then a softmax over the ten digit tokens. The trailing space of `# Output: ` is a token of its own (id 25); scoring `logits[-1]` at the digit ids would compare `"7"` rather than `" 7"`.
6. Measurement scale: 50 programs per cell (20 for the ceiling curve) over five distances; window sweep 64/128/256/512 at 1,646 and 4,051 tokens; counterfactual and deletion ablations on the three-hop set.
7. Engineering: 24 automated tests, including "explicit full mask == no mask" and "random=W has the same per-row degree as window=W"; a lab self-check that verifies the 4-D mask pass-through on a randomly initialised model.

Results

Table 1 — Gate (n = 40): trivial 90% (77–96), leak 93% (80–97), real 15% (7–29).

Table 2 — Three-hop ceiling by source length: 96 B 23% (n = 40), 160 B 30% (n = 53), 256 B 20% (n = 60), 512 B 7% (n = 60).

Table 3 — Full attention, depth × distance (n = 20):

| Hops | 73 tok | 216 | 736 | 2,816 |
|---|---:|---:|---:|---:|
| 1 | 100% | 100% | 100% | 100% |
| 2 | 45% | 20% | 20% | 30% |
| 3 | 20% | 5% | 5% | 5% |

Table 4 — Six patterns × five distances (n = 50):

| Distance (tok) | full | window=64 | window=256 | random=64 | gold_union=64 | block=4 |
|---:|---:|---:|---:|---:|---:|---:|
| 73 | 98% | 68% | 98% | 72% | 98% | 6% |
| 151 | 100% | 6% | 100% | 6% | 100% | 8% |
| 450 | 100% | 16% | 16% | 2% | 100% | 10% |
| 1,646 | 100% | 6% | 14% | 12% | 22% | 14% |
| 4,051 | 98% | 0% | 12% | 6% | 18% | 4% |

Table 5 — Same connection count, local window versus random placement:

| Distance (tok) | window=64 | random=64 | Connections (both) |
|---:|---:|---:|---:|
| 73 | 68% | 72% | 2,656 |
| 151 | 6% | 6% | 7,648 |
| 450 | 16% | 2% | 26,784 |
| 1,646 | 6% | 12% | 103,328 |
| 4,051 | 0% | 6% | 257,248 |

Table 6 — Same 4,051-token length, definition at the start versus next to the query:

| Condition | Distance (tok) | full | window=64 | gold_union=64 | random=64 |
|---|---:|---:|---:|---:|---:|
| definition at the start | ≈4,040 | 98% | 0% | 18% | 6% |
| definition next to the query | ≈46 | 98% | 90% | 72% | 6% |

Table 7 — Window sweep at 4,051 tokens:

| Pattern | Connections | Share of full | Accuracy |
|---|---:|---:|---:|
| full | 8,207,326 | 100% | 98% |
| gold_union=64 | 403,867 | 4.9% | 18% |
| gold_union=128 | 654,651 | 8.0% | 24% |
| gold_union=256 | 1,143,931 | 13.9% | 60% |
| gold_union=512 | 2,073,339 | 25.3% | 78% |

Table 8 — Budget-matched control at 4,051 tokens (`window=100` and `random=100` are within 2% of `gold_union=64`):

| Pattern | Connections | Accuracy | Paired vs `gold_union=64` |
|---|---:|---:|---|
| full | 8,207,326 | 98% | — |
| gold_union=64 | 403,867 | 18% | — |
| window=100 | 400,150 | 6% | 7:1, p = 0.070 |
| random=100 | 400,150 | 12% | 8:5, p = 0.58 |

Four findings. (i) Distance does not damage the ability under full attention (98–100% at every distance up to 4,051 tokens). (ii) A mask that does not contain the evidence cannot recover, whatever its shape; at long distance a window and a random mask both sit at the floor and are **indistinguishable**, which is not the same as "shape does not matter". In the near condition, at the same budget, the window beats random **42:0** (p ≈ 4.5×10⁻¹³) precisely because it contains the evidence. (iii) Making all evidence global restores full accuracy up to ≈450 tokens but only 18–22% at 1,646–4,051; the same-length / short-distance control lifts the same masks from 0% to 90% (window=64) and from 18% to 72% (completeness control), so the decay is caused by distance rather than by sequence length. (iv) That control uses 30–57% more connections than a window of the same width, so a budget-matched run followed: at equal budget it still dominates up to 450 tokens (46:0, 41:0) but from 1,646 tokens it is indistinguishable from a random mask (p = 0.11–0.58), and the "0% → 18%" step at 4,051 cannot be decomposed into connections versus visibility.

Two smaller results: fixed structural anchors (every line start, the first four tokens) do not help — at 4,051 tokens every line start costs 17.5% of the connections for 6% accuracy — and the completeness control is **not an accuracy upper bound**, since at gap 46 it scores 72% against a plain window's 90%.

Problems and decisions

1. The three-hop task has no ceiling at 3B (15%), so it cannot support an attention comparison. Decided: measure it once, report it as a hard slice, and move the main table to the one-hop distance-controlled set. This is the same "pass the gate first" rule Week 2 ended with.
2. The 4 KB visibility experiment was first run on the three-hop data and is at chance under full attention (3.3%); it is recorded as uninterpretable and replaced by 4 k-token needle programs. The counterfactual and deletion ablations on the three-hop set (14.4% / 8.9%) are uninterpretable for the same reason.
3. A counterfactual label could leave 0–9 and crashed the readout (index 10 into a 10-way softmax). Fixed: try several deltas, accept only labels that stay in 0–9, skip and count otherwise; one regression test over 60 programs, plus a full check of all 540 test programs.
4. The needle generator first assumed 7.5 tokens per filler line, but Qwen tokenises every digit, so a line is ≈13 tokens; target 4,000 produced 6,969. Fixed: measure one filler line, solve the count directly, then correct in at most four steps (targets 20/100/400/1,600/4,000 → 66/144/443/1,639/4,044 code tokens).
5. 7B was not used: the lab home quota left 16 GB free against 15.2 GB of weights, and 19.5 GB of VRAM is tight at 4 k tokens. All results are 3B and are labelled as such throughout.
6. Masking does not save compute: the 4-D mask runs through dense SDPA. Logical connections are reported as a budget, never as a speed-up.
7. `gold_union` was not budget-matched: it uses 30–57% more connections than a window of the same width, so the "0% → 18%" step changed two things at once. Fixed by adding `window=100` / `random=100` (within 2% of its connection count, Table 8). The wording "the loss comes from the budget, not from placement" was withdrawn: the long-distance cells are at the floor and cannot support it, and the 4,051 cells cannot decompose the two factors.
8. `random=W` was seeded from `crc32(row["id"])`. Adding the `start/end` suffix to the ids changed all 250 seeds, so the committed random numbers could not be reproduced. Fixed: the seed now comes from the program's `seed` field and the mask width; the `random` runs were repeated. Only the random columns moved, which also served as a check that nothing else was affected.
9. The report called `gold_union` an "accuracy upper bound". At gap 46 it scores 72% against a window's 90% (paired 12:3, p = 0.035), so the name was wrong; it is an evidence-completeness control.

Verification

24 automated tests pass (causality, receptive field against BFS, block-diagonal cut-off, random==window budget, span→token mapping, digit tokens, mutation range, statistics).
The mask path is verified functionally: an explicit full mask reproduces the unmasked logits to 1e-4, and a window mask changes them.
The gate separates a broken readout from a hard task: leak 93% shows a visible digit can be copied; real 15% shows the task itself is hard.
All needle labels are re-derived by the Week 1 interpreter at generation time; the 540 Week 1 programs survive counterfactual mutation with labels kept in 0–9.
Every table is produced by the commands in section 7 of the report; per-item records are committed under `week3/results/`; the directory-to-table map is `week3/RESULTS.md`.

Artifacts

`w3_core.py`, `w3_run.py`, `w3_gate.py`, `w3_needle.py`, `w3_curve.py`, `w3_preflight.py`, `lab_check.py`, `make_bundle.py`, `test_week3.py`, `run_all.sh`, `README.md`, `RESULTS.md`, `LAB_SETUP_ZH.md`, `Week3_Implementation_Guide_ZH.md`, `Week3_Experiment_Report_ZH.md`, `Week3_Experiment_Report_EN.md`, `_ref/tokenizer/`, `data/` (4 sets), `preflight/`, `results/` (per-run records; see `RESULTS.md`). Week 1 dependency: `dataset.py` and `data/{test,validation}.jsonl`.

Next

1. Stop tuning mask parameters: with frozen weights the long-distance loss is not recoverable by making evidence visible.
2. Move to training-time adaptation: LoRA-tune the 3B model while training with the `window=64` / `gold_union=64` masks, and test whether the 18% at 4 k can be raised above 80%.
3. Bring in the Week 2 analyser: measure the dependency-distance distribution of real code; if most dependencies are ≤ 512 tokens, "window 512 + a few global anchors" is adequate in practice.
4. Optional mechanism probe: under full attention, measure the attention mass from the readout onto the gold tokens as a function of distance.
