# Week 3 implementation notes (revised): visibility experiments on a frozen Qwen

Project: efficient attention for long-context code. The first two weeks' output is in `report/Week1-2_Experiment_Report_ZH.md`.

**The three-step framework is right**: fix a ceiling first, map the evidence onto the model's own positions next, and only then change visibility. But the original plan, executed literally, produces three results that cannot be interpreted — not because it was written carelessly, but because three assumptions do not hold on a real tokenizer and a real model. Every point below comes with measured numbers, and the revised order of execution is at the end.

Every number here was produced on this machine with the real Qwen2.5-Coder tokenizer and real 0.5B/1.5B weights. **They validate the pipeline and give magnitudes, not conclusions**: 20–80 programs per cell can only tell you whether a step would be wasted.

---

## 1. Five things that must change

### 1.1 Window attention is not an information cut-off — compute the receptive field first

The original plan assumed that a window of 64 means a distant `v051 = -1` is invisible. That is wrong. Stacking L layers of causal attention with window W gives a maximum reach of

```
RF = 1 + L × (W − 1)
```

Qwen2.5-Coder-7B has 28 layers, so W=64 gives **RF = 1,765 tokens**. Information is relayed through intermediate tokens, exactly like a convolutional receptive field.

Measured on your test set with the real tokenizer (`week3/preflight/preflight.md`):

| Source bytes | Programs | Mean Qwen tokens | Evidence token share | Readout → earliest evidence |
|---:|---:|---:|---:|---:|
| 256 | 180 | 203 | 30.8% | 172 |
| 1,024 | 180 | 886 | 6.9% | 703 |
| 4,096 (paired lengthening) | 180 | 3,693 | 1.6% | 2,949 |

Share of programs where the furthest evidence **cannot reach** the readout within L layers:

| Source bytes | W=8 | W=16 | W=32 | W=64 | W=128 | W=256 |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 22% | 0% | 0% | 0% | 0% | 0% |
| 1,024 | 98% | 91% | 9% | 0% | 0% | 0% |
| 4,096 | 100% | 100% | 100% | 94% | 9% | 0% |

(7B, 28 layers; the 3B model has 36 layers, a larger RF, and each column shifts towards 0%.)

**Conclusion**: at 256 B, any W ≥ 16 leaves no program where the information cannot arrive; at 1 KB even W=64 gives 0%. **Only at 4 KB does W=64 become a genuine information cut-off.**

There is a counter-intuitive measurement here that tells you what to change:

| one-hop programs, 1.5B, full attention | 73 tok | 216 | 736 | 2,816 |
|---|---:|---:|---:|---:|
| accuracy | 90% | 95% | 95% | 95% |

**One hop, and a 2,800-token distance costs nothing.** Yet the same model:

| one-hop, window=64 | 73 | 216 | 736 | 2,816 |
|---|---:|---:|---:|---:|
| accuracy | 60% | 5% | 10% | 20% |
| evidence reachable within L layers | 100% | 100% | 100% | 73% |

At 216 tokens the window drops accuracy to 5% **even though the evidence is "reachable" by the RF calculation**. Therefore:

- reachability is a **necessary** condition, not a predictor of accuracy; the real break is whether the readout can see the evidence **in one step**;
- step 3 measures the right thing, but the result cannot be explained by connection counts alone: record three quantities — permitted connections, evidence reachable within L layers, and **evidence directly visible to the readout** (`gold_direct_frac`, already implemented).

### 1.2 The readout position is one token off

The original plan said: append `print(result)` and `# Output: `, then compare the model's scores for the digits 0–9. On the real tokenizer:

```
"# Output: "  →  ['#', 'ĠOutput', ':', 'Ġ']      ← the trailing space is its own token (id 25)
"7"           →  ['7']                            ← Qwen pre-tokenisation makes each digit one token
" 7"          →  ['Ġ', '7']                       ← there is no merged "Ġ7"
```

So after `prefix + " " + digit`, the candidate at `logits[-1]` is **a space, not a digit**. Taking a softmax over the ten digit tokens directly compares `7` rather than ` 7`, which is not the position the model would naturally continue from.

The correct approach (`encode_prompt` in `w3_core.py` implements it):

```python
ids = tok(prompt_text, add_special_tokens=False)["input_ids"]   # prompt_text ends with "# Output:"
ids.append(tok.encode(" ", add_special_tokens=False)[0])        # put the space token into the input
logits = model(input_ids=torch.tensor([ids])).logits[0, -1]     # read the ten digits after the space
```

Measured, this change is decisive:

| 1.5B, completion style | trivial (one-line arithmetic) | leak (answer in a comment) | real (your 256 B programs) |
|---|---:|---:|---:|
| accuracy | 100% | 92% | 5% |

`leak` 92% shows the readout is right (a visible digit can be copied); `trivial` 100% shows the prompt format is fine; `real` 5% shows **the task itself is beyond this model**, not that the pipeline is broken. Repeating the same 0.5B gate with a chat template gives `leak` 15% and `trivial` 0% — hence **the main tables use completion style**, and chat is only a robustness check.

One more caveat for the report: first-digit scoring reads a model answer of "10" as 1. Your labels are 0–9, so a wrong answer stays wrong, but the score for 1 is slightly inflated; run a generative decode separately if you do error analysis.

### 1.3 Use the 540-program test set for main tables, not the 120-program validation set

The original plan used "the existing 120 256-byte validation programs". Those were for **checkpoint selection** in Week 1, 40 per family, and a single cell has a 95% interval about ±15 points wide. A frozen model has no selection process; the validation set should only be used to choose the prompt style. Main tables use the test set (180 per length, 60 per family) and keep three lengths for the length axis.

### 1.4 Four controls are missing, so differences cannot be attributed

The original plan only compared "accuracy under four attention patterns". When accuracy drops you cannot tell whether the evidence was cut off, whether the smaller visible range degraded the model overall, or whether the model never learned this mask. All four controls are cheap:

| Control | How | Answers |
|---|---|---|
| `gold_union` (upper bound) | window ∪ all gold evidence tokens made global | with very few connections but the evidence one step away, how much survives? |
| `block=4` (true cut-off) | one block per 4 lines, never visible across | is the structural prior alone enough? (it should collapse) |
| `gold-value` (counterfactual) | change one integer literal in a gold line, recompute the label with the Week-1 interpreter | is the model computing, or guessing the format? |
| `gold-drop` (deletion) | delete the gold line and see whether the prediction changes | does the answer change when that evidence is gone? |

Measured on 1.5B (one hop, 20 per cell):

| Distance | full | window=64 | window=256 | gold_union=64 | block=4 |
|---:|---:|---:|---:|---:|---:|
| 73 tok | 90% | 60% | 90% | 90% | 0% |
| 216 | 95% | 5% | 95% | 80% | 5% |
| 736 | 95% | 10% | 15% | 75% | 15% |
| 2,816 | 95% | 20% | 25% | 45% | 15% |

At 736 and 2,816 tokens `gold_union` uses only 25% and 7% of full's connections and lifts accuracy from 10%/20% to 75%/45% — **that is the main line of the "efficient attention" story**: not "look at fewer tokens" but "when you look at fewer, you must keep the evidence inside". `block=4` is a true cut-off and falls to chance, as expected.

Note that at 2,816 tokens `gold_union` only recovers to 45%: the evidence is one step away, but the 64-token window itself still hurts. Report those two losses separately or a reviewer will ask.

### 1.5 The most important point: your dataset ties "hops" to "distance"

The original plan asked "when given enough information, which kinds of question can Qwen answer?". Measured, that axis is the wrong one:

| Dependency hops (no distractors) | 1 hop | 2 hops | 3 hops |
|---|---:|---:|---:|
| 0.5B | 95% | — | 5% |
| 1.5B | 100% | 40% | 5% |

| By family (1.5B, 256 B, full attention) | assignment chain | function call | conditional branch |
|---|---:|---:|---:|
| accuracy | 10% | 7% | 17% |

**All three of your families are three-hop, and at ≤1.5B they all sit near 10%.** Comparing four attention patterns at that ceiling can only produce "all 10%", which is indistinguishable from "attention has no effect". That is not a data error: the Week-1 task is simply harder than this model size.

The one-hop task, by contrast, is a perfect platform: **ceiling 95%, distance independently stretchable to 2,800 tokens, and it collapses as soon as the window changes.** Week 3 must therefore separate two factors:

- **depth** (1/2/3 hops) controls "can the model do this at all";
- **distance** (20/100/400/1600 tokens) controls "must the evidence be fetched from far away".

`week3/w3_needle.py` already generates data on that design (`data/needle_test.jsonl`, 3 hop counts × 4 distances × 20 programs, labels re-derived by the Week-1 interpreter). The original 540 programs are kept and reported separately as the "hard slice".

---

## 2. Revised order of execution

Every step has a go/no-go. **If it does not pass, stop and fix it there; do not run on** — Week 1 already paid that tuition once (comparisons made near a random baseline cannot be interpreted).

### Step 0 · half a day: gate

```bash
cd week3
python -m unittest -v test_week3                 # 24 tests, no weights needed
python w3_gate.py --model <7B path> --style completion --limit 40
```

Criteria (put them in the report):

| Level | Meaning | Threshold |
|---|---|---|
| trivial | one line `result = x + y` | ≥ 80% |
| leak | real program + the answer written in a comment | ≥ 80% |
| real | real 256 B programs | record only, no threshold |

`leak` below threshold = wrong readout or template; fix it, do not move on. Measured reference: 0.5B 95/100/5, 1.5B 100/92/5.

### Step 1 · 1 day: ceiling curve

```bash
python w3_curve.py    --model <7B> --lengths 96 160 256 512 1024   # the existing three-hop task
python w3_needle.py   --distances 20 100 400 1600 --hops 1 2 3 --per 20
python w3_run.py      --model <7B> --data-dir data --split needle_test --lengths 0 \
                      --group-by family_id --modes full --no-reach --out results/needle_7b
```

Output: two tables — accuracy by (byte length × family), accuracy by (hops × distance). **This is one of Week 3's real main results**: it decides which task configuration everything else uses.

Criterion: at least one (hops, distance) cell ≥ 70%. If 7B is still ≤ 20% on three-hop 256 B, move the main table to one-hop needle and demote the original dataset to "hard slice", explaining the switch in the report.

### Step 2 · 1 day: evidence dependence (not "can it see", but "does it use it")

```bash
# counterfactual: change one constant, recompute the label
python w3_run.py --model <7B> --split test --lengths 256 --style completion \
                  --ablate gold-value --modes full --out results/cf_7b
# deletion: the evidence is gone, the prediction should change
python w3_run.py --model <7B> --split test --lengths 256 --style completion \
                  --ablate gold-drop  --modes full --out results/drop_7b
```

Criteria: under the counterfactual accuracy should barely drop (the model follows the code); after deletion the agreement with the clean prediction should fall to ~10% (the model really uses that evidence). If neither holds, the premise "long-distance evidence" is itself in question — write that up before discussing attention.

The original step 2's "map source ranges to Qwen token positions" is not a separate experiment: it is the measurement infrastructure for this step and step 3, already in `w3_core.span_token_positions()` — **gold byte offsets → character offsets in the prompt → token positions**, with a test asserting that every mapped position really falls inside the gold text.

### Step 3 · 1.5 days: visibility comparison

Run the four cheapest conditions first, then decide whether to extend:

```bash
python w3_run.py --model <7B> --data-dir data --split needle_h1 --lengths 0 \
  --group-by family_id --modes full,window=64,gold_union=64,block=4 \
  --out results/vis_7b
```

Then extend as needed:

| Pattern | Definition | What it means now |
|---|---|---|
| `full` | causal full visibility | ceiling |
| `window=W` | each position sees the previous W | local attention; choose W from the RF |
| `local_global=W:linestart:K` | window + every K-th line start global | Longformer-style; **global tokens only help if they sit between the evidence and the query** |
| `block=L` | one block per L lines, never visible across | true cut-off, the structural-prior upper bound |
| `gold_union=W` | window + gold evidence global | an oracle bound, not a method |

There is an easy mistake in `local_global`: if the global positions are all at the beginning (`first:4`), they are pure read slots and **cannot carry evidence forwards**. Measured, on 4 KB `local_global=32:first:4` uses only 1.9% of the connections, but the fraction reachable within L layers is 29–36%. To be useful the global set should be anchors between the evidence and the query (line starts, function definition lines, the `result` line).

Quantities that must be reported together (`summary.csv` already has them):

| Column | Meaning | Why it is mandatory |
|---|---|---|
| `accuracy` + Wilson interval | main metric | n=20/cell, differences under 5 points are uninterpretable |
| `mean_connections` / share | permitted (query, key) pairs | "how much is saved" against full |
| `mean_gold_visible_frac` | fraction of evidence reachable within L layers | **the information-theoretic bound** |
| `mean_gold_direct_frac` | fraction of evidence visible to the readout in one step | the only quantity that tracked accuracy in practice |
| wall clock | time | state up front that **the mask does not speed anything up**, see §4 |

Criterion: `full` and `window=256` (when 256 ≥ sequence length) must agree item by item, as proof that the mask is wired; `block=4` must fall to chance. If either fails, the implementation is wrong — that is not a scientific finding.

### Step 4 · 1 day: write

Three things must be written separately: whether the pipeline is trustworthy (step 0 gate), whether the task has a ceiling (step 1), and what visibility changed (step 3). Week 2's lesson was that "the pipeline is fine" and "the task was not learned" must be reported apart; the same holds here.

---

## 3. Table templates

**Table A — ceiling (step 1)**

| Task configuration | Hops | Distance (tokens) | n | Accuracy | Wilson 95% |
|---|---:|---:|---:|---:|---|

**Table B — visibility (step 3, one per length)**

| Pattern | Connection share | Evidence direct | Evidence within L | Accuracy | McNemar p vs full |
|---|---:|---:|---:|---:|---:|

**Table C — controls (step 2)**

| Condition | Accuracy | Agreement with clean | Note |
|---|---:|---:|---|

---

## 4. Engineering details and known pitfalls

1. **A 4-D mask can simply be passed as `model(attention_mask=bool4d)`.** In this transformers version (4.57.1) `_preprocess_mask_arguments` returns a 4-D input unchanged, SDPA receives it with `is_causal=False`, and True = may attend. The test `test_explicit_full_mask_equals_no_mask` pins this down: an explicit full mask and no mask agree to 1e-4.
2. **The mask does not speed anything up; say so up front.** Week 1 measured it: logical connections fall 56.5–87.9% while dense SDPA wall clock barely moves. The timing column in Week 3 reproduces that, it is not a new result. Real speed-ups need a different execution (block-sparse kernels) or fewer tokens (input selection) — a different line of work.
3. **Memory is smaller than you would think.** The 4-D bool mask takes the memory-efficient path and never materialises L×L: at 3,670 tokens, 0.5B peaks at 1.18 GB with a mask and 1.17 GB without. 7B (15.2 GB of weights) runs within 8k, but start at 4 KB.
4. **The platform is an AMD card (RX 7900 XTX 24 GB, torch 2.10.0+rocm)** with no flash-attn; `attn_implementation` is `sdpa`. To inspect attention mass with `output_attentions` you must switch to `eager` (sdpa does not support it).
5. **Do not modify Week 1's `dataset.py`.** Its name pool has only 256 entries, so `make_sample` raises `StopIteration` at 4 KB. Lengthen with `w3_core.lengthen_program()`: insert irrelevant assignment lines, shift the gold offsets, and re-derive labels with the Week-1 interpreter; 180×2 programs already pass.
6. **Deleting tokens re-tokenises.** When a whole line is deleted the newline is its own token and tokenisation barely changes, but positions shift; pass `position_ids` explicitly if original positions must be preserved. Deletion and masking are **two mechanisms**; do not compare them in one table.
7. **Compute estimate**: 0.5B at 3.7k tokens takes about 1.5 s per program (including tokenisation, mask and reachability); 7B is roughly 10–15× slower. 540 programs × 5 patterns × 3.7k tokens on 7B is 10–15 hours — **do not run it in one go**: screen the patterns on 80 programs first, then run everything.

---

## 5. Time budget (35 hours)

| Day | Task | Hours | Acceptance |
|---|---|---:|---|
| 1 | environment, weights, gate, prompt style | 7 | `leak ≥ 80%`, trivial ≥ 80% |
| 2 | ceiling curves (three-hop + needle 1/2/3 hops × 4 distances) | 7 | a table that selects the main task configuration |
| 3 | counterfactual and deletion controls + 4 KB generation and preflight | 7 | the causal dependence is explicit; the RF table is done |
| 4 | visibility comparison (full/window/global/block/gold_union) | 7 | table B filled, mask wiring verified |
| 5 | write the report and re-check | 7 | pipeline, ceiling and visibility written in three separate sections |

If only 7 hours are available: the step 0 gate plus `full / window=64 / gold_union=64` on one-hop needle (80 programs). Those three rows already show "how much visibility costs" and "how much keeping the evidence buys back".

---

## 6. New answers to your three questions

| Original question | What was wrong | Replace with |
|---|---|---|
| When given enough information, which kinds of question can Qwen answer? | grouping by family, 60 per cell; all three measured near 10%, the axis was wrong | group by **hops** first, then by **distance**; family is descriptive only |
| `v051 = -1` is the distant key definition — can the model see it directly? | under full attention of course it can; "seeing" is not "using" | record `gold_direct_frac`; then test dependence with counterfactual and deletion |
| How much answering ability survives under different visible ranges? | right question, but accuracy alone cannot attribute the difference | add the `gold_union` bound, the `block` cut-off, and the same-length `window=256` identity check |

---

## 7. File list

| File | Purpose |
|---|---|
| `w3_core.py` | data, tokenizer, prompt, digit readout, span→token mapping, four masks, reachability, statistics |
| `w3_run.py` | frozen-model evaluation: modes, ablations, per-item CSV, summary |
| `w3_gate.py` | gate: trivial / leak / real |
| `w3_curve.py` | ceiling curve (byte length × family) |
| `w3_needle.py` | distance-controlled data generation (hops × distance) |
| `w3_preflight.py` | weight-free preflight: token lengths, RF table, mask geometry |
| `w3_paired.py` | paired exact McNemar over `per_item.csv` |
| `w3_reach.py` | can the value token reach the readout under a mask |
| `test_week3.py` | 24 tests, including "explicit full mask == no mask" |
| `preflight/preflight.md` | the numbers in §1.1 |
| `data/needle_test.jsonl` | 3 hop counts × 4 distances × 20 programs |
| `results/` | gate and initial run records |
