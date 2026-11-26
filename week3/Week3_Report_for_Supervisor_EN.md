# Week 3 Report: Attention-Visibility Experiments with a Frozen Qwen

*(Supervisor-facing version. Each mask is explained on a minimal example before any data; every table is followed by the specific cells that support the stated conclusion.)*

---

## 1. Summary in one page

With the weights **frozen** and the prompt, programs and answer scoring held fixed, changing only *which tokens the model may attend to*:

- Under full attention the model answers correctly **98%** of the time even when the definition it needs sits **4,051 tokens** away — distance itself does not damage the ability.
- Cutting each position's view to **64 tokens** drops accuracy to **0–16%**.
- An **evidence-completeness control** (`gold_union`: every evidence token made global, using gold annotations) restores 98–100% up to 450 tokens but only **18–22% at 1,646 and beyond**. That control uses **30–57% more connections** than a window of the same width, so a **budget-matched** comparison was run (Table 10): up to 450 tokens the evidence position is still decisive (100% vs 8–18%; 46:0 and 41:0), while from 1,646 tokens on it is **indistinguishable from a matched-budget random mask** (p = 0.11–0.58). It guarantees that information is *available*; it is **not an accuracy upper bound** (72% at gap 46, below a plain window's 90%).
- Holding sequence length fixed at 4,051 tokens and moving the definition next to the query lifts the same masks from 0% to **90%** — the failure is caused by **distance**, not by sequence length.
- **Deployable fixed anchors** (every line start, the first four tokens) do not help: at 4,051 tokens every line start costs **17.5%** of the connections and still gives **6%**.

**Implication.** Applying a sparse mask to a dense model at inference time cannot preserve long-range retrieval; to preserve it, the model has to be adapted during training.

---

## 2. Six masks: principle and a minimal example

### 2.1 A six-line program

```
line 1   a = 7            <- definition (gold)
line 2   b = 3            <- irrelevant
line 3   c = 9            <- irrelevant
line 4   d = a - 2        <- computation (gold)
line 5   e = b + 1        <- irrelevant
line 6   result = d       <- query: the model reads the answer here
```

The answer is 5 (`7 - 2`). To get it right the model must use lines 1, 4 and 6. Lines 1 and 4 are the "evidence"; the report calls them **gold**. They are recorded by the generator and independently re-checked by the Week-1 analyser; they never enter the model's input.

Every mask below changes only **which positions can see which**, never the token sequence.

### 2.2 The six masks on this program

| Mask | Rule | Line 6 (readout) can directly see | Answers correctly? |
|---|---|---|---|
| `full` | every position sees itself and everything before | 1,2,3,4,5,6 | yes |
| `window=2` | each position sees the previous 2 lines | 5,6 | **no** (lines 1, 4 missing) |
| `random=2` | each row sees 2 positions, chosen at random | e.g. 2,6 | **no** (lines 1, 4 missing) |
| `block=3` | one block per 3 lines; blocks never see each other | 4,5,6 | **no** (line 1 missing) |
| `local_global=2` | `window=2` **∪ every line start made global** (a fixed rule, no labels) | 5,6 plus the **first token** of lines 1 and 4 (`a`, `d`) | **no** (the names are visible, not the values) |
| `gold_union=2` | `window=2` **∪ every gold line made global** (from annotations) | 1,4,5,6 (whole lines) | **yes** |

This table is the whole principle (paper provenance in §2.6):

- `window` and `random` are **equal-budget** placements (both see 2 lines), one by position, one at random;
- `block` is a **true cut-off**: blocks cannot exchange information even through depth;
- `local_global` and `gold_union` **share one mask mechanism** (window + a few global tokens); they differ only in **how the global set is chosen** — a fixed rule versus gold annotations. The former is deployable but does not know where the evidence is; the latter does, and is therefore a diagnostic control rather than a method;
- the `local_global` row is not hypothetical: the value `7` is the third token of `a = 7`, so a line-start global exposes `a`. The real experiment shows the same effect — at 4,051 tokens every line start costs 17.5% of the connections and gives 6% (Table 8).

> For readability this section counts **lines**; the real experiments count **tokens** (a Python line is about 13 Qwen tokens, because every digit is a separate token). The rules are identical.

### 2.3 After lengthening the program

```
line1 a = 7      d01=3 d02=8 ... d99=2      d = a - 2      result = d
     ^ far definition (gold)   ^ 100 irrelevant assignments  ^ computation (gold)  ^ readout
     |<----------------- about 1,300 tokens ---------------->|
```

- `window=64`: the readout sees only the last 64 tokens, **line 1 is not among them**.
- `random=64`: 64 random positions; line 1 is **very unlikely** to be among them.
- `block=4`: lines 1 and 6 are in different blocks, **never visible**.
- `local_global=64`: line 1's *first token* is global, but **the first token is not the value**; the value is about the 10th token of the line and is still not directly visible.
- `gold_union=64`: lines 1 and 4 are made global **as whole lines**, so they are visible however far away they are.

### 2.4 What `gold_union` is, and why it matters

`gold_union` = "window + every evidence token made global". It **guarantees that no information is missing**:

> If everything the answer needs is placed in front of the model, can it still answer?

**A naming correction first.** Early drafts called this an "accuracy upper bound"; that is wrong. In the near condition it scores **72%**, below a plain window's **90%** (paired 12:3, p = 0.035). It guarantees **information completeness** (all required tokens can be read directly), not **maximum accuracy** — making tokens global also changes the attention pattern and can itself hurt. The report therefore calls it the **evidence-completeness control**, and uses "oracle" only to say that annotations were used.

Three things must be said:

1. **It is a diagnostic control, not a method.** It uses gold annotations (which lines are evidence); those do not exist at deployment. It only answers "is the information sufficient"; it is not a usable attention scheme.
2. **It differs from a plain window not only in "evidence visible" but also in connection count.** At 4,051 tokens `window=64` has 257,248 connections and `gold_union=64` has 403,867 (**+57%**). The budget-matched control (`window=100` / `random=100`, about 400,150) is Table 10: with the budget equalised, `gold_union` is still clearly better up to 450 tokens (46:0, 41:0) but indistinguishable from a random baseline from 1,646 tokens on (p = 0.11–0.58). The source of the "0% → 18%" step at 4,051 therefore **cannot be decomposed** into connections versus evidence visibility — `gold_union` changes three things at once (count, position, global read/write).
3. **It only supports one direction of inference.** If `gold_union` cannot recover, then "making the evidence visible" is not enough; it does not prove that missing information is the only cause, because the mask itself can degrade performance (72% vs 90% at gap 46 is direct evidence).

**How does the model know which tokens are gold? It does not.** The annotation lives in the dataset's `gold_static_spans` field (recorded by the generator, independently re-checked by the Week-1 analyser) and, like the answer, is evaluation-side information that never enters the prompt. When building the mask the harness maps those byte spans to token positions and does exactly one thing: it sets the corresponding entries of the boolean mask to `True`. The model receives a True/False matrix; it cannot tell which columns are evidence, it is merely permitted to look. That is why `gold_union` "cheats": it does not leak the answer (the answer token is not in the input) but it does leak *which positions matter*.

The deployable substitute is the Week-2 dependency analyser: a reverse traversal of dependency edges from `result` locates the evidence statements **without any annotation**. `gold_union` replaces that prediction with the truth, which measures "how much comes back if the prediction is perfect".

**How it differs from `local_global` (the only difference): how the global set is chosen.** Both call the same mask construction: a window plus a few global tokens (a global token can read everything before it and is read by every later position).

| | `local_global` | `gold_union` |
|---|---|---|
| Source of the global set | fixed rule: line starts, first N tokens | gold annotations: evidence-statement tokens |
| Needs labels | no | **yes** |
| Deployable | yes | no |
| Role in this report | deployable baseline (Table 8) | evidence-completeness control (Tables 4, 7, 10) |
| Connections at 4,051 tokens | 273,190 (first 4 tokens) | 403,867 |

They are not the same thing: `local_global` is *a rule*, `gold_union` is *the same mechanism fitted with annotations*. Conflating them hides the most important comparison in this report — **spending more connections on structurally chosen global tokens still does not recover the ability (Table 8)**.

### 2.5 Operational definitions

- The mask is a 4-D boolean tensor, `True` = may attend, passed straight to `model(attention_mask=...)`. Before any number is reported it is verified that an explicit `full` mask reproduces the unmasked logits to within 1e-4.
- The "64" in `window=64` counts **tokens**, not lines; in the needle data one line `d1234500001 = 3` is about 13 tokens (every digit is its own token).
- The model has 36 layers, so a window mask can in principle relay information through depth: receptive field = 1 + 36 × (64 − 1) = 2,269 tokens. That is theoretical reachability; **the model does not exploit it in practice** (Table 4).
- The token sequence is identical under every mask; only visibility changes.

### 2.6 Where the masks come from

These masks are not ours. Following the supervisor's request, only the four references already in the literature review are cited.

| Mask in this report | Source | Form in the original paper | Our use and differences |
|---|---|---|---|
| `full` | Vaswani et al., 2017 [1] | standard full attention, O(n²) | ceiling baseline |
| `window=W` | the sliding window of Longformer (Beltagy et al., 2020) [2] | each position sees the neighbouring W tokens | identical; no attention sink |
| `local_global=W:k` | same mechanism as Longformer's **global attention** [2]; a code-domain design of the same kind is SASA (Liu T. et al., 2022) [4] | a few tokens are global: they read earlier positions and are read by all later ones | same mechanism; the **global set follows a fixed rule** (line starts, first k tokens) and needs no labels |
| `gold_union=W` | **no corresponding paper**; borrows the global mechanism of [2] but takes the global set from gold annotations | — | an **evidence-completeness control**, uses labels, not deployable; same mechanism as `local_global`, different selection |
| `random=W` | the **random attention** component of BigBird (Zaheer et al., 2020) [3] | random connections outside the window | only the "random positions" idea; our version is an **equal-budget control**, not a reproduction of BigBird |
| `block=L` | **no corresponding paper**; a fixed block-diagonal control in this project | within-block full, between-block none | only used to provide a "true cut-off" lower bound |

Three notes:

1. **`gold_union` and `local_global` must be read separately.** They share the "window + global tokens" mechanism and differ only in where the global set comes from: annotations (the mechanism of [2] plus labels) versus a fixed rule (deployable). The former is a control, the latter a baseline; see the comparison table in §2.4 and Table 8 in §4.
2. **`random=W` is a control variable, not a BigBird reproduction.** In BigBird the random edges are one component of the sparse pattern; our random mask keeps the per-row visible count exactly equal to `window=W` and only changes positions, which separates "seeing less" from "seeing the wrong places".
3. **The literature consensus is that sparse attention is trained with its pattern.** Longformer [2] and BigBird [3] both use their sparse patterns during training; §4 of this report tests what happens when a dense model is masked only at inference time.

---

## 3. Setup

| Item | Value |
|---|---|
| Model | Qwen2.5-Coder-3B-Instruct (36 layers, hidden 2,048, 16 heads / 2 KV heads), **weights frozen, no training** |
| Precision / device | bfloat16, NVIDIA RTX 4000 Ada (19.5 GB), torch 2.6.0+cu124, transformers 4.57.1 |
| Dataset A | Week-1 three-hop programs: 256/512/1,024 bytes, 180 each (all 2,100 samples audited) |
| Dataset B | distance-controlled one-hop programs: one dependency chain, the definition placed at a target distance of 20/100/400/1,600/4,000 tokens, 50 programs per cell |
| Readout | program + `\nprint(result)\n# Output:` + one space token, then a softmax over the ten digit tokens |
| Statistics | 50 programs per cell (20 for Tables 2–3), Wilson 95% intervals in brackets; **differences below ±10 points are not interpreted**; paired comparisons use exact McNemar |

**Gate.** Before comparing any masks, the readout is checked: one-line arithmetic (`trivial`), a real program with the answer written into a comment (`leak`), and the real 256-byte programs (`real`).

Dataset B is structurally the same as the "needle in a haystack" long-context evaluations: a definition is placed at a controlled distance and the model must retrieve it. The difference is that the answer is not a verbatim copy but **one integer operation along the dependency chain, emitted as a digit 0–9**, so the ten-way readout gives a unique judgement without human or model grading.

---

## 4. Results, table by table

### Table 1 — Gate (n = 40)

| Level | Accuracy | Wilson 95% | What this table shows |
|---|---:|---|---|
| trivial | 90% | 77–96% | the readout and prompt format work |
| leak | 93% | 80–97% | if the answer appears in the input, the model can copy it → later drops are not "cannot read out" |
| real | 15% | 7–29% | **the three-hop task is already near chance** (10%) |

**Conclusion 1**: the pipeline is trustworthy (trivial and leak are high), but the Week-1 three-hop task has no ceiling at 3B and cannot be used to compare attention patterns.

### Table 2 — Ceiling curve on the three-hop task (20 per family)

| Source bytes | n | Accuracy |
|---|---:|---:|
| 96 | 40 | 23% |
| 160 | 53 | 30% |
| 256 | 60 | 20% |
| 512 | 60 | 7% |

**Conclusion 2**: all four lengths land in 7–30%; none is usable. Comparing masks on a 20% baseline would produce differences that cannot be interpreted.

### Table 3 — Full attention, depth × distance (n = 20)

| Hops | 73 tok | 216 | 736 | 2,816 |
|---|---:|---:|---:|---:|
| 1 | 100% | 100% | 100% | **100%** |
| 2 | 45% | 20% | 20% | 30% |
| 3 | 20% | 5% | 5% | 5% |

- the one-hop row is **100% everywhere**, including 2,816 tokens: the model handles long-distance single-hop retrieval;
- two hops drop to 20–45%, three hops return to chance: **depth is the bottleneck**;
- so every subsequent experiment uses one-hop data and treats distance as the variable.

**Conclusion 3**: depth is the bottleneck, not distance; the one-hop task is the right platform.

### Table 4 — Six masks × five distances (n = 50, one hop)

| Distance (tok) | full | window=64 | window=256 | random=64 | gold_union=64 | block=4 |
|---:|---:|---:|---:|---:|---:|---:|
| 73 | 98% | 68% | 98% | 72% | 98% | 6% |
| 151 | 100% | 6% | 100% | 6% | 100% | 8% |
| 450 | 100% | 16% | 16% | 2% | 100% | 10% |
| 1,646 | 100% | 6% | 14% | 12% | 22% | 14% |
| 4,051 | 98% | 0% | 12% | 6% | 18% | 4% |

*The `random=64` column is the re-run with the fixed seed (the earlier implementation seeded from the program id, which was not reproducible; see §6).*

Reading the columns:

1. **`full` is 98–100%**, including 4,051 tokens: distance does not damage the ability.
2. **At 73 tokens `window=256` equals `full`**: 256 covers the whole sequence, confirming that the mask is wired as designed.
3. **At 151 and 450, `gold_union` is 100% while `window=64` is 6%/16%**: in these two cells, almost everything depends on whether the distant definition is inside the visible set. **But this is not an equal-budget comparison**: `gold_union=64` uses 30–50% more connections (151 tok: 9,967 vs 7,648; 450 tok: 40,166 vs 26,784), so visibility and budget change together.
4. **At 1,646 and 4,051, `gold_union` falls to 22%/18%**: the evidence is 100% visible and the model still fails. The conclusion is the conservative one: **more visibility and more connections together still do not recover the ability**.
5. **`block=4` stays at 4–14%**: the expected behaviour of a true cut-off.

**Conclusion 4**: up to 450 tokens accuracy is roughly binary — evidence visible means correct, invisible means wrong; but these cells are not budget-matched (see Table 10). **Conclusion 5**: from 1,646 tokens on, even making the evidence global leaves only 18–22%.

### Table 5 — Local window vs random placement at the same connection count (n = 50)

| Distance (tok) | window=64 | random=64 | Connections (both) | Paired McNemar (window-only / random-only) | p |
|---:|---:|---:|---:|---|---:|
| 73 | 68% | 72% | 2,656 | 7 / 9 | 0.80 |
| 151 | 6% | 6% | 7,648 | 3 / 3 | 1.00 |
| 450 | 16% | 2% | 26,784 | 8 / 1 | 0.039 |
| 1,646 | 6% | 12% | 103,328 | 3 / 6 | 0.51 |
| 4,051 | 0% | 6% | 257,248 | 0 / 3 | 0.25 |

The two columns have **exactly the same connection count**; only the placement differs. **This does not establish equivalence**: from 151 tokens on, both masks sit near the 0–16% floor, where a paired test has no power. At 73 tokens the window covers almost the entire sequence (73 tokens), so the two are close by construction.

The p = 0.039 at 450 tokens should not be read on its own: with a **different random mask** (before the seed fix) the same cell was 7:3, p = 0.34, so this small difference is sensitive to the random draw and is not a stable effect.

**Conclusion 6**: at distances where **both** masks are at the floor they are indistinguishable; those cells therefore **cannot** support "shape does not matter". All they support is the weaker statement: **a mask that does not contain the evidence cannot recover the ability, whatever its shape.**

What separates "containing the evidence" from "shape" is the near-condition control in Table 6 and the budget-matched control in Table 10: at the same 4,051 tokens and the same ~400k connections, `gold_union=64` is still clearly above both matched baselines at 151 and 450 tokens (151: 46:0, p ≈ 2.8×10⁻¹⁴; 450: 41:0, p ≈ 9.1×10⁻¹³), while at 1,646 and 4,051 the differences fall to p = 0.07–0.58.

### Table 6 — Same length, only the definition moves (n = 50)

| Condition | Distance (tok) | full | window=64 | gold_union=64 | random=64 |
|---|---:|---:|---:|---:|---:|
| definition at the start | ≈4,040 | 98% | **0%** | **18%** | 6% |
| definition next to the query | ≈46 | 98% | **90%** | **72%** | 6% |

Length, mask structure and total token count are unchanged; only the position of the definition moves.

**Conclusion 7**: the collapse at 4 k is caused by **distance**, not by "a long sequence under a sparse mask"; otherwise the two rows' `window=64` would be the same number.

(Paired evidence: `window=64` vs `random=64` is 45/50 against 3/50 in the near condition, discordant **42:0**, p ≈ 4.5×10⁻¹³; in the far condition it is 0/50 against 3/50 (0:3, p = 0.25).)

(A counter-intuitive detail, reported as a pairing: at distance 46, `window=64` gets 45/50 and `gold_union=64` 36/50; discordant counts are 12 only-window and 3 only-gold, exact two-sided McNemar **p = 0.035**, uncorrected. Two overlapping Wilson intervals are not evidence of "no difference" — the paired test is the right one. The direction is also counter-intuitive: `gold_union`'s visible set is a superset of the window's, yet it does worse. Extra global rows may dilute attention, or this may be small-sample noise; no conclusion is drawn, and the cell needs replication on a fresh 50 programs.)

### Table 7 — Window sweep at 4,051 tokens (n = 50)

| Pattern | Connections | Share of full | Accuracy |
|---|---:|---:|---:|
| full | 8,207,326 | 100% | 98% |
| gold_union=64 | 403,867 | 4.9% | 18% |
| gold_union=128 | 654,651 | 8.0% | 24% |
| gold_union=256 | 1,143,931 | 13.9% | 60% |
| gold_union=512 | 2,073,339 | 25.3% | 78% |

**Conclusion 8**: recovery needs a window of the same order as the distance. At 4 k, reaching 78% costs **25.3% of the connections**, and since masking saves no computation (see §5) that ratio has no practical value.

**Limitation of this table**: it sweeps one mask family (window width); accuracy rises monotonically with connections, which is clean. It **cannot** answer "what is evidence visibility worth", because `gold_union=64` and `window=64` use different connection counts (403,867 vs 257,248). That question is answered by Table 10.

### Table 8 — Deployable fixed anchors (n = 50)

Tables 4–7 are all "content-blind" masks except `gold_union`. This table adds the only deployable option that uses structural information: fixing particular **positions** as global (no labels needed).

| Distance (tok) | full | first 4 tokens | every line start | every 4th line start | `gold_union=64` |
|---:|---:|---:|---:|---:|---:|
| 73 | 98% | 98% | 98% | 98% | 98% |
| 151 | 100% | 6% | 8% | 6% | 100% |
| 450 | 100% | 12% | 12% | 12% | 100% |
| 1,646 | 100% | 12% | 14% | 14% | 22% |
| 4,051 | 98% | 6% | 6% | 6% | **18%** |

Connection comparison at 4,051 tokens: `every line start` uses **17.5%** of the connections for 6% accuracy; `gold_union=64` uses **4.9%** of the connections for 18%. **3.6× the connections, one third of the accuracy.**

**The accurate mechanism statement.** `linestart` makes the **first token of each line** global (the first token of the variable name); the value `7` of the definition line, about the tenth token of that line, is **not made global directly**. In principle the value can travel two hops (a later global line start can read all earlier tokens, and the readout can read that line start), yet the model's measured accuracy in these 50 programs is only 6–14%. So what can be said is: **when the value token is not in the directly visible set, accuracy is low under this condition**; it cannot be claimed that the model "cannot see the value at all". Likewise, the problem with the first-4-token variant is not that the value is invisible, but that the global positions precede the evidence and cannot carry later information backwards.

**Conclusion 9**: fixed structural anchors are not a deployable substitute for the completeness control. The line-start variant makes only the name's first token global; the first-4-token variant sits before the evidence and cannot relay forwards (§2.3). Turning the control's advantage into a deployable scheme requires **predicting which tokens carry the value** — the job of the Week-2 dependency analyser, not of any fixed structural rule.

### Table 9 — Paired comparisons (same 50 programs)

Conditions are **paired** (the same program under different masks), so an exact McNemar test is used rather than two independent Wilson intervals. Only comparisons the conclusions depend on are listed.

| Data | Comparison | Left only | Right only | p |
|---|---|---:|---:|---:|
| near (gap 46, 4,051 tok) | `window=64` vs `random=64` | 42 | 0 | 4.5×10⁻¹³ |
| near | `window=64` vs `gold_union=64` | 12 | 3 | **0.035** |
| near | `gold_union=64` vs `random=64` | 33 | 0 | 2.3×10⁻¹⁰ |
| far 4,051 (start) | `window=64` vs `gold_union=64` | 0 | 9 | 0.0039 |
| far 1,646 (start) | `window=64` vs `gold_union=64` | 2 | 10 | 0.039 |
| far 450 (start) | `window=64` vs `gold_union=64` | 0 | 42 | 4.5×10⁻¹³ |
| far 151 (start) | `window=64` vs `random=64` | 3 | 3 | 1.0 |
| far 450 (start) | `window=64` vs `random=64` | 8 | 1 | 0.039 |
| identity check | `full` vs `window=256` (151 tok) | 0 | 0 | 1.0 |
| sweep 4,051 | `gold_union=512` vs `gold_union=64` | 30 | 0 | 1.9×10⁻⁹ |
| sweep 4,051 | `gold_union=256` vs `gold_union=64` | 21 | 0 | 9.5×10⁻⁷ |
| **matched 151** | `gold_union=64` vs `window=100` | 46 | 0 | 2.8×10⁻¹⁴ |
| **matched 450** | `gold_union=64` vs `window=100` | 41 | 0 | 9.1×10⁻¹³ |
| **matched 1,646** | `gold_union=64` vs `window=100` | 8 | 2 | 0.11 |
| **matched 4,051** | `gold_union=64` vs `window=100` | 7 | 1 | 0.070 |
| **matched 151** | `gold_union=64` vs `random=100` | 48 | 0 | 7.1×10⁻¹⁵ |
| **matched 450** | `gold_union=64` vs `random=100` | 47 | 0 | 1.4×10⁻¹⁴ |
| **matched 1,646** | `gold_union=64` vs `random=100` | 10 | 5 | 0.30 |
| **matched 4,051** | `gold_union=64` vs `random=100` | 8 | 5 | 0.58 |

Three notes:

1. **About 20 paired comparisons were run, none corrected for multiplicity.** p = 0.035, 0.039 and 0.070 do not survive Bonferroni correction and are only leads to be replicated.
2. `full` vs `window=256` is identical at 151 tokens (0:0, p = 1) — the identity check that the mask is wired correctly.
3. At 450 tokens the `window=64` vs `random=64` significance moved from 0.34 to 0.039 when the random mask changed, which is itself a symptom of a small sample near the floor.

### Table 10 — Budget-matched control (`window=100` / `random=100` vs `gold_union=64`)

In the earlier tables `gold_union=64` has 30–57% more connections than `window=64`. This table equalises the budget: `window=100` and `random=100` are within 2% of `gold_union=64` at every distance.

| Distance (tok) | full | window=64 | **window=100** | **random=100** | **gold_union=64** | Connections (w100 / gold) |
|---:|---:|---:|---:|---:|---:|---|
| 73 | 98% | 68% | 98% | 98% | 98% | 2,701 / 2,701 |
| 151 | 100% | 6% | **8%** | **4%** | **100%** | 10,150 / 9,967 |
| 450 | 100% | 16% | **18%** | **6%** | **100%** | 40,050 / 40,166 |
| 1,646 | 100% | 6% | **10%** | **12%** | **22%** | 159,650 / 160,962 |
| 4,051 | 98% | 0% | **6%** | **12%** | **18%** | 400,150 / 403,867 |

Two regimes:

- **Up to 450 tokens, position is decisive even at equal budget.** At 151 and 450, `gold_union` beats both matched baselines 46:0 and 41:0 (p < 10⁻¹²); `window=100` adds 33% (7,648 → 10,150) and 50% (26,784 → 40,050) more connections and still only moves from 6% to 8–18%.
- **From 1,646 tokens on, the differences fall to noise.** 8:2 (p = 0.11) and 10:5 (p = 0.30) at 1,646; 7:1 (p = 0.070) and 8:5 (p = 0.58) at 4,051. At long distance "making the evidence global" shows **no demonstrable advantage over a matched-budget random mask**; 18% versus 12% cannot be separated at this sample size.

**Table 10's conclusion**: on the 4,051-token one-hop task the frozen model answers **9/50** with `gold_union=64` and **49/50** with full attention; `window=100` gives 3/50 and `random=100` 6/50. Even allowing direct access to all annotated evidence, this mask recovers only part of the ability.

**These 50 programs cannot decompose the "0% → 18%" step.** `gold_union` changes the connection count, the position and the global read/write pattern at the same time, and 9/3/6 are indistinguishable at n = 50 (p = 0.070 / 0.58). The report therefore does **not** claim that "most of the gain comes from connections"; what it can say is that **under a matched budget no demonstrable advantage of evidence visibility is observed at long distance**. Whether training-time adaptation can close the gap to full attention is the next question.

---

## 5. Conclusions and the cells that support them

| # | Conclusion | Supporting tables and cells |
|---|---|---|
| 1 | the measurement pipeline is trustworthy | Table 1: trivial 90%, leak 93% |
| 2 | the three-hop task has no ceiling | Table 1 real 15% (7–29%); Table 2, four lengths 7–30% |
| 3 | depth is the bottleneck, distance is not | Table 3: one hop 100% vs three hops 5–20% |
| 4 | distance itself does not damage the ability | Table 3 one-hop row; Table 4 `full` column 98–100% from 73 to 4,051 |
| 5 | up to 450 tokens position is decisive, **also at equal budget** | Table 4: 100% vs 6–16%; Table 10: 100% vs 8–18% (46:0, 41:0) |
| 6 | from 1,646 tokens on, evidence visibility shows **no demonstrable advantage**: matched-budget differences are not significant | Table 10: 22% vs 10–12% (p = 0.11 / 0.30); 18% vs 6–12% (p = 0.070 / 0.58) |
| 7 | **a mask without the evidence cannot recover**; at long distance both masks are at the floor and **indistinguishable**, so "shape does not matter" does not follow | Table 5 + Table 9: five cells p ≥ 0.25, both at 0–16% from 151 tokens on |
| 8 | in the near condition the window beats random at equal budget (42:0, p ≈ 4.5×10⁻¹³), **because it contains the evidence** | Table 6 + Table 9 |
| 9 | the failure is caused by distance | Table 6: same length, `window=64` 0% → 90%, `gold_union` 18% → 72% |
| 10 | recovery needs a window of the same order as the distance | Table 7 + Table 9: 512 vs 64 is 30:0 (p ≈ 1.9×10⁻⁹) |
| 11 | fixed structural anchors cannot replace the completeness control | Table 8: 17.5% connections → 6%; `gold_union=64` 4.9% → 18% (not budget-matched either) |
| 12 | **post-hoc sparse masking is not viable** | follows from 5 + 6 + 9: visibility is necessary but not sufficient |
| 13 | paired data require a paired test: the near `window` vs `gold_union` Wilson intervals overlap, yet the pairing is 12:3 (p = 0.035) | Table 6 + Table 9 (uncorrected, to be replicated) |
| 14 | **`gold_union` is not an accuracy upper bound**: 72% in the near condition, below the window's 90%; it guarantees completeness only | Table 6 + Table 9 |
| 15 | **the budget-matched control is done**: up to 450 tokens position remains decisive at equal budget (46:0, 41:0); from 1,646 tokens no demonstrable advantage, and 9/3/6 at 4,051 cannot decompose the "0% → 18%" step | Table 10 (`window=100` / `random=100` within 2% of `gold_union=64`) |

---

## 6. Threats to validity

| Threat | Effect | Handling |
|---|---|---|
| 50 programs per cell | differences below ±10 points are uninterpretable | Wilson intervals everywhere; no cross-distance claims |
| a single model scale (3B) | conclusions may hold only for this size | stated explicitly; 7B was not run because the lab home quota left 16 GB against 15.2 GB of weights |
| synthetic data | real code may have a different distance distribution | described as diagnostic data only; the real-code distance distribution is future work |
| `gold_union` uses gold annotations | not deployable | labelled "evidence-completeness control" everywhere, and explicitly **not an accuracy upper bound** |
| **the visibility comparison was not budget-matched** | `gold_union=64` has 30–57% more connections than `window=64` | **addressed**: Table 10 adds `window=100` / `random=100` (about 400,150 connections, within 2%); conclusions 5/6/15 rewritten |
| **`random=W` was not reproducible** | the seed came from `crc32(row["id"])`; adding the `start/end` suffix changed all 250 seeds, so a re-run produced a different random graph | fixed: the seed now comes from the program's `seed` field and the mask width, independent of the id. The `random` columns were re-run; `window`, `block`, `gold_union`, `local_global` and `full` are unaffected |
| no multiplicity correction | about 20 paired comparisons; p = 0.035 / 0.039 / 0.070 do not survive correction | stated in Table 9; these are leads, not conclusions |
| small paired samples | at n = 50, 15 discordant pairs put p near 0.03 | discordant counts and p are both listed |
| the two Week-2 selectors are indistinguishable on this data | 250 one-hop programs, three budgets: `structure_topk` and `evidence_groups` select **identical blocks** (250/250), so this week involves no selector comparison | this week measures information visibility, not selector quality; the 150/250 at 64 bytes is a **feasibility ratio** (at d100/d400 the remote evidence spans two 64-byte blocks and needs 128 bytes), and on feasible programs both are 100% complete |

---

## 7. Next steps

The two methodological patches left over from this week (the seed re-run and the budget-matched control) are done; the ordering below reflects that.

1. **Stop tuning mask parameters.** With frozen weights the long-distance loss cannot be repaired by making evidence visible, and the matched-budget control further shows that at long distance even "evidence visible" is not a measurable gain.
2. **Move to training-time adaptation.** LoRA-tune the 3B model while training with the `window=64` / `gold_union=64` masks, and test whether the 18% at 4 k can be raised above 80%.
3. **Measure `analyzer_union` instead of comparing the two selectors again.** Replace `gold_union`'s gold spans with the spans produced by the Week-2 `analyze()` under the same mask rule and budget, which quantifies the cost of an imperfect analyser. The two selectors pick identical blocks on this data (§6), so comparing them yields nothing; what carries information is **the gap between analyser spans and gold spans**.
4. **Measure the dependency-distance distribution of real code.** If almost all dependencies are ≤512 tokens, "window 512 + a few global anchors" is adequate in practice — a stronger claim than "no loss on a synthetic task".
5. **Optional mechanism probe.** Under full attention, measure the attention mass from the readout onto the gold tokens as a function of distance, to give direct evidence for "visible but unusable".

---

## 8. References

Only the four already in the literature review; nothing else is expanded.

[1] Vaswani, A., et al. *Attention Is All You Need.* NeurIPS 2017. arXiv:1706.03762

[2] Beltagy, I., Peters, M. E., Cohan, A. *Longformer: The Long-Document Transformer.* arXiv:2004.05150, 2020.

[3] Zaheer, M., et al. *Big Bird: Transformers for Longer Sequences.* NeurIPS 2020. arXiv:2007.14062

[4] Liu, T., et al. *Understanding Long Programming Languages with Structure-Aware Sparse Attention.* SIGIR 2022. arXiv:2205.13730

*`gold_union` and `block` have no corresponding paper: they are, respectively, this project's evidence-completeness control and its true cut-off control. Citations identify the origin of each attention pattern only; they are not an argument for the novelty of this week's method.*

---

## Appendix — reproduction

```bash
python lab_check.py                     # environment and the 4-D mask pass-through
python -m unittest -v test_week3        # 24 tests

python w3_gate.py  --model <3B> --style completion --limit 40
python w3_curve.py --model <3B> --lengths 96 160 256 512 --per 20
python w3_needle.py --distances 20 100 400 1600 --hops 1 2 3 --per 20
python w3_run.py --model <3B> --data-dir data --split needle_test --lengths 0 \
    --group-by family_id --modes full --no-reach --out results/needle_full

# the 250-program, five-distance set used by tables 4, 8 and 10
python w3_needle.py --distances 20 100 400 1600 4000 --hops 1 --per 50 --split needle_long
python w3_run.py --model <3B> --data-dir data --split needle_h1 --lengths 0 \
    --group-by family_id --no-reach \
    --modes full,window=64,window=256,gold_union=64,block=4,random=64 --out results/vis_h1
python w3_run.py --model <3B> --data-dir data --split needle_h1 --lengths 0 \
    --group-by family_id --no-reach \
    --modes full,local_global=64:first:4,local_global=64:linestart:1,local_global=64:linestart:4 \
    --out results/vis_localglobal
python w3_run.py --model <3B> --data-dir data --split needle_h1 --lengths 0 \
    --group-by family_id --no-reach \
    --modes full,window=64,window=100,random=100,gold_union=64 --out results/vis_matched

# same length, short distance
python w3_needle.py --distances 4000 --hops 1 --per 50 --position end --split needle_end
python w3_run.py --model <3B> --data-dir data --split needle_h1_end --lengths 0 \
    --group-by family_id --no-reach \
    --modes full,window=64,gold_union=64,random=64 --out results/vis_end

# window sweep
python w3_run.py --model <3B> --data-dir data --split needle_h1 --lengths 0 \
    --group-by family_id --no-reach \
    --modes full,gold_union=64,gold_union=128,gold_union=256,gold_union=512 \
    --out results/vis_goldwin

# paired tests (every p-value in Table 9)
python w3_paired.py --run results/vis_h1_final --left window=64 --right random=64
python w3_paired.py --run results/vis_h1_final --left window=64 --right gold_union=64
python w3_paired.py --run results/vis_end_4k   --left window=64 --right random=64
python w3_paired.py --run results/vis_end_4k   --left window=64 --right gold_union=64
python w3_paired.py --run results/vis_goldwin  --left gold_union=512 --right gold_union=64
python w3_paired.py --run results/vis_matched  --left gold_union=64 --right window=100
python w3_paired.py --run results/vis_matched  --left gold_union=64 --right random=100
```

**On reproducing `random=W`**: the seed is derived from the program's own `seed` field and the mask width, never from the id string. Earlier versions seeded from the id, so adding the `start/end` suffix changed all 250 seeds and those numbers could not be reproduced; the `random` figures in this report were re-run with the fixed code (`results/vis_h1_final`, `results/vis_end_4k`). `window`, `block`, `gold_union`, `local_global` and `full` are unaffected.

Per-program records are committed under `week3/results/`; the mapping from directory to table is in `week3/RESULTS.md`.
