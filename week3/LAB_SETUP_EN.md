# Running the Week-3 experiments on a lab machine

Target machine: Linux + NVIDIA GPU, with network access (or a way to copy the weights). The whole pipeline **trains nothing** — it is forward passes only.

Five steps, in order: copy the files → build the environment → get the weights → self-check → one command to run everything. The troubleshooting table and "what to bring back" are at the end.

---

## 1. What to copy

The minimal set is about **14 MB** (the 4 GB of local small models is not needed):

```
week3/                       # except _ref/qwen05b and _ref/qwen15b
  _ref/tokenizer/            # 11 MB, works offline, must be copied
  *.py  *.sh  *.md
week1/dataset.py             # the label interpreter, used by w3_needle / counterfactuals
week1/data/test.jsonl        # 540 programs
week1/data/validation.jsonl  # 360 programs
```

`preflight/preview_l4096.jsonl` can be left out (`w3_preflight.py` regenerates it), and so can `results/`.

```bash
# on the source machine
tar czf week3_lab.tgz week3 week1/dataset.py week1/data/test.jsonl week1/data/validation.jsonl \
    --exclude='week3/_ref/qwen05b' --exclude='week3/_ref/qwen15b' \
    --exclude='week3/__pycache__' --exclude='week3/results'

# copy it over
scp week3_lab.tgz you@lab-gpu:~/
ssh you@lab-gpu
mkdir -p ~/proj && cd ~/proj && tar xzf ~/week3_lab.tgz && cd week3
```

The directory layout must have `proj/week1/...` and `proj/week3/...` side by side — the scripts find `week1` through a relative path.

---

## 2. Build the environment

Python 3.9+ is enough (I have run it on 3.14; 3.10/3.11 are fine too).

```bash
conda create -n week3 python=3.11 -y && conda activate week3
pip install torch --index-url https://download.pytorch.org/whl/cu124   # cu121/cu124 depending on the driver
pip install "transformers==4.57.1" "tokenizers>=0.22" numpy
```

Pinning the version matters: this code relies on **transformers 4.57.x** passing a 4-D bool mask straight through to SDPA. Other versions may rewrite the mask. `lab_check.py` verifies this functionally with a randomly initialised small Qwen2, so you do not have to trust the version number — if it says PASS, it really works.

You do not need flash-attn, Apex, or `pip install -e .`.

---

## 3. Weights

7B is about **15.2 GB**; a local directory is simplest:

```bash
# option A: direct Hugging Face access
pip install -U huggingface_hub
HF_HOME=/data/hf hf download Qwen/Qwen2.5-Coder-7B-Instruct

# option B: through a mirror
HF_ENDPOINT=https://hf-mirror.com HF_HOME=/data/hf hf download Qwen/Qwen2.5-Coder-7B-Instruct

# option C: ModelScope
pip install modelscope
modelscope download --model Qwen/Qwen2.5-Coder-7B-Instruct --local_dir /data/models/qwen25coder7b
```

The directory should contain `config.json`, `model.safetensors` (or shards) and `tokenizer*.json`.

Choosing by memory:

| VRAM | Use | Note |
|---|---|---|
| 16 GB | `Qwen2.5-Coder-3B-Instruct` + `--dtype float16` | 7B in bf16 does not fit |
| 24 GB | 7B bf16 | 4 KB programs (3.7k tokens) run fine |
| 40/80 GB | 7B, 8 KB possible | still recommend batch=1 |

Older cards (V100/2080Ti, sm_70/75) have no bf16 — use `--dtype float16`; `lab_check.py` will tell you.

---

## 4. Self-check

```bash
export MODEL=/data/models/qwen25coder7b
python lab_check.py --model "$MODEL"
python -m unittest -v test_week3        # 24 tests, no weights needed
```

`lab_check` prints python/torch/transformers, the GPU and its memory, the required files, the tokenizer, and **whether the 4-D mask really passes through**; the last line is `Environment OK`. If anything shows `[FAIL]`, fix it with the table below before downloading 15 GB.

---

## 5. One command to run everything

```bash
bash run_all.sh
```

Or step by step (`run_all.sh` does exactly this, and every step tees into `results/$TAG/`):

| Step | Command | Estimate (24 GB card, 7B) |
|---|---|---|
| 0 self-check | `python lab_check.py --model $MODEL` | 10 s |
| 0 tests | `python -m unittest -v test_week3` | 15 s |
| 0 preflight (no weights) | `python w3_preflight.py --lengths 256 1024 --make-length 4096 --sample 8` | 1–2 min |
| 1 gate | `python w3_gate.py --model $MODEL --style completion --limit 40` | 1–2 min |
| 2 ceiling | `python w3_curve.py --model $MODEL --lengths 96 160 256 512 --per 20` | 3–5 min |
| 3 depth × distance | read the committed `data/needle_test.jsonl`, then `python w3_run.py --model $MODEL --data-dir data --split needle_test --lengths 0 --group-by family_id --modes full --no-reach --out results/needle_full` | 8–12 min |
| 4 visibility | `python w3_run.py --model $MODEL --data-dir data --split needle_h1 --lengths 0 --group-by family_id --modes full,window=64,window=256,gold_union=64,block=4,random=64 --out results/vis_h1` | 10–15 min |
| 5 evidence dependence | `--ablate gold-value` and `--ablate gold-drop` once each (256 B, 180 programs) | 2–3 min each |
| 6 4 KB | `python w3_run.py ... --data-dir preflight --split preview_l4096 --lengths 4096 --limit 60 --modes full,window=64,gold_union=64` | 8–12 min |

`bash run_all.sh` takes about **45–70 minutes** in total; `SKIP_CURVE=1 SKIP_4K=1 bash run_all.sh` about 25 minutes.

If the gate fails, `run_all.sh` **stops at step 1** (exit code 2) rather than wasting the rest of the time.

---

## 6. What you should see afterwards

These are the magnitudes I measured on 1.5B, for judging "did it run correctly" — they are not 7B conclusions:

| Check | Expectation |
|---|---|
| `gate` trivial / leak | both ≥ 0.8; 7B should be near 1.0 |
| `needle_full`, one hop, four distances | 80–95%, and **not falling with distance** |
| `needle_full`, three hops | clearly lower; if 7B is also ≤ 20%, move the main table to one hop |
| `vis_h1`, `window=64` vs `full` | a large drop from 400 tokens on |
| `vis_h1`, `gold_union=64` | clearly above `window=64`, but usually **below** `full` |
| `vis_h1`, `block=4` | down to ~10% |
| `full` vs `window=256` (sequence < 256) | identical item by item — proof that the mask is wired |

If `window=64` does not drop at any distance, the mask is almost certainly not taking effect: look at line 9 of `lab_check` and at `test_explicit_full_mask_equals_no_mask`.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `gate` leak < 0.8 | the prompt/readout is wrong, not the model | make sure `--style completion` is used; make sure the space token (`Ġ`, id 25) is fed in before reading digits; run `--limit 4` and print the top-5 tokens once |
| `[FAIL] 4D mask passthrough` | the transformers version rewrote the mask | `pip install transformers==4.57.1`; if that is impossible, use `--attn eager` (eager goes through `masked_fill` and is version-insensitive, but uses more memory above 4 KB) |
| `CUDA out of memory` (4 KB) | 7B weights + KV + activations | switch to 3B, or `--dtype float16`, or `--limit 30`; the mask itself is a few MB and is not the cause |
| `StopIteration` in `make_sample` | the Week-1 name pool has only 256 entries | do not use `make_sample` for 4 KB; use `w3_preflight.py --make-length 4096` (`lengthen_program`) |
| `FileNotFoundError: week1/data/...` | only `week3/` was copied | copy `week1/dataset.py` and `week1/data/*.jsonl`, or pass `--data-dir` |
| `Can't load tokenizer` | `_ref/tokenizer` was not copied | copy that 11 MB directory, or point `--model` at a weights directory with a tokenizer and change `TOK_DIR` in the code |
| Absurdly slow (>10 s/program) | running on CPU | `export CUDA_VISIBLE_DEVICES=0` and check `torch.cuda.is_available()`; line 3 of `lab_check` shows the device |
| A multi-GPU machine uses only one | the scripts are single-GPU | expected. To parallelise, split by `--out` directory and `CUDA_VISIBLE_DEVICES` across different splits |

---

## 8. Cluster (Slurm)

`run_all.sh` assumes no interactive session; hand it to sbatch directly:

```bash
#!/bin/bash
#SBATCH --job-name=week3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_%j.log
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate week3
export MODEL=/data/models/qwen25coder7b
export CUDA_VISIBLE_DEVICES=0
bash run_all.sh
```

`TAG=${SLURM_JOB_ID} bash run_all.sh` aligns the output directory with the job id.

---

## 9. What to bring back

```bash
tar czf week3_results.tgz results/*/summary.csv results/*/meta.json results/*/*.log \
    preflight/preflight.md data/needle_test.jsonl
```

Usually a few MB. `per_item.csv` is only needed when a per-item re-check is required (tens to hundreds of KB per run).

Once the results are back, read three tables: `gate.json` (the pipeline), `needle_full/summary.csv` (the ceiling) and `vis_h1/summary.csv` (visibility). Those three decide whether Week 4 keeps working on masks or goes back and changes the task.
