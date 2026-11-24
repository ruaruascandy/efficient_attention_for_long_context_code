# 在实验室机器上跑第 3 周实验

目标机器：Linux + NVIDIA GPU + 能联网（或能拷权重）。全部流程**不训练模型**，只做前向。

按顺序做五件事：拷文件 → 建环境 → 下权重 → 自检 → 一条命令跑完。最后是排错表和"带什么回来"。

---

## 1. 拷什么

最小集合约 **14 MB**（不需要那 4 GB 的本地小模型）：

```
week3/                       # 除了 _ref/qwen05b、_ref/qwen15b
  _ref/tokenizer/            # 11 MB，离线可用，必须带
  *.py  *.sh  *.md
week1/dataset.py             # 标签解释器，w3_needle / 反事实要用
week1/data/test.jsonl        # 540 条
week1/data/validation.jsonl  # 360 条
```

`preflight/preview_l4096.jsonl` 可以不带（`w3_preflight.py` 会重新生成），`results/` 也可以不带。

```bash
# 在源机器上打包
tar czf week3_lab.tgz week3 week1/dataset.py week1/data/test.jsonl week1/data/validation.jsonl \
    --exclude='week3/_ref/qwen05b' --exclude='week3/_ref/qwen15b' \
    --exclude='week3/__pycache__' --exclude='week3/results'

# 拷过去
scp week3_lab.tgz you@lab-gpu:~/
ssh you@lab-gpu
mkdir -p ~/proj && cd ~/proj && tar xzf ~/week3_lab.tgz && cd week3
```

目录结构必须是 `proj/week1/...` 和 `proj/week3/...` 并列——脚本用相对路径找 `week1`。

---

## 2. 建环境

Python 3.9+ 就够（我在 3.14 上跑过，3.10/3.11 也行）。

```bash
conda create -n week3 python=3.11 -y && conda activate week3
pip install torch --index-url https://download.pytorch.org/whl/cu124   # 按驱动选 cu121/cu124
pip install "transformers==4.57.1" "tokenizers>=0.22" numpy
```

版本固定很重要：这套代码依赖 **transformers 4.57.x** 把 4D bool mask 原样透传给 SDPA 的行为。别的版本可能改写 mask。`lab_check.py` 会用一个随机初始化的小 Qwen2 做功能验证，不靠版本号猜——它说 PASS 就是真的能用。

不需要 flash-attn、不需要 Apex、不需要 `pip install -e .`。

---

## 3. 权重

7B 约 **15.2 GB**，放本地目录最省事：

```bash
# 方式 A：能直连 Hugging Face
pip install -U huggingface_hub
HF_HOME=/data/hf hf download Qwen/Qwen2.5-Coder-7B-Instruct

# 方式 B：走镜像
HF_ENDPOINT=https://hf-mirror.com HF_HOME=/data/hf hf download Qwen/Qwen2.5-Coder-7B-Instruct

# 方式 C：ModelScope
pip install modelscope
modelscope download --model Qwen/Qwen2.5-Coder-7B-Instruct --local_dir /data/models/qwen25coder7b
```

拿到目录里应有 `config.json`、`model.safetensors`（或分片）、`tokenizer*.json`。

大小选择：

| 显存 | 用哪个 | 备注 |
|---|---|---|
| 16 GB | `Qwen2.5-Coder-3B-Instruct` + `--dtype float16` | 7B bf16 装不下 |
| 24 GB | 7B bf16 | 4 KB 程序（3.7k token）可以跑 |
| 40/80 GB | 7B，可上 8 KB | 仍然建议 batch=1 |

老卡（V100/2080Ti，sm_70/75）没有 bf16，用 `--dtype float16`；`lab_check.py` 会提示。

---

## 4. 自检

```bash
export MODEL=/data/models/qwen25coder7b
python lab_check.py --model "$MODEL"
python -m unittest -v test_week3        # 22 项，不需要权重
```

`lab_check` 会打印 python/torch/transformers、GPU 与显存、必需文件、tokenizer、**4D mask 透传是否真的生效**，最后一行是 `Environment OK`。有 `[FAIL]` 就先按下面排错，别急着下 15 GB。

---

## 5. 一条命令跑完

```bash
bash run_all.sh
```

或者分步来（`run_all.sh` 里就是这些，每步都会 tee 到 `results/$TAG/`）：

| 步骤 | 命令 | 预计（24 GB 卡、7B） |
|---|---|---|
| 0 自检 | `python lab_check.py --model $MODEL` | 10 s |
| 0 测试 | `python -m unittest -v test_week3` | 15 s |
| 0 预检（无需权重） | `python w3_preflight.py --lengths 256 1024 --make-length 4096 --sample 8` | 1–2 min |
| 1 门禁 | `python w3_gate.py --model $MODEL --style completion --limit 40` | 1–2 min |
| 2 天花板 | `python w3_curve.py --model $MODEL --lengths 96 160 256 512 --per 20` | 3–5 min |
| 3 深度×距离 | `python w3_needle.py --distances 20 100 400 1600 --hops 1 2 3 --per 20` 然后 `python w3_run.py --model $MODEL --data-dir data --split needle_test --lengths 0 --group-by family_id --modes full --no-reach --out results/needle_full` | 8–12 min |
| 4 可见性 | `python w3_run.py --model $MODEL --data-dir data --split needle_h1 --lengths 0 --group-by family_id --modes full,window=64,window=256,gold_union=64,block=4 --out results/vis_h1` | 10–15 min |
| 5 证据依赖 | `--ablate gold-value` 与 `--ablate gold-drop` 各一遍（256 B，180 条） | 各 2–3 min |
| 6 4 KB | `python w3_run.py ... --data-dir preflight --split preview_l4096 --lengths 4096 --limit 60 --modes full,window=64,gold_union=64` | 8–12 min |

`bash run_all.sh` 全程约 **45–70 分钟**；`SKIP_CURVE=1 SKIP_4K=1 bash run_all.sh` 约 25 分钟。

门禁不通过时 `run_all.sh` 会 **停在第 1 步**（退出码 2），不会浪费后面的时间。

---

## 6. 跑完应该看到什么

这些是我在 1.5B 上实测的量级，用来判断"是不是跑对了"，不是 7B 的结论：

| 检查 | 期望 |
|---|---|
| `gate` 的 trivial / leak | 都 ≥ 0.8；7B 应该接近 1.0 |
| `needle_full` 里 1 跳、4 个距离 | 80–95%，且**不随距离下降** |
| `needle_full` 里 3 跳 | 明显更低；若 7B 也 ≤ 20%，主表切到 1 跳 |
| `vis_h1` 的 `window=64` 对 `full` | 距离 ≥ 400 token 时大幅下降 |
| `vis_h1` 的 `gold_union=64` | 明显高于 `window=64`，但通常**低于** `full` |
| `vis_h1` 的 `block=4` | 掉到 ~10% |
| `full` 与 `window=256`（序列 < 256 时） | 逐条一致——这是 mask 接通的证明 |

如果 `window=64` 在任何距离上都没掉，八成是 mask 没生效：回看 `lab_check` 的第 9 行和 `test_explicit_full_mask_equals_no_mask`。

---

## 7. 排错表

| 症状 | 原因 | 处理 |
|---|---|---|
| `gate` 里 leak < 0.8 | prompt/读口不对，不是模型不行 | 确认用的是 `--style completion`；确认把 `Ġ`(id 25) 放进输入再读数字；先跑 `--limit 4` 打印一次 top-5 token 看看 |
| `[FAIL] 4D mask passthrough` | transformers 版本改写了 mask | `pip install transformers==4.57.1`；装不上就改用 `--attn eager`（eager 走 `masked_fill`，对版本不敏感，但 4 KB 以上更吃显存） |
| `CUDA out of memory`（4 KB） | 7B 权重 + KV + 激活 | 换 3B；或 `--dtype float16`；或 `--limit 30`；mask 本身只有几 MB，不是主因 |
| `StopIteration` in `make_sample` | 第 1 周生成器名字池只有 256 个 | 不要用 `make_sample` 造 4 KB；用 `w3_preflight.py --make-length 4096`（`lengthen_program`） |
| `FileNotFoundError: week1/data/...` | 只拷了 `week3/` | 补拷 `week1/dataset.py` 与 `week1/data/*.jsonl`，或给 `--data-dir` |
| `Can't load tokenizer` | `_ref/tokenizer` 没带 | 带上 11 MB 的那个目录，或设 `--model` 指向含 tokenizer 的权重目录后改代码里的 `TOK_DIR` |
| 慢得离谱（>10 s/条） | 跑在 CPU 上 | `export CUDA_VISIBLE_DEVICES=0`，确认 `torch.cuda.is_available()`；`lab_check` 第 3 行会显示设备 |
| 多卡机器只用了 1 张 | 脚本是单卡 | 正常。要并行就按 `--out` 分目录、`CUDA_VISIBLE_DEVICES` 分卡跑不同 split |

---

## 8. 集群（Slurm）

`run_all.sh` 不假设交互式环境，直接丢给 sbatch：

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

`TAG=${SLURM_JOB_ID} bash run_all.sh` 可以把输出目录和作业号对齐。

---

## 9. 带什么回来

```bash
tar czf week3_results.tgz results/*/summary.csv results/*/meta.json results/*/*.log \
    preflight/preflight.md data/needle_test.jsonl
```

通常几 MB。`per_item.csv` 只在需要复核逐条预测时才带（每个 run 几十到几百 KB）。

拿到结果后看三张表：`gate.json`（管线）、`needle_full/summary.csv`（天花板）、`vis_h1/summary.csv`（可见性）。这三张表决定第 4 周是继续做 mask，还是回去改任务。
