# week3 — 冻结 Qwen 的可见性实验（可运行脚手架）

配套说明见 [Week3_Implementation_Guide_ZH.md](Week3_Implementation_Guide_ZH.md)；**换到实验室机器上跑**见 [LAB_SETUP_ZH.md](LAB_SETUP_ZH.md)。这里只讲怎么跑。

本目录里的代码**不训练任何模型**。它把第 1 周的数据、第 2 周的依赖分析、Qwen 的真实 tokenizer 和一个冻结的 Qwen 接在一起，测四件事：

1. 读出口是否可用（门禁）；
2. 任务在哪个体量上有天花板（深度 × 距离）；
3. 模型是否真的依赖远处的证据（反事实 / 删除）；
4. 改变注意力可见范围后还剩多少能力（full / window / local-global / block / gold-union）。

## 在新机器上先跑这两行

```bash
python lab_check.py --model <权重目录>    # 环境 + 数据 + 4D mask 透传的功能验证
bash run_all.sh                            # 上面四件事，全程约 45–70 分钟
```

## 不需要模型权重就能跑

```bash
cd week3
python -m unittest -v test_week3        # 22 项：tokenizer、span→token、mask、可达性、统计
python w3_preflight.py --lengths 256 1024 --make-length 4096 --sample 8 --layers 28 36
```

`w3_preflight.py` 会写 `preflight/preflight.md`：程序在 Qwen token 下有多长、证据离读出口多远、每种 mask 的连接数与可达率。**先看这张表再决定窗口大小**，否则很容易在 256 B 上跑一个"其实什么都没切断"的窗口。

tokenizer 已经放在 `_ref/tokenizer/`，离线可用。

## 需要一个 Qwen 权重

以下命令把 `<模型>` 换成 HF 名字（需要联网）或本地目录。

```bash
# 0. 门禁：读口 + 简单算术 + 抄答案
python w3_gate.py --model <模型> --style completion --limit 40

# 1. 天花板曲线
python w3_curve.py  --model <模型> --lengths 96 160 256 512 1024
python w3_needle.py --distances 20 100 400 1600 --hops 1 2 3 --per 20
python w3_run.py    --model <模型> --data-dir data --split needle_h1 --lengths 0 \
                    --group-by family_id --modes full --out results/needle_<tag>

# 3. 可见性对比
python w3_run.py --model <模型> --data-dir data --split needle_h1 --lengths 0 \
  --group-by family_id --modes full,window=64,window=256,gold_union=64,block=4 \
  --out results/vis_<tag>
```

每个 `--out` 目录里有 `per_item.csv`（逐条）、`summary.csv`（按组）、`meta.json`（配置与耗时）。

## 已经跑过的验证（`results/`）

| 文件 | 内容 |
|---|---|
| `results/gate.json`, `gate_15b.json` | 0.5B / 1.5B 门禁：trivial 95%/100%，leak 100%/92%，real 5%/5% |
| `results/curve_15b.csv` | 1.5B 在 3 跳任务上 96–512 B 全部 ≈10% |
| `results/needle_15b/` | 1.5B，full attention：1 跳 90–95%（到 2,816 token），2 跳 5–40%，3 跳 5–10% |
| `results/needle_masks_15b/` | 同一批 1 跳数据上的 full / window=64 / window=256 / gold_union=64 / block=4 |

这些是管线验证，不是论文结论：每格 20 条。

## 文件

| 文件 | 作用 |
|---|---|
| `Week3_Experiment_Report_ZH.md` | 本周实验报告：门禁、天花板、可见性、紧邻对照、窗口扫描 |
| `lab_check.py` | 新机器自检：环境、显存、必需文件、4D mask 是否真的生效 |
| `run_all.sh` | 一条命令跑完 0–6 步，门禁不过就停 |
| `LAB_SETUP_ZH.md` | 实验室机器上的完整步骤、显存/时间预算、排错表、Slurm 模板 |
| `w3_core.py` `w3_run.py` | 核心：数据/tokenizer/prompt/mask/可达性；冻结模型评测 |
| `w3_gate.py` `w3_curve.py` `w3_needle.py` `w3_preflight.py` | 门禁、天花板曲线、距离可控数据、无权重预检 |

## 本地权重

`_ref/qwen05b/`（约 1 GB）和 `_ref/qwen15b/`（约 3 GB）只用于跑通管线（`--random-model` 连权重都不需要）。不想留 4 GB 的话直接删掉，命令里换成你自己的模型路径即可。

## 已知坑

- `"# Output: "` 末尾的空格是**独立 token**（id 25），必须把它放进输入再在下一个位置读 10 个数字，否则读的是 `"7"` 而不是 `" 7"`。
- 窗口不是信息切断：叠 L 层后感受野是 `1 + L×(W−1)`。256 B ≈ 203 token，W=64 时 RF=1765，什么都切不断。要看效果至少用 4 KB。
- 4D bool mask 直接传给 `model(attention_mask=...)`，`True` = 可参与。本版本 transformers 会原样透传给 SDPA（`is_causal=False`）。
- mask **不会**加速：稠密 SDPA 下只是把部分权重压到 0。要加速得换执行（块稀疏内核）或减 token。
- 平台是 AMD ROCm（RX 7900 XTX 24 GB）：没有 flash-attn；要看注意力权重必须 `--attn eager`。
