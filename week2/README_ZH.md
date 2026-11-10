# 第 2 周：可重复运行的基线

四层因果 Transformer，从零训练，读入一段受限 Python 源码，预测最后变量 `result` 的整数值（0–9）。模型不执行代码，只在 UTF-8 字节序列上做分类。

- 模型：4 层、hidden 128、4 头、FFN 512、827,658 参数、末尾 CLS 读出
- 数据：1,200 训练 / 360 验证 / 540 测试（256/512/1,024 字节），SHA-256 见 `data/manifest.json`
- 记录：答案准确率、逐样例预测、CPU 耗时与进程内存；CUDA 同步计时与峰值显存代码路径已写入

## 安装

推荐 Python 3.12。本仓库结果来自 PyTorch 2.8.0+cpu。

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1             # Windows
# source .venv/bin/activate                # Linux / macOS

python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install "psutil>=5.9,<8"
```

GPU 用对应 CUDA 版本的 wheel；绘图需要额外 `pip install matplotlib`。

## 最短验证路线

在本目录（`week2/`）下执行，数据路径是相对当前目录的：

```bash
python -m unittest -v test_baseline
python audit_data.py
python evaluate.py --checkpoint <path-to-best.pt> --output rechecked
```

仓库里没有权重文件（13 MB，未纳入版本管理），最后一条需要一个已训练好的 `best.pt`。

## 完整重跑

```bash
python run_experiments.py --output runs --seeds 11 22 33 --epochs 8
```

三个种子全部训练完成后才打开测试集。已有输出目录会被拒绝覆盖。

## 第一张结果表

三种子均值 ± 种子间标准差，测试集每长度 180 条：

| 源码字节 | Full 准确率 | Window 推理消融 | 随机期望 |
|---:|---:|---:|---:|
| 256（训练长度） | **8.52%** ± 0.32 | 9.81% ± 1.40 | 10.00% |
| 512（未训练长度） | 9.63% ± 1.16 | 9.26% ± 1.28 | 10.00% |
| 1024（未训练长度） | 11.67% ± 1.47 | 9.07% ± 1.16 | 10.00% |

CPU 摊销耗时（batch 16）：256 / 512 / 1024 字节分别约 3.34 / 9.32 / 25.82 ms 每样例。逐种子原始记录由 `run_experiments.py` 写入 `runs/`。

**这张表说明的是：管线能跑通，但模型还没学会任务。** 训练集本身的准确率也只有 10%–15%，验证损失贴着 `ln(10)≈2.303`，属于欠拟合，不是过拟合。30 条样例能拟合到 100%，说明梯度、保存和加载都正常。

Window 一列是把 Full 权重在推理时换成 64 位置窗口遮罩，**没有重新训练**，只能用于诊断，不能当作公平的窗口模型成绩。

## 目录

| 文件 | 作用 |
|---|---|
| `dataset.py` | 数据生成器 + 受限 AST 解释器（只用来造/验标签） |
| `model.py` | 四层因果 Transformer、字节分词、遮罩 |
| `train.py` / `evaluate.py` / `evaluation.py` | 训练（只读 train/val）、一次性测试评估、计时 |
| `run_experiments.py` | 三个种子，全部训练完才评估 |
| `predict.py` | 用已保存权重预测一段代码 |
| `audit_data.py` | 全量数据审查（标签、去重、SHA-256、证据一致性） |
| `test_baseline.py` | 9 项正确性检查 |
| `selection/` | 依赖分析、块选择与模型输入适配器原型，见 `selection/README.md` |
| `data/` `examples/` | 数据与示例 |

## 已知限制

1. 数据是合成的诊断数据，不是 benchmark；24 个模板骨架，语言覆盖很小。
2. 模型接近随机水平，当前结果不能用于论证任何选择策略的优劣。
3. CPU 的耗时不能外推到 GPU；逻辑连接数不等于实际计算量，代码仍是稠密 SDPA。
