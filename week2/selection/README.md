# Week 2：代码依赖与证据组选取原型

Python 3.10+；仅使用标准库。详细中文说明见同目录 `Week2_Implementation_Guide_ZH.md`。

```bash
python -m unittest -v test_week2
python week2_mvp.py --samples 300 --output out
```

在Windows终端或Linux/macOS终端都可以运行。根据安装情况将 `python` 换成 `python3`。

## 实现内容

- `generate_sample()`：生成赋值链、函数调用、条件返回三种合成Python任务。
- `analyze()`：对明确受限的语法提取粗粒度依赖，映射到固定字节token块。
- `build_groups()`：为源码节点生成有界依赖闭包候选组。
- `select()`：词法Top-k、共享结构信息的Top-k、考虑新增成本的组选择。
- `evaluate()`：输出静态证据完整率、召回率、预算可行比例及分析/选择耗时。
- `model_adapter.pack_selected()`：仅将选中内容与原始位置送入后续模型。

默认每块64字节token，远端预算128/256/384；本地查询块始终可见。这里不是开源LLM的分词器。迁移到其他模型时重新分词与映射。

## 输出

`summary.csv` 是汇总；`per_sample.csv` 是逐样例记录；`example_trace.json` 展示一个样例的完整链路；`smoke_samples.jsonl` 保存代码及仅用于评价的标签。`config.json` 记录配置。

随包的 `smoke_results/` 是300样例实际运行结果。它只是通路验证数据，不作为模型训练/测试集或公开benchmark。10项自动检查通过；没有运行神经模型训练、GPU推理或稀疏Attention加速测试。

## 必须理解的限制

函数体只能是单个return或含两个return分支的if/else；函数整体是一个粗粒度节点。模块绑定不能重复，函数内部不能继续调用其他函数。循环、类、属性/索引、递归、别名等未支持，遇到时应报错。

gold证据是生成器独立记录的保守静态集合，包含两个分支；不是最小动态切片。选择器不接收gold、答案或执行结果。测试只执行本地生成器生成的固定语法程序来核对标签，分析器不执行传入源码。

组选择器是固定贪心启发式，不是训练好的路由器；词法Top-k不是任何稀疏Attention论文的正式复现。当前简单任务中结构Top-k与证据组持平，不能报告分组创新获得提升。先理解该结果，再扩展任务与模型实验。
