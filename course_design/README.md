# 深度学习期末课程设计

题目：

《基于卷积神经网络的植物病害识别及网络结构参数对比研究》

课程设计与平时实验分开管理。`src/` 已复制稳定 CNN 工程，后续修改不会影响平时实验
提交版。当前研究复用三组已经按相同规范运行的正式结果，并在 `results/` 独立生成
课程设计汇总。

## 已完成

- `3x3` 与 `5x5` 卷积核对比。
- 3 个与 4 个卷积块对比。
- 三模型准确率、参数量、时间、吞吐量和泛化差距汇总。
- 自建 CNN 与 ImageNet 预训练 ResNet18 全层微调对比。
- 四模型准确率、Loss、参数量、时间、吞吐量和泛化差距汇总。
- 课程设计研究设计和报告初稿。

## 主要结果

| 模型 | 测试 Accuracy | 参数量 |
| --- | ---: | ---: |
| 3 blocks, 3x3 | 86.89% | 115,110 |
| 4 blocks, 3x3 | 94.02% | 427,174 |
| 4 blocks, 5x5 | 95.05% | 1,116,838 |
| ResNet18, ImageNet fine-tune | 98.12% | 11,196,006 |

## 生成汇总

```powershell
.\venv\Scripts\python.exe course_design\src\summarize_research.py
```

ResNet18 正式实验使用 10 Epoch、Batch Size 32、Adam、学习率 0.001，并微调全部
网络参数。其测试 Accuracy 比自建 `4 blocks, 3x3` 提高 4.10 个百分点。
