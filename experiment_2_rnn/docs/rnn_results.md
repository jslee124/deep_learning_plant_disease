# 实验二结果：RNN 隐藏维度对比

## 控制变量

两组实验均使用 MNIST、固定随机种子 42、相同的 `54000/6000/10000`
训练/验证/测试划分、10 Epoch、Batch Size 128、Adam 和学习率 0.001。
唯一主动改变的变量是隐藏层维度。

## 实际结果

| 指标 | hidden=64 | hidden=128 |
| --- | ---: | ---: |
| 最佳 Epoch | 8 | 10 |
| 最佳验证 Accuracy | 94.18% | 96.88% |
| 独立测试 Accuracy | 94.34% | 97.07% |
| 最终训练 Accuracy | 94.35% | 96.81% |
| 测试 Loss | 0.2023 | 0.1124 |
| 参数量 | 6,666 | 21,514 |
| 总训练时间 | 201.01 秒 | 165.34 秒 |
| 平均每 Epoch | 19.90 秒 | 16.52 秒 |
| 吞吐量 | 2,686.38 样本/秒 | 3,266.03 样本/秒 |

隐藏维度 128 的测试准确率提高了 2.73 个百分点，测试 Loss 也更低，说明 64 维隐藏
状态在当前任务中容量偏小，而 128 维能够保存更丰富的笔画与行间结构信息。参数量增加
到 3.23 倍，但总量仍只有约 2.15 万，属于很小的教学模型。

本次 128 维实验的实测速度反而更快，但不能据此断言扩大隐藏维度会提升训练效率。
64 维实验是本次会话中第一次读取 MNIST 并运行 RNN，可能受到数据下载、磁盘缓存、
CUDA 预热和后台负载影响。严谨的效率结论需要预热后重复多次并报告均值和标准差。

## 泛化分析

64 维模型最终训练 Accuracy 为 94.35%，测试 Accuracy 为 94.34%，差距约 0.01 个
百分点。128 维模型测试 Accuracy 比最终训练 Accuracy 高 0.26 个百分点，这种很小的
负差距可由训练批次随机性和测试集难度差异造成，并不表示模型在未知数据上神奇地优于
训练数据。两组都没有出现明显过拟合。

## 生成产物

每组运行目录包含：

- `models/best_rnn.pth`：最佳验证模型。
- `models/latest_checkpoint.pth`：可恢复训练的最新检查点。
- `logs/final_result.json`：配置、硬件和最终指标。
- `logs/training_history.csv`：逐 Epoch 指标。
- `figures/loss_curve.png` 与 `acc_curve.png`。
- `figures/confusion_matrix.png`。
- `figures/prediction_examples.png`。

自动对比产物位于
`experiment_2_rnn/outputs/comparisons/rnn_hidden_64_baseline_vs_rnn_hidden_128_comparison/`。
