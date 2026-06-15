# 实验二：基于循环神经网络的 MNIST 手写数字分类

本实验把每张 `28x28` MNIST 图片看作 28 个时间步的序列，每个时间步输入一行
28 个像素，使用 PyTorch 自建基础 RNN 完成十分类。

```text
experiment_2_rnn/
├── src/
├── docs/
├── outputs/
├── requirements.txt
└── README.md
```

正式对比只改变隐藏层维度 `64` 与 `128`，其余数据划分、随机种子、Epoch、
Batch Size、优化器和学习率完全一致。

## 运行

```powershell
.\venv\Scripts\python.exe experiment_2_rnn\src\train.py `
  --experiment-name rnn_hidden_64_baseline --hidden-size 64

.\venv\Scripts\python.exe experiment_2_rnn\src\train.py `
  --experiment-name rnn_hidden_128_comparison --hidden-size 128 --no-download

.\venv\Scripts\python.exe experiment_2_rnn\src\compare_runs.py `
  rnn_hidden_64_baseline rnn_hidden_128_comparison
```

每组实验独立保存最佳模型、检查点、训练历史、Loss/Accuracy 曲线、混淆矩阵、
预测示例、硬件信息和最终指标。

## 正式结果

| 指标 | hidden=64 | hidden=128 |
| --- | ---: | ---: |
| 最佳验证 Accuracy | 94.18% | 96.88% |
| 独立测试 Accuracy | 94.34% | 97.07% |
| 参数量 | 6,666 | 21,514 |
| 总训练时间 | 201.01 秒 | 165.34 秒 |

128 维模型的测试准确率提高 2.73 个百分点。完整解释见
`docs/rnn_results.md`；计时受到首次运行缓存和 CUDA 预热影响，不把第二组更快解释为
隐藏维度越大训练越快。
