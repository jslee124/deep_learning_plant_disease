# 实验二 RNN 学习目录

主题：基于循环神经网络的 MNIST 手写数字分类。

这个实验的关键不是“MNIST 多难”，而是通过一个简单任务理解 RNN 的时间步、序列输入和隐藏状态。

## 学习路线

1. [01 项目任务与整体流程](01_task_overview.md)
2. [02 MNIST 数据集与固定划分](02_mnist_dataset.md)
3. [03 图片如何转换成序列](03_image_as_sequence.md)
4. [04 RNN 基础知识](04_rnn_basic_theory.md)
5. [05 MnistRNN 模型代码详解](05_model_code_explained.md)
6. [06 训练流程与梯度裁剪](06_training_process.md)
7. [07 隐藏维度对比实验结果分析](07_hidden_size_results.md)
8. [08 CNN 与 RNN 对比](08_cnn_vs_rnn.md)
9. [09 高频追问与自测题](09_questions_and_self_test.md)

## 最低掌握目标

学完后你至少要能讲清楚：

- MNIST 图片形状为什么是 `[1, 28, 28]`。
- 为什么可以把一张图片按行看成长度为 28 的序列。
- RNN 每个时间步输入什么。
- 隐藏状态是什么。
- 为什么使用最后一个时间步做分类。
- `hidden_size=128` 为什么比 `hidden_size=64` 准确率更高，但参数量也更多。
