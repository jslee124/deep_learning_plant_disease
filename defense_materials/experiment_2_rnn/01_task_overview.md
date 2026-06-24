# 01 项目任务与整体流程

## 1. 这个实验到底做什么

这个实验用基础 RNN 对 MNIST 手写数字图片进行分类。
对应文件：`experiment_2_rnn/src/models.py`

对应代码片段：`MnistRNN` 先把图片变序列，再用 RNN 和线性层输出 10 类 logits。

```python
sequence = images.squeeze(1)
outputs, _ = self.rnn(sequence)
final_output = outputs[:, -1, :]
return self.classifier(final_output)
```

输入是一张手写数字图片。

输出是数字类别：

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

也就是 10 分类任务。

## 2. 一句话解释给老师听

> 本实验把每张 `28x28` MNIST 图片按行看作长度为 28 的序列，每个时间步输入一行 28 个像素，使用基础 RNN 完成 0 到 9 十分类，并比较隐藏层维度 64 和 128 对准确率和参数量的影响。

这句话有四个重点：

- 图片按行转序列。
- 每个时间步输入一行。
- 使用基础 RNN。
- 对比隐藏维度。

## 3. 为什么这个实验不是为了证明 RNN 最适合图片

MNIST 是图片任务，CNN 通常更适合。

但是本实验的目的不是证明 RNN 比 CNN 好，而是学习 RNN：

- 什么是序列。
- 什么是时间步。
- 什么是隐藏状态。
- 隐藏维度如何影响模型能力。

所以把图片按行转成序列，是一种教学设计。

## 4. 整体流程

```text
MNIST 图片
  ↓
ToTensor + Normalize
  ↓
[B, 1, 28, 28]
  ↓
squeeze(1)
  ↓
[B, 28, 28]
  ↓
RNN 逐行读取
  ↓
输出每个时间步隐藏表示 [B, 28, hidden_size]
  ↓
取最后时间步 [B, hidden_size]
  ↓
Linear 分类层
  ↓
[B, 10] 类别得分
  ↓
CrossEntropyLoss 训练
  ↓
验证集保存最佳模型
  ↓
测试集最终评估
```

## 5. 项目核心文件

```text
experiment_2_rnn/src/dataset.py
```

下载和读取 MNIST，划分训练集、验证集、测试集。

```text
experiment_2_rnn/src/models.py
```

定义 `MnistRNN`。

```text
experiment_2_rnn/src/train.py
```

完整训练入口。

```text
experiment_2_rnn/src/compare_runs.py
```

比较 `hidden=64` 与 `hidden=128`。

```text
experiment_2_rnn/src/utils.py
```

保存日志、画曲线、画混淆矩阵和预测样例。

## 6. 本章你要记住

- 实验二是 RNN 教学实验。
- 图片被按行转成序列。
- 序列长度是 28。
- 每个时间步输入 28 个像素。
- 输出是 10 个数字类别得分。

