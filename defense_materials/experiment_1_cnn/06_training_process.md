# 06 训练流程与实验规范

## 1. 训练代码位置

```text
experiment_1_cnn/src/train.py
```

这个文件是实验主入口。

它完成：

- 读取参数。
- 构建数据。
- 创建模型。
- 训练和验证。
- 保存最佳模型。
- 测试集评估。
- 保存结果图表和日志。

## 2. 命令行参数为什么重要

代码支持：

```bash
python experiment_1_cnn/src/train.py --experiment-name cnn_kernel_3x3_baseline_v2 --kernel-size 3
python experiment_1_cnn/src/train.py --experiment-name cnn_kernel_5x5_comparison --kernel-size 5
```

这样不用改代码就能做不同实验。

实验名称会创建独立目录，避免结果互相覆盖。

## 3. 一个 Epoch 是什么

一个 Epoch 表示模型完整看完一遍训练集。

如果训练集有 70,295 张图片，Batch Size 是 32，那么一个 Epoch 大约有：

```text
70295 / 32 ≈ 2197 个 batch
```

模型会在每个 batch 上更新一次参数。

## 4. 一个 batch 的训练过程

代码逻辑：

```python
optimizer.zero_grad()
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

逐步解释：

### optimizer.zero_grad()

清空上一批数据留下的梯度。

PyTorch 默认梯度会累加，所以每个 batch 前要清空。

### logits = model(images)

前向传播。

输入图片，得到每个类别的预测得分。

### loss = criterion(logits, labels)

用损失函数衡量预测和真实标签的差距。

### loss.backward()

反向传播。

计算每个参数对 loss 的影响，也就是梯度。

### optimizer.step()

优化器根据梯度更新参数。

## 5. CrossEntropyLoss

本实验是多分类任务，一张图片只有一个正确类别。

所以使用：

```python
nn.CrossEntropyLoss()
```

它接收：

```text
logits: [B, 38]
labels: [B]
```

其中 labels 是类别编号。

注意：

`CrossEntropyLoss` 内部已经包含 softmax，所以训练时不需要手动 softmax。

## 6. Adam 优化器

Adam 根据梯度更新参数，同时会自适应调整每个参数的学习步长。

本实验默认：

```text
learning_rate = 0.001
optimizer = Adam
```

作为课程实验，Adam 通常比较稳定，适合做基线。

## 7. 训练和评估模式的区别

训练时：

```python
model.train()
```

验证/测试时：

```python
model.eval()
```

区别：

- Dropout 训练时开启，评估时关闭。
- BatchNorm 训练时用当前 batch 统计，评估时用累计统计。

验证/测试还会使用：

```python
with torch.no_grad():
```

表示不计算梯度，节省显存和计算。

## 8. 为什么保存最佳模型

代码逻辑：

```python
if val_acc > best_val_acc:
    torch.save(checkpoint, config.best_model_path)
```

原因：

最后一轮不一定最好。

你的 `3x3` 基线中，第 9 轮验证准确率最高，第 10 轮反而下降。

所以最终测试应该加载验证集最好的模型，而不是最后一轮模型。

## 9. latest checkpoint 是什么

项目还保存：

```text
latest_checkpoint.pth
```

它用于中断恢复。

如果训练中断，可以用：

```bash
--resume
```

继续训练，不需要从第 1 轮重来。

## 10. 实验记录保存了什么

训练结束后会保存：

```text
models/best_simple_cnn.pth
models/latest_checkpoint.pth
logs/run_config.json
logs/final_result.json
logs/training_history.csv
figures/loss_curve.png
figures/accuracy_curve.png
figures/confusion_matrix.png
```

这些文件能支持你写报告和答辩。

## 11. 实验规范亮点

你可以在答辩中主动说：

> 我的实验使用固定随机种子、独立验证集和测试集、按实验名称隔离输出目录，并自动记录配置、硬件、参数量、训练时间和最终指标。这些设计保证了实验结果更容易复现，也方便做公平对比。
