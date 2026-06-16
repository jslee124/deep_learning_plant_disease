# 06 训练流程与梯度裁剪

## 1. 训练代码位置

```text
experiment_2_rnn/src/train.py
```

这个文件完成：

- 参数读取。
- 数据加载。
- 模型创建。
- 训练和验证。
- 保存最佳模型。
- 测试评估。
- 绘制曲线、混淆矩阵和预测样例。

## 2. 训练命令

正式实验：

```powershell
.\venv\Scripts\python.exe experiment_2_rnn\src\train.py `
  --experiment-name rnn_hidden_64_baseline --hidden-size 64

.\venv\Scripts\python.exe experiment_2_rnn\src\train.py `
  --experiment-name rnn_hidden_128_comparison --hidden-size 128 --no-download
```

对比命令：

```powershell
.\venv\Scripts\python.exe experiment_2_rnn\src\compare_runs.py `
  rnn_hidden_64_baseline rnn_hidden_128_comparison
```

## 3. run_epoch 函数

代码中训练、验证和测试共用：

```python
run_epoch(...)
```

如果传入 `optimizer`，就是训练。

如果不传入 `optimizer`，就是验证或测试。

## 4. 一个 batch 的训练步骤

```python
optimizer.zero_grad()
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
optimizer.step()
```

逐步解释：

### 清空梯度

```python
optimizer.zero_grad()
```

避免上一批梯度累加到当前批。

### 前向传播

```python
logits = model(images)
```

输入 `[B, 1, 28, 28]`，输出 `[B, 10]`。

### 计算损失

```python
loss = criterion(logits, labels)
```

使用 `CrossEntropyLoss` 计算多分类损失。

### 反向传播

```python
loss.backward()
```

计算参数梯度。

### 梯度裁剪

```python
clip_grad_norm_(..., max_norm=5.0)
```

限制梯度范数，防止梯度爆炸。

### 更新参数

```python
optimizer.step()
```

用 Adam 更新参数。

## 5. 为什么 RNN 需要注意梯度问题

RNN 的梯度要沿时间步反向传播。

如果序列较长，梯度可能：

- 越传越小，叫梯度消失。
- 越传越大，叫梯度爆炸。

本实验序列长度 28，不算很长，但加入梯度裁剪是一个稳妥设计。

答辩回答：

> 基础 RNN 在序列反向传播时可能出现梯度爆炸，所以训练中使用梯度裁剪限制梯度范数，提高稳定性。

## 6. 保存最佳模型

代码逻辑：

```python
if val_acc > best_val_acc:
    torch.save(..., config.best_model_path)
```

验证集准确率最高的模型会保存为：

```text
best_rnn.pth
```

训练结束后加载这个模型做测试。

这样比直接用最后一轮更规范。

## 7. 输出文件

每组实验保存：

```text
models/best_rnn.pth
models/latest_checkpoint.pth
logs/run_config.json
logs/final_result.json
logs/training_history.csv
figures/loss_curve.png
figures/acc_curve.png
figures/confusion_matrix.png
figures/prediction_examples.png
```

这些文件用于报告和答辩分析。

## 8. 本章你要记住

- RNN 训练也是前向传播、计算损失、反向传播、更新参数。
- 使用 CrossEntropyLoss 做 10 分类。
- 使用 Adam 优化器。
- 使用梯度裁剪防止梯度爆炸。
- 使用验证集保存最佳模型。
