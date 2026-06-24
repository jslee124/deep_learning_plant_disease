# 05 MnistRNN 模型代码详解

## 1. 模型代码位置

```text
experiment_2_rnn/src/models.py
```

核心类：
对应文件：`experiment_2_rnn/src/models.py`

对应代码片段：模型类就是 `MnistRNN`。

```python
class MnistRNN(nn.Module):
    """把 28x28 图片按行看作长度为 28 的序列。"""
```

```python
class MnistRNN(nn.Module)
```

## 2. 初始化参数

```python
def __init__(
    self,
    input_size: int = 28,
    hidden_size: int = 64,
    num_layers: int = 1,
    num_classes: int = 10,
    dropout: float = 0.0,
):
```

参数含义：

| 参数 | 含义 |
| --- | --- |
| `input_size` | 每个时间步输入特征数，本实验是 28 |
| `hidden_size` | 隐藏状态维度，对比 64 和 128 |
| `num_layers` | RNN 层数，本实验是 1 |
| `num_classes` | 类别数，MNIST 是 10 |
| `dropout` | 多层 RNN 时使用，本实验一层所以为 0 |

## 3. RNN 层

代码：

```python
self.rnn = nn.RNN(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_first=True,
    nonlinearity="tanh",
    dropout=dropout if num_layers > 1 else 0.0,
)
```

### input_size=28

每个时间步输入一行 28 个像素。

### hidden_size=64 或 128

每个时间步输出一个隐藏向量。

如果 hidden_size 是 64：

```text
每个时间步输出 64 维向量
```

如果 hidden_size 是 128：

```text
每个时间步输出 128 维向量
```

### batch_first=True

输入格式使用：

```text
[batch, sequence_length, input_size]
```

也就是：

```text
[B, 28, 28]
```

如果不设置 `batch_first=True`，PyTorch 默认格式会是 `[sequence_length, batch, input_size]`，不如现在直观。

### nonlinearity="tanh"

基础 RNN 内部使用 tanh 激活。

tanh 会把值压到 -1 到 1 之间。

## 4. 分类层

代码：

```python
self.classifier = nn.Linear(hidden_size, num_classes)
```

作用：

```text
[B, hidden_size] -> [B, 10]
```

也就是把最后时间步的隐藏表示映射为 10 个数字类别得分。
对应文件：`experiment_2_rnn/src/models.py`

对应代码片段：最后一步由线性层输出 10 个类别得分。

```python
self.classifier = nn.Linear(hidden_size, num_classes)
return self.classifier(final_output)
```

## 5. forward 函数

代码：

```python
def forward(self, images: torch.Tensor) -> torch.Tensor:
    sequence = images.squeeze(1)
    outputs, _ = self.rnn(sequence)
    final_output = outputs[:, -1, :]
    return self.classifier(final_output)
```

逐步解释：

### 第一步：图片变序列

```python
sequence = images.squeeze(1)
```

```text
[B, 1, 28, 28] -> [B, 28, 28]
```

### 第二步：送入 RNN

```python
outputs, _ = self.rnn(sequence)
```

输出：

```text
[B, 28, hidden_size]
```

### 第三步：取最后时间步

```python
final_output = outputs[:, -1, :]
```

输出：

```text
[B, hidden_size]
```

### 第四步：分类

```python
self.classifier(final_output)
```

输出：

```text
[B, 10]
```

## 6. 张量形状总表

| 阶段 | 张量形状 |
| --- | --- |
| DataLoader 输出 | `[B, 1, 28, 28]` |
| 去掉通道维度 | `[B, 28, 28]` |
| RNN 全部时间步输出 | `[B, 28, hidden_size]` |
| 最后时间步输出 | `[B, hidden_size]` |
| 分类层输出 | `[B, 10]` |

这张表必须背熟。

## 7. 你要能现场讲的一段话

> 我的 RNN 模型先把 MNIST 输入从 `[B, 1, 28, 28]` 转成 `[B, 28, 28]`，其中 28 个时间步对应 28 行，每个时间步输入 28 个像素。RNN 输出所有时间步的隐藏表示，我取最后一个时间步作为整张图片的表示，再通过线性层输出 10 个数字类别的 logits。


