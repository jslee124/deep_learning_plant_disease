# 05 SimpleCNN 模型代码详解

## 1. 模型代码位置

```text
experiment_1_cnn/src/models.py
```

核心类：

```python
class SimpleCNN(nn.Module)
```

这个模型故意没有写得很复杂，因为课程实验更重要的是能解释每一层。

## 2. 初始化参数

```python
def __init__(
    self,
    num_classes: int,
    dropout: float = 0.3,
    conv_kernel_size: int = 3,
    conv_blocks: int = 4,
):
```

含义：

| 参数 | 作用 |
| --- | --- |
| `num_classes` | 输出类别数量，本实验是 38 |
| `dropout` | Dropout 比例，默认 0.3 |
| `conv_kernel_size` | 卷积核大小，可设为 3 或 5 |
| `conv_blocks` | 卷积块数量，可设为 2、3、4 |

## 3. 为什么检查卷积核必须是奇数

代码：

```python
if conv_kernel_size < 1 or conv_kernel_size % 2 == 0:
    raise ValueError("卷积核大小必须是正奇数，例如 3 或 5。")
```

原因：

奇数卷积核有明确中心点，例如 `3x3` 中心是第 2 行第 2 列，`5x5` 中心是第 3 行第 3 列。

配合：

```python
padding = conv_kernel_size // 2
```

可以保持卷积前后尺寸不变。

## 4. 通道数量如何变化

代码：

```python
channel_options = [32, 64, 128, 256]
```

如果使用 4 个卷积块，通道变化是：

```text
3 -> 32 -> 64 -> 128 -> 256
```

含义：

- 输入图片有 3 个 RGB 通道。
- 越深的层通道数越多，可以表达更多复杂特征。
- 空间尺寸逐渐变小，通道数逐渐增加，这是 CNN 常见设计。

## 5. feature_layers 如何构建

代码用循环生成卷积块：

```python
for block_index, out_channels in enumerate(channel_options[:conv_blocks]):
    feature_layers.extend([
        nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ])
    if block_index < conv_blocks - 1:
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    in_channels = out_channels
```

每个卷积块：

```text
Conv2d -> BatchNorm2d -> ReLU
```

前 3 个卷积块后面有池化，最后一个没有。

## 6. 为什么最后用 AdaptiveAvgPool2d

代码：

```python
feature_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
```

它会把每个通道的特征图压缩成 `1x1`。

例如：

```text
[B, 256, 28, 28] -> [B, 256, 1, 1]
```

好处：

- 全连接层输入维度固定。
- 减少参数量。
- 强迫模型总结每个通道的整体响应。

## 7. 分类器结构

代码：

```python
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Dropout(p=dropout),
    nn.Linear(in_channels, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(p=dropout),
    nn.Linear(128, num_classes),
)
```

逐层解释：

`Flatten`：

```text
[B, 256, 1, 1] -> [B, 256]
```

第一个 `Dropout`：减少过拟合。

`Linear(in_channels, 128)`：把 256 维特征压到 128 维。

`ReLU`：增加非线性。

第二个 `Dropout`：进一步减少过拟合。

`Linear(128, num_classes)`：输出 38 个类别得分。

## 8. forward 函数

代码：

```python
def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x
```

含义：

```text
图片 -> 卷积特征提取 -> 分类器 -> 类别 logits
```

注意：返回的是 logits，不是概率。

训练时直接交给 `CrossEntropyLoss`。

预测展示时才用 `softmax` 转成概率。

## 9. 张量形状完整变化

输入：

```text
[B, 3, 224, 224]
```

4 个卷积块时：

| 阶段 | 输出形状 |
| --- | --- |
| 输入 | `[B, 3, 224, 224]` |
| Conv1 | `[B, 32, 224, 224]` |
| Pool1 | `[B, 32, 112, 112]` |
| Conv2 | `[B, 64, 112, 112]` |
| Pool2 | `[B, 64, 56, 56]` |
| Conv3 | `[B, 128, 56, 56]` |
| Pool3 | `[B, 128, 28, 28]` |
| Conv4 | `[B, 256, 28, 28]` |
| AdaptiveAvgPool | `[B, 256, 1, 1]` |
| Flatten | `[B, 256]` |
| Linear 输出 | `[B, 38]` |

## 10. 你要能现场讲的一段话

> 我的模型是一个教学型 SimpleCNN。输入是 `[B, 3, 224, 224]` 的 RGB 图片。前面用 4 个卷积块逐步提取特征，通道数从 32 增加到 256，池化层逐步降低空间尺寸。然后用自适应平均池化把每个通道压缩成一个值，再经过全连接层输出 38 个类别的 logits。
