# 04 CNN 基础知识

## 1. CNN 为什么适合图片

图片有空间结构。

相邻像素之间往往相关，例如：

- 叶片边缘由一串相邻像素形成。
- 病斑由局部颜色和纹理变化形成。
- 叶脉由连续线状结构形成。

CNN 的卷积层正是用小窗口扫描图片，学习这些局部特征。
对应文件：`experiment_1_cnn/src/models.py`

对应代码片段：卷积层在 `SimpleCNN.__init__()` 中创建，`kernel_size` 决定小窗口大小。

```python
nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=padding)
```

## 2. 卷积层做什么

卷积层可以理解成多个特征检测器。

每个卷积核在图片上滑动，对局部区域做加权求和。

训练前卷积核参数是随机的。

训练后，不同卷积核可能学会检测不同特征：

- 边缘。
- 纹理。
- 颜色变化。
- 病斑形状。

## 3. 通道是什么

输入图片有 3 个通道：

```text
R, G, B
```

卷积层输出也有通道。

例如第一层输出 32 个通道：

```text
[B, 32, 224, 224]
```

可以理解为模型提取出了 32 种不同的低级特征图。
对应文件：`experiment_1_cnn/src/models.py`

对应代码片段：第一层输入通道是 3，输出通道来自 `channel_options` 的第一个值 32。

```python
channel_options = [32, 64, 128, 256]
in_channels = 3
for block_index, out_channels in enumerate(channel_options[:conv_blocks]):
    feature_layers.extend([
        nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ])
```

## 4. 卷积核大小

`3x3` 卷积核一次看 9 个像素位置。

`5x5` 卷积核一次看 25 个像素位置。

卷积核越大：

- 单层看到的局部范围越大。
- 参数量越多。
- 计算量越大。
- 可能捕捉更大范围纹理。

参数量公式：

```text
参数量 = 输出通道数 × 输入通道数 × 卷积核高 × 卷积核宽 + bias
```

如果同样输入输出通道，`5x5` 的卷积核面积是 25，`3x3` 是 9，所以 `5x5` 参数更多。
对应文件：`experiment_1_cnn/src/train.py`、`experiment_1_cnn/src/models.py`

对应代码片段：命令行参数控制卷积核大小，模型构建时传给 `Conv2d`。

```python
parser.add_argument("--kernel-size", type=int, default=config.conv_kernel_size, help="卷积核大小，如 3 或 5")

model = build_model(
    conv_kernel_size=config.conv_kernel_size,
    conv_blocks=config.conv_blocks,
)
```

## 5. 感受野是什么

感受野指某个神经元能看到原图中多大范围。

浅层神经元感受野小，常关注局部边缘和纹理。

深层神经元经过多层卷积和池化，感受野变大，可以关注更大区域和更抽象的类别特征。

植物病害识别中，感受野很重要，因为病斑不只是一个像素，而是包含一定区域的颜色、纹理和形状。

## 6. Padding 是什么

卷积如果不加 padding，图片边缘会被裁掉，特征图尺寸会变小。

本项目使用：

```python
padding = conv_kernel_size // 2
```

这样：

```text
3x3 -> padding 1
5x5 -> padding 2
```

可以保持卷积前后高宽不变。

这对对比实验很重要，因为你想主要比较卷积核大小，而不是让特征图尺寸也跟着变化。
对应文件：`experiment_1_cnn/src/models.py`

对应代码片段：`padding = conv_kernel_size // 2` 会让 `3x3` 得到 padding 1，`5x5` 得到 padding 2。

```python
padding = conv_kernel_size // 2
nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=padding)
```

## 7. ReLU 做什么

ReLU 公式：

```text
ReLU(x) = max(0, x)
```

它把负数变成 0，正数保持不变。

作用是引入非线性。

没有非线性，多层神经网络即使叠很多层，本质上也很难表达复杂关系。
对应文件：`experiment_1_cnn/src/models.py`

对应代码片段：每个卷积块中 `Conv2d` 后面接 `BatchNorm2d` 和 `ReLU`。

```python
feature_layers.extend([
    nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=padding),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
])
```

## 8. 池化层做什么

本项目使用最大池化：

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

它会把高宽减半。

例如：

```text
[B, 32, 224, 224] -> [B, 32, 112, 112]
```

作用：

- 降低计算量。
- 扩大后续层感受野。
- 保留局部最显著特征。
对应文件：`experiment_1_cnn/src/models.py`

对应代码片段：除了最后一个卷积块，前面的卷积块后面都会添加最大池化。

```python
if block_index < conv_blocks - 1:
    feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
```

## 9. BatchNorm 做什么

BatchNorm 会对中间特征进行归一化，让每层输入分布更稳定。

好处：

- 训练更稳定。
- 收敛可能更快。
- 对学习率不那么敏感。

答辩时不用讲公式，知道作用即可。
对应文件：`experiment_1_cnn/src/models.py`

对应代码片段：BatchNorm 放在卷积层之后、ReLU 之前。

```python
nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=padding),
nn.BatchNorm2d(out_channels),
nn.ReLU(inplace=True),
```

## 10. Dropout 做什么

Dropout 在训练时随机让一部分神经元输出为 0。

作用是减少模型对某些神经元的依赖，降低过拟合风险。

注意：

- 训练时 Dropout 开启。
- 验证和测试时 Dropout 关闭。
对应文件：`experiment_1_cnn/src/models.py`、`experiment_1_cnn/src/train.py`

对应代码片段：Dropout 定义在分类器中；训练时 `model.train()` 开启，验证/测试时 `model.eval()` 关闭。

```python
nn.Dropout(p=dropout)

model.train()
model.eval()
```

## 11. 全连接层做什么

前面的卷积层负责提取特征。

全连接层负责根据这些特征进行分类。

在你的模型中，最后输出：

```text
[B, 38]
```

表示每张图片对应 38 个类别得分。
对应文件：`experiment_1_cnn/src/models.py`

对应代码片段：最后的全连接层输出 `num_classes` 个类别得分，也就是 logits。

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









