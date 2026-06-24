# 02 MNIST 数据集与固定划分

## 1. MNIST 是什么

MNIST 是经典手写数字数据集。

每张图片是一个灰度手写数字。

类别是：

```text
0 到 9
```

总共 10 类。

## 2. 图片形状

单张 MNIST 图片形状：

```text
[1, 28, 28]
```

含义：

- `1`：灰度通道。
- 第一个 `28`：高度。
- 第二个 `28`：宽度。

一个 batch 的形状：

```text
[B, 1, 28, 28]
```

本实验默认 Batch Size 是 128，所以常见输入是：

```text
[128, 1, 28, 28]
```

## 3. 数据读取代码

代码位置：

```text
experiment_2_rnn/src/dataset.py
```

核心代码：

```python
datasets.MNIST(data_dir, train=True, transform=transform, download=download)
datasets.MNIST(data_dir, train=False, transform=transform, download=download)
```

`train=True` 读取官方训练集。

`train=False` 读取官方测试集。

## 4. 预处理

代码：

```python
transforms.ToTensor()
transforms.Normalize((0.1307,), (0.3081,))
```

### ToTensor

把图片转成张量，并把像素范围从 0 到 255 转为 0 到 1。

### Normalize

使用 MNIST 常用均值和标准差：

```text
mean = 0.1307
std = 0.3081
```

因为 MNIST 是灰度图，只有一个通道，所以均值和标准差都只有一个数。
对应文件：`experiment_2_rnn/src/dataset.py`

对应代码片段：MNIST 的预处理只包含灰度图的单通道均值和标准差。

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
```

## 5. 数据划分

MNIST 官方训练集：

```text
60,000 张
```

官方测试集：

```text
10,000 张
```

你的项目把官方训练集拆成：

| 数据集 | 数量 |
| --- | ---: |
| 训练集 | 54,000 |
| 验证集 | 6,000 |
| 测试集 | 10,000 |

代码使用：

```python
random_split(full_train, [train_size, config.validation_size], generator=generator)
```

其中：

```text
validation_size = 6000
seed = 42
```

## 6. 为什么要从训练集中拆验证集

因为官方测试集应该只用于最终评估。

训练过程中需要一个不更新参数的数据集来选择最佳模型，这就是验证集。

所以流程是：

```text
训练集：更新参数
验证集：选择最佳 Epoch
测试集：最终评价
```

## 7. 固定随机种子的作用

固定随机种子后，每次拆出的训练集和验证集一致。

这对对比实验很重要。

如果 `hidden=64` 和 `hidden=128` 使用不同数据划分，就很难判断准确率差异到底来自模型还是数据。

## 8. 本章你要能回答的问题

### 问：MNIST 输入形状是什么？

答：

> 单张图片是 `[1, 28, 28]`，一个 batch 是 `[B, 1, 28, 28]`。

### 问：为什么要划分验证集？

答：

> 验证集用于模型选择，测试集只用于最终评价，避免测试集参与调参。

### 问：为什么固定随机种子？

答：

> 固定随机种子可以保证两组对比实验使用相同划分，使比较更公平。
对应文件：`experiment_2_rnn/src/dataset.py`

对应代码片段：`random_split` 使用固定 seed 的 generator。

```python
generator = torch.Generator().manual_seed(config.seed)
train_dataset, val_dataset = random_split(
    full_train,
    [train_size, config.validation_size],
    generator=generator,
)
```


