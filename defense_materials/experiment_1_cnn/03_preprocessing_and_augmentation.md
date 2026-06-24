# 03 图像预处理与数据增强

## 1. 为什么图片不能直接送进模型

模型需要的是数值张量，不是普通图片文件。

一张图片从硬盘读取出来后，要经过几个步骤：

```text
读取图片
  ↓
调整尺寸
  ↓
转成 Tensor
  ↓
归一化
  ↓
变成 [C, H, W]
```

一个 batch 送进模型时形状是：
对应文件：`experiment_1_cnn/src/dataset.py`

对应代码片段：图片经过 transform 后由 `DataLoader` 组成 batch。

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
)
```

```text
[B, 3, 224, 224]
```

## 2. 训练集 transform

代码位置：

```text
experiment_1_cnn/src/dataset.py
```

训练集使用：

```python
transforms.Resize((image_size + 32, image_size + 32))
transforms.RandomResizedCrop(image_size)
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomRotation(degrees=15)
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
transforms.ToTensor()
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

## 3. 每个 transform 的作用

### Resize

先把图片放大到 `image_size + 32`，也就是 256x256。

这样后面的随机裁剪有空间可裁。

### RandomResizedCrop

随机裁剪出 224x224 的区域。

作用：

- 模拟叶片在图片中位置略有变化。
- 减少模型死记固定位置。
- 增强模型鲁棒性。

### RandomHorizontalFlip

以 0.5 概率水平翻转。

植物叶片左右翻转后类别通常不变，所以这是合理的数据增强。

### RandomRotation

随机旋转最多 15 度。

叶片拍摄时角度可能不同，轻微旋转有助于模型适应角度变化。

### ColorJitter

随机改变亮度、对比度、饱和度。

植物图片可能受光照、相机、环境影响。颜色扰动可以让模型不要过度依赖某一种固定亮度。

### ToTensor

把 PIL 图片变成 PyTorch 张量。

变化：

```text
原始图片: [H, W, C], 像素 0 到 255
Tensor: [C, H, W], 像素 0 到 1
```

### Normalize

对每个通道做标准化：

```text
标准化后 = (原值 - mean) / std
```

本项目使用 ImageNet 常用均值和标准差：

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

它能让输入分布更稳定，通常有利于训练。

## 4. 验证集和测试集 transform

验证和测试使用：

```python
transforms.Resize((image_size, image_size))
transforms.ToTensor()
transforms.Normalize(...)
```

没有随机裁剪、随机翻转、随机旋转、颜色扰动。
对应文件：`experiment_1_cnn/src/dataset.py`

对应代码片段：验证和测试使用 `get_eval_transform()`，只保留确定性处理。

```python
return transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

## 5. 为什么验证和测试不做随机增强

验证和测试的目标是评价模型。

如果每次验证都随机改变图片，那么同一个模型每次评估的数据都不一样，准确率会有随机波动。

这会导致：

- 结果不稳定。
- 两组实验不公平。
- 难以复现。

所以：

```text
训练集可以随机增强。
验证集和测试集必须保持确定性。
```

## 6. 为什么训练 Accuracy 有时低于验证 Accuracy

这在你的实验中出现过。

原因不是代码错了，而是训练阶段更难：

- 训练集使用随机增强，图片被裁剪、旋转、变色。
- 训练阶段 Dropout 会随机关闭部分神经元。
- 验证和测试阶段没有这些随机扰动。

所以验证/测试 Accuracy 可能高于训练 Accuracy。

答辩时不要慌，可以这样说：

> 训练阶段启用了数据增强和 Dropout，输入更困难，模型也处于随机失活状态；验证和测试阶段关闭这些随机因素，因此验证准确率高于训练准确率是可能出现的。

## 7. 本章你要能回答的问题

### 问：为什么图片要 Resize 到 224x224？

答：

> 模型需要固定尺寸输入。统一到 224x224 后，一个 batch 中的图片才能组成同一个张量。

### 问：数据增强有什么作用？

答：

> 数据增强通过旋转、裁剪、翻转和颜色扰动生成更多变化，减少模型死记训练集，提高泛化能力。

### 问：Normalize 的目的是什么？

答：

> Normalize 让输入分布更稳定，使模型训练更容易收敛。


