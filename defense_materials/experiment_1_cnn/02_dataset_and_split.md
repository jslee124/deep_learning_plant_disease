# 02 数据集、ImageFolder 与数据划分

## 1. 为什么先讲数据

深度学习项目不是从模型开始的，而是从数据开始的。

如果数据目录不对、类别编号不一致、训练集和测试集混用，那么模型即使能跑，结果也不可信。

本实验的数据处理代码在：

```text
experiment_1_cnn/src/dataset.py
```

## 2. ImageFolder 的规则

代码使用：

```python
torchvision.datasets.ImageFolder
```

它的规则很简单：

```text
一个子文件夹 = 一个类别
文件夹名 = 类别名
```

例如：

```text
data/train/
├── Apple___Apple_scab/
├── Apple___healthy/
├── Tomato___Late_blight/
└── ...
```

`ImageFolder` 会自动扫描这些子文件夹，并建立类别到数字编号的映射：

```text
Apple___Apple_scab -> 0
Apple___healthy -> 1
...
```

模型训练时不直接使用文字类别，而是使用数字标签。
对应文件：`experiment_1_cnn/src/dataset.py`

对应代码片段：`ImageFolder` 会读取类别文件夹，并通过 `classes` 保存类别名顺序。

```python
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
class_names = train_dataset.classes
```

## 3. 训练集、验证集、测试集分别干什么

### 训练集

训练集用于更新模型参数。

在训练阶段，模型会根据训练集图片预测类别，然后通过损失函数计算错误，再用反向传播修改参数。

### 验证集

验证集不用于更新参数。

它用于观察模型在没直接训练的数据上的表现，并选择最佳模型。

本项目中，如果某一轮验证准确率最高，就保存这一轮模型为 `best_simple_cnn.pth`。

### 测试集

测试集只在最后使用。

它用于最终评价模型泛化能力。

不能用测试集调参数，也不能用测试集选择最佳 Epoch。

## 4. 为什么不能混用测试集

如果你用测试集挑模型，就等于模型选择过程已经看过测试集了。

这样最后测试准确率会偏乐观，不能代表真实泛化能力。

答辩时可以这样说：

> 测试集必须只用于最终评价。如果用测试集参与调参或模型选择，就会造成数据泄漏，使测试结果失去客观性。

## 5. 你的项目如何适配 Kaggle 数据

Kaggle 的 New Plant Diseases Dataset 通常有：

```text
train/
valid/
test/test/
```

问题是：`test/test` 中通常没有按类别整理的标签目录，不适合直接用于 `ImageFolder` 测试。

你的代码处理方式是：

1. 使用 `train` 作为训练集。
2. 识别 `valid` 作为带标签数据。
3. 按类别把 `valid` 稳定拆成验证集和独立测试集。
4. 使用固定随机种子保证拆分可复现。

相关函数：

```python
_find_split_root()
_stratified_split_indices()
_validate_same_classes()
build_dataloaders()
```

## 6. 按类别拆分是什么意思

按类别拆分，也叫 stratified split。

意思是每个类别都按相似比例分到验证集和测试集。

为什么要这样？

假设某类图片本来就少，如果随机拆分，可能验证集里有这个类，但测试集里没有，或者反过来。

按类别拆分能让验证集和测试集的类别分布更接近。
对应文件：`experiment_1_cnn/src/dataset.py`

对应代码片段：`_stratified_split_indices()` 会逐个类别拆分索引。

```python
for class_id in sorted(set(targets)):
    class_indices = [index for index, target in enumerate(targets) if target == class_id]
    shuffled = torch.randperm(len(class_indices), generator=generator).tolist()
    test_size = max(1, min(len(class_indices) - 1, round(len(class_indices) * test_ratio)))
```

## 7. 本实验正式数据规模

正式基线记录中：

| 数据划分 | 数量 |
| --- | ---: |
| 训练集 | 70,295 |
| 验证集 | 8,790 |
| 测试集 | 8,782 |
| 类别数 | 38 |

这说明实验不是玩具数据，而是一个比较完整的多分类图像任务。

## 8. DataLoader 是什么

`DataLoader` 的作用是把数据集包装成可以训练的批次。

它负责：

- 每次取出一个 batch。
- 训练集打乱顺序。
- 多进程读取数据。
- 把图片和标签组织成张量。

训练集：

```python
DataLoader(train_dataset, batch_size=32, shuffle=True)
```

验证和测试：

```python
shuffle=False
```

因为验证和测试不需要打乱，保持固定顺序更方便复现和分析。

## 9. 本章你要能回答的问题

### 问：`ImageFolder` 怎么知道类别？

答：

> 它根据子文件夹确定类别，一个子文件夹就是一个类别，文件夹名就是类别名。

### 问：为什么要有验证集？

答：

> 验证集用于选择最佳模型和观察泛化表现，不能用训练集直接判断模型好坏。

### 问：为什么测试集不能参与训练？

答：

> 测试集代表未知数据，只能用于最终评价。如果参与训练或调参，会造成数据泄漏。

### 问：为什么按类别拆分 valid？

答：

> 为了让验证集和测试集中每个类别比例接近，避免某些类别分布不均导致评价不稳定。


