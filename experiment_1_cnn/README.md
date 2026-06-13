# 实验一：基于卷积神经网络的植物病害图像分类

本目录用于《深度学习》课程的卷积神经网络平时实验。当前已完成 CNN 基线模型，
后续将在这里完成卷积核或卷积层数对比，不与循环神经网络实验和期末课程设计混用输出。

## 已实现功能

- 数据集读取
- 训练集数据增强
- 自建 CNN 模型
- 训练循环
- 验证循环
- 独立测试集评估
- 保存验证集表现最好的模型
- 通过实验名称隔离每次训练输出
- 自动记录实验参数、数据划分、硬件信息和训练效率
- 可配置 `3x3`/`5x5` 卷积核及 2/3/4 个卷积块
- 自动绘制 Loss 曲线
- 自动绘制 Accuracy 曲线
- 自动生成混淆矩阵
- 单张图片预测

## 项目结构

```text
experiment_1_cnn/
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── config.py
│   └── utils.py
├── outputs/
│   ├── models/
│   ├── logs/
│   ├── figures/
│   └── reports/
├── requirements.txt
└── README.md
```

数据集统一放在仓库根目录的 `data/` 中，避免在多个实验目录中重复保存大型数据集。

## 数据集格式

推荐使用标准的 `train/val/test` 结构：

```text
data/
├── train/
│   ├── Apple_scab/
│   └── Apple_healthy/
├── val/
│   ├── Apple_scab/
│   └── Apple_healthy/
└── test/
    ├── Apple_scab/
    └── Apple_healthy/
```

如果暂时没有划分好的数据集，也可以直接放成：

```text
data/
├── Apple_scab/
└── Apple_healthy/
```

程序会自动按 70%/15%/15% 划分训练集、验证集和测试集。

本项目已经适配 Kaggle 的 `vipoooool/new-plant-diseases-dataset`。该数据集解压后
训练集和验证集通常位于较深目录中，并且验证集目录名是 `valid`，程序会自动识别。
该数据集自带的 `test/test` 是少量无类别标签图片，不适合直接做 ImageFolder 测试集；
程序会按类别、固定随机种子将 `valid` 平分为验证集和独立测试集，不再复用同一批
图片。当前数据共 38 类，`valid` 共 17572 张图片，默认约一半用于验证、一半用于测试。

## 安装依赖

```bash
pip install -r requirements.txt
```

如果已经创建虚拟环境，先激活虚拟环境后再安装依赖。

## 训练模型

在仓库根目录 `deep_learning_plant_disease/` 下运行：

```bash
python experiment_1_cnn/src/train.py --experiment-name baseline_k3
```

也可以覆盖默认参数：

```bash
python experiment_1_cnn/src/train.py --experiment-name baseline_k3 --epochs 10 --kernel-size 3
```

关闭数据增强：

```bash
python experiment_1_cnn/src/train.py --no-augmentation
```

每次训练保存在独立目录，不会覆盖其他实验：

```text
experiment_1_cnn/outputs/runs/baseline_k3/models/best_simple_cnn.pth
experiment_1_cnn/outputs/runs/baseline_k3/figures/
experiment_1_cnn/outputs/runs/baseline_k3/logs/
experiment_1_cnn/outputs/logs/experiment_results.csv
experiment_1_cnn/outputs/logs/hardware_benchmark.csv
```

卷积核单变量对比应保持其他参数一致：

```bash
python experiment_1_cnn/src/train.py --experiment-name kernel_3x3 --kernel-size 3
python experiment_1_cnn/src/train.py --experiment-name kernel_5x5 --kernel-size 5
```

如果 GPU 利用率较低，可以为两组对比实验统一增加 `--num-workers 4`。训练每个 Epoch
都会保存 `latest_checkpoint.pth` 和训练历史；若训练意外中断，可使用原实验名继续：

```bash
python experiment_1_cnn/src/train.py --experiment-name kernel_3x3 --kernel-size 3 --num-workers 4 --resume
```

当前正式 `3x3` 基线实验为 `cnn_kernel_3x3_baseline_v2`。在 10 Epoch、Batch Size 32、
Adam、学习率 0.001 和 4 个卷积块的配置下，最佳验证 Accuracy 为 93.96%，独立测试
Accuracy 为 94.02%。详细记录见 `docs/stage_4_baseline_results.md`。

## 评估模型

```bash
python experiment_1_cnn/src/evaluate.py --model-path experiment_1_cnn/outputs/runs/baseline_k3/models/best_simple_cnn.pth --split test
```

可选 `--split train`、`--split val`、`--split test`。

## 单张图片预测

```bash
python experiment_1_cnn/src/predict.py --model-path experiment_1_cnn/outputs/runs/baseline_k3/models/best_simple_cnn.pth --image-path path/to/leaf.jpg
```

显示置信度最高的前 3 个类别：

```bash
python experiment_1_cnn/src/predict.py --model-path experiment_1_cnn/outputs/runs/baseline_k3/models/best_simple_cnn.pth --image-path path/to/leaf.jpg --top-k 3
```

## 课堂讲解要点

### dataset.py

这个模块负责读取图片并做数据增强。这样设计是为了让训练、验证、测试都使用统一的数据处理流程。

老师可能问：为什么验证集和测试集不做随机增强？

回答：验证和测试要反映模型在固定数据上的真实性能，如果加入随机增强，每次评估图片都不同，结果不稳定。

### models.py

这个模块实现自建 CNN。卷积层提取局部纹理和病斑特征，池化层降低特征图尺寸并减少计算量，ReLU 提供非线性表达能力，全连接层根据提取到的特征输出类别得分。

老师可能问：为什么 CNN 适合图像分类？

回答：图像中的边缘、纹理、颜色斑点等局部特征很重要，卷积层可以自动学习这些局部特征，并通过多层组合形成更高级的病害特征。

### train.py

这个模块负责完整训练流程。CrossEntropyLoss 用于多分类损失计算，Adam 根据梯度自动更新模型参数。

老师可能问：为什么要保存验证集表现最好的模型？

回答：最后一个 Epoch 不一定是泛化能力最好的模型。用验证集 Accuracy 保存最佳模型，可以减少过拟合带来的影响。

### evaluate.py

这个模块用于训练后独立评估模型，并生成混淆矩阵。

老师可能问：为什么要看混淆矩阵？

回答：Accuracy 只能说明总体正确率，混淆矩阵能看出具体哪些类别容易被错分，更适合做实验结果分析。

### predict.py

这个模块用于单张图片预测，适合现场演示。

老师可能问：预测时为什么要使用和测试集一致的预处理？

回答：训练和评估时输入分布要尽量一致，否则模型看到的图片尺度和归一化方式不同，预测结果可能不稳定。
