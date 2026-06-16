# 实验一 CNN 学习目录

主题：基于卷积神经网络的植物病害图像分类。

建议按顺序阅读，不要跳着看。每一章只解决一个问题。

## 学习路线

1. [01 项目任务与整体流程](01_task_overview.md)
2. [02 数据集、ImageFolder 与数据划分](02_dataset_and_split.md)
3. [03 图像预处理与数据增强](03_preprocessing_and_augmentation.md)
4. [04 CNN 基础知识](04_cnn_basic_theory.md)
5. [05 SimpleCNN 模型代码详解](05_model_code_explained.md)
6. [06 训练流程与实验规范](06_training_process.md)
7. [07 卷积核对比实验结果分析](07_kernel_comparison_results.md)
8. [08 特征图可视化与单张预测](08_feature_maps_and_prediction.md)
9. [09 高频追问与自测题](09_questions_and_self_test.md)

## 最低掌握目标

学完后你至少要能讲清楚：

- CNN 为什么适合图像分类。
- 图片如何变成 `[B, 3, 224, 224]`。
- `Conv2d`、`BatchNorm2d`、`ReLU`、`MaxPool2d`、`Linear` 分别做什么。
- 为什么对比 `3x3` 和 `5x5` 卷积核时要控制其他变量。
- 为什么 `5x5` 准确率更高，但参数量和训练时间也更高。
- 为什么用验证集保存最佳模型，测试集只做最终评价。
