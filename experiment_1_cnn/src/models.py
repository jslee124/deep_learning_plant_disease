"""模型结构模块。

本阶段实现自建 CNN 基线模型。模型故意保持清晰、适中，不追求复杂结构，
方便向老师解释每一层的作用。
"""

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """适合教学演示的简单卷积神经网络。

    输入图片形状通常为 [batch_size, 3, 224, 224]：
    - batch_size 表示一次送入模型的图片数量；
    - 3 表示 RGB 三个颜色通道；
    - 224 x 224 表示图片高和宽。
    """

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # 卷积层作用：
            # 卷积核会在图片上滑动，提取局部特征，例如边缘、纹理、斑点、病斑形状等。
            # out_channels=32 表示这一层学习 32 种不同的局部特征。
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            # ReLU 激活函数作用：
            # 给模型加入非线性能力。如果没有激活函数，多层网络本质上仍接近线性变换，
            # 难以学习复杂的植物病害图像特征。
            nn.ReLU(inplace=True),
            # 池化层作用：
            # 降低特征图尺寸，减少计算量，同时保留主要特征。
            # MaxPool 会保留局部区域中响应最强的特征，常用于图像分类 CNN。
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 自适应平均池化会把任意大小的特征图压缩成 1x1。
            # 这样全连接层输入维度固定，后续即使改 image_size，模型也更稳。
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            # 全连接层作用：
            # 前面的卷积层负责提取图像特征，全连接层负责把这些特征组合起来，
            # 输出每个类别的得分。例如“健康叶片”“某种病害叶片”各对应一个得分。
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：图片 -> 特征提取 -> 分类得分。"""

        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(model_name: str = "simple_cnn", num_classes: int = 2, dropout: float = 0.3) -> nn.Module:
    """根据模型名称创建模型。

    参数:
        model_name: 模型名称。本阶段使用 simple_cnn。
        num_classes: 分类类别数，由数据集类别文件夹数量决定。
        dropout: Dropout 比例，用于减少过拟合。
    """

    name = model_name.lower()
    if name in {"simple_cnn", "cnn", "baseline_cnn"}:
        return SimpleCNN(num_classes=num_classes, dropout=dropout)

    raise ValueError(f"暂不支持模型: {model_name}。第二阶段请使用 simple_cnn。")
