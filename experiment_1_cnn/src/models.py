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

    def __init__(
        self,
        num_classes: int,           # 分类类别数，由数据集类别文件夹数量决定
        dropout: float = 0.3,       # Dropout 比例，训练时以 30% 概率随机丢弃神经元，防止过拟合。值越大正则化越强
        conv_kernel_size: int = 3,  # 卷积核大小（3x3）。必须是正奇数。通过 padding = k//2 保持特征图尺寸不变，方便对比不同核大小的效果。
        conv_blocks: int = 4,       # 卷积块数量，决定堆叠多少层卷积。代码中从 [32, 64, 128, 256] 里取前 N 个通道数，例如 4 就是 4 个卷积块，通道数依次为 32→64→128→256，每块之间用 MaxPool2d 下采样。
    ):
        super().__init__()

        if conv_kernel_size < 1 or conv_kernel_size % 2 == 0:
            raise ValueError("卷积核大小必须是正奇数，例如 3 或 5。")
        channel_options = [32, 64, 128, 256]
        if not 2 <= conv_blocks <= len(channel_options):
            raise ValueError("conv_blocks 当前支持 2、3 或 4。")

        padding = conv_kernel_size // 2
        feature_layers = []
        in_channels = 3
        for block_index, out_channels in enumerate(channel_options[:conv_blocks]):
            # padding=k//2 让 3x3 和 5x5 卷积都保持卷积前后的高宽不变，
            # 对比时只改变卷积核大小，不额外改变特征图尺寸。
            # 循环搭建卷积块，每一轮做的事：
            # - Conv2d：用 3x3 卷积提取图像特征（比如检测边缘、纹理、斑点）
            # - BatchNorm2d：把数据归一化，让训练更稳定、更快
            # - ReLU：激活函数，把负数变成 0，给模型加入非线性能力
            # - MaxPool2d(非最后一轮)：2x2 池化把图缩小一半（224→112→56→28），降低计算量同时保留主要特征
            feature_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            if block_index < conv_blocks - 1:
                feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        feature_layers.append(nn.AdaptiveAvgPool2d((1, 1))) # 自适应池化层，将特征图大小缩小为 1x1，方便后续全连接层处理。
        self.features = nn.Sequential(*feature_layers)   # nn.Sequential 将多个层组合成一个模块，按顺序执行。

        # 全连接层作用：
        # 前面的卷积层负责提取图像特征，全连接层负责把这些特征组合起来，
        # 输出每个类别的得分。例如“健康叶片”“某种病害叶片”各对应一个得分。
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：图片 -> 特征提取 -> 分类得分。"""

        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(
    model_name: str = "simple_cnn",
    num_classes: int = 2,
    dropout: float = 0.3,
    conv_kernel_size: int = 3,
    conv_blocks: int = 4,
) -> nn.Module:
    """根据模型名称创建模型。

    参数:
        model_name: 模型名称。本阶段使用 simple_cnn。
        num_classes: 分类类别数，由数据集类别文件夹数量决定。
        dropout: Dropout 比例，用于减少过拟合。
    """

    name = model_name.lower()
    if name in {"simple_cnn", "cnn", "baseline_cnn"}:
        return SimpleCNN(
            num_classes=num_classes,
            dropout=dropout,
            conv_kernel_size=conv_kernel_size,
            conv_blocks=conv_blocks,
        )

    raise ValueError(f"暂不支持模型: {model_name}。第二阶段请使用 simple_cnn。")
