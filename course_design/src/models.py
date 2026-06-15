"""课程设计模型结构：自建 CNN 与 ResNet18 迁移学习。"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class SimpleCNN(nn.Module):
    """适合教学演示的简单卷积神经网络。

    输入图片形状通常为 [batch_size, 3, 224, 224]：
    - batch_size 表示一次送入模型的图片数量；
    - 3 表示 RGB 三个颜色通道；
    - 224 x 224 表示图片高和宽。
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.3,
        conv_kernel_size: int = 3,
        conv_blocks: int = 4,
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

        feature_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*feature_layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            # 全连接层作用：
            # 前面的卷积层负责提取图像特征，全连接层负责把这些特征组合起来，
            # 输出每个类别的得分。例如“健康叶片”“某种病害叶片”各对应一个得分。
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
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> nn.Module:
    """根据模型名称创建模型。

    参数:
        model_name: simple_cnn 或 resnet18。
        num_classes: 分类类别数，由数据集类别文件夹数量决定。
        dropout: Dropout 比例，用于减少过拟合。
        pretrained: ResNet18 是否加载 ImageNet 预训练权重。
        freeze_backbone: 是否冻结 ResNet18 主干，仅训练最后的分类层。
    """

    name = model_name.lower()
    if name in {"simple_cnn", "cnn", "baseline_cnn"}:
        return SimpleCNN(
            num_classes=num_classes,
            dropout=dropout,
            conv_kernel_size=conv_kernel_size,
            conv_blocks=conv_blocks,
        )

    if name in {"resnet18", "resnet_18"}:
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        input_features = model.fc.in_features
        model.fc = nn.Linear(input_features, num_classes)

        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False
            for parameter in model.fc.parameters():
                parameter.requires_grad = True

        return model

    raise ValueError(f"暂不支持模型: {model_name}。请使用 simple_cnn 或 resnet18。")
