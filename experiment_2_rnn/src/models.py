"""基础循环神经网络模型。"""

import torch
from torch import nn


class MnistRNN(nn.Module):
    """把 28x28 图片按行看作长度为 28 的序列。"""

    def __init__(
        self,
        input_size: int = 28,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # 输入 [B, 1, 28, 28] -> 序列 [B, 28, 28]。
        sequence = images.squeeze(1)
        outputs, _ = self.rnn(sequence)
        # 最后一个时间步包含模型按行读取整张图片后形成的序列表示。
        final_output = outputs[:, -1, :]
        return self.classifier(final_output)


def build_model(
    input_size: int = 28,
    hidden_size: int = 64,
    num_layers: int = 1,
    num_classes: int = 10,
    dropout: float = 0.0,
) -> nn.Module:
    return MnistRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    )
