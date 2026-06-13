"""通用工具函数。

这里放多个脚本都会用到的小工具，例如固定随机种子、选择训练设备、创建目录、
保存类别名称、绘制曲线和混淆矩阵。
"""

import json
import random
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


def ensure_dir(path: str | Path):
    """确保目录存在。

    训练前创建 outputs/models、outputs/figures 等目录，可以避免保存文件时报错。
    """

    Path(path).mkdir(parents=True, exist_ok=True)


def prepare_output_dirs(config):
    """创建项目输出目录。"""

    ensure_dir(config.model_dir)
    ensure_dir(config.log_dir)
    ensure_dir(config.figure_dir)
    ensure_dir(config.report_dir)


def set_seed(seed: int = 42):
    """固定随机种子，让实验结果更容易复现。

    深度学习中仍然可能存在少量非确定性，但固定随机种子能让数据划分、参数初始值
    等步骤尽量一致，方便比较不同实验。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """自动选择训练设备。

    优先使用 NVIDIA CUDA；如果是在 Apple Silicon Mac 上且 PyTorch 支持 MPS，
    则使用 mps；否则退回 CPU。
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_optimizer(model, optimizer_name: str, learning_rate: float):
    """创建优化器。

    Adam 是本阶段默认优化器。它会为不同参数自动调整学习步长，通常比普通 SGD
    更容易在初学实验中得到稳定结果。后续课程设计可以在这里切换 SGD、RMSprop
    做优化器对比实验。
    """

    name = optimizer_name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    raise ValueError(f"暂不支持优化器: {optimizer_name}")


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    """根据模型输出计算预测正确数量。

    logits 是模型最后一层的原始输出，不需要手动做 softmax。
    取最大值所在位置就是预测类别。
    """

    predictions = torch.argmax(logits, dim=1)
    correct = int((predictions == labels).sum().item())
    total = int(labels.size(0))
    return correct, total


def save_class_names(class_names: list[str], path: str | Path):
    """保存类别名称，预测单张图片时需要把类别编号还原成文字类别。"""

    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, ensure_ascii=False, indent=2)


def load_class_names(path: str | Path) -> list[str]:
    """读取类别名称。"""

    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def config_to_dict(config) -> dict:
    """将配置转换成可保存的普通字典。"""

    data = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}


def plot_training_curves(history: dict, figure_dir: str | Path):
    """绘制 Loss 曲线和 Accuracy 曲线。

    Loss 曲线能观察模型是否还在继续学习；Accuracy 曲线能观察分类正确率是否提升。
    如果训练集效果很好但验证集效果变差，通常说明模型可能过拟合。
    """

    figure_dir = Path(figure_dir)
    ensure_dir(figure_dir)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_dir / "accuracy_curve.png", dpi=200)
    plt.close()


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    save_path: str | Path,
    title: str = "Confusion Matrix",
):
    """绘制混淆矩阵。

    混淆矩阵能看出模型容易把哪一类错分成哪一类，比单独看 Accuracy 更适合做
    实验结果分析。
    """

    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    matrix = confusion_matrix(y_true, y_pred)
    fig_size = max(6, min(16, len(class_names) * 0.7))

    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
