"""RNN 实验通用工具。"""

import csv
import json
import platform
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def prepare_output_dirs(config):
    for path in (config.model_dir, config.log_dir, config.figure_dir):
        ensure_dir(path)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_hardware_info(device):
    return {
        "operating_system": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available",
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda or "Not available",
        "pytorch_version": torch.__version__,
        "training_device": str(device),
    }


def config_to_dict(config):
    data = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}


def save_json(data, path):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_csv_row(path, row):
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def current_time_text():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save_history_csv(history, path):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])
        for index in range(len(history["train_loss"])):
            writer.writerow(
                [
                    index + 1,
                    history["train_loss"][index],
                    history["train_acc"][index],
                    history["val_loss"][index],
                    history["val_acc"][index],
                    history["epoch_seconds"][index],
                ]
            )


def plot_training_curves(history, figure_dir):
    figure_dir = Path(figure_dir)
    ensure_dir(figure_dir)
    epochs = range(1, len(history["train_loss"]) + 1)
    for metric, ylabel in (("loss", "Loss"), ("acc", "Accuracy")):
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history[f"train_{metric}"], marker="o", label=f"Train {ylabel}")
        plt.plot(epochs, history[f"val_{metric}"], marker="o", label=f"Validation {ylabel}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"RNN Training and Validation {ylabel}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / f"{metric}_curve.png", dpi=200)
        plt.close()


def plot_confusion(y_true, y_pred, class_names, path):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("MNIST Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_prediction_examples(images, labels, predictions, path):
    images = images.cpu()
    fig, axes = plt.subplots(2, 5, figsize=(11, 5))
    for index, axis in enumerate(axes.flat):
        axis.imshow(images[index, 0], cmap="gray")
        color = "#237a57" if labels[index] == predictions[index] else "#b84233"
        axis.set_title(f"True {labels[index]} / Pred {predictions[index]}", color=color, pad=6)
        axis.axis("off")
    fig.suptitle("MNIST RNN Prediction Examples")
    fig.subplots_adjust(top=0.86, bottom=0.04, hspace=0.32, wspace=0.08)
    fig.savefig(path, dpi=200)
    plt.close(fig)
