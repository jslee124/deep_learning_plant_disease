"""模型训练入口。

本脚本实现第二阶段 CNN 基线训练流程：
数据读取 -> 模型创建 -> 损失函数 -> 优化器 -> 训练循环 -> 验证循环
-> 保存最佳模型 -> 测试集评估 -> 绘制曲线和混淆矩阵。
"""

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from config import config
from dataset import build_dataloaders
from models import build_model
from utils import (
    accuracy_from_logits,
    config_to_dict,
    get_device,
    get_optimizer,
    plot_confusion_matrix,
    plot_training_curves,
    prepare_output_dirs,
    save_class_names,
    set_seed,
)


def parse_args():
    """读取命令行参数，方便做不同实验。"""

    parser = argparse.ArgumentParser(description="训练植物病害 CNN 分类模型")
    parser.add_argument("--data-dir", type=str, default=None, help="数据集目录，默认使用项目 data/")
    parser.add_argument("--epochs", type=int, default=config.epochs, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=config.batch_size, help="Batch Size")
    parser.add_argument("--lr", type=float, default=config.learning_rate, help="学习率")
    parser.add_argument("--image-size", type=int, default=config.image_size, help="输入图片尺寸")
    parser.add_argument("--optimizer", type=str, default=config.optimizer_name, choices=["Adam", "SGD", "RMSprop"])
    parser.add_argument("--dropout", type=float, default=config.dropout, help="Dropout 比例")
    parser.add_argument("--no-augmentation", action="store_true", help="关闭数据增强")
    return parser.parse_args()


def apply_args_to_config(args):
    """用命令行参数覆盖默认配置。"""

    if args.data_dir is not None:
        config.data_dir = Path(args.data_dir)
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.image_size = args.image_size
    config.optimizer_name = args.optimizer
    config.dropout = args.dropout
    config.use_augmentation = not args.no_augmentation


def train_one_epoch(model, data_loader, criterion, optimizer, device, epoch: int):
    """训练一个 Epoch。

    一个 Epoch 表示模型完整看完一遍训练集。训练阶段需要：
    1. 前向传播：模型根据图片输出类别得分。
    2. 计算损失：比较预测结果和真实标签的差距。
    3. 反向传播：计算每个参数应该如何调整。
    4. 优化器更新参数：让模型下一次预测更接近真实标签。
    """

    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} Train", leave=False)
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        correct, total = accuracy_from_logits(logits, labels)
        running_loss += loss.item() * batch_size
        running_correct += correct
        running_total += total

        progress_bar.set_postfix(
            loss=running_loss / running_total,
            acc=running_correct / running_total,
        )

    return running_loss / running_total, running_correct / running_total


def evaluate_one_epoch(model, data_loader, criterion, device, desc: str):
    """验证或测试一个 Epoch。

    验证/测试阶段只评估模型效果，不更新参数，所以使用 torch.no_grad() 节省显存。
    """

    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=desc, leave=False)
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)

            batch_size = labels.size(0)
            correct, total = accuracy_from_logits(logits, labels)
            running_loss += loss.item() * batch_size
            running_correct += correct
            running_total += total
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

            progress_bar.set_postfix(
                loss=running_loss / running_total,
                acc=running_correct / running_total,
            )

    return (
        running_loss / running_total,
        running_correct / running_total,
        all_labels,
        all_predictions,
    )


def save_history_csv(history: dict, save_path: Path):
    """保存每个 Epoch 的训练记录，方便后续写实验分析。"""

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for index in range(len(history["train_loss"])):
            writer.writerow(
                [
                    index + 1,
                    history["train_loss"][index],
                    history["train_acc"][index],
                    history["val_loss"][index],
                    history["val_acc"][index],
                ]
            )


def main():
    args = parse_args()
    apply_args_to_config(args)

    set_seed(config.seed)
    prepare_output_dirs(config)
    device = get_device()

    print(f"当前训练设备: {device}")
    print(f"数据目录: {config.data_dir}")

    train_loader, val_loader, test_loader, class_names = build_dataloaders(config)
    num_classes = len(class_names)
    save_class_names(class_names, config.class_names_path)

    model = build_model(
        model_name=config.model_name,
        num_classes=num_classes,
        dropout=config.dropout,
    ).to(device)

    # CrossEntropyLoss 作用：
    # 它适合多分类任务，会把模型输出的类别得分和真实标签进行比较。
    # 分错且信心很高时 Loss 会很大；分对且信心很高时 Loss 会变小。
    # PyTorch 的 CrossEntropyLoss 内部已经包含 softmax，所以训练时不需要手动 softmax。
    criterion = nn.CrossEntropyLoss()

    # Adam 优化器作用：
    # 优化器根据 loss.backward() 得到的梯度更新模型参数。
    # Adam 会自动调整每个参数的更新幅度，通常收敛比较稳定，适合作为基线实验。
    optimizer = get_optimizer(model, config.optimizer_name, config.learning_rate)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, _, _ = evaluate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            desc=f"Epoch {epoch} Val",
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch}/{config.epochs}] "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_name": config.model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "num_classes": num_classes,
                "class_names": class_names,
                "config": config_to_dict(config),
            }
            torch.save(checkpoint, config.best_model_path)
            print(f"已保存当前最佳模型: {config.best_model_path}")

    total_time = time.time() - start_time

    print("加载最佳模型进行测试集评估...")
    checkpoint = torch.load(config.best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, y_true, y_pred = evaluate_one_epoch(
        model,
        test_loader,
        criterion,
        device,
        desc="Test",
    )

    print(f"测试集 Loss: {test_loss:.4f}")
    print(f"测试集 Accuracy: {test_acc:.4f}")
    print(f"总训练时间: {total_time:.2f} 秒")

    plot_training_curves(history, config.figure_dir)
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_path=config.figure_dir / config.confusion_matrix_name,
        title="Test Confusion Matrix",
    )
    save_history_csv(history, config.log_dir / "training_history.csv")

    print(f"Loss 曲线已保存到: {config.figure_dir / config.loss_curve_name}")
    print(f"Accuracy 曲线已保存到: {config.figure_dir / config.accuracy_curve_name}")
    print(f"混淆矩阵已保存到: {config.figure_dir / config.confusion_matrix_name}")
    print(f"训练历史已保存到: {config.log_dir / 'training_history.csv'}")


if __name__ == "__main__":
    main()
