"""模型训练入口。

本脚本实现第二阶段 CNN 基线训练流程：
数据读取 -> 模型创建 -> 损失函数 -> 优化器 -> 训练循环 -> 验证循环
-> 保存最佳模型 -> 测试集评估 -> 绘制曲线和混淆矩阵。
"""

import argparse
import csv
import re
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from config import config
from dataset import build_dataloaders
from models import build_model
from utils import (
    accuracy_from_logits,
    append_csv_row,
    config_to_dict,
    current_time_text,
    get_device,
    get_hardware_info,
    get_optimizer,
    plot_confusion_matrix,
    plot_training_curves,
    prepare_output_dirs,
    save_json,
    save_class_names,
    set_seed,
)


def parse_args():
    """读取命令行参数，方便做不同实验。"""

    parser = argparse.ArgumentParser(description="训练植物病害 CNN 分类模型")
    parser.add_argument("--data-dir", type=str, default=None, help="数据集目录，默认使用项目 data/")
    parser.add_argument("--epochs", type=int, default=config.epochs, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=config.batch_size, help="Batch Size")
    parser.add_argument("--num-workers", type=int, default=config.num_workers, help="DataLoader 工作进程数")
    parser.add_argument("--lr", type=float, default=config.learning_rate, help="学习率")
    parser.add_argument("--image-size", type=int, default=config.image_size, help="输入图片尺寸")
    parser.add_argument("--optimizer", type=str, default=config.optimizer_name, choices=["Adam", "SGD", "RMSprop"])
    parser.add_argument("--dropout", type=float, default=config.dropout, help="Dropout 比例")
    parser.add_argument("--kernel-size", type=int, default=config.conv_kernel_size, help="卷积核大小，如 3 或 5")
    parser.add_argument("--conv-blocks", type=int, default=config.conv_blocks, choices=[2, 3, 4], help="卷积块数量")
    parser.add_argument("--experiment-name", type=str, default=None, help="唯一实验名称，用于隔离输出")
    parser.add_argument("--resume", action="store_true", help="从该实验的 latest_checkpoint.pth 继续训练")
    parser.add_argument("--no-augmentation", action="store_true", help="关闭数据增强")
    return parser.parse_args()


def apply_args_to_config(args):
    """用命令行参数覆盖默认配置。"""

    if args.data_dir is not None:
        config.data_dir = Path(args.data_dir)
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.num_workers = args.num_workers
    config.learning_rate = args.lr
    config.image_size = args.image_size
    config.optimizer_name = args.optimizer
    config.dropout = args.dropout
    config.conv_kernel_size = args.kernel_size
    config.conv_blocks = args.conv_blocks
    config.use_augmentation = not args.no_augmentation
    if config.conv_kernel_size < 1 or config.conv_kernel_size % 2 == 0:
        raise ValueError("--kernel-size 必须是正奇数，例如 3 或 5。")
    if config.num_workers < 0:
        raise ValueError("--num-workers 不能小于 0。")
    if args.resume and not args.experiment_name:
        raise ValueError("使用 --resume 时必须同时指定 --experiment-name。")
    raw_name = args.experiment_name or (
        f"cnn_k{config.conv_kernel_size}_b{config.conv_blocks}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    experiment_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("._")
    if not experiment_name:
        raise ValueError("实验名称至少需要包含一个字母、数字、下划线、点或短横线。")
    config.configure_run(experiment_name)
    if args.resume:
        if not config.latest_model_path.exists():
            raise FileNotFoundError(f"没有找到可恢复检查点: {config.latest_model_path}")
    elif config.run_dir.exists() and any(config.run_dir.iterdir()):
        raise FileExistsError(
            f"实验目录已存在且非空: {config.run_dir}。请使用新的 --experiment-name，避免覆盖结果。"
        )


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
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

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

    train_loader, val_loader, test_loader, class_names, dataset_info = build_dataloaders(config)
    num_classes = len(class_names)
    save_class_names(class_names, config.class_names_path)
    hardware_info = get_hardware_info(device)
    run_started_at = current_time_text()
    run_config_path = config.log_dir / "run_config.json"
    if not args.resume:
        save_json(
            {
                "started_at": run_started_at,
                "config": config_to_dict(config),
                "dataset": dataset_info,
                "hardware": hardware_info,
            },
            run_config_path,
        )

    model = build_model(
        model_name=config.model_name,
        num_classes=num_classes,
        dropout=config.dropout,
        conv_kernel_size=config.conv_kernel_size,
        conv_blocks=config.conv_blocks,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

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
    best_epoch = 0
    epoch_times = []
    start_epoch = 1
    previous_training_seconds = 0.0

    if args.resume:
        latest_checkpoint = torch.load(config.latest_model_path, map_location=device)
        model.load_state_dict(latest_checkpoint["model_state_dict"])
        optimizer.load_state_dict(latest_checkpoint["optimizer_state_dict"])
        history = latest_checkpoint["history"]
        epoch_times = latest_checkpoint["epoch_times"]
        best_val_acc = latest_checkpoint["best_val_acc"]
        best_epoch = latest_checkpoint["best_epoch"]
        previous_training_seconds = latest_checkpoint["training_seconds"]
        start_epoch = latest_checkpoint["epoch"] + 1
        run_started_at = latest_checkpoint.get("run_started_at", run_started_at)
        if start_epoch > config.epochs:
            print(f"检查点已完成 {latest_checkpoint['epoch']} 个 Epoch，直接进行最终测试。")
        else:
            print(f"从 Epoch {start_epoch} 恢复训练，之前最佳验证准确率: {best_val_acc:.4f}")

    start_time = time.time()

    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start = time.time()
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
        epoch_times.append(time.time() - epoch_start)

        print(
            f"Epoch [{epoch}/{config.epochs}] "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
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

        current_training_seconds = time.time() - start_time
        latest_checkpoint = {
            "epoch": epoch,
            "model_name": config.model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "num_classes": num_classes,
            "class_names": class_names,
            "history": history,
            "epoch_times": epoch_times,
            "training_seconds": previous_training_seconds + current_training_seconds,
            "run_started_at": run_started_at,
            "config": config_to_dict(config),
        }
        torch.save(latest_checkpoint, config.latest_model_path)
        save_history_csv(history, config.log_dir / "training_history.csv")
        print(f"已保存恢复检查点: {config.latest_model_path}")

    total_time = previous_training_seconds + (time.time() - start_time)

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

    samples_per_second = (
        dataset_info["train_samples"] * config.epochs / total_time if total_time > 0 else 0.0
    )
    completed_at = current_time_text()
    result_row = {
        "experiment_name": config.experiment_name,
        "started_at": run_started_at,
        "completed_at": completed_at,
        "model_name": config.model_name,
        "kernel_size": config.conv_kernel_size,
        "conv_blocks": config.conv_blocks,
        "parameters": parameter_count,
        "train_samples": dataset_info["train_samples"],
        "val_samples": dataset_info["val_samples"],
        "test_samples": dataset_info["test_samples"],
        "split_strategy": dataset_info["split_strategy"],
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "optimizer": config.optimizer_name,
        "dropout": config.dropout,
        "augmentation": config.use_augmentation,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "final_train_loss": history["train_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "test_loss": test_loss,
        "test_acc": test_acc,
        "total_seconds": total_time,
        "mean_epoch_seconds": sum(epoch_times) / len(epoch_times),
        "samples_per_second": samples_per_second,
    }
    append_csv_row(config.experiment_results_path, result_row)
    append_csv_row(
        config.hardware_benchmark_path,
        {
            "experiment_name": config.experiment_name,
            "completed_at": completed_at,
            **hardware_info,
            "parameters": parameter_count,
            "total_seconds": total_time,
            "mean_epoch_seconds": sum(epoch_times) / len(epoch_times),
            "samples_per_second": samples_per_second,
        },
    )
    save_json(
        {
            "result": result_row,
            "hardware": hardware_info,
            "dataset": dataset_info,
        },
        config.log_dir / "final_result.json",
    )

    print(f"Loss 曲线已保存到: {config.figure_dir / config.loss_curve_name}")
    print(f"Accuracy 曲线已保存到: {config.figure_dir / config.accuracy_curve_name}")
    print(f"混淆矩阵已保存到: {config.figure_dir / config.confusion_matrix_name}")
    print(f"训练历史已保存到: {config.log_dir / 'training_history.csv'}")
    print(f"实验汇总已追加到: {config.experiment_results_path}")


if __name__ == "__main__":
    main()
