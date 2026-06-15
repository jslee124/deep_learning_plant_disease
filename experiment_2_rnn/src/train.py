"""训练基础 RNN，并完整记录 MNIST 对比实验结果。"""

import argparse
import re
import time
from datetime import datetime

import torch
from torch import nn
from tqdm import tqdm

from config import config
from dataset import build_dataloaders
from models import build_model
from utils import (
    append_csv_row,
    config_to_dict,
    current_time_text,
    get_device,
    get_hardware_info,
    plot_confusion,
    plot_prediction_examples,
    plot_training_curves,
    prepare_output_dirs,
    save_history_csv,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="训练 MNIST 基础 RNN")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--hidden-size", type=int, default=config.hidden_size)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--lr", type=float, default=config.learning_rate)
    parser.add_argument("--num-workers", type=int, default=config.num_workers)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def apply_args(args):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.experiment_name).strip("._")
    if not name:
        raise ValueError("实验名称不能为空。")
    if args.hidden_size <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("hidden-size、epochs 和 batch-size 必须大于 0。")
    config.configure_run(name)
    config.hidden_size = args.hidden_size
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.num_workers = args.num_workers
    if args.resume and not config.latest_model_path.exists():
        raise FileNotFoundError(f"没有找到检查点: {config.latest_model_path}")
    if not args.resume and config.run_dir.exists() and any(config.run_dir.iterdir()):
        raise FileExistsError(f"实验目录非空，为避免覆盖请更换名称: {config.run_dir}")


def run_epoch(model, loader, criterion, device, optimizer=None, desc="Evaluate"):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += int((predictions == labels).sum().item())
            total_samples += batch_size
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    return total_loss / total_samples, total_correct / total_samples, all_labels, all_predictions


def main():
    args = parse_args()
    apply_args(args)
    set_seed(config.seed)
    prepare_output_dirs(config)
    device = get_device()

    train_loader, val_loader, test_loader, class_names, dataset_info = build_dataloaders(
        config, download=not args.no_download
    )
    model = build_model(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    hardware = get_hardware_info(device)
    started_at = current_time_text()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_seconds": []}
    best_val_acc = -1.0
    best_epoch = 0
    start_epoch = 1
    previous_seconds = 0.0

    if args.resume:
        checkpoint = torch.load(config.latest_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = checkpoint["history"]
        best_val_acc = checkpoint["best_val_acc"]
        best_epoch = checkpoint["best_epoch"]
        start_epoch = checkpoint["epoch"] + 1
        previous_seconds = checkpoint["training_seconds"]
        started_at = checkpoint["started_at"]
    else:
        save_json(
            {"started_at": started_at, "config": config_to_dict(config), "dataset": dataset_info, "hardware": hardware},
            config.log_dir / "run_config.json",
        )

    print(f"训练设备: {device} | 隐藏维度: {config.hidden_size} | 参数量: {parameter_count:,}")
    training_start = time.time()
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer, f"Epoch {epoch} Train"
        )
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, device, desc=f"Epoch {epoch} Val")
        epoch_seconds = time.time() - epoch_start
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_seconds"].append(epoch_seconds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config_to_dict(config),
                    "class_names": class_names,
                    "best_val_acc": best_val_acc,
                },
                config.best_model_path,
            )

        elapsed = previous_seconds + time.time() - training_start
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "training_seconds": elapsed,
                "started_at": started_at,
                "config": config_to_dict(config),
            },
            config.latest_model_path,
        )
        save_history_csv(history, config.log_dir / "training_history.csv")
        print(
            f"Epoch {epoch}/{config.epochs} | train {train_acc:.2%} | val {val_acc:.2%} | "
            f"{epoch_seconds:.2f}s"
        )

    total_seconds = previous_seconds + time.time() - training_start
    checkpoint = torch.load(config.best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, y_true, y_pred = run_epoch(model, test_loader, criterion, device, desc="Test")

    sample_images, sample_labels = next(iter(test_loader))
    with torch.no_grad():
        sample_predictions = model(sample_images[:10].to(device)).argmax(dim=1).cpu().tolist()
    plot_prediction_examples(
        sample_images[:10], sample_labels[:10].tolist(), sample_predictions, config.figure_dir / "prediction_examples.png"
    )
    plot_training_curves(history, config.figure_dir)
    plot_confusion(y_true, y_pred, class_names, config.figure_dir / "confusion_matrix.png")

    result = {
        "experiment_name": config.experiment_name,
        "started_at": started_at,
        "completed_at": current_time_text(),
        "model_name": "basic_rnn",
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "parameters": parameter_count,
        "train_samples": dataset_info["train_samples"],
        "val_samples": dataset_info["val_samples"],
        "test_samples": dataset_info["test_samples"],
        "split_strategy": dataset_info["split_strategy"],
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "optimizer": config.optimizer_name,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "final_train_loss": history["train_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "test_loss": test_loss,
        "test_acc": test_acc,
        "total_seconds": total_seconds,
        "mean_epoch_seconds": sum(history["epoch_seconds"]) / len(history["epoch_seconds"]),
        "samples_per_second": dataset_info["train_samples"] * config.epochs / total_seconds,
        "train_test_gap_pp": (history["train_acc"][-1] - test_acc) * 100,
    }
    append_csv_row(config.experiment_results_path, result)
    append_csv_row(
        config.hardware_benchmark_path,
        {
            "experiment_name": config.experiment_name,
            "completed_at": result["completed_at"],
            **hardware,
            "parameters": parameter_count,
            "total_seconds": total_seconds,
            "mean_epoch_seconds": result["mean_epoch_seconds"],
            "samples_per_second": result["samples_per_second"],
        },
    )
    save_json({"result": result, "hardware": hardware, "dataset": dataset_info}, config.log_dir / "final_result.json")
    print(f"测试准确率: {test_acc:.2%} | 最佳验证准确率: {best_val_acc:.2%}")
    print(f"结果目录: {config.run_dir}")


if __name__ == "__main__":
    main()
