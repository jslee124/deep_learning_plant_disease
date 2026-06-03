"""模型评估入口。

训练完成后，可以单独运行本脚本在验证集或测试集上评估模型，并重新生成混淆矩阵。
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from config import config
from dataset import build_dataloaders
from models import build_model
from utils import accuracy_from_logits, get_device, plot_confusion_matrix, prepare_output_dirs


def parse_args():
    parser = argparse.ArgumentParser(description="评估植物病害 CNN 分类模型")
    parser.add_argument("--model-path", type=str, default=str(config.best_model_path), help="模型权重路径")
    parser.add_argument("--data-dir", type=str, default=None, help="数据集目录，默认使用项目 data/")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="评估哪个数据集")
    parser.add_argument("--batch-size", type=int, default=config.batch_size, help="Batch Size")
    parser.add_argument("--image-size", type=int, default=config.image_size, help="输入图片尺寸")
    return parser.parse_args()


def evaluate_model(model, data_loader, criterion, device, desc: str):
    """评估模型并收集预测结果。"""

    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=desc):
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

    return running_loss / running_total, running_correct / running_total, all_labels, all_predictions


def main():
    args = parse_args()
    if args.data_dir is not None:
        config.data_dir = Path(args.data_dir)
    config.batch_size = args.batch_size
    config.image_size = args.image_size
    config.use_augmentation = False

    prepare_output_dirs(config)
    device = get_device()

    train_loader, val_loader, test_loader, class_names = build_dataloaders(config)
    loader_map = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    data_loader = loader_map[args.split]

    checkpoint = torch.load(args.model_path, map_location=device)
    checkpoint_class_names = checkpoint.get("class_names", class_names)
    model_name = checkpoint.get("model_name", config.model_name)
    dropout = checkpoint.get("config", {}).get("dropout", config.dropout)

    model = build_model(
        model_name=model_name,
        num_classes=len(checkpoint_class_names),
        dropout=float(dropout),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # CrossEntropyLoss 用于衡量多分类预测与真实标签之间的差距。
    criterion = nn.CrossEntropyLoss()
    loss, accuracy, y_true, y_pred = evaluate_model(
        model,
        data_loader,
        criterion,
        device,
        desc=f"Evaluate {args.split}",
    )

    save_path = config.figure_dir / f"{args.split}_confusion_matrix.png"
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=checkpoint_class_names,
        save_path=save_path,
        title=f"{args.split.title()} Confusion Matrix",
    )

    print(f"{args.split} Loss: {loss:.4f}")
    print(f"{args.split} Accuracy: {accuracy:.4f}")
    print(f"混淆矩阵已保存到: {save_path}")


if __name__ == "__main__":
    main()
