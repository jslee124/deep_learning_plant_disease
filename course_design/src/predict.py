"""单张图片预测入口。

训练完成后，可以用本脚本加载最佳模型，对单张叶片图片进行预测。
"""

import argparse
from pathlib import Path
from typing import cast

import torch
from PIL import Image

from config import config
from dataset import get_eval_transform
from models import build_model
from utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="使用训练好的 CNN 模型预测单张图片")
    parser.add_argument("--image-path", type=str, required=True, help="待预测图片路径")
    parser.add_argument("--model-path", type=str, required=True, help="用于预测的模型权重路径")
    parser.add_argument("--image-size", type=int, default=config.image_size, help="输入图片尺寸")
    parser.add_argument("--top-k", type=int, default=3, help="显示置信度最高的前 K 个类别")
    return parser.parse_args()


def load_model(model_path: str | Path, device: torch.device):
    """加载训练好的模型和类别名称。"""

    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    model_name = checkpoint.get("model_name", config.model_name)
    dropout = checkpoint.get("config", {}).get("dropout", config.dropout)
    conv_kernel_size = checkpoint.get("config", {}).get("conv_kernel_size", 3)
    conv_blocks = checkpoint.get("config", {}).get("conv_blocks", 4)

    model = build_model(
        model_name=model_name,
        num_classes=len(class_names),
        dropout=float(dropout),
        conv_kernel_size=int(conv_kernel_size),
        conv_blocks=int(conv_blocks),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def predict_image(model, image_path: str | Path, class_names: list[str], image_size: int, top_k: int, device):
    """预测单张图片。

    softmax 会把模型输出的类别得分转换为概率，概率越高表示模型越相信该类别。
    """

    transform = get_eval_transform(image_size)
    image = Image.open(image_path).convert("RGB")
    # transform 最后包含 ToTensor，因此这里的结果是形状为 [3, H, W] 的 torch.Tensor。
    tensor = cast(torch.Tensor, transform(image))
    image_tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    top_k = min(top_k, len(class_names))
    values, indices = torch.topk(probabilities, k=top_k)
    results = []
    for value, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
        results.append((class_names[index], value))
    return results


def main():
    args = parse_args()
    device = get_device()
    model, class_names = load_model(args.model_path, device)
    results = predict_image(
        model=model,
        image_path=args.image_path,
        class_names=class_names,
        image_size=args.image_size,
        top_k=args.top_k,
        device=device,
    )

    print(f"图片路径: {args.image_path}")
    print("预测结果:")
    for rank, (class_name, confidence) in enumerate(results, start=1):
        print(f"Top {rank}: {class_name}，置信度 {confidence:.4f}")


if __name__ == "__main__":
    main()
