"""Visualize how a trained CNN transforms one plant image."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from config import config
from dataset import build_dataloaders
from models import build_model
from utils import get_device


NORMALIZE_MEAN = np.array([0.485, 0.456, 0.406])
NORMALIZE_STD = np.array([0.229, 0.224, 0.225])


def parse_args():
    parser = argparse.ArgumentParser(description="绘制 CNN 各卷积层特征图")
    parser.add_argument("--model-path", type=Path, required=True, help="最佳模型检查点")
    parser.add_argument("--sample-index", type=int, default=0, help="测试集样本索引")
    parser.add_argument("--data-dir", type=Path, default=None, help="数据集根目录")
    parser.add_argument("--output-path", type=Path, default=None, help="特征图保存路径")
    return parser.parse_args()


def denormalize(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(array * NORMALIZE_STD + NORMALIZE_MEAN, 0, 1)


def collect_conv_activations(model: nn.Module, image: torch.Tensor):
    activations = {}
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(
                module.register_forward_hook(
                    lambda _module, _inputs, output, layer_name=name: activations.__setitem__(
                        layer_name, output.detach().cpu()
                    )
                )
            )

    with torch.no_grad():
        logits = model(image)

    for hook in hooks:
        hook.remove()
    return logits, activations


def select_feature_maps(activation: torch.Tensor):
    feature_tensor = activation[0]
    mean_map = feature_tensor.mean(dim=0).numpy()
    channel_energy = feature_tensor.abs().mean(dim=(1, 2))
    channel_index = int(torch.argmax(channel_energy).item())
    strongest_map = feature_tensor[channel_index].numpy()
    return mean_map, strongest_map, channel_index


def save_figure(
    original: np.ndarray,
    activations: dict,
    true_label: str,
    predicted_label: str,
    confidence: float,
    output_path: Path,
):
    columns = len(activations) + 1
    fig, axes = plt.subplots(2, columns, figsize=(3.2 * columns, 6.2))

    axes[0, 0].imshow(original)
    axes[0, 0].set_title(f"Input\nTrue: {true_label}")
    axes[1, 0].imshow(original)
    axes[1, 0].set_title(f"Prediction: {predicted_label}\nConfidence: {confidence:.2%}")

    for column, (layer_name, activation) in enumerate(activations.items(), start=1):
        mean_map, strongest_map, channel_index = select_feature_maps(activation)
        axes[0, column].imshow(mean_map, cmap="viridis")
        axes[0, column].set_title(f"{layer_name}\nMean activation")
        axes[1, column].imshow(strongest_map, cmap="magma")
        axes[1, column].set_title(f"Strongest channel #{channel_index}")

    for axis in axes.flat:
        axis.axis("off")

    fig.suptitle("CNN Feature Maps: From Pixels to Disease Features", fontsize=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"没有找到模型检查点: {args.model_path}")

    device = get_device()
    checkpoint = torch.load(args.model_path, map_location=device)
    checkpoint_config = checkpoint.get("config", {})

    if args.data_dir is not None:
        config.data_dir = args.data_dir
    elif checkpoint_config.get("data_dir"):
        config.data_dir = Path(checkpoint_config["data_dir"])
    config.image_size = int(checkpoint_config.get("image_size", config.image_size))
    config.batch_size = 1
    config.num_workers = 0
    config.use_augmentation = False

    _, _, test_loader, class_names, _ = build_dataloaders(config)
    test_dataset = test_loader.dataset
    if not 0 <= args.sample_index < len(test_dataset):
        raise IndexError(f"sample-index 必须在 0 到 {len(test_dataset) - 1} 之间")

    image, label_index = test_dataset[args.sample_index]
    checkpoint_class_names = checkpoint.get("class_names", class_names)
    model = build_model(
        model_name=checkpoint.get("model_name", config.model_name),
        num_classes=len(checkpoint_class_names),
        dropout=float(checkpoint_config.get("dropout", config.dropout)),
        conv_kernel_size=int(checkpoint_config.get("conv_kernel_size", 3)),
        conv_blocks=int(checkpoint_config.get("conv_blocks", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logits, activations = collect_conv_activations(model, image.unsqueeze(0).to(device))
    probabilities = torch.softmax(logits, dim=1)[0]
    predicted_index = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_index].item())

    output_path = args.output_path or args.model_path.parents[1] / "figures" / (
        f"feature_maps_sample_{args.sample_index}.png"
    )
    save_figure(
        original=denormalize(image),
        activations=activations,
        true_label=checkpoint_class_names[label_index],
        predicted_label=checkpoint_class_names[predicted_index],
        confidence=confidence,
        output_path=output_path,
    )

    metadata = {
        "model_path": str(args.model_path),
        "sample_index": args.sample_index,
        "true_label": checkpoint_class_names[label_index],
        "predicted_label": checkpoint_class_names[predicted_index],
        "confidence": confidence,
        "correct": predicted_index == label_index,
        "activation_shapes": {
            name: list(activation.shape) for name, activation in activations.items()
        },
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"特征图已保存到: {output_path}")
    print(
        f"真实类别: {metadata['true_label']} | 预测类别: {metadata['predicted_label']} | "
        f"置信度: {confidence:.2%}"
    )


if __name__ == "__main__":
    main()
