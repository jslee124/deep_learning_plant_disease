"""汇总课程设计的自建 CNN 与 ResNet18 实验。"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = Path(__file__).resolve().parents[1] / "results"
RUN_SOURCES = [
    ("experiment_1_cnn", "course_design_conv_blocks_3", "自建 CNN: 3 blocks, 3x3", "CNN 3B 3x3"),
    ("experiment_1_cnn", "cnn_kernel_3x3_baseline_v2", "自建 CNN: 4 blocks, 3x3", "CNN 4B 3x3"),
    ("experiment_1_cnn", "cnn_kernel_5x5_comparison", "自建 CNN: 4 blocks, 5x5", "CNN 4B 5x5"),
    ("course_design", "resnet18_imagenet_finetune", "ResNet18: ImageNet fine-tune", "ResNet18 FT"),
]


def load_run(project_name, run_name, label, chart_label):
    path = (
        WORKSPACE_ROOT
        / project_name
        / "outputs"
        / "runs"
        / run_name
        / "logs"
        / "final_result.json"
    )
    with path.open("r", encoding="utf-8") as file:
        result = json.load(file)["result"]
    return {
        "experiment_name": result["experiment_name"],
        "label": label,
        "chart_label": chart_label,
        "model_name": result["model_name"],
        "pretrained": result.get("pretrained", False),
        "kernel_size": result["kernel_size"],
        "conv_blocks": result["conv_blocks"],
        "parameters": result["parameters"],
        "trainable_parameters": result.get("trainable_parameters", result["parameters"]),
        "best_val_acc": result["best_val_acc"],
        "final_train_acc": result["final_train_acc"],
        "test_acc": result["test_acc"],
        "test_loss": result["test_loss"],
        "total_seconds": result["total_seconds"],
        "mean_epoch_seconds": result["mean_epoch_seconds"],
        "samples_per_second": result["samples_per_second"],
        "val_test_gap_pp": abs(result["best_val_acc"] - result["test_acc"]) * 100,
    }


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [load_run(*source) for source in RUN_SOURCES]
    fields = list(rows[0].keys())
    with (RESULT_DIR / "cnn_research_summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (RESULT_DIR / "cnn_research_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    labels = [row["chart_label"] for row in rows]
    colors = ["#b9a44c", "#2c7772", "#cf6f3d", "#315c8c"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    charts = [
        ("test_acc", "Test Accuracy (%)", 100),
        ("parameters", "Parameters (million)", 1 / 1_000_000),
        ("mean_epoch_seconds", "Mean Epoch Time (s)", 1),
        ("samples_per_second", "Throughput (images/s)", 1),
        ("test_loss", "Test Loss", 1),
        ("val_test_gap_pp", "Validation-Test Gap (pp)", 1),
    ]
    for axis, (key, title, scale) in zip(axes.flat, charts):
        axis.bar(labels, [row[key] * scale for row in rows], color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Plant Disease Model Comparison")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "cnn_research_summary.png", dpi=200)
    plt.close(fig)

    shallow, baseline, large_kernel, resnet = rows
    depth_gain = (baseline["test_acc"] - shallow["test_acc"]) * 100
    kernel_gain = (large_kernel["test_acc"] - baseline["test_acc"]) * 100
    transfer_gain = (resnet["test_acc"] - baseline["test_acc"]) * 100
    transfer_parameter_ratio = resnet["parameters"] / baseline["parameters"]
    transfer_epoch_time_change = (
        resnet["mean_epoch_seconds"] / baseline["mean_epoch_seconds"] - 1
    ) * 100
    content = f"""# 课程设计当前对比结果

| 模型 | 验证 Accuracy | 测试 Accuracy | 测试 Loss | 参数量 | 平均 Epoch | 吞吐量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 自建 CNN: 3 blocks, 3x3 | {shallow['best_val_acc']:.2%} | {shallow['test_acc']:.2%} | {shallow['test_loss']:.4f} | {shallow['parameters']:,} | {shallow['mean_epoch_seconds']:.2f}s | {shallow['samples_per_second']:.2f}/s |
| 自建 CNN: 4 blocks, 3x3 | {baseline['best_val_acc']:.2%} | {baseline['test_acc']:.2%} | {baseline['test_loss']:.4f} | {baseline['parameters']:,} | {baseline['mean_epoch_seconds']:.2f}s | {baseline['samples_per_second']:.2f}/s |
| 自建 CNN: 4 blocks, 5x5 | {large_kernel['best_val_acc']:.2%} | {large_kernel['test_acc']:.2%} | {large_kernel['test_loss']:.4f} | {large_kernel['parameters']:,} | {large_kernel['mean_epoch_seconds']:.2f}s | {large_kernel['samples_per_second']:.2f}/s |
| ResNet18: ImageNet 全层微调 | {resnet['best_val_acc']:.2%} | {resnet['test_acc']:.2%} | {resnet['test_loss']:.4f} | {resnet['parameters']:,} | {resnet['mean_epoch_seconds']:.2f}s | {resnet['samples_per_second']:.2f}/s |

## 主要发现

- 在 `3x3` 卷积核下，从 3 个卷积块增加到 4 个卷积块，测试 Accuracy 提高
  {depth_gain:.2f} 个百分点，说明更深的分层特征提取对 38 类病害识别非常重要。
- 在 4 个卷积块下，从 `3x3` 改为 `5x5`，测试 Accuracy 提高
  {kernel_gain:.2f} 个百分点，但参数量和训练成本明显增加。
- ResNet18 使用 ImageNet 预训练权重并微调全部网络，测试 Accuracy 比自建
  `4 blocks, 3x3` 提高 {transfer_gain:.2f} 个百分点；参数量为其
  {transfer_parameter_ratio:.2f} 倍，平均每 Epoch 时间增加
  {transfer_epoch_time_change:.2f}%。
- ResNet18 的验证-测试差距为 {resnet['val_test_gap_pp']:.3f} 个百分点，说明在
  当前独立测试划分上保持了较稳定的泛化表现。
- 单次计时容易受到 DataLoader 启动、磁盘缓存、GPU 温度和后台负载影响。当前计时
  作为实测记录保留，但模型效率排序需要预热后重复运行才能形成强结论。
- 训练集 Accuracy 来自启用随机增强和 Dropout 的训练态统计，可能低于验证和测试
  Accuracy，因此不把该数值直接解释为常规的训练-测试泛化误差。
"""
    (RESULT_DIR / "cnn_research_summary.md").write_text(content, encoding="utf-8")
    print(f"课程设计汇总已保存到: {RESULT_DIR}")


if __name__ == "__main__":
    main()
