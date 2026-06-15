"""Compare completed CNN runs and export report-ready artifacts."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="汇总两个 CNN 实验的核心指标")
    parser.add_argument("run_a", help="第一个实验名称或 final_result.json 路径")
    parser.add_argument("run_b", help="第二个实验名称或 final_result.json 路径")
    parser.add_argument("--output-dir", type=Path, default=None, help="对比结果输出目录")
    return parser.parse_args()


def resolve_result_path(value: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path

    result_path = PROJECT_ROOT / "outputs" / "runs" / value / "logs" / "final_result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"没有找到实验结果: {result_path}")
    return result_path


def load_result(value: str) -> dict:
    path = resolve_result_path(value)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    result = payload["result"]
    return {
        "experiment_name": result["experiment_name"],
        "kernel_size": result["kernel_size"],
        "parameters": result["parameters"],
        "best_epoch": result["best_epoch"],
        "best_val_acc": result["best_val_acc"],
        "test_acc": result["test_acc"],
        "val_test_gap_pp": abs(result["best_val_acc"] - result["test_acc"]) * 100,
        "total_seconds": result["total_seconds"],
        "mean_epoch_seconds": result["mean_epoch_seconds"],
        "samples_per_second": result["samples_per_second"],
    }


def calculate_changes(baseline: dict, comparison: dict) -> dict:
    return {
        "test_accuracy_change_pp": (comparison["test_acc"] - baseline["test_acc"]) * 100,
        "validation_accuracy_change_pp": (comparison["best_val_acc"] - baseline["best_val_acc"]) * 100,
        "parameter_ratio": comparison["parameters"] / baseline["parameters"],
        "mean_epoch_time_change_percent": (
            comparison["mean_epoch_seconds"] / baseline["mean_epoch_seconds"] - 1
        )
        * 100,
        "throughput_change_percent": (
            comparison["samples_per_second"] / baseline["samples_per_second"] - 1
        )
        * 100,
        "val_test_gap_change_pp": comparison["val_test_gap_pp"] - baseline["val_test_gap_pp"],
    }


def save_csv(rows: list[dict], path: Path):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_chart(rows: list[dict], path: Path):
    labels = [f"{row['kernel_size']}x{row['kernel_size']}" for row in rows]
    colors = ["#2f6b4f", "#d8783f"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].bar(labels, [row["test_acc"] * 100 for row in rows], color=colors)
    axes[0, 0].set_title("Test Accuracy")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_ylim(90, 96)

    axes[0, 1].bar(labels, [row["parameters"] / 1_000_000 for row in rows], color=colors)
    axes[0, 1].set_title("Trainable Parameters")
    axes[0, 1].set_ylabel("Parameters (million)")

    axes[1, 0].bar(labels, [row["mean_epoch_seconds"] for row in rows], color=colors)
    axes[1, 0].set_title("Mean Epoch Time")
    axes[1, 0].set_ylabel("Seconds")

    axes[1, 1].bar(labels, [row["val_test_gap_pp"] for row in rows], color=colors)
    axes[1, 1].set_title("Validation-Test Gap")
    axes[1, 1].set_ylabel("Absolute gap (percentage points)")

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle("CNN Kernel Size Comparison", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_markdown(rows: list[dict], changes: dict, path: Path):
    a, b = rows
    content = f"""# CNN 卷积核对比结果

## 指标汇总

| 指标 | {a['kernel_size']}x{a['kernel_size']} | {b['kernel_size']}x{b['kernel_size']} |
| --- | ---: | ---: |
| 最佳验证准确率 | {a['best_val_acc']:.2%} | {b['best_val_acc']:.2%} |
| 独立测试准确率 | {a['test_acc']:.2%} | {b['test_acc']:.2%} |
| 可训练参数量 | {a['parameters']:,} | {b['parameters']:,} |
| 总训练时间 | {a['total_seconds']:.2f} 秒 | {b['total_seconds']:.2f} 秒 |
| 平均每 Epoch | {a['mean_epoch_seconds']:.2f} 秒 | {b['mean_epoch_seconds']:.2f} 秒 |
| 训练吞吐量 | {a['samples_per_second']:.2f} 张/秒 | {b['samples_per_second']:.2f} 张/秒 |
| 验证-测试泛化差距 | {a['val_test_gap_pp']:.3f} 个百分点 | {b['val_test_gap_pp']:.3f} 个百分点 |

## 结论

- {b['kernel_size']}x{b['kernel_size']} 的测试准确率比 {a['kernel_size']}x{a['kernel_size']}
  高 {changes['test_accuracy_change_pp']:.2f} 个百分点。
- 参数量变为 {changes['parameter_ratio']:.2f} 倍，平均每 Epoch 时间增加
  {changes['mean_epoch_time_change_percent']:.2f}%，吞吐量变化
  {changes['throughput_change_percent']:.2f}%。
- 这里的泛化差距定义为最佳验证准确率与独立测试准确率的绝对差。两组差距都很小，
  说明模型选择结果在未参与选择的测试集上保持稳定。
"""
    path.write_text(content, encoding="utf-8")


def main():
    args = parse_args()
    rows = [load_result(args.run_a), load_result(args.run_b)]
    output_dir = args.output_dir or (
        PROJECT_ROOT
        / "outputs"
        / "comparisons"
        / f"{rows[0]['experiment_name']}_vs_{rows[1]['experiment_name']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    changes = calculate_changes(rows[0], rows[1])
    save_csv(rows, output_dir / "comparison.csv")
    save_chart(rows, output_dir / "comparison.png")
    save_markdown(rows, changes, output_dir / "comparison.md")
    (output_dir / "comparison.json").write_text(
        json.dumps({"experiments": rows, "changes": changes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"对比结果已保存到: {output_dir}")
    print(f"测试准确率变化: {changes['test_accuracy_change_pp']:+.2f} 个百分点")
    print(f"参数量倍数: {changes['parameter_ratio']:.2f}x")
    print(f"平均 Epoch 时间变化: {changes['mean_epoch_time_change_percent']:+.2f}%")


if __name__ == "__main__":
    main()
