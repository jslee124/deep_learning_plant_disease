"""汇总隐藏层维度对比实验。"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_result(run_name):
    path = PROJECT_ROOT / "outputs" / "runs" / run_name / "logs" / "final_result.json"
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)["result"]


def main():
    parser = argparse.ArgumentParser(description="比较两组 RNN 隐藏维度实验")
    parser.add_argument("baseline")
    parser.add_argument("comparison")
    args = parser.parse_args()
    rows = [load_result(args.baseline), load_result(args.comparison)]
    output_dir = PROJECT_ROOT / "outputs" / "comparisons" / f"{args.baseline}_vs_{args.comparison}"
    output_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "experiment_name", "hidden_size", "parameters", "best_val_acc", "test_acc",
        "total_seconds", "mean_epoch_seconds", "samples_per_second", "train_test_gap_pp",
    ]
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in rows])

    base, comparison = rows
    changes = {
        "test_accuracy_change_pp": (comparison["test_acc"] - base["test_acc"]) * 100,
        "parameter_ratio": comparison["parameters"] / base["parameters"],
        "mean_epoch_time_change_percent": (comparison["mean_epoch_seconds"] / base["mean_epoch_seconds"] - 1) * 100,
        "throughput_change_percent": (comparison["samples_per_second"] / base["samples_per_second"] - 1) * 100,
        "train_test_gap_change_pp": comparison["train_test_gap_pp"] - base["train_test_gap_pp"],
    }
    (output_dir / "comparison.json").write_text(
        json.dumps({"experiments": rows, "changes": changes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    labels = [str(row["hidden_size"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    metrics = [
        ("test_acc", "Test Accuracy (%)", 100),
        ("parameters", "Parameters", 1),
        ("mean_epoch_seconds", "Mean Epoch Time (s)", 1),
        ("train_test_gap_pp", "Train-Test Gap (pp)", 1),
    ]
    for axis, (key, title, scale) in zip(axes.flat, metrics):
        axis.bar(labels, [row[key] * scale for row in rows], color=["#266f73", "#d38345"])
        axis.set_title(title)
        axis.set_xlabel("Hidden Size")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("MNIST RNN Hidden Size Comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=200)
    plt.close(fig)

    markdown = f"""# RNN 隐藏维度对比

| 指标 | hidden=64 | hidden=128 |
| --- | ---: | ---: |
| 最佳验证准确率 | {base['best_val_acc']:.2%} | {comparison['best_val_acc']:.2%} |
| 测试准确率 | {base['test_acc']:.2%} | {comparison['test_acc']:.2%} |
| 参数量 | {base['parameters']:,} | {comparison['parameters']:,} |
| 平均 Epoch 时间 | {base['mean_epoch_seconds']:.2f} 秒 | {comparison['mean_epoch_seconds']:.2f} 秒 |
| 吞吐量 | {base['samples_per_second']:.2f} 样本/秒 | {comparison['samples_per_second']:.2f} 样本/秒 |
| 训练-测试差距 | {base['train_test_gap_pp']:.2f} pp | {comparison['train_test_gap_pp']:.2f} pp |

隐藏维度从 64 增加到 128 后，测试准确率变化
{changes['test_accuracy_change_pp']:+.2f} 个百分点，参数量变为 {changes['parameter_ratio']:.2f} 倍，
平均 Epoch 时间变化 {changes['mean_epoch_time_change_percent']:+.2f}%。
"""
    (output_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    print(f"对比结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
