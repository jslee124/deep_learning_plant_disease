"""项目配置文件。

这个文件把训练中经常修改的参数集中起来，避免把学习率、Batch Size、
Epoch 等参数散落在多个脚本里。后续做参数对比实验时，优先改这里或通过
命令行参数覆盖这里的默认值。
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    """训练配置。

    dataclass 的作用类似一个清晰的参数表：字段名就是参数含义，默认值就是
    基线实验的默认设置。这样写也方便后续把参数记录到 CSV 实验日志中。
    """

    project_root: Path = Path(__file__).resolve().parents[1]
    workspace_root: Path = project_root.parent
    # 两个平时实验和课程设计共用仓库根目录中的数据，避免复制大型数据集。
    data_dir: Path = workspace_root / "data"
    output_dir: Path = project_root / "outputs"
    experiment_name: str = "baseline"
    run_dir: Path = output_dir / "runs" / experiment_name
    model_dir: Path = run_dir / "models"
    log_dir: Path = run_dir / "logs"
    figure_dir: Path = run_dir / "figures"
    report_dir: Path = run_dir / "reports"

    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001

    # Windows 初学环境下 num_workers=0 最稳定，避免多进程 DataLoader 报错。
    # 如果训练很稳定、CPU 较强，可以尝试改成 2 或 4 提高数据读取速度。
    num_workers: int = 0
    seed: int = 42

    model_name: str = "simple_cnn"
    pretrained: bool = False
    freeze_backbone: bool = False
    optimizer_name: str = "Adam"
    use_augmentation: bool = True
    dropout: float = 0.3
    conv_kernel_size: int = 3
    conv_blocks: int = 4
    # Kaggle 数据集只有 train/valid 带标签，因此把 valid 按类别平分为验证集和测试集。
    test_ratio_from_val: float = 0.5

    best_model_name: str = "best_simple_cnn.pth"
    latest_model_name: str = "latest_checkpoint.pth"
    class_names_name: str = "class_names.json"
    loss_curve_name: str = "loss_curve.png"
    accuracy_curve_name: str = "accuracy_curve.png"
    confusion_matrix_name: str = "confusion_matrix.png"

    @property
    def best_model_path(self) -> Path:
        return self.model_dir / self.best_model_name

    @property
    def latest_model_path(self) -> Path:
        return self.model_dir / self.latest_model_name

    @property
    def class_names_path(self) -> Path:
        return self.model_dir / self.class_names_name

    @property
    def experiment_results_path(self) -> Path:
        return self.output_dir / "logs" / "experiment_results.csv"

    @property
    def hardware_benchmark_path(self) -> Path:
        return self.output_dir / "logs" / "hardware_benchmark.csv"

    def configure_run(self, experiment_name: str):
        """按实验名称创建独立输出路径，避免不同实验互相覆盖。"""

        self.experiment_name = experiment_name
        self.run_dir = self.output_dir / "runs" / experiment_name
        self.model_dir = self.run_dir / "models"
        self.log_dir = self.run_dir / "logs"
        self.figure_dir = self.run_dir / "figures"
        self.report_dir = self.run_dir / "reports"


config = TrainConfig()
