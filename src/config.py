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
    data_dir: Path = project_root / "data"
    output_dir: Path = project_root / "outputs"
    model_dir: Path = output_dir / "models"
    log_dir: Path = output_dir / "logs"
    figure_dir: Path = output_dir / "figures"
    report_dir: Path = output_dir / "reports"

    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001

    # Windows 初学环境下 num_workers=0 最稳定，避免多进程 DataLoader 报错。
    # 如果训练很稳定、CPU 较强，可以尝试改成 2 或 4 提高数据读取速度。
    num_workers: int = 0
    seed: int = 42

    model_name: str = "simple_cnn"
    optimizer_name: str = "Adam"
    use_augmentation: bool = True
    dropout: float = 0.3

    best_model_name: str = "best_simple_cnn.pth"
    class_names_name: str = "class_names.json"
    loss_curve_name: str = "loss_curve.png"
    accuracy_curve_name: str = "accuracy_curve.png"
    confusion_matrix_name: str = "confusion_matrix.png"

    @property
    def best_model_path(self) -> Path:
        return self.model_dir / self.best_model_name

    @property
    def class_names_path(self) -> Path:
        return self.model_dir / self.class_names_name


config = TrainConfig()
