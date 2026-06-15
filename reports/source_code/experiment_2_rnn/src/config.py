"""MNIST RNN 实验配置。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    project_root: Path = Path(__file__).resolve().parents[1]
    workspace_root: Path = project_root.parent
    data_dir: Path = workspace_root / "data" / "MNIST"
    output_dir: Path = project_root / "outputs"

    experiment_name: str = "baseline"
    run_dir: Path = output_dir / "runs" / experiment_name
    model_dir: Path = run_dir / "models"
    log_dir: Path = run_dir / "logs"
    figure_dir: Path = run_dir / "figures"

    input_size: int = 28
    sequence_length: int = 28
    hidden_size: int = 64
    num_layers: int = 1
    num_classes: int = 10
    dropout: float = 0.0

    batch_size: int = 128
    epochs: int = 10
    learning_rate: float = 0.001
    optimizer_name: str = "Adam"
    num_workers: int = 0
    validation_size: int = 6000
    seed: int = 42

    def configure_run(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.run_dir = self.output_dir / "runs" / experiment_name
        self.model_dir = self.run_dir / "models"
        self.log_dir = self.run_dir / "logs"
        self.figure_dir = self.run_dir / "figures"

    @property
    def best_model_path(self) -> Path:
        return self.model_dir / "best_rnn.pth"

    @property
    def latest_model_path(self) -> Path:
        return self.model_dir / "latest_checkpoint.pth"

    @property
    def experiment_results_path(self) -> Path:
        return self.output_dir / "logs" / "experiment_results.csv"

    @property
    def hardware_benchmark_path(self) -> Path:
        return self.output_dir / "logs" / "hardware_benchmark.csv"


config = TrainConfig()
