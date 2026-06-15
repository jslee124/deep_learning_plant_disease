"""MNIST 数据读取与固定划分。"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def build_dataloaders(config, download: bool = True):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    data_dir = Path(config.data_dir)
    full_train = datasets.MNIST(data_dir, train=True, transform=transform, download=download)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform, download=download)

    train_size = len(full_train) - config.validation_size
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, config.validation_size],
        generator=generator,
    )

    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    dataset_info = {
        "name": "MNIST",
        "data_root": str(data_dir.resolve()),
        "split_strategy": "official_train_fixed_54000_6000_and_official_test",
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "num_classes": 10,
        "input_shape": [1, 28, 28],
        "sequence_shape": [28, 28],
    }
    return train_loader, val_loader, test_loader, [str(i) for i in range(10)], dataset_info
