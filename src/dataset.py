"""数据集读取与数据增强模块。

本项目使用 torchvision.datasets.ImageFolder 读取图片。ImageFolder 的规则是：
一个子文件夹代表一个类别，文件夹名就是类别名。

推荐数据目录：

data/
  train/
    Apple_scab/
    Apple_healthy/
  val/
    Apple_scab/
    Apple_healthy/
  test/
    Apple_scab/
    Apple_healthy/

如果暂时没有 train/val/test 三个子目录，也可以直接把类别文件夹放在 data/ 下，
程序会自动按 70%/15%/15% 划分训练、验证、测试集。
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_train_transform(image_size: int, use_augmentation: bool = True):
    """构建训练集图像预处理。

    数据增强只用于训练集，目的是让模型看到更多“变化后的图片”，例如轻微旋转、
    随机裁剪、水平翻转等，从而提升泛化能力。
    """

    if use_augmentation:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_eval_transform(image_size: int):
    """构建验证集和测试集图像预处理。

    验证和测试阶段不做随机增强，否则每次评估图片都可能不同，结果不稳定。
    """

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _has_class_folders(path: Path) -> bool:
    """判断目录下是否存在类别子文件夹。"""

    return path.exists() and any(child.is_dir() for child in path.iterdir())


def _split_dataset_indices(dataset_size: int, seed: int):
    """在只有 data/类别名/图片 的情况下自动划分数据集。"""

    train_size = int(dataset_size * 0.7)
    val_size = int(dataset_size * 0.15)
    test_size = dataset_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size : train_size + val_size + test_size]
    return train_indices, val_indices, test_indices


def build_dataloaders(config):
    """构建训练、验证、测试 DataLoader。

    DataLoader 的作用是按 Batch 读取图片，并在训练时自动打乱训练集顺序。
    返回值包含三个 DataLoader 和 class_names，后续创建模型时需要类别数量。
    """

    data_dir = Path(config.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    train_transform = get_train_transform(config.image_size, config.use_augmentation)
    eval_transform = get_eval_transform(config.image_size)

    if _has_class_folders(train_dir):
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        class_names = train_dataset.classes

        if not _has_class_folders(val_dir):
            raise FileNotFoundError("已检测到 data/train，但缺少 data/val 验证集目录。")
        if not _has_class_folders(test_dir):
            raise FileNotFoundError("已检测到 data/train，但缺少 data/test 测试集目录。")

        val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
        test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    elif _has_class_folders(data_dir):
        base_train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
        base_eval_dataset = datasets.ImageFolder(data_dir, transform=eval_transform)
        class_names = base_train_dataset.classes

        train_indices, val_indices, test_indices = _split_dataset_indices(len(base_train_dataset), config.seed)
        train_dataset = Subset(base_train_dataset, train_indices)
        val_dataset = Subset(base_eval_dataset, val_indices)
        test_dataset = Subset(base_eval_dataset, test_indices)

    else:
        raise FileNotFoundError(
            "没有找到图片数据。请按 README 中的格式放入 data/train、data/val、data/test，"
            "或直接在 data/ 下放置类别子文件夹。"
        )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, class_names
