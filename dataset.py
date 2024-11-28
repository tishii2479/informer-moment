import argparse
from typing import Any

import torch

from Informer2020.exp.exp_informer import get_data


def to_dataloader(
    dataset: torch.utils.data.Dataset, args: argparse.Namespace, flag: str
) -> torch.utils.data.DataLoader:
    dataset = DatasetWithIndex(dataset=dataset)

    if flag == "test":
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    elif flag == "pred":
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )

    return data_loader


def load_dataset(args: argparse.Namespace) -> tuple[
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
]:
    """
    データセットを読み込む
    返り値: (
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
    )
    """
    return (
        get_data(args=args, flag="train")[0],
        get_data(args=args, flag="val")[0],
        get_data(args=args, flag="test")[0],
    )


class DatasetWithIndex(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset) -> None:
        self.dataset = dataset

    def __getitem__(self, index: int) -> tuple[int, Any]:
        return (index, self.dataset[index])

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore
