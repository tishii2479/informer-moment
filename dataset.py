import argparse
from typing import Any

import torch

from Informer2020.exp.exp_informer import get_data


def to_dataloader(
    dataset: torch.utils.data.Dataset, args: argparse.Namespace, flag: str
) -> torch.utils.data.DataLoader:
    if flag == "test":
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        base_index = 0
    elif flag == "pred":
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        base_index = 1_000_000_000
    elif flag == "val":
        # y_predのcacheが効かなくなるので、シャッフルしていない、本当は良くない
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        base_index = 2_000_000_000
    elif flag == "train":
        # y_predのcacheが効かなくなるので、シャッフルしていない、本当は良くない
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        base_index = 3_000_000_000
    else:
        assert False

    dataset = DatasetWithIndex(dataset=dataset, base_index=base_index)
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
    def __init__(self, dataset: torch.utils.data.Dataset, base_index: int) -> None:
        self.dataset = dataset
        self.base_index = base_index

    def __getitem__(self, index: int) -> tuple[int, Any]:
        return (self.base_index + index, self.dataset[index])

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore
