import argparse
from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from Informer2020.exp.exp_informer import get_data


def to_dataloader(
    dataset: torch.utils.data.Dataset, args: argparse.Namespace, flag: str
) -> torch.utils.data.DataLoader:
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
]:
    """
    データセットを読み込む
    返り値: (
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
    )
    """
    return (get_data(args=args, flag="train")[0], get_data(args=args, flag="test")[0])


class Dataset_Custom(torch.utils.data.Dataset):
    def __init__(
        self,
        data: np.ndarray,
        flag: str = "train",
        size: Optional[tuple[int, int, int]] = None,
        scale: bool = True,
    ) -> None:
        # size [seq_len, label_len, pred_len]
        # info
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.__read_data__(data)

    def __read_data__(self, data: np.ndarray) -> None:
        self.scaler = StandardScaler()
        num_train = int(len(data) * 0.7)
        num_test = int(len(data) * 0.2)
        num_vali = len(data) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.reshape(-1, 1))
            data = self.scaler.transform(data.reshape(-1, 1))

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index: int) -> tuple:
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y, np.zeros(1), np.zeros(1)

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)
