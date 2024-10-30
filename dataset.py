import argparse

import torch

from informer.data.data_loader import Dataset_ETT_hour  # type: ignore
from informer.data.data_loader import Dataset_ETT_minute  # type: ignore
from informer.data.data_loader import Dataset_Pred  # type: ignore


# informer/exp/exp_informer.pyから移植
def get_data(args: argparse.Namespace, flag: str) -> torch.utils.data.Dataset:
    data_dict = {
        "ETTh1": Dataset_ETT_hour,
        "ETTh2": Dataset_ETT_hour,
        "ETTm1": Dataset_ETT_minute,
        "ETTm2": Dataset_ETT_minute,
    }
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != "timeF" else 1

    if flag == "test":
        freq = args.freq
    elif flag == "pred":
        freq = args.detail_freq
        Data = Dataset_Pred
    else:
        freq = args.freq
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        inverse=args.inverse,
        timeenc=timeenc,
        freq=freq,
        cols=args.cols,
    )
    return data_set


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
    return (get_data(args=args, flag="train"), get_data(args=args, flag="test"))
