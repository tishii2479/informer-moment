import sys
sys.path.append("Informer2020") # workaround

import argparse
import random
import tqdm
import numpy as np
import torch
import pandas as pd
import os

import dataset
from model.informer_model import InformerModel
from model.model import Model
from model.moment_model import MomentModel
from propose import ProposedModelWithMoe
from evaluation import evaluate_mse, evaluate_nll

from main_informer import parse_args


def set_seed(seed: int) -> None:
    # random
    random.seed(seed)

    # numpy
    np.random.seed(seed)

    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_moment_model(args: argparse.Namespace) -> Model:
    return MomentModel(param="AutonLab/MOMENT-1-large", pred_len=args.pred_len)


def load_informer_model(args: argparse.Namespace) -> Model:
    return InformerModel(args, checkpoint_path=INFORMER_CKPT_PATH)


def load_proposed_model(moment_model: Model, informer_model: Model,input_size: int,train_dataset: torch.utils.data.Dataset, args: argparse.Namespace) -> Model:
    model = ProposedModelWithMoe(moment_model=moment_model, informer_model=informer_model,input_size=input_size)
    model.train(train_dataset=train_dataset,args=args)
    return model


def main():
    set_seed(0)

    args = parse_args(ARG_STR)
    save_file_name = args.checkpoints
    if not os.path.exists(save_file_name):
        os.mkdir(save_file_name)
    print("args:", args)

    train_dataset, test_dataset = dataset.load_dataset(args=args)
    input_size = args.seq_len
    moment_model = load_moment_model(args=args)
    informer_model = load_informer_model(args=args)

    proposed_model = load_proposed_model(moment_model, informer_model, input_size, train_dataset, args)

    # 予測結果の評価
    print("args:", args)
    results = {}

    for method, model in {
        "informer": informer_model,
        "moment": moment_model,
        "proposed": proposed_model,
    }.items():
        print(f"testing: {method}")
        test_dataloader = dataset.to_dataloader(test_dataset, args, "test")

        y_pred = []
        y_true = []
        for batch in tqdm.tqdm(test_dataloader):
            y_pred.append(model.predict_distr(batch).detach().tolist())
            y_true.append(batch[1][:, -1].squeeze().detach().tolist())

        y_pred, y_true = np.array(y_pred).reshape(-1, 2), np.array(y_true).flatten()
        results[method] = {
            "mse": evaluate_mse(y_pred, y_true),
            "nll": evaluate_nll(y_pred, y_true),
        }
        print(results[method])

        np.save(f"checkpoints/y_pred_{method}.npy", y_pred)

    print(results)
    pd.DataFrame(results).to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()