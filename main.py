import sys

sys.path.append("Informer2020")

# Informer2020をimportするためのwork-around
# 本来は https://qiita.com/siida36/items/b171922546e65b868679 などに従うべき
if True:  # noqa: E402
    import argparse
    import pickle
    import random
    from typing import Optional

    import numpy as np
    import pandas as pd
    import torch
    import tqdm

    import dataset
    from evaluation import evaluate_mse, evaluate_nll
    from Informer2020.main_informer import parse_args
    from model.informer_model import InformerModel
    from model.model import Model
    from model.moment_model import MomentModel
    from propose import ProposedModel, ProposedModelWithMoe

Y_PRED_PATH_PLACEHOLDER = "checkpoints/{method}_{data}_y_pred.pkl"


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


def load_moment_model(
    args: argparse.Namespace, y_pred_path: Optional[str] = None
) -> MomentModel:
    return MomentModel(
        param="AutonLab/MOMENT-1-large", pred_len=args.pred_len, y_pred_path=y_pred_path
    )


def load_moment_model_finetuned(
    args: argparse.Namespace,
    train_dataset: torch.utils.data.Dataset,
    valid_dataset: torch.utils.data.Dataset,
    y_pred_path: Optional[str] = None,
) -> MomentModel:
    model = load_moment_model(args=args, y_pred_path=y_pred_path)
    model.fine_tuning(
        train_dataset=train_dataset, valid_dataset=valid_dataset, args=args
    )
    return model


def load_informer_model(
    args: argparse.Namespace,
    use_saved_model: bool = False,
    y_pred_path: Optional[str] = None,
) -> InformerModel:
    return InformerModel(
        args=args, use_saved_model=use_saved_model, y_pred_path=y_pred_path
    )


def load_proposed_model(
    moment_model: Model, informer_model: Model, lmda: float
) -> ProposedModel:
    model = ProposedModel(
        moment_model=moment_model, informer_model=informer_model, lmda=lmda
    )
    return model


def load_proposed_model_with_moe(
    moment_model: Model,
    informer_model: Model,
    input_size: int,
    train_dataset: torch.utils.data.Dataset,
    valid_dataset: torch.utils.data.Dataset,
    args: argparse.Namespace,
    lr: float,
    weight_decay: float,
    epochs: int,
) -> ProposedModelWithMoe:
    model = ProposedModelWithMoe(
        moment_model=moment_model,
        informer_model=informer_model,
        input_size=input_size,
    )
    model.train(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        args=args,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
    )
    return model


def main() -> None:
    args = parse_args()

    set_seed(0)

    print("args:", args)

    train_dataset, valid_dataset, test_dataset = dataset.load_dataset(args=args)
    input_size = args.seq_len

    moment_model = load_moment_model(
        args=args,
        y_pred_path=(
            Y_PRED_PATH_PLACEHOLDER.format(method="moment", data=args.data)
            if args.use_y_pred_cache
            else None
        ),
    )
    informer_model = load_informer_model(
        args=args,
        use_saved_model=False,
        y_pred_path=(
            Y_PRED_PATH_PLACEHOLDER.format(method="informer", data=args.data)
            if args.use_y_pred_cache
            else None
        ),
    )
    proposed_model = load_proposed_model(
        moment_model, informer_model, lmda=args.proposed_lmda
    )
    proposed_model_with_moe = load_proposed_model_with_moe(
        moment_model,
        informer_model,
        input_size,
        train_dataset,
        valid_dataset,
        args,
        lr=args.proposed_moe_lr,
        weight_decay=args.proposed_moe_weight_decay,
        epochs=args.proposed_moe_epochs,
    )

    print("args:", args)
    results = {}

    for method, model in {
        "informer": informer_model,
        "moment": moment_model,
        "proposed": proposed_model,
        "proposed+moe": proposed_model_with_moe,
    }.items():
        print(f"testing: {method}")
        test_dataloader = dataset.to_dataloader(test_dataset, args, "test")

        y_pred = []
        y_true = []
        for index, batch in tqdm.tqdm(test_dataloader):
            y_pred.append(model.predict_distr(index, batch).detach().tolist())
            y_true.append(batch[1][:, -1].squeeze().detach().tolist())

        y_pred, y_true = np.array(y_pred).reshape(-1, 2), np.array(y_true).flatten()
        results[method] = {
            "mse": evaluate_mse(y_pred, y_true),
            "nll": evaluate_nll(y_pred, y_true),
        }
        print(results[method])

    df = pd.DataFrame(results)
    df.to_csv(f"data/results_{args.data}.csv")
    print(df)

    with open(
        Y_PRED_PATH_PLACEHOLDER.format(method="informer", data=args.data), "wb"
    ) as f:
        pickle.dump(informer_model.y_pred, f)

    with open(
        Y_PRED_PATH_PLACEHOLDER.format(method="moment", data=args.data), "wb"
    ) as f:
        pickle.dump(moment_model.y_pred, f)


if __name__ == "__main__":
    main()
