from typing import Callable

import numpy as np
import torch

from model import Model


def generate_new_x(x: np.ndarray, params: dict) -> np.ndarray:
    """
    入力xをランダムに変動させる
    引数:
        x: 元の入力ベクトル
    返り値:
        xをランダムに変動させた新たな入力ベクトル
    """
    window_size = params["window_size"]
    left_index = np.random.randint(0, len(x) - window_size)
    mean_value = np.mean(x[left_index : left_index + window_size])
    x[left_index : left_index + window_size] = mean_value
    return x


def generate_new_batch(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    sample_size: int,
    generate_new_x: Callable[[np.ndarray, dict], np.ndarray],
    params: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    バッチをランダムに変動させる
    """
    (batch_x, batch_y, batch_x_mark, batch_y_mark) = batch
    new_x = []
    for i in range(len(batch_x)):
        for _ in range(sample_size):
            _x = batch_x[i].detach().numpy()
            new_x.append(generate_new_x(_x, params).tolist())

    return (
        torch.tensor(new_x),
        batch_y.clone().detach().repeat(sample_size, 1, 1),
        batch_x_mark.clone().detach().repeat(sample_size, 1, 1),
        batch_y_mark.clone().detach().repeat(sample_size, 1, 1),
    )


def predict_distr_by_sampling(
    model: Model,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    sample_size = 10
    params = {"window_size": 10}
    batch = generate_new_batch(
        batch=batch,
        sample_size=sample_size,
        generate_new_x=generate_new_x,
        params=params,
    )
    pred = model.predict(batch)

    m = torch.mean(input=pred.reshape(-1, sample_size), dim=1)
    s = torch.std(input=pred.reshape(-1, sample_size), dim=1)
    return torch.stack((m, s), dim=1)


def predict_distr(
    model: Model,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    modelで入力xに対する予測値の正規分布を推定する
    引数:
        x[i] = i番目のデータの入力（ベクトル）
    返り値:
        各データに対する正規分布の推定パラメータ（平均, 標準偏差）
    """
    return predict_distr_by_sampling(model, batch)


class ProposedModel(Model):
    """
    提案モデル
    """

    def __init__(self, moment_model: Model, informer_model: Model):
        self.moment_model = moment_model
        self.informer_model = informer_model

    def predict(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        y1 = self.moment_model.predict(batch)
        y2 = self.informer_model.predict(batch)
        y3 = (y1 + y2) / 2
        return y3

    def predict_distr(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        y1 = self.moment_model.predict_distr(batch)
        y2 = self.informer_model.predict_distr(batch)
        y3 = (y1 + y2) / 2
        return y3
