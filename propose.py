import argparse
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

import dataset
from model.model import Model


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
    sample_size = 5  # 10に変更
    params = {"window_size": 30}  # 30に変更
    batch = generate_new_batch(
        batch=batch,
        sample_size=sample_size,
        generate_new_x=generate_new_x,
        params=params,
    )
    pred = model.predict(
        None,
        batch,
    )

    m = torch.mean(input=pred.reshape(-1, sample_size), dim=1)
    s = torch.var(input=pred.reshape(-1, sample_size), dim=1)
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

    def __init__(self, moment_model: Model, informer_model: Model, lmda: float):
        self.moment_model = moment_model
        self.informer_model = informer_model
        self.lmda = lmda

    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset,
        args: argparse.Namespace,
    ) -> None:
        pass

    def predict(
        self,
        index: Optional[torch.Tensor],
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        y1 = self.moment_model.predict(index=index, batch=batch)
        y2 = self.informer_model.predict(index=index, batch=batch)
        y3 = self.lmda * y1 + (1 - self.lmda) * y2
        return y3

    def predict_distr(
        self,
        index: Optional[torch.Tensor],
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        y1 = self.moment_model.predict_distr(index=index, batch=batch)
        y2 = self.informer_model.predict_distr(index=index, batch=batch)
        y3 = self.lmda * y1 + (1 - self.lmda) * y2
        return y3


# モデルの定義
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 32)  # 入力層から隠れ層
        self.fc2 = torch.nn.Linear(32, 64)  # 隠れ層から隠れ層
        self.fc3 = torch.nn.Linear(64, 32)  # 隠れ層から隠れ層
        self.fc4 = torch.nn.Linear(32, output_size)  # 隠れ層から出力層

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.fc1(x))  # ReLU活性化関数
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.relu(self.fc3(h2))
        y = torch.sigmoid(self.fc4(h3))  # 出力層
        return y


class ProposedModelWithMoe(Model):
    """
    提案モデル
    """

    def __init__(self, moment_model: Model, informer_model: Model, input_size: int):
        self.moment_model = moment_model
        self.informer_model = informer_model

        self.weight_model = SimpleNN(input_size, 1).float()

    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset,
        args: argparse.Namespace,
        lr: float,
        weight_decay: float,
        epochs: int,
    ) -> None:
        train_dataloader = dataset.to_dataloader(train_dataset, args, "train")
        valid_dataloader = dataset.to_dataloader(valid_dataset, args, "val")

        criterion = nn.GaussianNLLLoss()
        optimizer = optim.Adam(
            self.weight_model.parameters(), lr=lr, weight_decay=weight_decay
        )

        def run_epoch(dataloader: torch.utils.data.DataLoader, is_eval: bool) -> float:
            total_loss = 0.0
            for index, batch in tqdm.tqdm(dataloader):
                _, batch_y, _, _ = batch
                labels = batch_y[:, -1].squeeze().float()
                optimizer.zero_grad()
                outputs = self.predict_distr(index, batch)
                mu = outputs[:, 0]
                sigma = outputs[:, 1]
                loss = criterion(mu, labels, sigma)

                if not is_eval:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

            total_loss /= len(dataloader)
            return total_loss

        for epoch in range(epochs):
            self.weight_model.train()
            train_loss = run_epoch(train_dataloader, False)
            self.weight_model.eval()
            valid_loss = run_epoch(valid_dataloader, True)
            print(
                f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, "
                f"Valid Loss: {valid_loss:.4f}"
            )

    def predict(
        self,
        index: Optional[torch.Tensor],
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_x, _, _, _ = batch
        y1 = self.moment_model.predict(index=index, batch=batch)
        y2 = self.informer_model.predict(index=index, batch=batch)
        w = self.weight_model.forward(batch_x.squeeze().float())
        y3 = w * y1 + (1 - w) * y2
        return y3

    def predict_distr(
        self,
        index: Optional[torch.Tensor],
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_x, _, _, _ = batch
        y1 = self.moment_model.predict_distr(index=index, batch=batch)
        y2 = self.informer_model.predict_distr(index=index, batch=batch)
        w = self.weight_model.forward(batch_x.squeeze().float())
        y3 = w * y1 + (1 - w) * y2
        return y3
