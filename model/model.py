import abc

import torch


class Model(abc.ABC):
    @abc.abstractmethod
    def predict(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        入力xをもとに実測値の点推定をする
        引数:
            x[i] = i番目のデータの入力（ベクトル）
        返り値:
            各データに対する点推定
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def predict_distr(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        入力xをもとに予測値の正規分布を推定する
        引数:
            x[i] = i番目のデータの入力（ベクトル）
        返り値:
            各データに対する正規分布の推定パラメータ（平均, 標準偏差）
        """
        raise NotImplementedError()


class MockModel(Model):
    """
    テスト用のモックとして使うモデル
    """

    def predict(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        # 入力によらず0を予測値として返す
        batch_x, _, _, _ = batch
        return torch.tensor([0 for _ in range(len(batch_x))])

    def predict_distr(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        # 入力によらず、[0, 1]を予測値として返す
        batch_x, _, _, _ = batch
        return torch.tensor([[0, 1] for _ in range(len(batch_x))])
