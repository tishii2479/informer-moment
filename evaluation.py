import unittest

import numpy as np
from scipy.stats import norm


def evaluate_mse(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    """
    予測結果y_predの平均二乗誤差（MSE）を評価する関数
    引数:
        y_pred[i] := (y_test[i]の平均の予測値, y_test[i]の標準偏差の予測値)
        y_test[i] := i番目のデータの実測値
    返り値:
        y_predとy_testの平均二乗誤差（MSE）
    """
    MSE = 0
    for i in range(len(y_pred)):
        MSE += (y_pred[i][0] - y_test[i]) ** 2
    MSE = MSE / len(y_pred)
    return MSE


def evaluate_nll(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    """
    予測結果y_predの負の対数尤度を評価する関数
    引数:
        y_pred[i] := (y_test[i]の平均の予測値, y_test[i]の標準偏差の予測値)
        y_test[i] := i番目のデータの実測値
    返り値:
        y_predのy_testに対する負の対数尤度
    """

    NLL = 0
    for i in range(len(y_pred)):
        NLL -= norm.logpdf(y_test[i], loc=y_pred[i][0], scale=np.sqrt(y_pred[i][1]))
    NLL = NLL / len(y_pred)
    return NLL


class Test(unittest.TestCase):
    def test_evaluate_mse(self) -> None:
        self.assertAlmostEqual(
            evaluate_mse(np.array([[2, 1], [3, 1]]), np.array([1, 2])), 1
        )

    def test_evaluate_nll(self) -> None:
        self.assertAlmostEqual(
            evaluate_nll(np.array([[2, 1], [3, 1]]), np.array([1, 2])),
            1.4189385332046727,
        )


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
