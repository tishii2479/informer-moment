import unittest

import torch
from momentfm import MOMENTPipeline

import config
import dataset
from model import Model
from propose import predict_distr


class MomentModel(Model):
    T = 512

    def __init__(self, param: str, pred_len: int) -> None:
        self.model = MomentModel.from_pretrained(param)
        self.pred_len = pred_len

    @classmethod
    def from_pretrained(
        cls,
        param: str,
    ) -> MOMENTPipeline:
        model = MOMENTPipeline.from_pretrained(
            param,
            model_kwargs={
                "task_name": "reconstruction",
            },
        )
        model.init()
        return model

    def predict(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        batch_x, _, _, _ = batch
        seq_len = batch_x.shape[1]
        x_enc = torch.cat(
            (
                batch_x,
                torch.zeros((batch_x.shape[0], self.T - seq_len, batch_x.shape[2])),
            ),
            dim=1,
        ).permute(0, 2, 1)
        mask = torch.cat(
            (torch.ones(seq_len), torch.zeros(self.T - seq_len)),
            dim=0,
        ).repeat(x_enc.shape[0], 1)

        output = self.model.forward(x_enc=x_enc, input_mask=mask)
        pred = output.reconstruction.squeeze()[
            :, x_enc.shape[1] + self.pred_len - 1
        ]  # 最終時刻からpred_len先の結果を使用する
        return pred

    def predict_distr(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return predict_distr(self, batch)


class Test(unittest.TestCase):
    def test_output_shape(self) -> None:
        args = config.ARGS
        train_dataset, _ = dataset.load_dataset(config.ARGS)
        model = MomentModel(param="AutonLab/MOMENT-1-large", pred_len=args.pred_len)
        train_dataloader = dataset.to_dataloader(train_dataset, args, flag="train")
        for batch in train_dataloader:
            y = model.predict_distr(batch)
            self.assertEqual(y.shape, torch.Size((args.batch_size, 2)))
            break


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
