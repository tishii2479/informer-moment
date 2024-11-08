import torch
from momentfm import MOMENTPipeline

from model.model import Model
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

        output = self.model.forward(x_enc=x_enc.float(), input_mask=mask.float())
        pred = output.reconstruction.squeeze()[
            :, x_enc.shape[1] + self.pred_len - 1
        ]  # 最終時刻からpred_len先の結果を使用する
        return pred

    def predict_distr(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return predict_distr(self, batch)
