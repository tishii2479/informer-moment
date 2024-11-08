import argparse

import torch

from Informer2020.models.model import Informer
from model.model import Model
from propose import predict_distr


def load_default_informer(args: argparse.Namespace) -> Informer:
    e_layers = args.e_layers if args.model == "informer" else args.s_layers
    return Informer(
        args.enc_in,
        args.dec_in,
        args.c_out,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.factor,
        args.d_model,
        args.n_heads,
        e_layers,  # args.e_layers,
        args.d_layers,
        args.d_ff,
        args.dropout,
        args.attn,
        args.embed,
        args.freq,
        args.activation,
        args.output_attention,
        args.distil,
        args.mix,
        # args.device,
        device="cpu",
    ).float()


class InformerModel(Model):
    def __init__(self, args: argparse.Namespace, checkpoint_path: str) -> None:
        self.model = load_default_informer(args)
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.args = args

    def predict(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        self.model.eval()

        (batch_x, batch_y, batch_x_mark, batch_y_mark) = batch
        preds, _ = self.process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
        preds = preds.squeeze()[:, -1]  # 最終時刻の予測のみ使用する

        return preds

    def predict_distr(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        return predict_distr(self, batch)

    def process_one_batch(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_mark: torch.Tensor,
        batch_y_mark: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()

        # decoder input
        dec_inp = torch.zeros(
            [batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]
        ).float()
        dec_inp = torch.cat(
            [batch_y[:, : self.args.label_len, :], dec_inp], dim=1
        ).float()
        # encoder - decoder
        outputs = self.model.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if self.args.features == "MS" else 0
        batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

        return outputs, batch_y
