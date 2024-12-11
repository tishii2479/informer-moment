import argparse
import pickle
from typing import Optional

import torch

from Informer2020.exp.exp_informer import Exp_Informer, get_checkpoint_path
from Informer2020.main_informer import to_setting_str
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
    def __init__(
        self,
        args: argparse.Namespace,
        y_pred_path: Optional[str] = None,
    ) -> None:
        setting = to_setting_str(args=args, itr=0)
        path = get_checkpoint_path(args=args, setting=setting)

        try:
            with open(y_pred_path, "rb") as f:
                self.y_pred = pickle.load(f)
        except:  # noqa: E722
            self.y_pred = {}

        try:
            self.model = load_default_informer(args=args)
            self.model.load_state_dict(torch.load(path))
        except:  # noqa: E722
            # y_predのデータがある場合は、学習済みのinformerがなくても良い
            if len(self.y_pred) == 0:
                print("=" * 30 + "\nstart training informer\n" + "=" * 30)
                self.model = Exp_Informer(args=args).train(setting=setting)

        self.model.eval()
        self.args = args

    def predict(
        self,
        index: Optional[torch.Tensor],
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if index is not None:
            index = [
                -(1 + i.item()) for i in index
            ]  # predictはキャッシュのキーに負のindexを使う、だいぶ良くない
            if sum([i not in self.y_pred for i in index]) == 0:
                return torch.stack([self.y_pred[i] for i in index])

        (batch_x, batch_y, batch_x_mark, batch_y_mark) = batch
        preds, _ = self.process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
        preds = preds.squeeze()[:, -1]  # 最終時刻の予測のみ使用する

        y = preds.detach().clone()

        if index is not None:
            for i, _y in zip(index, y):
                self.y_pred[i] = _y
        return y

    def predict_distr(
        self,
        index: Optional[torch.Tensor],
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if index is not None:
            index = [i.item() for i in index]
            if sum([i not in self.y_pred for i in index]) == 0:
                return torch.stack([self.y_pred[i] for i in index])

        y = predict_distr(self, batch)
        if index is not None:
            for i, _y in zip(index, y):
                self.y_pred[i] = _y
        return y

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
