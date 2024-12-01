import argparse
import pickle
from typing import Optional

import torch
from momentfm import MOMENTPipeline
from momentfm.utils.masking import Masking  # kengo - 追加
from tqdm import tqdm  # kengo - 追加

import dataset
from model.model import Model
from propose import predict_distr


class MomentModel(Model):
    T = 512

    def __init__(
        self, param: str, pred_len: int, y_pred_path: Optional[str] = None
    ) -> None:
        self.model = MomentModel.from_pretrained(param)
        self.model.eval()
        self.pred_len = pred_len
        try:
            with open(y_pred_path, "rb") as f:
                self.y_pred = pickle.load(f)
        except:  # noqa: E722
            self.y_pred = {}

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
        return pred.detach().clone()

    def predict_distr(
        self,
        index: torch.Tensor,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        index = [i.item() for i in index]
        if sum([i not in self.y_pred for i in index]) == 0:
            return torch.stack([self.y_pred[i] for i in index])

        y = predict_distr(self, batch)
        for i, _y in zip(index, y):
            self.y_pred[i] = _y
        return y

    def fine_tuning(
        self,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset,
        args: argparse.Namespace,
        lr: float = 1e-4,
        mask_ratio: float = 0.3,
    ):  # 追加
        train_dataloader = dataset.to_dataloader(
            train_dataset=train_dataset, args=args, flag="train"
        )
        # valid_dataloader = dataset.to_dataloader(
        #     valid_dataset=valid_dataset, args=args, flag="val"
        # )

        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        mask_generator = Masking(mask_ratio=mask_ratio)  # Mask 30% of patches randomly

        for _, batch in tqdm(train_dataloader):
            batch_x, batch_y, batch_masks, _ = batch
            seq_len = batch_x.shape[1]
            x_enc = torch.cat(
                (
                    batch_x,
                    torch.zeros((batch_x.shape[0], self.T - seq_len, batch_x.shape[2])),
                ),
                dim=1,
            ).permute(0, 2, 1)
            labels = batch_y[:, -1].squeeze().float()
            n_channels = batch_x.shape[1]

            batch_masks = batch_masks.to(device).long()
            batch_masks = batch_masks.repeat_interleave(n_channels, axis=0)

            # Randomly mask some patches of data
            mask = (
                mask_generator.generate_mask(x=x_enc, input_mask=batch_masks)
                .to(device)
                .long()
            )

            # Forward
            output = self.model(x_enc=x_enc, input_mask=batch_masks, mask=mask)

            # Compute loss
            recon_loss = criterion(output.reconstruction, labels)
            observed_mask = batch_masks * (1 - mask)
            masked_loss = observed_mask * recon_loss

            loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)

            print(f"loss: {loss.item()}")

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
