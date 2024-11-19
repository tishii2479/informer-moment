import unittest

import torch
from momentfm import MOMENTPipeline

from momentfm.utils.masking import Masking #kengo - 追加

import config
import dataset
from model import Model
from propose import predict_distr

from tqdm import tqdm #kengo - 追加

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
    
    def fine_tuning(self, test_dataloader, lr: float = 1e-4, mask_ratio: float = 0.3): # 追加
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        criterion = torch.nn.MSELoss() 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        mask_generator = Masking(mask_ratio=mask_ratio) # Mask 30% of patches randomly

        for batch in tqdm(test_dataloader):
            batch_x, batch_y, batch_masks, _ = batch
            labels = batch_y[:, -1].squeeze().float()
            n_channels = batch_x.shape[1]
            
            # Reshape to [batch_size * n_channels, 1, window_size]
            batch_x = batch_x.reshape((-1, 1, 512)) 
            
            batch_masks = batch_masks.to(device).long()
            batch_masks = batch_masks.repeat_interleave(n_channels, axis=0)
            
            # Randomly mask some patches of data
            mask = mask_generator.generate_mask(
                x=batch_x, input_mask=batch_masks).to(device).long()

            # Forward
            output = self.model(x_enc=batch_x, input_mask=batch_masks, mask=mask) 
            
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
