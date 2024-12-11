import torch
import tqdm

from model.model import Model


def calc_sigma(data_loader: torch.utils.data.DataLoader, model: Model) -> float:
    mse_loss = torch.nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for index, batch in tqdm.tqdm(data_loader):
            y_pred = model.predict(index, batch)
            y_true = batch[1][:, -1].squeeze()

            loss = mse_loss(y_pred, y_true)
            total_loss += loss.item()

    sigma = total_loss / len(data_loader)
    return sigma
