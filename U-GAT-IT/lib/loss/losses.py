import torch
import torch.nn as nn


class LossWithValue(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, x, value):
        target = torch.empty_like(x).fill_(value).type_as(x)
        return self.loss_fn(x, target)


class MSEWithValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, x, value):
        target = torch.empty_like(x).fill_(value).type_as(x)
        return self.mse(x, target)


class BCELogitsWithValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_logits = nn.BCEWithLogitsLoss()

    def forward(self, x, value):
        target = torch.empty_like(x).fill_(value)
        return self.bce_logits(x, target)