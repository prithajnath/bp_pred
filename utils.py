import torch.nn as nn


class PPGDownsampler(nn.Module):
    """
    15000 -[k=10,s=5,p=3]-> 3000 -[k=12,s=6,p=3]-> 500
    """

    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            # (batch, 1, 15000) -> (batch, 32, 3000)
            nn.Conv1d(1, 32, kernel_size=10, stride=5, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(32),
            # (batch, 32, 3000) -> (batch, d_model, 500)
            nn.Conv1d(32, d_model, kernel_size=12, stride=6, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
        )

    def forward(self, x):
        # x: (batch, 15000)
        x = x.unsqueeze(1)        # (batch, 1, 15000)
        x = self.net(x)           # (batch, d_model, 500)
        return x.permute(0, 2, 1) # (batch, 500, d_model)
