import numpy as np
import pandas as pd
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split

import matplotlib.pyplot as plt
import seaborn as sns


DATA_DIR = "data"

dfs = [pd.read_parquet(f) for f in glob(f"{DATA_DIR}/*.parquet")]
df = pd.concat(dfs, ignore_index=True)


def classify_raw_abp(abp: float):
    if abp <= 90:
        return 0
    if 90 < abp <= 130:
        return 1
    if abp > 130:
        return 2


abp_classified = df["ABP_Raw"].apply(
    lambda raw_bp_seq: [classify_raw_abp(i) for i in raw_bp_seq]
)

X = np.stack(df["PPG_F"].values, dtype=np.float32)
y = np.stack(abp_classified, dtype=np.float32)

full_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

# 80/20 split
n_train = int(0.8 * len(full_dataset))
train_dataset, test_dataset = random_split(
    full_dataset,
    [n_train, len(full_dataset) - n_train],
    generator=torch.Generator().manual_seed(42),
)

batch_size = 3

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        # output
        # 0, 1, 2
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, ppg):

        output, (hidden, cell) = self.lstm(ppg)

        # output: (seq len, batch, hidden dim)
        return self.fc(output)


if __name__ == "__main__":

    lstm_model = LSTMClassifier(
        input_dim=1,
        hidden_dim=128,
        num_layers=2,
        output_dim=3,
        dropout=0.3,
    )

    device = "mps"
    lstm_model.to(device)
    lr = 0.0001

    optimizer = optim.Adam(lstm_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    NUM_EPOCHS = 100

    trace = {"train_loss": [], "val_loss": []}
    lstm_model.train()

    for epoch in range(NUM_EPOCHS):

        running_loss = 0

        for batch_idx, (seqs, bps) in enumerate(train_loader):
            seqs, bps = seqs.to(device), bps.to(device)

            optimizer.zero_grad()
            # lstm expects (batch, seq_len, 1), we have (batch, seq_len)
            outputs = lstm_model(seqs.unsqueeze(-1))

            # output is (batch, seq_len, 3)
            # CE wants (batch, 3, seq_len)
            loss = criterion(outputs.permute(0, 2, 1), bps)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"  Epoch {epoch + 1} | batch {batch_idx}/{len(train_loader)} | loss={loss.item():.4f}"
                )

        train_loss = running_loss / len(train_loader)
        trace["train_loss"].append(train_loss)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | train_loss={train_loss:.4f}")

    sns.lineplot(x=range(NUM_EPOCHS), y=trace["train_loss"])
    plt.savefig("lstm_train_loss.png")
