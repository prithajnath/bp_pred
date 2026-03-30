import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import (
    X_mean,
    X_std,
    train_loader,
    val_loader,
    y_mean,
    y_std,
)


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, ppg):

        output, _ = self.lstm(ppg)

        # squeeze out last dim -> (batch, seq_len)
        return self.fc(output).squeeze(-1)


if __name__ == "__main__":

    lstm_model = LSTM(
        input_dim=1,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
    )

    device = "cuda" if torch.cuda.is_available() else "mps"
    lstm_model.to(device)
    lr = 0.0001

    optimizer = optim.Adam(lstm_model.parameters(), lr=lr)
    criterion = nn.HuberLoss()

    NUM_EPOCHS = 10

    trace = {"train_loss": [], "val_loss": []}
    lstm_model.train()

    for epoch in range(NUM_EPOCHS):

        running_loss = 0

        for batch_idx, (seqs, bps) in enumerate(train_loader):
            seqs, bps = seqs.to(device), bps.to(device)

            optimizer.zero_grad()
            # lstm expects (batch, seq_len, 1), we have (batch, seq_len)
            outputs = lstm_model(seqs.unsqueeze(-1))

            loss = criterion(outputs, bps)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"  Epoch {epoch + 1} | batch {batch_idx}/{len(train_loader)} | loss={loss.item():.4f}"
                )

        train_loss = running_loss / len(train_loader)
        trace["train_loss"].append(train_loss)

        # Validation
        lstm_model.eval()
        val_loss = 0
        with torch.no_grad():
            for seqs, bps in val_loader:
                seqs, bps = seqs.to(device), bps.to(device)
                outputs = lstm_model(seqs.unsqueeze(-1))
                loss = criterion(outputs, bps)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        trace["val_loss"].append(val_loss)
        lstm_model.train()

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

    torch.save(
        {
            "model_state": lstm_model.state_dict(),
            "X_mean": float(X_mean),
            "X_std": float(X_std),
            "y_mean": float(y_mean),
            "y_std": float(y_std),
        },
        "lstm_checkpoint.pt",
    )
    print("Saved lstm_checkpoint.pt")

    # train vs val loss
    epochs = range(NUM_EPOCHS)
    fig, ax = plt.subplots()
    sns.lineplot(x=epochs, y=trace["train_loss"], ax=ax, label="train")
    sns.lineplot(x=epochs, y=trace["val_loss"], ax=ax, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Validation Loss")
    plt.savefig("lstm_train_val_loss.png")
    plt.close()
