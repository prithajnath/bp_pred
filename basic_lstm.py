import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import X_mean, X_std, train_loader, val_loader, y_mean, y_std
from utils import PPGDownsampler


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.ppg_downsampler = PPGDownsampler(d_model=hidden_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, ppg):

        ppg = self.ppg_downsampler(ppg)  # (batch, 500, hidden_dim)

        output, _ = self.lstm(ppg)  # (batch, seq_len, hidden_dim)
        # we have (batch, seq_len, 2)
        # we want (batch, 2) before passing it to the linear layer
        return self.fc(output.mean(dim=1)).squeeze(-1)


if __name__ == "__main__":

    lstm_model = LSTM(
        input_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
    )

    device = "cuda" if torch.cuda.is_available() else "mps"
    # device = "cpu"
    lstm_model.to(device)
    lr = 0.0001

    optimizer = optim.Adam(lstm_model.parameters(), lr=lr)
    criterion = nn.HuberLoss()

    NUM_EPOCHS = 100
    EARLY_STOP_PATIENCE = 25

    trace = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    lstm_model.train()

    for epoch in range(NUM_EPOCHS):

        running_loss = 0

        for batch_idx, (ppg_seqs, poincare_imgs, abps) in enumerate(train_loader):
            seqs, abps = ppg_seqs.to(device), abps.to(device)

            optimizer.zero_grad()
            outputs = lstm_model(seqs)

            loss = criterion(outputs, abps)
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
            for ppg_seqs, poincare_imgs, abps in val_loader:
                seqs, abps = ppg_seqs.to(device), abps.to(device)
                outputs = lstm_model(seqs)
                loss = criterion(outputs, abps)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        trace["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": lstm_model.state_dict(),
                    "X_mean": float(X_mean),
                    "X_std": float(X_std),
                    "y_mean": y_mean.tolist(),
                    "y_std": y_std.tolist(),
                },
                "lstm_checkpoint.pt",
            )

            print(f"Saved best model (val_loss={best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(
                    f"Early stopping triggered after {epoch + 1} epochs with no improvement."
                )
                break

        lstm_model.train()

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

    # train vs val loss
    epochs = range(len(trace["train_loss"]))
    fig, ax = plt.subplots()
    sns.lineplot(x=epochs, y=trace["train_loss"], ax=ax, label="train")
    sns.lineplot(x=epochs, y=trace["val_loss"], ax=ax, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Validation Loss")
    plt.savefig("lstm_train_val_loss.png")
    plt.close()
