import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import X_mean, X_std, train_loader, val_loader, y_mean, y_std
from transformer_nld import (
    NUM_PPG_TOKENS,
    PositionalEncoding,
    TransformerEncoderLayer,
)
from utils import PPGDownsampler


class BasicTransformer(nn.Module):
    """
    Ablation baseline: identical PPG encoding to DualStreamTransformer but
    with no Poincaré sequence.

    Input:  (batch, 15000) PPG
    Output: (batch, 2)     mean [SBP, DBP] over the 2-min window
    """

    def __init__(self, d_model=128, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.ppg_downsampler = PPGDownsampler(d_model)
        self.pos_encoding = PositionalEncoding(
            d_model, dropout, max_len=NUM_PPG_TOKENS + 10
        )
        self.transformer = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.bp_head = nn.Linear(d_model, 2)

    def forward(self, ppg_seq):
        # ppg_seq: (batch, 15000)
        tokens = self.ppg_downsampler(ppg_seq)  # (batch, 500, d_model)
        tokens = self.pos_encoding(tokens)
        for layer in self.transformer:
            tokens = layer(tokens)
        return self.bp_head(tokens.mean(dim=1))  # (batch, 2)


if __name__ == "__main__":
    model = BasicTransformer(d_model=128, num_heads=4, num_layers=4, dropout=0.3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.HuberLoss()
    NUM_EPOCHS = 100
    EARLY_STOP_PATIENCE = 25

    trace = {"train_loss": [], "val_loss": []}
    best_val, no_improve = float("inf"), 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running = 0.0

        for batch_idx, (ppg_seqs, _poincare, abps) in enumerate(train_loader):
            ppg_seqs = ppg_seqs.to(device)
            abps = abps.to(device)

            optimizer.zero_grad()
            loss = criterion(model(ppg_seqs), abps)
            loss.backward()
            optimizer.step()
            running += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"  Epoch {epoch+1} | batch {batch_idx}/{len(train_loader)} | loss={loss.item():.4f}"
                )

        train_loss = running / len(train_loader)
        trace["train_loss"].append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ppg_seqs, _poincare, abps in val_loader:
                ppg_seqs = ppg_seqs.to(device)
                abps = abps.to(device)
                val_loss += criterion(model(ppg_seqs), abps).item()
        val_loss /= len(val_loader)
        trace["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f}"
        )

        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "X_mean": float(X_mean),
                    "X_std": float(X_std),
                    "y_mean": y_mean.tolist(),
                    "y_std": y_std.tolist(),
                },
                "transformer_checkpoint.pt",
            )
            print(f"  -> Saved best model (val={best_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    epochs = range(len(trace["train_loss"]))
    fig, ax = plt.subplots()
    sns.lineplot(x=epochs, y=trace["train_loss"], ax=ax, label="train")
    sns.lineplot(x=epochs, y=trace["val_loss"], ax=ax, label="val")
    ax.set(
        xlabel="Epoch",
        ylabel="Loss",
        title="Basic Transformer (no Poincaré) — Train vs Val Loss",
    )
    plt.savefig("basic_transformer_train_val_loss.png")
    plt.close()
