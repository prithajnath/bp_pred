import math

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_loader import (
    NUM_POINCARE,
    X_mean,
    X_std,
    train_loader,
    val_loader,
    y_mean,
    y_std,
)

FREQUENCY = 125  # Hz
# SEQ_LEN = 15000  ( 12 * 1250)
# NUM_POINCARE = 4
NUM_PPG_TOKENS = 500  # after 2-stage CNN downsampling: 15000 -> 3000 -> 500


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            FREQUENCY,
            torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens,
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        if mask is not None:
            scores = scores + mask
        self.attention_weights = F.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention = ScaledDotProductAttention(dropout)
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_k = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_v = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def transpose_qkv(self, X):
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, queries, keys, values):
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))
        output = self.attention(queries, keys, values)
        return self.W_o(self.transpose_output(output))


class PPGDownsampler(nn.Module):
    """
    15000 -[stride 5]-> 3000 -[stride 5]-> 600 -[stride 4]-> 150
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
        x = x.unsqueeze(1)  # (batch, 1, 15000)
        x = self.net(x)  # (batch, d_model, 150)
        return x.permute(0, 2, 1)  # (batch, 150, d_model)


class PoincareCNN(nn.Module):
    """Encodes a single 32×32 Poincaré histogram to a feature vector."""

    def __init__(self, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16->8
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class PoincareSequenceEncoder(nn.Module):

    def __init__(self, d_model, cnn_features=64):
        super().__init__()
        self.cnn = PoincareCNN(out_features=cnn_features)
        self.proj = nn.Linear(cnn_features, d_model)

    def forward(self, x):
        # x: (batch, num_poincare, 1, 32, 32)
        batch, T, C, H, W = x.shape
        x = x.view(batch * T, C, H, W)  # (batch*T, 1, 32, 32)
        feats = self.cnn(x)  # (batch*T, cnn_features)
        feats = feats.view(batch, T, -1)  # (batch, T, cnn_features)
        return self.proj(feats)  # (batch, T, d_model)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attn(x, x, x)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


# 150 PPG tokens -> 12 sub-window predictions by folding groups of 12.5 -> use
# a learned linear to pool from 150 to 12 sub-window embeddings.


class DualStreamTransformer(nn.Module):
    """

    Stream 1: 15,000-sample PPG -> CNN downsampler -> 150 tokens
    Stream 2: 4 Poincaré images -> CNN sequence encoder -> 4 tokens

    The 154-token sequence is processed by a transformer encoder.
    The 150 PPG output tokens are pooled to 12 sub-window embeddings,
    each predicting SBP and DBP for the corresponding 10s window.

    Output: (batch, 12, 2)  — [SBP, DBP] per sub-window
    """

    def __init__(self, d_model=128, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()

        self.ppg_downsampler = PPGDownsampler(d_model)

        self.poincare_encoder = PoincareSequenceEncoder(d_model)

        # Positional encoding over the full 154-token sequence
        self.pos_encoding = PositionalEncoding(
            d_model, dropout, max_len=NUM_PPG_TOKENS + NUM_POINCARE + 10
        )

        # Transformer encoder
        self.transformer = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # Global average pool all tokens. predict mean SBP+DBP
        self.bp_head = nn.Linear(d_model, 2)  # -> [mean SBP, mean DBP]

    def forward(self, ppg_seq, poincare_seq):
        # ppg_seq:      (batch, 15000)
        # poincare_seq: (batch, 4, 1, 32, 32)

        ppg_tokens = self.ppg_downsampler(ppg_seq)  # (batch, 150, d_model)
        poincare_tokens = self.poincare_encoder(poincare_seq)  # (batch, 4,   d_model)

        # Poincaré tokens first (low-freq context), then PPG tokens
        tokens = torch.cat(
            [poincare_tokens, ppg_tokens], dim=1
        )  # (batch, 154, d_model)
        tokens = self.pos_encoding(tokens)

        for layer in self.transformer:
            tokens = layer(tokens)

        # Global average pool over all 154 tokens → (batch, d_model)
        pooled = tokens.mean(dim=1)

        return self.bp_head(pooled)  # (batch, 2)


model = DualStreamTransformer(
    d_model=128,
    num_heads=4,
    num_layers=4,
    dropout=0.3,
)
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.HuberLoss()
    # criterion = nn.CrossEntropyLoss()
    NUM_EPOCHS = 100
    EARLY_STOP_PATIENCE = 25

    trace = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (ppg_seqs, poincare_imgs, abps) in enumerate(train_loader):
            ppg_seqs = ppg_seqs.to(device)  # (batch, 15000)
            poincare_imgs = poincare_imgs.to(device)  # (batch, 4, 1, 32, 32)
            abps = abps.to(device)  # (batch, 2)

            optimizer.zero_grad()
            outputs = model(ppg_seqs, poincare_imgs)
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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ppg_seqs, poincare_imgs, abps in val_loader:
                ppg_seqs = ppg_seqs.to(device)
                poincare_imgs = poincare_imgs.to(device)
                abps = abps.to(device)
                outputs = model(ppg_seqs, poincare_imgs)
                val_loss += criterion(outputs, abps).item()
        val_loss /= len(val_loader)
        trace["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "X_mean": float(X_mean),
                    "X_std": float(X_std),
                    "y_mean": y_mean.tolist(),  # [sbp_mean, dbp_mean]
                    "y_std": y_std.tolist(),  # [sbp_std,  dbp_std]
                },
                "transformer_nld_checkpoint.pt",
            )
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(
                    f"Early stopping at epoch {epoch + 1} (no improvement for {EARLY_STOP_PATIENCE} epochs)"
                )
                break
    print("Saved transformer_nld_checkpoint.pt")

    epochs = range(len(trace["train_loss"]))  # becasue we're dropping out early
    fig, ax = plt.subplots()
    sns.lineplot(x=epochs, y=trace["train_loss"], ax=ax, label="train")
    sns.lineplot(x=epochs, y=trace["val_loss"], ax=ax, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Validation Loss")
    plt.savefig("transformer_nld_train_val_loss.png")
    plt.close()
