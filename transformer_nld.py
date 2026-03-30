import math

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_loader import X_mean, X_std, train_loader, val_loader, y_mean, y_std

FRQUENCY = 1250  # Hz
SEQ_LEN = 1250  # 1 second of data at 1250 Hz


class PositionalEncoding(nn.Module):
    """Positional encoding using sine and cosine functions."""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P matrix
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            FRQUENCY, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention - the core attention mechanism used in Transformers.

    Computes: Attention(Q,K,V) = softmax(QK^T / √d_k) V

    This is more efficient than additive attention (just matrix multiplications),
    and works well in practice when properly scaled.
    """

    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask=None, valid_lens=None):
        """
        Apply scaled dot-product attention.

        Args:
            queries: Query vectors, shape (batch, num_queries, d_k)
            keys: Key vectors, shape (batch, num_keys, d_k)
            values: Value vectors, shape (batch, num_keys, d_v)
            mask: Optional additive mask (e.g. causal), shape (seq_len, seq_len)
            valid_lens: Optional mask for padding, shape (batch,) or (batch, num_queries)

        Returns:
            Output: Weighted sum of values, shape (batch, num_queries, d_v)
        """
        d = queries.shape[-1]  # d_k: dimension of queries/keys

        # ========================================================================
        # STEP 1: Compute attention scores (Q @ K^T / √d_k)
        # ========================================================================
        # Matrix multiply: (batch, num_queries, d_k) @ (batch, d_k, num_keys)
        #                = (batch, num_queries, num_keys)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)

        # ========================================================================
        # STEP 2: Apply mask and softmax to get attention weights
        # ========================================================================
        if mask is not None:
            scores = scores + mask

        self.attention_weights = F.softmax(scores, dim=-1)
        # Shape: (batch, num_queries, num_keys), each row sums to 1.0

        # ========================================================================
        # STEP 3: Compute weighted sum of values
        # ========================================================================
        # Matrix multiply: (batch, num_queries, num_keys) @ (batch, num_keys, d_v)
        #                = (batch, num_queries, d_v)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention = ScaledDotProductAttention(dropout)

        # STEP 1: Learnable projection matrices (one set shared across all heads)
        # Each will be split into multiple heads during forward pass
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=bias)  # Query projection
        self.W_k = nn.Linear(num_hiddens, num_hiddens, bias=bias)  # Key projection
        self.W_v = nn.Linear(num_hiddens, num_hiddens, bias=bias)  # Value projection

        # STEP 4: Output projection (after concatenating all heads)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def transpose_qkv(self, X):
        """
        Reshape and transpose for parallel computation of multiple attention heads.

        Input:  (batch_size, seq_len, num_hiddens)
        Output: (batch_size * num_heads, seq_len, num_hiddens / num_heads)

        This splits the num_hiddens dimension into num_heads separate "heads",
        then merges batch and heads dimensions for parallel processing.
        """
        # Split into heads: (batch, seq_len, num_heads, head_dim)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)

        # Transpose: (batch, num_heads, seq_len, head_dim)
        X = X.permute(0, 2, 1, 3)

        # Merge batch and heads: (batch * num_heads, seq_len, head_dim)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        """
        Reverse the operation of transpose_qkv.

        Input:  (batch_size * num_heads, seq_len, num_hiddens / num_heads)
        Output: (batch_size, seq_len, num_hiddens)
        """
        # Split batch*heads back: (batch, num_heads, seq_len, head_dim)
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])

        # Transpose: (batch, seq_len, num_heads, head_dim)
        X = X.permute(0, 2, 1, 3)

        # Concatenate heads: (batch, seq_len, num_hiddens)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, queries, keys, values, tgt_mask=None, valid_lens=None):
        """
        Forward pass implementing 4-step multi-head attention.

        Args:
            queries: (batch, query_len, num_hiddens)
            keys: (batch, key_len, num_hiddens)
            values: (batch, key_len, num_hiddens)
            valid_lens: Optional sequence lengths for masking

        Returns:
            output: (batch, query_len, num_hiddens)
        """
        # ============ STEP 1: Project Q, K, V ============
        # Transform using learned projection matrices
        queries = self.W_q(queries)  # (batch, query_len, num_hiddens)
        keys = self.W_k(keys)  # (batch, key_len, num_hiddens)
        values = self.W_v(values)  # (batch, key_len, num_hiddens)

        # Reshape for multi-head: split num_hiddens into num_heads
        queries = self.transpose_qkv(queries)  # (batch*heads, query_len, head_dim)
        keys = self.transpose_qkv(keys)  # (batch*heads, key_len, head_dim)
        values = self.transpose_qkv(values)  # (batch*heads, key_len, head_dim)

        # Handle masking for multiple heads
        if valid_lens is not None:
            # Repeat valid_lens for each head
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0
            )

        # ============ STEP 2: Compute attention for each head (in parallel) ============
        # Each of the num_heads processes its portion independently
        output = self.attention(
            queries, keys, values, mask=tgt_mask, valid_lens=valid_lens
        )
        # output shape: (batch*heads, query_len, head_dim)

        # Store attention weights for visualization
        self.attention_weights = self.attention.attention_weights

        # ============ STEP 3: Concatenate all heads ============
        output_concat = self.transpose_output(output)  # (batch, query_len, num_hiddens)

        # ============ STEP 4: Apply output projection ============
        return self.W_o(output_concat)  # (batch, query_len, num_hiddens)


# Create a simple Transformer-style encoder with multi-head attention
class SimpleTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers=2, dropout=0.1):
        super().__init__()
        # self.embedding = nn.Embedding(vocab_size, d_model)
        self.intput_proj = nn.Linear(input_dim, d_model)  # Project input to d_model
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=SEQ_LEN)

        self.layers = nn.ModuleList(
            [MultiHeadAttention(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, x):
        # x = self.embedding(x)
        x = self.intput_proj(x)  # Project input to d_model
        x = self.pos_encoding(x)
        self.tgt_mask = torch.triu(
            torch.ones(x.size(1), x.size(1), device=x.device) * float("-inf"),
            diagonal=1,
        )
        for attn, norm in zip(self.layers, self.norms):
            x = norm(x + attn(x, x, x, tgt_mask=None))  # Residual connection

        return x


class SimpleTransformerDecoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers=2, dropout=0.1):
        super().__init__()
        # self.embedding = nn.Embedding(vocab_size, d_model)
        self.input_proj = nn.Linear(input_dim, d_model)  # Project input to d_model
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=SEQ_LEN)
        self.tgt_mask = None

        self.self_attns = nn.ModuleList(
            [MultiHeadAttention(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.cross_attns = nn.ModuleList(
            [MultiHeadAttention(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.norms1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, x, enc_output):
        # x = self.embedding(x)
        x = self.input_proj(x)  # Project input to d_model
        x = self.pos_encoding(x)

        self.tgt_mask = torch.triu(
            torch.ones(x.size(1), x.size(1), device=x.device) * float("-inf"),
            diagonal=1,
        )
        for self_attn, cross_attn, norm1, norm2 in zip(
            self.self_attns, self.cross_attns, self.norms1, self.norms2
        ):
            x = norm1(
                x + self_attn(x, x, x, tgt_mask=self.tgt_mask)
            )  # masked self-attention
            x = norm2(x + cross_attn(x, enc_output, enc_output))  # cross-attention

        return x


class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = SimpleTransformerEncoder(
            input_dim, d_model, num_heads, num_layers, dropout
        )
        # self.decoder = SimpleTransformerDecoder(
        #     input_dim, d_model, num_heads, num_layers, dropout
        # )
        self.fc_out = nn.Linear(d_model, 1)  # Final output layer for regression

    def forward(self, src):
        enc_output = self.encoder(src)
        # dec_output = self.decoder(tgt, enc_output)
        return self.fc_out(enc_output).squeeze(-1)  # Output shape: (batch, seq_len)


if __name__ == "__main__":

    transformer = SimpleTransformer(
        input_dim=1,
        d_model=128,
        num_heads=2,
        num_layers=2,
        dropout=0.3,
    )

    device = "cuda" if torch.cuda.is_available() else "mps"
    transformer.to(device)
    lr = 0.0001

    optimizer = optim.Adam(transformer.parameters(), lr=lr)
    criterion = nn.HuberLoss()

    NUM_EPOCHS = 10

    trace = {"train_loss": [], "val_loss": []}
    transformer.train()

    for epoch in range(NUM_EPOCHS):

        running_loss = 0

        for batch_idx, (seqs, abps) in enumerate(train_loader):
            seqs, abps = seqs.to(device), abps.to(device)

            optimizer.zero_grad()
            # transformer expects (batch, seq_len), we have (batch, seq_len)
            outputs = transformer(seqs.unsqueeze(-1))

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
        transformer.eval()
        val_loss = 0
        with torch.no_grad():
            for seqs, abps in val_loader:
                seqs, abps = seqs.to(device), abps.to(device)
                outputs = transformer(seqs.unsqueeze(-1))
                loss = criterion(outputs, abps)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        trace["val_loss"].append(val_loss)
        transformer.train()

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

    torch.save(
        {
            "model_state": transformer.state_dict(),
            "X_mean": float(X_mean),
            "X_std": float(X_std),
            "y_mean": float(y_mean),
            "y_std": float(y_std),
        },
        "transformer_checkpoint.pt",
    )
    print("Saved transformer_checkpoint.pt")

    # train vs val loss
    epochs = range(NUM_EPOCHS)
    fig, ax = plt.subplots()
    sns.lineplot(x=epochs, y=trace["train_loss"], ax=ax, label="train")
    sns.lineplot(x=epochs, y=trace["val_loss"], ax=ax, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Validation Loss")
    plt.savefig("transformer_train_val_loss.png")
    plt.close()
