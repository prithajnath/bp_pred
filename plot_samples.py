import matplotlib.pyplot as plt
import numpy as np
import torch

from basic_lstm import LSTM
from data_loader import test_dataset, y_mean, y_std

CHECKPOINT = "lstm_checkpoint.pt"
N_SAMPLES = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)

model = LSTM(input_dim=1, hidden_dim=128, num_layers=2, dropout=0.0)
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

# Sample N_SAMPLES random indices from the test set
rng = np.random.default_rng(seed=0)
indices = rng.choice(len(test_dataset), size=N_SAMPLES, replace=False)

fig, axes = plt.subplots(N_SAMPLES, 1, figsize=(14, 3 * N_SAMPLES), sharex=False)

with torch.no_grad():
    for ax, idx in zip(axes, indices):
        seq, bp = test_dataset[int(idx)]
        seq_in = seq.unsqueeze(0).unsqueeze(-1).to(device)  # (1, seq_len, 1)
        pred = model(seq_in).squeeze(0).cpu().numpy() * y_std + y_mean
        actual = bp.numpy() * y_std + y_mean

        time_axis = np.arange(len(actual)) / 125  # seconds at 125 Hz
        ax.plot(time_axis, actual, label="Actual", alpha=0.8)
        ax.plot(time_axis, pred, label="Predicted", alpha=0.8)
        for thresh, label in [(90, "Low/Normal"), (130, "Normal/High")]:
            ax.axhline(thresh, color="grey", linestyle="--", linewidth=0.7)
        ax.set_ylabel("BP (mmHg)")
        ax.set_title(f"Test sample {idx}")
        ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("Time (s)")
plt.suptitle("Predicted vs Actual BP", y=1.01)
plt.tight_layout()
plt.savefig("lstm_samples.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved lstm_samples.png")
