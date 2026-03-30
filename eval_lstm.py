import os

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from scipy.signal import find_peaks
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from torch.utils.data import DataLoader

from basic_lstm import LSTM
from data_loader import test_dataset, y_mean, y_std

CHECKPOINT = "lstm_checkpoint.pt"

_proc = psutil.Process(os.getpid())


def mem(label):
    rss = _proc.memory_info().rss / 1024**3
    print(f"[MEM] {label}: {rss:.2f} GB RSS")


mem("after basic_lstm import")

test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)

model = LSTM(input_dim=1, hidden_dim=128, num_layers=2, dropout=0.0)
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()
mem("after model load")


# 125 Hz → min 0.5s between beats (max 120 bpm), min 5 mmHg prominence
PEAK_DISTANCE = 62
PEAK_PROMINENCE = 5


def extract_sbp_dbp(waveform):
    """Return (median_sbp, median_dbp) for a single ABP waveform (mmHg)."""
    peaks, _ = find_peaks(waveform, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
    troughs, _ = find_peaks(
        -waveform, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE
    )
    sbp = float(np.median(waveform[peaks])) if len(peaks) > 0 else float(waveform.max())
    dbp = (
        float(np.median(waveform[troughs]))
        if len(troughs) > 0
        else float(waveform.min())
    )
    return sbp, dbp


def classify_beat(sbp, dbp):
    """ACC/AHA 2017: 0=Normal, 1=Stage1, 2=Stage2"""
    if sbp >= 140 or dbp >= 90:
        return 2
    if sbp >= 130 or dbp >= 80:
        return 1
    return 0


BEAT_CLASS_NAMES = [
    "Normal (<130/<80)",
    "Stage 1 (130-139/80-89)",
    "Stage 2 (≥140/≥90)",
]

# Beat-level accumulator (one classification per 10s window)
beat_cm = np.zeros((3, 3), dtype=np.int64)

# MAE accumulators
all_mae = []
all_sbp_errors = []
all_dbp_errors = []

print(f"Evaluating on {len(test_dataset)} test samples...")
with torch.no_grad():
    for i, (seqs, bps) in enumerate(test_loader):
        seqs = seqs.to(device)
        outputs = model(seqs.unsqueeze(-1)).cpu().numpy()  # (batch, seq_len)
        bps_np = bps.numpy()

        for j in range(outputs.shape[0]):
            pred_seq = outputs[j] * y_std + y_mean
            actual_seq = bps_np[j] * y_std + y_mean

            # Point-by-point MAE for this sample
            all_mae.append(np.mean(np.abs(pred_seq - actual_seq)))

            pred_sbp, pred_dbp = extract_sbp_dbp(pred_seq)
            actual_sbp, actual_dbp = extract_sbp_dbp(actual_seq)

            all_sbp_errors.append(abs(pred_sbp - actual_sbp))
            all_dbp_errors.append(abs(pred_dbp - actual_dbp))

            beat_cm[
                classify_beat(actual_sbp, actual_dbp), classify_beat(pred_sbp, pred_dbp)
            ] += 1

        if i % 100 == 0:
            rss = _proc.memory_info().rss / 1024**3
            print(f"  batch {i}/{len(test_loader)}  [{rss:.2f} GB RSS]")

print(f"\nOverall MAE:  {np.mean(all_mae):.2f} ± {np.std(all_mae):.2f} mmHg")
print(f"SBP MAE:      {np.mean(all_sbp_errors):.2f} ± {np.std(all_sbp_errors):.2f} mmHg")
print(f"DBP MAE:      {np.mean(all_dbp_errors):.2f} ± {np.std(all_dbp_errors):.2f} mmHg")

print("\nBeat-level classification")
y_true_rep = np.repeat([0, 1, 2], beat_cm.sum(axis=1))
y_pred_rep = np.concatenate([np.repeat([0, 1, 2], beat_cm[i]) for i in range(3)])
print(classification_report(y_true_rep, y_pred_rep, target_names=BEAT_CLASS_NAMES))

beat_cm_norm = beat_cm / beat_cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(beat_cm_norm, display_labels=BEAT_CLASS_NAMES).plot(
    ax=ax, colorbar=False, values_format=".2f"
)
ax.set_title("Beat-level Confusion Matrix (normalized)")
ax.tick_params(axis="x", labelrotation=15)
plt.tight_layout()
plt.savefig("lstm_eval.png", dpi=150)
plt.close()
print("\nSaved lstm_eval.png")
