import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from data_loader import test_dataset
from torch.utils.data import DataLoader
from transformer_nld import model

CHECKPOINT = "transformer_nld_checkpoint.pt"

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)


model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

# y_mean/y_std are [sbp, dbp] arrays
y_mean_np = np.array(ckpt["y_mean"], dtype=np.float32)  # (2,)
y_std_np = np.array(ckpt["y_std"], dtype=np.float32)  # (2,)


def classify_bp(sbp, dbp):
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

all_sbp_errors = []
all_dbp_errors = []
beat_cm = np.zeros((3, 3), dtype=np.int64)

print(f"Evaluating on {len(test_dataset)} test chunks...")
with torch.no_grad():
    for ppg_seqs, poincare_imgs, bps in test_loader:
        ppg_seqs = ppg_seqs.to(device)
        poincare_imgs = poincare_imgs.to(device)

        # outputs: (batch, 2) normalized — mean SBP/DBP over 2-min window
        outputs = model(ppg_seqs, poincare_imgs).cpu().numpy()
        bps_np = bps.numpy()  # (batch, 2) normalized

        pred = outputs * y_std_np + y_mean_np  # (batch, 2) mmHg
        actual = bps_np * y_std_np + y_mean_np  # (batch, 2) mmHg

        for j in range(pred.shape[0]):
            pred_sbp, pred_dbp = pred[j, 0], pred[j, 1]
            actual_sbp, actual_dbp = actual[j, 0], actual[j, 1]

            all_sbp_errors.append(abs(pred_sbp - actual_sbp))
            all_dbp_errors.append(abs(pred_dbp - actual_dbp))

            beat_cm[
                classify_bp(actual_sbp, actual_dbp), classify_bp(pred_sbp, pred_dbp)
            ] += 1

print(f"\nSBP MAE: {np.mean(all_sbp_errors):.2f} ± {np.std(all_sbp_errors):.2f} mmHg")
print(f"DBP MAE: {np.mean(all_dbp_errors):.2f} ± {np.std(all_dbp_errors):.2f} mmHg")

print("\nBP classification (per 2-min chunk)")
y_true_rep = np.repeat([0, 1, 2], beat_cm.sum(axis=1))
y_pred_rep = np.concatenate([np.repeat([0, 1, 2], beat_cm[i]) for i in range(3)])
print(
    classification_report(
        y_true_rep,
        y_pred_rep,
        labels=[0, 1, 2],
        target_names=BEAT_CLASS_NAMES,
        zero_division=0,
    )
)

beat_cm_norm = beat_cm / beat_cm.sum(axis=1, keepdims=True).clip(min=1)

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(beat_cm_norm, display_labels=BEAT_CLASS_NAMES).plot(
    ax=ax, colorbar=False, values_format=".2f"
)
ax.set_title("BP Classification Confusion Matrix (normalized, per 2-min chunk)")
ax.tick_params(axis="x", labelrotation=15)
plt.tight_layout()
plt.savefig("transformer_nld_eval.png", dpi=150)
plt.close()
print("\nSaved transformer_nld_eval.png")
