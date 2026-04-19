import numpy as np
import torch
from torch.utils.data import DataLoader

from basic_lstm import LSTM
from data_loader import test_dataset

CHECKPOINT = "lstm_checkpoint.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"


ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)

y_mean = np.array(ckpt["y_mean"], dtype=np.float32)
y_std = np.array(ckpt["y_std"], dtype=np.float32)

model = LSTM(input_dim=128, hidden_dim=128, num_layers=2, dropout=0.0)
model.load_state_dict(ckpt["model_state"])
model.to(device).eval()

loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
sbp_errors, dbp_errors = [], []
cm = np.zeros((2, 2), dtype=np.int64)

print(f"Evaluating on {len(test_dataset)} test samples...")
with torch.no_grad():
    for ppg, _poincare, bps in loader:
        pred = model(ppg.to(device)).cpu().numpy()
        actual = bps.numpy()
        pred = pred * y_std + y_mean
        actual = actual * y_std + y_mean
        for j in range(pred.shape[0]):
            sbp_errors.append(abs(pred[j, 0] - actual[j, 0]))
            dbp_errors.append(abs(pred[j, 1] - actual[j, 1]))


print(f"── LSTM (PPG only) ──")
print(f"SBP MAE: {np.mean(sbp_errors):.2f} ± {np.std(sbp_errors):.2f} mmHg")
print(f"DBP MAE: {np.mean(dbp_errors):.2f} ± {np.std(dbp_errors):.2f} mmHg")
