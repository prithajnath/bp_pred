import numpy as np
import torch
from torch.utils.data import DataLoader

from basic_transformer import BasicTransformer
from data_loader import test_dataset
from transformer_nld import DualStreamTransformer

NLD_CHECKPOINT = "transformer_nld_checkpoint.pt"
BASIC_CHECKPOINT = "transformer_checkpoint.pt"

CLASS_NAMES = ["Normal (<130/<80)", "Hypertensive (≥130/≥80)"]

device = "cuda" if torch.cuda.is_available() else "cpu"


def classify_bp(sbp, dbp):
    return 1 if (sbp >= 130 or dbp >= 80) else 0


def eval_nld():
    ckpt = torch.load(NLD_CHECKPOINT, map_location=device, weights_only=False)
    model = DualStreamTransformer(d_model=128, num_heads=4, num_layers=4, dropout=0.0)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    y_mean = np.array(ckpt["y_mean"], dtype=np.float32)
    y_std = np.array(ckpt["y_std"], dtype=np.float32)

    loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    sbp_errors, dbp_errors = [], []
    cm = np.zeros((2, 2), dtype=np.int64)

    with torch.no_grad():
        for ppg, poincare, bps in loader:
            pred = model(ppg.to(device), poincare.to(device)).cpu().numpy()
            actual = bps.numpy()
            pred = pred * y_std + y_mean
            actual = actual * y_std + y_mean
            for j in range(pred.shape[0]):
                sbp_errors.append(abs(pred[j, 0] - actual[j, 0]))
                dbp_errors.append(abs(pred[j, 1] - actual[j, 1]))
                cm[
                    classify_bp(actual[j, 0], actual[j, 1]),
                    classify_bp(pred[j, 0], pred[j, 1]),
                ] += 1

    return sbp_errors, dbp_errors, cm


#  basic transformer (2-min PPG only)


def eval_basic():
    ckpt = torch.load(BASIC_CHECKPOINT, map_location=device, weights_only=False)
    model = BasicTransformer(d_model=128, num_heads=4, num_layers=4, dropout=0.0)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    y_mean = np.array(ckpt["y_mean"], dtype=np.float32)
    y_std = np.array(ckpt["y_std"], dtype=np.float32)

    loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    sbp_errors, dbp_errors = [], []
    cm = np.zeros((2, 2), dtype=np.int64)

    with torch.no_grad():
        for ppg, _poincare, bps in loader:
            pred = model(ppg.to(device)).cpu().numpy()
            actual = bps.numpy()
            pred = pred * y_std + y_mean
            actual = actual * y_std + y_mean
            for j in range(pred.shape[0]):
                sbp_errors.append(abs(pred[j, 0] - actual[j, 0]))
                dbp_errors.append(abs(pred[j, 1] - actual[j, 1]))
                cm[
                    classify_bp(actual[j, 0], actual[j, 1]),
                    classify_bp(pred[j, 0], pred[j, 1]),
                ] += 1

    return sbp_errors, dbp_errors, cm


print("Evaluating NLD transformer (2-min windows)...")
nld_sbp_err, nld_dbp_err, nld_cm = eval_nld()

print("Evaluating basic transformer (10s windows)...")
basic_sbp_err, basic_dbp_err, basic_cm = eval_basic()

# Print metrics
for name, sbp_e, dbp_e, cm in [
    ("Basic transformer (PPG only)", basic_sbp_err, basic_dbp_err, basic_cm),
    ("NLD transformer  (PPG + Poincaré)", nld_sbp_err, nld_dbp_err, nld_cm),
]:
    print(f"\n── {name} ──")
    print(f"SBP MAE: {np.mean(sbp_e):.2f} ± {np.std(sbp_e):.2f} mmHg")
    print(f"DBP MAE: {np.mean(dbp_e):.2f} ± {np.std(dbp_e):.2f} mmHg")
