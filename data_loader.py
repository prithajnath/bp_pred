from glob import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = "data"
BATCH_SIZE = 3

# Split patient files before loading to avoid same-patient leakage across splits
rng = np.random.default_rng(42)
all_files = sorted(glob(f"{DATA_DIR}/*.parquet"))
shuffled = rng.permutation(all_files)
n_train_files = int(0.7 * len(shuffled))
n_val_files = int(0.15 * len(shuffled))
train_files = shuffled[:n_train_files]
val_files = shuffled[n_train_files : n_train_files + n_val_files]
test_files = shuffled[n_train_files + n_val_files :]

train_df = pd.concat([pd.read_parquet(f) for f in train_files], ignore_index=True)
val_df = pd.concat([pd.read_parquet(f) for f in val_files], ignore_index=True)
test_df = pd.concat([pd.read_parquet(f) for f in test_files], ignore_index=True)

X_train = np.stack(train_df["PPG_F"].values, dtype=np.float32)
y_train = np.stack(train_df["ABP_Raw"].values, dtype=np.float32)
X_val = np.stack(val_df["PPG_F"].values, dtype=np.float32)
y_val = np.stack(val_df["ABP_Raw"].values, dtype=np.float32)
X_test = np.stack(test_df["PPG_F"].values, dtype=np.float32)
y_test = np.stack(test_df["ABP_Raw"].values, dtype=np.float32)

# Normalize using train stats only
X_mean, X_std = X_train.mean(), X_train.std()
y_mean, y_std = y_train.mean(), y_train.std()

X_train = (X_train - X_mean) / X_std
y_train = (y_train - y_mean) / y_std
X_val = (X_val - X_mean) / X_std
y_val = (y_val - y_mean) / y_std
X_test = (X_test - X_mean) / X_std
y_test = (y_test - y_mean) / y_std

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
