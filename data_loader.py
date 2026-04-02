import functools
import os
from glob import glob

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from torch.utils.data import DataLoader, Dataset

DATA_DIR = os.getenv("DATA_DIR", "data")
BATCH_SIZE = 32

WINDOWS_PER_2MIN = 12  # 12 × 10s = 2 minutes
POINCARE_EVERY_N = 3  # one Poincaré image per 30s (every 3 sub-windows)
NUM_POINCARE = WINDOWS_PER_2MIN // POINCARE_EVERY_N  # 4 images per 2-min chunk
SUB_WIN_LEN = 1250  # samples per 10s window at 125 Hz
SEQ_LEN = WINDOWS_PER_2MIN * SUB_WIN_LEN  # 15,000 samples per 2-min chunk


def create_poincare_histogram(ppg_waveform, grid_size=32):
    """Generates a 32x32 2D density histogram of RR intervals from a PPG segment."""
    peaks, _ = find_peaks(ppg_waveform, distance=62, prominence=5)
    if len(peaks) < 3:
        return np.zeros((1, grid_size, grid_size), dtype=np.float32)
    rr_intervals = np.diff(peaks)
    x = rr_intervals[:-1]
    y = rr_intervals[1:]
    H, _, _ = np.histogram2d(x, y, bins=grid_size, range=[[40, 150], [40, 150]])
    return np.expand_dims(H.astype(np.float32), axis=0)


@functools.lru_cache(maxsize=8)
def _load_parquet(filepath):
    """Load only the columns we need; cached so each file is read once per process."""
    return (
        pd.read_parquet(filepath, columns=["WinSeqID", "PPG_F", "SegSBP", "SegDBP"])
        .sort_values("WinSeqID")
        .reset_index(drop=True)
    )


def _build_chunk_index(file_list):
    """
    Scan files (reading only WinSeqID) to find all valid 2-min chunks.
    Returns a list of (filepath, [row_indices]) — one entry per chunk.
    """
    chunks = []
    for f in file_list:
        df_ids = (
            pd.read_parquet(f, columns=["WinSeqID"])
            .sort_values("WinSeqID")
            .reset_index(drop=True)
        )
        win_ids = df_ids["WinSeqID"].values
        diffs = np.diff(win_ids)

        run_starts = [0]
        for i, d in enumerate(diffs):
            if d != 1:
                run_starts.append(i + 1)
        run_starts.append(len(win_ids))

        for start, end in zip(run_starts[:-1], run_starts[1:]):
            run_len = end - start
            for c in range(run_len // WINDOWS_PER_2MIN):
                row_start = start + c * WINDOWS_PER_2MIN
                chunks.append((f, list(range(row_start, row_start + WINDOWS_PER_2MIN))))

    return chunks


class BPDataset(Dataset):
    """
    Lazy dataset for 2-minute BP prediction chunks.

    Chunk indices are built at init (reads only WinSeqID column — fast).
    Full parquet files are loaded on demand in __getitem__ and cached in
    memory (up to 16 files) to avoid re-reading the same file repeatedly.
    """

    def __init__(self, file_list, X_mean, X_std, y_mean, y_std):
        self.X_mean = X_mean
        self.X_std = X_std
        self.y_mean = y_mean  # (2,) [sbp_mean, dbp_mean]
        self.y_std = y_std  # (2,) [sbp_std,  dbp_std]
        self.chunks = _build_chunk_index(file_list)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        filepath, row_indices = self.chunks[idx]
        df = _load_parquet(filepath)
        sub_df = df.iloc[row_indices]

        ppg_windows = np.stack(sub_df["PPG_F"].values).astype(np.float32)  # (12, 1250)
        ppg_seq = (ppg_windows.reshape(-1) - self.X_mean) / self.X_std  # (15000,)

        # Mean SBP/DBP over the 2-minute window → (2,)
        sbp_mean = sub_df["SegSBP"].values.astype(np.float32).mean()
        dbp_mean = sub_df["SegDBP"].values.astype(np.float32).mean()
        bp_seq = (np.array([sbp_mean, dbp_mean]) - self.y_mean) / self.y_std  # (2,)

        poincare_imgs = []
        for p in range(NUM_POINCARE):
            segment = ppg_windows[
                p * POINCARE_EVERY_N : (p + 1) * POINCARE_EVERY_N
            ].reshape(-1)
            poincare_imgs.append(create_poincare_histogram(segment))
        poincare_seq = np.stack(poincare_imgs)  # (4, 1, 32, 32)

        return (
            torch.from_numpy(ppg_seq),
            torch.from_numpy(poincare_seq),
            torch.from_numpy(bp_seq),
        )


def compute_normalization_stats(file_list, sample_files=30):
    """
    Estimate normalization stats from a sample of training files.
    Only reads SegSBP, SegDBP, and a few PPG rows — stays lightweight.
    """
    sampled = list(file_list)[:sample_files]
    sbp_vals, dbp_vals, ppg_vals = [], [], []
    for f in sampled:
        df = pd.read_parquet(f, columns=["SegSBP", "SegDBP", "PPG_F"])
        sbp_vals.extend(df["SegSBP"].values)
        dbp_vals.extend(df["SegDBP"].values)
        for row in df["PPG_F"].values[:10]:
            ppg_vals.extend(row)

    ppg_arr = np.array(ppg_vals, dtype=np.float32)
    X_mean = float(ppg_arr.mean())
    X_std = float(ppg_arr.std())
    y_mean = np.array([np.mean(sbp_vals), np.mean(dbp_vals)], dtype=np.float32)
    y_std = np.array([np.std(sbp_vals), np.std(dbp_vals)], dtype=np.float32)
    return X_mean, X_std, y_mean, y_std


# ---------------------------------------------------------------------------
# Split files and build datasets
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
all_files = sorted(glob(f"{DATA_DIR}/*.parquet"))
shuffled = rng.permutation(all_files)
n_train = int(0.70 * len(shuffled))
n_val = int(0.15 * len(shuffled))
train_files = shuffled[:n_train]
val_files = shuffled[n_train : n_train + n_val]
test_files = shuffled[n_train + n_val :]

print(
    f"Files — train: {len(train_files)} | val: {len(val_files)} | test: {len(test_files)}"
)
print("Computing normalization stats from training sample...")
X_mean, X_std, y_mean, y_std = compute_normalization_stats(train_files)
print(f"  PPG  mean={X_mean:.3f} std={X_std:.3f}")
print(f"  SBP  mean={y_mean[0]:.1f} std={y_std[0]:.1f} mmHg")
print(f"  DBP  mean={y_mean[1]:.1f} std={y_std[1]:.1f} mmHg")

print("Indexing chunks (no data loaded yet)...")
train_dataset = BPDataset(train_files, X_mean, X_std, y_mean, y_std)
val_dataset = BPDataset(val_files, X_mean, X_std, y_mean, y_std)
test_dataset = BPDataset(test_files, X_mean, X_std, y_mean, y_std)
print(
    f"Chunks — train: {len(train_dataset)} | val: {len(val_dataset)} | test: {len(test_dataset)}"
)

# num_workers>0: workers load data in parallel with GPU compute.
# lru_cache is per-worker (not shared), but overlap of I/O and GPU is worth it.
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)
