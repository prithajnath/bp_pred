import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

DATA_DIR = Path("data/pulsedb-balanced-training-and-testing")


def load_mat(path: Path) -> dict:
    """Load a .mat file, handling both legacy and v7.3 (HDF5) formats."""
    try:
        mat = sio.loadmat(path)
        return {k: v for k, v in mat.items() if not k.startswith("_")}
    except NotImplementedError:
        import mat73
        return mat73.loadmat(str(path))


def flatten(obj, prefix="") -> dict[str, np.ndarray]:
    """Recursively flatten nested dicts into {dotted.key: array} pairs."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, np.ndarray):
        out[prefix] = obj
    else:
        out[prefix] = np.asarray(obj)
    return out


def to_dataframe(arr: np.ndarray) -> pd.DataFrame:
    if arr.ndim == 1:
        return pd.DataFrame({"value": arr})
    elif arr.ndim == 2:
        return pd.DataFrame(arr)
    else:
        raise ValueError(f"Cannot convert array with {arr.ndim} dims to DataFrame")


def save_value(value, out_path: Path, label: str):
    """Save a value (array or nested dict) to parquet file(s)."""
    if isinstance(value, dict):
        leaves = flatten(value)
        arrays = {k: v for k, v in leaves.items() if isinstance(v, np.ndarray) and v.ndim <= 2}
        if arrays:
            # Try to combine into one DataFrame if shapes match
            shapes = {v.shape[0] if v.ndim > 0 else 1 for v in arrays.values()}
            if len(shapes) == 1:
                df = pd.DataFrame({k: (v if v.ndim == 1 else list(v)) for k, v in arrays.items()})
                df.to_parquet(out_path, index=False)
                print(f"  -> {out_path.name} ({len(arrays)} fields, {df.shape[0]} rows)")
            else:
                # Shapes differ — save each leaf separately
                for k, arr in arrays.items():
                    leaf_path = out_path.with_name(out_path.stem + f"_{k.replace('.', '_')}.parquet")
                    to_dataframe(arr).to_parquet(leaf_path, index=False)
                    print(f"  -> {leaf_path.name} {arr.shape}")
    else:
        arr = np.asarray(value)
        try:
            to_dataframe(arr).to_parquet(out_path, index=False)
            print(f"  -> {out_path.name} {arr.shape}")
        except ValueError as e:
            warnings.warn(f"Skipping '{label}': {e}")


def convert(mat_path: Path):
    print(f"Loading {mat_path.name} ...")
    data = load_mat(mat_path)

    for key, value in data.items():
        out_path = mat_path.with_name(f"{mat_path.stem}_{key}.parquet")
        save_value(value, out_path, key)

    mat_path.unlink()
    print(f"Deleted {mat_path.name}")


if __name__ == "__main__":
    mat_files = sorted(DATA_DIR.glob("*.mat"))
    if not mat_files:
        print("No .mat files found.")
    else:
        for f in mat_files:
            convert(f)
        print("Done.")
