from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DATA_DIR = Path("data")

# Fields that are stored as uint16 ASCII char arrays in the HDF5 refs
STRING_FIELDS = {"SubjectID", "CaseID"}
# Field whose scalar value is an ASCII char code (77='M', 70='F')
CHAR_FIELDS = {"Gender"}


def decode_ref(ref, f: h5py.File):
    """Dereference one HDF5 object reference and return a Python scalar, string, or list."""
    ds = f[ref]
    val = ds[()]

    # String stored as uint16 char array
    if ds.dtype == np.uint16 and val.size > 1:
        return "".join(chr(c) for c in val.flatten())

    # Scalar
    if val.size == 1:
        raw = val.flat[0]
        return float(raw) if ds.dtype.kind == "f" else int(raw)

    # Array (time-series or variable-length peaks/turns)
    return val.flatten().tolist()


def convert(mat_path: Path):
    out_path = mat_path.with_suffix(".parquet")

    with h5py.File(mat_path, "r") as f:
        sw = f["Subj_Wins"]
        fields = list(sw.keys())
        n_wins = sw[fields[0]].shape[1]

        rows = []
        for i in range(n_wins):
            row = {}
            for field in fields:
                ref = sw[field][0, i]
                value = decode_ref(ref, f)
                if field in CHAR_FIELDS and isinstance(value, int):
                    value = chr(value)
                row[field] = value
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    mat_path.unlink()
    print(f"{mat_path.name} -> {out_path.name} ({n_wins} windows)")


if __name__ == "__main__":
    mat_files = sorted(DATA_DIR.glob("*.mat"))
    if not mat_files:
        print("No .mat files found.")
    else:
        for f in mat_files:
            convert(f)
        print("Done.")
