import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Set to my external drive holding the data
DATA_DIR = Path(os.getenv("DATA_DIR", "D:\DL Data NEW")) or Path("data")

STRING_FIELDS = {"SubjectID", "CaseID"}
CHAR_FIELDS = {"Gender"}


def infer_field_types(sw: h5py.Group, f: h5py.File) -> dict[str, str]:
    """Determine 'string', 'scalar', or 'array' for each field using window 0."""
    types = {}
    for field in sw.keys():
        ref = sw[field][0, 0]

        # ref would sometimes be a float, so set the type appropriately
        if not isinstance(ref, h5py.Reference):
            types[field] = "scalar"
            continue

        ds = f[ref]
        if ds.dtype == np.uint16:
            types[field] = "string"
        elif ds.shape == (1, 1):
            types[field] = "scalar"
        else:
            types[field] = "array"
    return types


def convert(mat_path: Path):
    out_path = mat_path.with_suffix(".parquet")

    with h5py.File(mat_path, "r") as f:
        sw = f["Subj_Wins"]
        fields = list(sw.keys())
        n_wins = sw[fields[0]].shape[1]
        field_types = infer_field_types(sw, f)

        rows = []
        for i in range(n_wins):
            row = {}
            for field in fields:
                # Again because some ref fields were floats, had to adjust how cols were retrieved
                col = min(i, sw[field].shape[1] - 1)
                raw_ref = sw[field][0, col]
                ftype = field_types[field]

                # Direct numeric value
                if not isinstance(raw_ref, h5py.Reference):
                    raw = raw_ref
                    if field in CHAR_FIELDS:
                        row[field] = chr(int(raw))
                    else:
                        row[field] = float(raw) if isinstance(raw, float) else int(raw)
                    continue

                ds = f[raw_ref]
                val = ds[()].flatten()
                ftype = field_types[field]

                if ftype == "string":
                    row[field] = "".join(chr(c) for c in val)
                elif ftype == "scalar":
                    raw = val[0]
                    if field in CHAR_FIELDS:
                        row[field] = chr(int(raw))
                    else:
                        row[field] = float(raw) if ds.dtype.kind == "f" else int(raw)
                else:
                    row[field] = val.tolist()

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
