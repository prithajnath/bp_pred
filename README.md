# BP prediction

Download the dataset by running the `download_pulse_db.py` script. Currently it only download the (cleaned) vitalDB segments from Google Drive. You can set the number of subjects using the `--num-subjects` argument.

Download waveform data for the first 20 subjects

```
uv run download_pulse_db.py --num-subjects 20
```

This should download 20 `mat` files in the `data` directory. We need to convert them to parquet before we can use them in Pandas. You can run the `convert_mat_to_parquet.py` script and it will convert all of those files to parquet

```
uv run convert_mat_to_parquet.py
```

You should have something like this in your `data` dir

```
data
├── p005966.parquet
├── p005967.parquet
├── p005968.parquet
├── p005969.parquet
├── p005974.parquet
├── p005977.parquet
├── p005978.parquet
├── p005980.parquet
├── p005983.parquet
├── p006066.parquet
├── p006069.parquet
├── p006077.parquet
├── p006079.parquet
├── p006081.parquet
├── p006082.parquet
├── p006084.parquet
├── p006086.parquet
├── p006087.parquet
├── p006088.parquet
└── p006090.parquet

1 directory, 20 files

```
