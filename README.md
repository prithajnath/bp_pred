# BP prediction

## Downloading the dataset

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

## Data processing

All data processing is done by the `data_loader.py` file. It loads all parquet files for all patients and creates the following

1. 2-min windows of PPG sequence for every patient
2. Train/Val/Test splits

We train for calibration-free prediction: no subject in the training set appears in the test set. The 2-minute window was chosen to capture autonomic nervous system dynamics — specifically the LF HRV band (Mayer waves, ~7-25s) which are physiologically linked to blood pressure via baroreceptor feedback, which are not observable in a 10s window.

However, since the data is sampled at 125 Hz, a 2-minute window contains 125 × 120 = 15,000 samples. Self-attention scales quadratically with sequence length, so a 15,000-token sequence would require a 15,000 × 15,000 attention matrix — not feasible. We use a 1D CNN to downsample from 15,000 samples to 500 tokens before the transformer.

In addition to the PPG stream, we extract 4 Poincaré plots (one per 30s sub-window) as a second input stream. Each plot is a 32×32 density histogram of successive RR intervals, encoding sympatho-vagal balance, and is processed by a small CNN before being concatenated with the PPG tokens

## Transformer results

Run the `eval_transformer.py` file to run the trained models on the test set

```
── Basic transformer (PPG only) ──
SBP MAE: 15.35 ± 9.70 mmHg
DBP MAE: 9.38 ± 5.40 mmHg
── NLD transformer  (PPG + Poincaré) ──
SBP MAE: 14.07 ± 9.37 mmHg
DBP MAE: 7.75 ± 5.84 mmHg

```

Our goal was to predict BP calibration-free purely on PPG data. The benchmark in the paper for that is an SBP MAE of 14.39 mmHg and a DBP MAE of 6.57 mmHg. Our NLD transformer beat the systollic benchmark and came close to beating the diastolic benchmark (slightly worse).
