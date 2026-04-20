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

## Results

Run the `eval_transformer.py` file to run the trained transformer models on the test set, and run the `eval_lstm.py` to run the LSTM model on the test set.

Here are the combined results

```
── LSTM (PPG only) ──
SBP MAE: 20.54 ± 11.52 mmHg
DBP MAE: 15.88 ± 8.70 mmHg
── Basic transformer (PPG only) ──
SBP MAE: 15.35 ± 9.70 mmHg
DBP MAE: 9.38 ± 5.40 mmHg
── NLD transformer  (PPG + Poincaré) ──
SBP MAE: 14.07 ± 9.37 mmHg
DBP MAE: 7.75 ± 5.84 mmHg

GOAL:
── RNN (from paper, PPG only) ──
SBP MAE: 14.39 mmHg
DBP MAE: 6.57 mmHg

```

| Model                            | SBP MAE (mmHg)   | DBP MAE (mmHg)  |
| -------------------------------- | ---------------- | --------------- |
| LSTM (PPG only)                  | 20.54 ± 11.52    | 15.88 ± 8.70    |
| Basic Transformer (PPG only)     | 15.35 ± 9.70     | 9.38 ± 5.40     |
| NLD Transformer (PPG + Poincaré) | **14.07 ± 9.37** | **7.75 ± 5.84** |
| Paper RNN baseline (PPG only)    | 14.39            | 6.57            |

Our goal was to predict BP calibration-free purely on PPG data. The benchmark in the paper for that is an SBP MAE of 14.39 mmHg and a DBP MAE of 6.57 mmHg. **Our NLD transformer beat the systollic benchmark** and came close to beating the diastolic benchmark (slightly worse).

## NLD Transformer Architecture

```
=========================================================================================================
Layer (type:depth-idx)                                  Output Shape              Param #
=========================================================================================================
DualStreamTransformer                                   [32, 2]                   --
├─PPGDownsampler: 1-1                                   [32, 500, 128]            --
│    └─Sequential: 2-1                                  [32, 128, 500]            --
│    │    └─Conv1d: 3-1                                 [32, 32, 3000]            352
│    │    └─GELU: 3-2                                   [32, 32, 3000]            --
│    │    └─BatchNorm1d: 3-3                            [32, 32, 3000]            64
│    │    └─Conv1d: 3-4                                 [32, 128, 500]            49,280
│    │    └─GELU: 3-5                                   [32, 128, 500]            --
│    │    └─BatchNorm1d: 3-6                            [32, 128, 500]            256
├─PoincareSequenceEncoder: 1-2                          [32, 4, 128]              --
│    └─PoincareCNN: 2-2                                 [128, 64]                 --
│    │    └─Sequential: 3-7                             [128, 64]                 135,936
│    └─Linear: 2-3                                      [32, 4, 128]              8,320
├─PositionalEncoding: 1-3                               [32, 504, 128]            --
│    └─Dropout: 2-4                                     [32, 504, 128]            --
├─ModuleList: 1-4                                       --                        --
│    └─TransformerEncoderLayer: 2-5                     [32, 504, 128]            --
│    │    └─MultiHeadAttention: 3-8                     [32, 504, 128]            65,536
│    │    └─Dropout: 3-9                                [32, 504, 128]            --
│    │    └─LayerNorm: 3-10                             [32, 504, 128]            256
│    │    └─Sequential: 3-11                            [32, 504, 128]            131,712
│    │    └─Dropout: 3-12                               [32, 504, 128]            --
│    │    └─LayerNorm: 3-13                             [32, 504, 128]            256
│    └─TransformerEncoderLayer: 2-6                     [32, 504, 128]            --
│    │    └─MultiHeadAttention: 3-14                    [32, 504, 128]            65,536
│    │    └─Dropout: 3-15                               [32, 504, 128]            --
│    │    └─LayerNorm: 3-16                             [32, 504, 128]            256
│    │    └─Sequential: 3-17                            [32, 504, 128]            131,712
│    │    └─Dropout: 3-18                               [32, 504, 128]            --
│    │    └─LayerNorm: 3-19                             [32, 504, 128]            256
│    └─TransformerEncoderLayer: 2-7                     [32, 504, 128]            --
│    │    └─MultiHeadAttention: 3-20                    [32, 504, 128]            65,536
│    │    └─Dropout: 3-21                               [32, 504, 128]            --
│    │    └─LayerNorm: 3-22                             [32, 504, 128]            256
│    │    └─Sequential: 3-23                            [32, 504, 128]            131,712
│    │    └─Dropout: 3-24                               [32, 504, 128]            --
│    │    └─LayerNorm: 3-25                             [32, 504, 128]            256
│    └─TransformerEncoderLayer: 2-8                     [32, 504, 128]            --
│    │    └─MultiHeadAttention: 3-26                    [32, 504, 128]            65,536
│    │    └─Dropout: 3-27                               [32, 504, 128]            --
│    │    └─LayerNorm: 3-28                             [32, 504, 128]            256
│    │    └─Sequential: 3-29                            [32, 504, 128]            131,712
│    │    └─Dropout: 3-30                               [32, 504, 128]            --
│    │    └─LayerNorm: 3-31                             [32, 504, 128]            256
├─Linear: 1-5                                           [32, 2]                   258
=========================================================================================================
Total params: 985,506
Trainable params: 985,506
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 1.04
=========================================================================================================
Input size (MB): 2.44
Forward/backward pass size (MB): 833.95
Params size (MB): 3.94
Estimated Total Size (MB): 840.33
=========================================================================================================
```
