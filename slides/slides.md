---
theme: seriph
title: Cuff-less Blood Pressure Prediction from PPG
info: |
  Blood pressure prediction from photoplethysmography using a dual-stream transformer with Poincaré plot features.
class: text-center
drawings:
  persist: false
transition: slide-left
fonts:
  sans: Space Mono
  mono: Space Mono
  provider: google
background: /bg.gif
---

# Cuff-less Blood Pressure Prediction

From PPG signals using a dual-stream transformer

<div class="abs-br m-6 text-sm opacity-50">
  Prithaj Nath, Alex Stute · 2026
</div>

---

# The Problem

<v-click>

Blood pressure (BP) is a key cardiovascular health marker — but traditional measurement requires a cuff.

</v-click>

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

<v-click>

**Goal:** predict systolic (SBP) and diastolic (DBP) blood pressure continuously from a **photoplethysmogram (PPG)** — a wearable optical signal.

</v-click>

**Why it's hard:**
<v-clicks>

- BP varies across individuals in ways that aren't fully encoded in PPG morphology
- Models that see a subject at training time perform far better than on unseen subjects (**calibration gap**)
- Most prior work used inconsistent, messy datasets, making comparisons unfair

</v-clicks>

</div>
<div>

<v-click>

**Our framing:**
> Calibration-free prediction — no subject in the training set appears in the test set.

</v-click>

<v-click>

This is the harder, more clinically meaningful problem.

</v-click>

**Benchmark** (PulseDB paper RNN):
<v-clicks>

- SBP MAE: 14.39 mmHg
- DBP MAE: 6.57 mmHg

</v-clicks>

</div>
</div>

---

# How We Got Here

<!--
Learning normal morphology with DL is already hard enough, and framing the problem this way made it even harder.
-->

<div class="grid grid-cols-2 gap-8 mt-3">
<div>

<v-click>

We initially regressed the full **ABP waveform** and applied an if-else threshold to classify patients as normotensive or hypertensive.

</v-click>

<v-click>

This pipeline was opaque — waveform errors and threshold sensitivity compounded, and the final label told you nothing about how far off the model was.

</v-click>

<v-click>

**Reading the PulseDB paper revealed a simpler target:** regress SBP and DBP directly — no waveform, no thresholding, no category collapse.

</v-click>

<v-clicks>

- Waveform + if-else is a strictly harder pipeline with more failure modes
- MAE in mmHg is interpretable and comparable to clinical standards (AAMI: ±5 mmHg)
- The benchmark exists — regression is both more honest and more competitive

</v-clicks>

</div>
<div>

<v-click>

<img src="/confusion_matrix.png" class="w-full rounded" style="filter:invert(1) hue-rotate(180deg) brightness(0.85)" />
<p class="text-xs mt-1 opacity-60 text-center">67% of hypertensive cases misclassified as normal — the if-else threshold obscures what the model actually learned</p>

</v-click>

</div>
</div>

---

# Dataset: PulseDB

<v-click>

Wang et al. 2023 — *Frontiers in Digital Health*

</v-click>

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

<v-click>

**What it is:**  
5,245,454 cleaned 10-second segments of PPG + ECG + arterial BP waveforms from **5,361 subjects** across MIMIC-III and VitalDB.

</v-click>

**Why it matters:**
<v-clicks>

- Largest cleaned dataset for cuff-less BP benchmarking
- Subject IDs included → enables calibration-free splits
- Meets AAMI standard requirements (>85 subjects in test, >5% low/high BP range)
- Beat-to-beat characteristic points included

</v-clicks>

</div>
<div>

<v-click>

**Our subset:** VitalDB segments (ICU surgical patients)

</v-click>

**Calibration-free split:**
<v-clicks>

- Training subjects: 2,506
- Test subjects: 279 (completely disjoint)
- 2-minute windows of PPG, sampled at 125 Hz

</v-clicks>

**BP distribution:**
<v-clicks>

- SBP: 121.42 ± 22.10 mmHg
- DBP: 61.87 ± 13.01 mmHg

</v-clicks>

</div>
</div>

<!-- Our dataset consists of ~5.2 million 10-second segments of PPG, ECG, and BP waveforms from ~5300 subjects. This is the largest prepared dataset for cuff-less BP measurement and comes "ready to go". We are using a subset of the data labeled as VitalDB, which are ICU surgical patients, which was split into training and test subjects (as on slide). Blood pressure measurement distributions are as such:-->

---

# Why 2-Minute Windows?

<v-click>

A **10-second** window captures heartbeat morphology — but blood pressure is regulated by the autonomic nervous system over much longer timescales.

</v-click>

<div class="grid grid-cols-2 gap-6 mt-3">
<div>

<div class="text-xs opacity-50 mb-1">anatomy of a single PPG pulse</div>

<svg viewBox="0 0 390 165" class="w-full">
  <line x1="30" y1="120" x2="360" y2="120" stroke="#444" stroke-width="0.5"/>
  <v-click><path class="ppg-wave" d="M 40,120 C 50,120 70,25 110,25 C 148,25 165,77 200,77 C 215,77 225,63 240,63 C 280,63 310,120 340,120"
    fill="none" stroke="#c0524a" stroke-width="2.5" stroke-linecap="round"/></v-click>
  <g v-click>
    <circle cx="110" cy="25" r="4" fill="#c0524a"/>
    <line x1="110" y1="21" x2="110" y2="13" stroke="#888" stroke-width="1" stroke-dasharray="2,2"/>
    <text x="110" y="12" style="font-size:9px;font-family:monospace" fill="#ddd" dominant-baseline="auto" text-anchor="middle">systolic peak</text>
  </g>
  <g v-click>
    <circle cx="200" cy="77" r="4" fill="none" stroke="#f0c060" stroke-width="1.5"/>
    <line x1="204" y1="80" x2="228" y2="97" stroke="#f0c060" stroke-width="1"/>
    <text x="230" y="101" style="font-size:9px;font-family:monospace" fill="#f0c060" dominant-baseline="hanging">dicrotic notch</text>
  </g>
  <g v-click>
    <circle cx="240" cy="63" r="4" fill="none" stroke="#6aabff" stroke-width="1.5"/>
    <line x1="244" y1="60" x2="262" y2="47" stroke="#6aabff" stroke-width="1"/>
    <text x="264" y="47" style="font-size:9px;font-family:monospace" fill="#6aabff" dominant-baseline="middle">diastolic wave</text>
  </g>
  <g v-click>
    <line x1="40" y1="138" x2="340" y2="138" stroke="#666" stroke-width="1"/>
    <line x1="40" y1="133" x2="40" y2="143" stroke="#666" stroke-width="1"/>
    <line x1="340" y1="133" x2="340" y2="143" stroke="#666" stroke-width="1"/>
    <text x="190" y="155" style="font-size:8px;font-family:monospace" fill="#888" text-anchor="middle">one cardiac cycle (~860ms @ 70 bpm)</text>
  </g>
</svg>

</div>
<div>

<v-click>
<div class="text-xs opacity-50 mb-1">amplitude modulation across beats — Mayer waves</div>
</v-click>

<v-click>
<svg viewBox="0 0 415 120" class="w-full">
  <line x1="5" y1="105" x2="410" y2="105" stroke="#444" stroke-width="0.5"/>
  <!-- 8 beats with sinusoidally modulated amplitude -->
  <path class="mayer-wave" d="M 5,105 C 8,105 12,25 17,25 C 23,25 27,67 32,67 C 36,67 38,57 41,57 C 49,57 53,105 55,105 C 58,105 62,18 67,18 C 73,18 77,63 82,63 C 86,63 88,53 91,53 C 99,53 103,105 105,105 C 108,105 112,32 117,32 C 123,32 127,70 132,70 C 136,70 138,61 141,61 C 149,61 153,105 155,105 C 158,105 162,55 167,55 C 173,55 177,81 182,81 C 186,81 188,75 191,75 C 199,75 203,105 205,105 C 208,105 212,72 217,72 C 223,72 227,89 232,89 C 236,89 238,85 241,85 C 249,85 253,105 255,105 C 258,105 262,62 267,62 C 273,62 277,84 282,84 C 286,84 288,79 291,79 C 299,79 303,105 305,105 C 308,105 312,38 317,38 C 323,38 327,73 332,73 C 336,73 338,65 341,65 C 349,65 353,105 355,105 C 358,105 362,22 367,22 C 373,22 377,65 382,65 C 386,65 388,55 391,55 C 399,55 403,105 405,105"
    fill="none" stroke="#c0524a" stroke-width="1.8" stroke-linecap="round"/>
  <!-- slow envelope through systolic peaks -->
  <path class="mayer-envelope" d="M 17,25 C 42,22 42,18 67,18 C 92,18 92,32 117,32 C 142,32 142,55 167,55 C 192,55 192,72 217,72 C 242,72 242,62 267,62 C 292,62 292,38 317,38 C 342,38 342,22 367,22"
    fill="none" stroke="#f0c06080" stroke-width="1.5" stroke-dasharray="4,3"/>
  <!-- Mayer wave annotation -->
  <text x="190" y="14" style="font-size:8px;font-family:monospace" fill="#f0c060" text-anchor="middle" dominant-baseline="middle">Mayer wave envelope (~10–25s period)</text>
  <line x1="67" y1="16" x2="190" y2="16" stroke="#f0c06060" stroke-width="0.8"/>
  <line x1="317" y1="16" x2="190" y2="16" stroke="#f0c06060" stroke-width="0.8"/>
</svg>
</v-click>

<v-click>
<p class="text-xs mt-2 opacity-80">The slow envelope is the <strong>LF HRV band</strong> (0.04–0.15 Hz) — baroreceptor feedback modulating heart rate in sync with blood pressure. One full cycle takes 7–25 seconds. You need at least 2 minutes to capture enough cycles for the model to use it.</p>
</v-click>

</div>
</div>

<!--
The 2-minute window is justified by the physiology of baroreceptor feedback.

Baroreceptors in the aortic arch and carotid sinus detect changes in arterial pressure and send signals to the autonomic nervous system. The ANS responds by adjusting heart rate — this creates periodic oscillations in RR intervals at roughly 0.04–0.15 Hz, which is the LF HRV band.

These oscillations are what we see as Mayer waves: a slow amplitude modulation of the PPG signal with a period of about 7–25 seconds.

To capture even one full Mayer wave cycle you need up to 25 seconds. To give the model enough cycles to learn the pattern from — and to reliably estimate LF HRV — you need at least 2 minutes. That's why the window length is what it is.
-->

---

# Poincaré Plots as a Second Input

<v-click>

Encoding sympatho-vagal balance from RR-interval dynamics.

</v-click>

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

<v-click>

A **Poincaré plot** plots each RR interval against the next one. Its shape encodes:

</v-click>

<v-clicks>

- SD1 (short-term HRV) → parasympathetic activity
- SD2 (long-term HRV) → sympathetic activity

</v-clicks>

<v-click>

**Our approach:** 4 plots, one per 30-second sub-window of the 2-minute input. Each plot is a **32×32 density histogram** of successive RR intervals.

</v-click>

</div>
<div>

<v-click>

<img src="/poincare_example.png" class="w-full rounded" style="filter:invert(1) hue-rotate(180deg) brightness(0.85)" />

</v-click>

<v-click>

<p class="text-xs mt-2 opacity-70">Comet (healthy), torpedo, and complex patterns encode distinct sympatho-vagal states — the CNN learns to distinguish these directly from the density histogram. Adapted from Woo et al. (1992).</p>

</v-click>

</div>
</div>

---

# NLD Transformer Architecture

<ArchViz />

---

# Results

<table class="w-full mt-4" style="border-collapse:collapse;font-family:monospace;font-size:0.82em">
  <thead>
    <tr style="border-bottom:1px solid #444;color:#888">
      <th style="text-align:left;padding:6px 10px;font-weight:normal">Model</th>
      <th style="text-align:right;padding:6px 10px;font-weight:normal">SBP MAE (mmHg)</th>
      <th style="text-align:right;padding:6px 10px;font-weight:normal">DBP MAE (mmHg)</th>
    </tr>
  </thead>
  <tbody>
    <tr v-click style="color:#666">
      <td style="padding:5px 10px">LSTM (PPG only)</td>
      <td style="text-align:right;padding:5px 10px">20.54 ± 11.52</td>
      <td style="text-align:right;padding:5px 10px">15.88 ± 8.70</td>
    </tr>
    <tr v-click style="color:#888">
      <td style="padding:5px 10px">Basic Transformer (PPG only)</td>
      <td style="text-align:right;padding:5px 10px">15.35 ± 9.70</td>
      <td style="text-align:right;padding:5px 10px">9.38 ± 5.40</td>
    </tr>
    <tr v-click style="color:#666;border-top:1px dashed #333">
      <td style="padding:5px 10px">Paper RNN baseline (PPG only)</td>
      <td style="text-align:right;padding:5px 10px">14.39</td>
      <td style="text-align:right;padding:5px 10px">6.57</td>
    </tr>
    <tr v-click class="nld-row">
      <td style="padding:6px 10px;font-weight:bold">NLD Transformer (PPG + Poincaré)</td>
      <td style="text-align:right;padding:6px 10px;font-weight:bold">14.07 ± 9.37 ✓</td>
      <td style="text-align:right;padding:6px 10px;font-weight:bold">7.75 ± 5.84</td>
    </tr>
  </tbody>
</table>

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

<v-click>

**SBP: beat the benchmark**  
14.07 vs 14.39 mmHg — the NLD transformer outperforms the paper's RNN on systolic prediction, despite being calibration-free and trained on a subset of the data.

</v-click>

</div>
<div>

<v-click>

**DBP: close but not there**  
7.75 vs 6.57 mmHg — diastolic error is ~18% higher than the benchmark. Diastolic BP has smaller variance, making it harder to track across unseen subjects.

</v-click>

</div>
</div>


<!-- Our initial LSTM model did not do well at all, with very large Mean Absolute Errors that were nowhere close to the original paper's results (seen soon). Our basic transformer did much better than the LSTM model, but still not quite on the same level as the paper. Here are the paper results we were looking to do better than (lower is better). 

The NLD transformer as seen here slightly outperformed the paper's RNN model at predicting systolic blood pressure (the upper number in a blood pressure measurement). This is impactful because we only trained on a subset of the data the paper used and was not calibrated at all, so it is possible we could get an even lower MAE value. However, diastolic blood pressure was not quite beating the paper's benchmark. DBP has a small variance, so it is more difficult to track in unseen/test-subset subjects. -->

---

# Loss Curve

<div class="flex justify-center mt-4">
  <img src="/nld_transformer_val_loss.png" class="rounded" style="max-height:360px;filter:invert(1) hue-rotate(180deg) brightness(0.85)" />
</div>

<!-- Here we can see that our loss curves don't show immediate signs of over/under-fitting, further giving our model legs to stand on. Negligible drops in loss started occuring around epoch 12-15, and from there on it stayed around the same value. Our model was training until 25 epochs were met with little to no change in loss value, so we can arguably say that we only needed around 12-15 epochs to train our model since anything after that was not improving.-->

---

# What the Poincaré Feature Adds

<v-click>

Ablation: Basic Transformer vs NLD Transformer

</v-click>

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

<v-click>

**SBP improvement:**  
15.35 → 14.07 mmHg  
**−1.28 mmHg (−8.4%)**

</v-click>

<v-click>

**DBP improvement:**  
9.38 → 7.75 mmHg  
**−1.63 mmHg (−17.4%)**

</v-click>

<v-click>

The Poincaré feature helps more on DBP — consistent with the physiological story: sympatho-vagal balance has a stronger connection to diastolic than systolic pressure.

</v-click>

</div>
<div>

<v-click>

**Why it works:**

The model can't easily learn HRV dynamics from raw PPG alone — the CNN downsampler discards fine temporal structure. The Poincaré plots encode this information explicitly as a structured image, processed by a dedicated CNN before fusion.

</v-click>

</div>
</div>

---

# Limitations / Next Steps

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

<v-click>

**Why we missed the DBP benchmark**

</v-click>

<v-clicks>

- The CNN downsampler compresses 15,000 samples → 500 tokens (30× reduction) using strided convolution
- This was necessary — self-attention over 15k tokens requires a 15,000 × 15,000 attention matrix, which isn't feasible
- But it's lossy: features in the ~50–100ms range get averaged out
- The **dicrotic notch** sits in exactly this range — its timing encodes vascular stiffness and pulse wave velocity, which are physiologically closer to DBP than SBP
- This is likely why SBP improved more than DBP, and why DBP still trails the benchmark

</v-clicks>

</div>
<div>

<v-click>

**Next step: PapaGei embeddings**

</v-click>

<v-clicks>

- PapaGei is a PPG foundation model pretrained on large-scale waveform data
- It produces fixed-length embeddings that capture the full morphological signal — without requiring a hand-designed lossy downsampler
- Swapping the CNN downsampler for PapaGei embeddings would give the transformer richer input, potentially recovering the dicrotic notch timing that DBP estimation depends on
- This is the most promising path to closing the DBP gap

</v-clicks>

</div>
</div>

<!-- The NLD model likely missed the paper's DBP benchmark due to how we had to compress the initial data. This data compression was a 30x reduction of sample sizes, for without the reduction our attention matrix would be far too large to feasibly compute and work with. This compression however results in features within ~50-100ms being averaged out, and the data not having a large impact.

Notably, the dicrotic notch (whose timing is very useful in finding DBP) sits in this range that gets compressed. Important timing data was thus being compressed out of our data, and explains why our model missed the paper's benchmark.

To fix this, we found a pretrained model called PaPaGei, which is trained on large-scale waveform data. It is used to capture full wavefform signals without needing to downsample the data!

We hope that by swapping the CNN downsampler with the PaPaGei pretrained model, the transformer would have more in-depth input data and could recover the dicrotic notch timing integral to DBP estimation.-->

---

# Proposed Architecture: PapaGei

<PapaGeiViz />

<!-- Click through to get diagram info -->

---

# Preliminary Results: PapaGei

<div class="flex justify-center mt-4">
  <img src="/papagei_transformer_val_loss.png" class="rounded" style="max-height:360px;filter:invert(1) hue-rotate(180deg) brightness(0.85)" />
</div>

<!--
I tried implementing the pretrained PaPaGei model into our work, but without much success to be had. As seen here, overfitting was occurring throughout training, so the results are not likely that useful. My thought to counteract this was to unfreeze some of the layers in the pretrained PaPaGei model, in case it was not working well with our data subset, but that did not seem to help much either. 

It is possible that this could be a data quantity issue, as I perosnally was having some trouble downloading all of the data. I had hoped I had enough to accomplish some training, but it is possible I just need more of the data still. 

With some more work (and data), we hope this model can still be used to improve our results.
-->

---
layout: center
class: text-center
---

# Thank You

<div class="mt-10 text-sm opacity-50">
  PulseDB · Wang et al. 2023 · Frontiers in Digital Health
</div>
