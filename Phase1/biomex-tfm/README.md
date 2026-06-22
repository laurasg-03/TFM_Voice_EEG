# BIOMEX-TFM — EEG Biometrics & EEG→Voice Bridge

Master's thesis (TFM) codebase. The goal is to give a *personalised* voice to people
who cannot speak, derived from their neural activity. **Phase 1** asks whether an EEG
encodes individual traits; this repository covers Phase 1 plus the voice-processing
scaffolding for Phase 3.

- **Dataset:** BIOMEX — 51 subjects (26 M / 25 F), ages 16–61
- **Device:** Emotiv Epoc, 14 channels @ 128 Hz
- **Stack:** MNE · scikit-learn · PyTorch (EEGNet) · SpeechBrain (ECAPA-TDNN)

## Repository structure

```
biomex-tfm/
├── notebooks/        one notebook per stage, outputs kept (they are the evidence)
│   ├── 01_preprocessing.ipynb
│   ├── 02_demographics_part1.ipynb
│   ├── 03_features_by_condition.ipynb
│   ├── 04_eegnet_identity.ipynb
│   ├── 05_age_coherence.ipynb
│   ├── 06_handedness.ipynb
│   ├── 07_voice_embeddings.ipynb
│   └── 08_crossmodal.ipynb
├── src/
│   └── biomex_utils.py   reusable functions (SCAFFOLD — review before use)
├── results/          figures (.png) and tables (.csv) cited by the thesis
├── data/             raw EDF/WAV — NOT versioned (see .gitignore)
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Place the BIOMEX files under `data/` and set `BIOMEX_ROOT` in `01_preprocessing`.
The notebooks currently share state from the original (`metadata`, `feature_matrix`,
`merged`, …), so run them **in order 01 → 08** within a single session, or import from
`src/biomex_utils.py`. Making each notebook fully standalone is the next refactor.

| # | Notebook | What it does |
|---|----------|--------------|
| 01 | preprocessing | EDF loading + cleaning pipeline |
| 02 | demographics_part1 | sex / age / handedness (PSD + asymmetry) |
| 03 | features_by_condition | REC / REO / TASK feature sets |
| 04 | eegnet_identity | 32-dim subject embedding |
| 05 | age_coherence | functional connectivity + age |
| 06 | handedness | Frontal Alpha Asymmetry |
| 07 | voice_embeddings | ECAPA-TDNN (192-dim) |
| 08 | crossmodal | EEG ↔ voice bridge |

## Results (honest summary)

| Target | Result | Defensible? |
|--------|--------|-------------|
| Sex from EEG | ~0.60–0.67 balanced acc. (chance 0.50) | Yes — modest but above chance |
| Age (group level) | gamma coherence ρ = −0.39, p = 0.005 (robust) | Yes — as a group trend |
| Age (individual) | classification at chance; regression worse than dummy | No |
| Handedness | at chance everywhere | No — only 5 left-handers |
| EEG identity (EEGNet) | 73% within-session; 0% cross-condition (REC→TASK) | With strong caveats |
| Voice identity (ECAPA) | cross-session Top-1 = 100% | Yes — robust |
| EEG ↔ voice bridge | not significant (ρ ≈ −0.06 to +0.02) | Not shown in this data |

## Data

Raw EEG/voice recordings are not included (size and privacy). Drop them into `data/`.

## Roadmap

- [ ] Make notebooks runnable standalone (decouple shared state)
- [ ] Honest cross-session EEGNet test (train G10 → test G04)
- [ ] Phase 2: speech decoding from silent-speech EEG
