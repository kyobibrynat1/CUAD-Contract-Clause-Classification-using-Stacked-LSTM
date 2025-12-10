# Legal Contract Clause Classification using Stacked LSTM

## CCS 248 – Artificial Neural Networks Final Project

---

## Problem Statement

**Automated classification of legal contract clauses** (e.g., parties, governing law, effective/expiration dates).

---

## Why Deep Learning?
- Captures sequential context in clauses
- Handles varied phrasing across contracts
- Learns beyond keyword lookup

---

## Solution
- **Model**: 2-layer BiLSTM with attention pooling and dropout
- **Tokenization**: Custom tokenizer (10k vocab), max length 11 tokens (85th percentile of filtered snippets)
- **Balancing**: Class weights + optional sampler
- **Framework**: PyTorch

### Architecture
```
Embedding (200)
BiLSTM(128×2) → Dropout 0.25
BiLSTM(96×2)  → Attention pooling
Linear → 7 classes
```

---

## Dataset
- Source: CUAD v1 `master_clauses.csv` (flattened snippets)
- Total snippets: 1,899 after filtering
- Classes: 7 clause types (min 5 samples each)
- Vocab: ~2.6k; OOV: 0%; Max length: 11 tokens

---

## Training & Tuning
- Split: 70/15/15 stratified (train/val/test)
- Optimizers tried (batch 64, epochs 25, patience 6):
  - Adam lr ∈ {5e-4, 8e-4, 1e-3}, wd=1e-4
  - RMSprop lr=8e-4
- Schedulers: ReduceLROnPlateau (factor 0.5, patience 2)
- Gradient clipping: max norm 1.0

### Results (run2)
- Best: RMSprop lr=8e-4 → **Test accuracy ≈ 57.9%** (meets 50% target)
- Per-class: strong on Parties / Document Name / Renewal Term; weak on Agreement Date / Effective Date; moderate on Governing Law; Expiration Date captured via overlap.
- Artifacts: `artifacts_run2/` holds tokenizer, label classes, confusion matrix, classification report; models in `trained_models_run2/`; results in `experiment_results_run2.csv`.

---

## Baseline
- TF-IDF + Logistic Regression (sanity check on label quality).

---

## How to Run
1) Install deps:
```bash
pip install torch numpy pandas scikit-learn h5py
```
2) Open and run `Untitled-6.ipynb` end-to-end. It will:
   - Load `master_clauses.csv`
   - Clean/tokenize, split, and train the BiLSTM+attention
   - Save models to `trained_models_run2/`
   - Save metrics to `experiment_results_run2.csv`
   - Save artifacts to `artifacts_run2/`

---

## Files
```
ANNFINAL/
├── Untitled-6.ipynb          # Main PyTorch notebook
├── w.ipynb                   # Keras reference notebook
├── README.md
├── experiment_results_run2.csv
├── trained_models_run2/      # Saved PyTorch checkpoints (.pt, .h5)
├── artifacts_run2/           # Tokenizer, labels, confusion matrix, reports
├── trained_models/           # Prior run models
├── artifacts/               # Prior run artifacts
└── CUAD_v1/
    ├── CUAD_v1.json
    ├── CUAD_v1_README.txt
    ├── master_clauses.csv
    └── full_contract_txt/
```

---

## References
- CUAD: Hendrycks et al., 2021, arXiv:2103.06268 (https://www.atticusprojectai.org/cuad)

---

## Author
CCS 248 – Artificial Neural Networks (Dec 2025)
