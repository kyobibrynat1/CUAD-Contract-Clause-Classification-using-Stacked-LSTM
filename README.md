# Legal Contract Clause Classification using Stacked LSTM

## CCS 248 – Artificial Neural Networks Final Project

---

## Problem Statement

**Automated Classification of Legal Contract Clauses**

Lawyers and legal professionals spend hours manually reading and categorizing contract clauses. This project automates that process using deep learning, specifically classifying legal contract clauses into predefined categories (e.g., governing law, termination, confidentiality, indemnity, etc.).

### Real-World Application
- **Contract Review Automation**: Speed up legal due diligence by automatically identifying and categorizing key contract provisions
- **Risk Analysis**: Quickly identify high-risk clauses (limitation of liability, indemnification) across large contract portfolios
- **Legal Research**: Help lawyers find precedent clauses across thousands of historical contracts

---

## Why Deep Learning?

Traditional keyword-matching or rule-based approaches fail to understand context and nuance in legal language. Deep learning, particularly LSTMs, can:
- **Understand sequential context** — Legal clauses have long-range dependencies (e.g., "subject to Section 3.2 above")
- **Capture semantic meaning** — Distinguish between similar phrases used in different legal contexts
- **Handle variations** — Recognize the same clause type written in different styles across contracts

---

## Solution: Stacked Bidirectional LSTM

### Architecture (as in notebook)
```
Input (Tokenized Clause Text)
    ↓
Embedding Layer (128 dimensions)
    ↓
Bidirectional LSTM Layer 1 (64 units × 2 directions)
    ↓
Dropout (0.30)
    ↓
Bidirectional LSTM Layer 2 (48 units × 2 directions)
    ↓
Dropout (0.30)
    ↓
Fully Connected Layer (10 classes)
    ↓
Softmax → Predicted Clause Type
```

### Why Stacked LSTM?
- **Two-layer architecture** captures both low-level patterns (legal terminology) and high-level structure (clause organization)
- **Bidirectional processing** reads clauses forward and backward to understand full context
- **Dropout regularization** prevents overfitting on legal jargon

---

## Dataset

**CUAD v1 (Contract Understanding Atticus Dataset)**
- **Source**: https://www.atticusprojectai.org/cuad
- **Size**: 510 commercial legal contracts, 13,000+ expert-labeled clauses
- **Labels**: 41 different clause types annotated by experienced attorneys
- **Quality**: Attorney-reviewed annotations, no privacy concerns (public contracts), no bias issues

### Dataset Statistics
- **Total samples**: ~13,000 labeled clause contexts
- **Top 10 clause types used**: most frequent clause types in CUAD v1 (selected dynamically in the notebook and printed for transparency)

### Data Validation
✅ **No PII (Personally Identifiable Information)** — public contracts only  
✅ **No bias concerns** — diverse contract types (M&A, licensing, employment, etc.)  
✅ **High-quality labels** — annotated by legal professionals  

---

## Methodology

### 1. Data Preprocessing
- **Text cleaning**: Lowercase, remove special characters, normalize whitespace
- **Tokenization**: Custom vocabulary (10,000 most common words)
- **Padding**: Sequences padded to the 60th percentile length and capped at 512 tokens to avoid OOM

### 2. Train/Validation/Test Split
- **Training**: 70% (~9,100 samples)
- **Validation**: 15% (~1,950 samples)
- **Test**: 15% (~1,950 samples)
- Stratified split to maintain class distribution

### 3. Hyperparameter Tuning (current notebook)

Multiple configurations tested:

| Config | Optimizer | Learning Rate | Weight Decay | Batch Size | Epochs |
|--------|-----------|---------------|--------------|------------|--------|
| 1      | Adam      | 0.0003        | 1e-4         | 32         | 12     |
| 2      | Adam      | 0.0008        | 0.0          | 32         | 15     |
| 3      | Adam      | 0.0001        | 1e-4         | 32         | 12     |
| 4      | RMSprop   | 0.0002        | 0.0          | 32         | 12     |

**Additional training features**:
- Gradient clipping (max norm = 1.0)
- ReduceLROnPlateau scheduler (factor=0.5, patience=2)
- Class weights enabled to counter imbalance
- Early stopping (patience=4) to prevent wasted epochs

### 4. Baseline Comparison
- **TF-IDF + Logistic Regression** baseline to validate label quality
- Helps determine if low performance is due to model vs. data issues

---

## Tools & Technologies

- **Deep Learning Framework**: PyTorch 2.x
- **Data Processing**: NumPy, Pandas
- **Tokenization**: Custom tokenizer (built from scratch, no pretrained embeddings)
- **Evaluation**: Scikit-learn (accuracy, precision, recall, F1, confusion matrix)
- **Hardware**: GPU acceleration (CUDA) when available

**No pretrained models used** — all training from scratch as required.

---

## Project Requirements Compliance

✅ **Specific problem identified**: Automated legal clause classification  
✅ **Deep Neural Network**: Stacked bidirectional LSTM (2 layers)  
✅ **Dataset validated**: CUAD v1, publicly available, attorney-annotated  
✅ **Optimizer tuning**: Testing Adam, RMSprop with multiple learning rates  
✅ **Documentation**: Training configs, results, and test scores recorded  
✅ **Target accuracy**: 50-60% (to be validated with baseline and model training)  
✅ **Trained from scratch**: No pretrained embeddings or transfer learning  

---

## Expected Results

- **Training Accuracy**: 60-70% (typical for legal text classification)
- **Validation Accuracy**: 50-60%
- **Test Accuracy**: 50-60% (**target for course requirement**)

### Success Metrics
- Overall accuracy ≥ 50%
- Macro F1 score (to account for class imbalance)
- Per-class recall (ensure minority classes are learned)

---

## Files in Repository

```
ANNFINAL/
├── Untitled-6.ipynb           # Main training notebook
├── README.md                   # This file
├── experiment_results.csv      # Hyperparameter tuning results
├── trained_models/             # Saved model checkpoints
│   ├── model_1.pt
│   ├── model_2.pt
│   └── model_3.pt
└── CUAD_v1/                    # Dataset folder
    ├── CUAD_v1.json
    ├── CUAD_v1_README.txt
    ├── master_clauses.csv
    └── full_contract_txt/
```

---

## How to Run

1. **Install dependencies**:
   ```bash
   pip install torch numpy pandas scikit-learn
   ```

2. **Open notebook**:
   ```bash
   jupyter notebook Untitled-6.ipynb
   ```

3. **Run all cells** to:
   - Load CUAD dataset
   - Preprocess and tokenize clauses
   - Train stacked LSTM with different optimizers
   - Evaluate on test set
   - Save results to `experiment_results.csv`

---

## References

- **CUAD Dataset**: Hendrycks, D., et al. (2021). "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review." arXiv:2103.06268
- **Dataset Source**: https://www.atticusprojectai.org/cuad

---

## Author

**Course**: CCS 248 – Artificial Neural Networks  
**Date**: December 2025  
**Objective**: Train a deep neural network from scratch to achieve 50-60% accuracy on legal clause classification
