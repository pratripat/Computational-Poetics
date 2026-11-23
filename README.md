# Computational Poetics: Hindi Metrical Scansion & Syllable Grouping

Computational analysis and automatic metrical scansion (Laghu-Guru weight assignment and word-formation / 2-mātrā grouping) for Hindi text using classical Chandas prosody rules and Machine Learning / Deep Learning sequence models.

---

## Overview

The scansion pipeline operates in two core stages:
1. **Stage 1 — Syllabification & Weight Assignment (Laghu / Guru):**
   - Segments Devanagari words into aksharas (syllables).
   - Applies classical Sanskrit/Hindi prosodic rules (short/long matras, anusvara, chandrabindu, visarga, and saṁyoga conjunct cluster closure rules) to assign **Laghu (1)** or **Guru (2)**.
   - The rule-based algorithm achieves **~99.9%** exact weight match against the 13,000+ word corpus.
2. **Stage 2 — Word-Formation & Bracket Grouping (2-Mātrā Units):**
   - Groups syllables into bracketed metrical units (e.g. `[[1, 1], [2], [1, 1]]`).
   - Purely deterministic rules struggle with edge cases (~57% accuracy), while Machine Learning and Deep BiLSTM sequence models resolve contextual grouping boundaries with high precision (~97% accuracy).

---

## Which File to Run?

### 1. Algorithmic / Rule-Based Pipeline

If you want to run the deterministic, rule-based prosody engine and view algorithmic scansion results:

* **Evaluate full scansion & grouping on corpus:**
  ```bash
  python3 hindi_scansion_algorithm.py corpus.json
  ```
  Evaluates both Laghu/Guru weight assignment and rule-based right-to-left 2-mātrā grouping on the full corpus, printing accuracy stats and sample failure cases.

* **Quick weight-only benchmark:**
  ```bash
  python3 evaluate_on_corpus.py corpus.json
  ```
  Runs `prosody_rules.py` against `corpus.json` and reports flat weight sequence accuracy and detailed mismatch diagnostics.

* **Alternative pipeline runner:**
  ```bash
  python3 hindi_prosody_pipeline.py corpus.json
  ```
  Runs the end-to-end rule-based pipeline and prints side-by-side comparisons of gold vs. predicted brackets.

---

### 2. Machine Learning & Deep Learning Pipelines

If you want to train, evaluate, or interact with the ML models:

* **Interactive Word Tester (Notebook):**
  * Open and run [`hindi_scansion_interactive.ipynb`](hindi_scansion_interactive.ipynb).
  * Loads the trained model (`trained_model/syllable_grouping_model.joblib`) and provides an interactive prompt where you can type any Hindi word to inspect its syllables, weights, heaviness type, and predicted bracket groupings.

* **Deep Bidirectional LSTM (BiLSTM) Model (Notebook):**
  * Open and run [`syllable_grouping_bilstm.ipynb`](syllable_grouping_bilstm.ipynb).
  * Uses PyTorch to train a 2-layer BiLSTM with multi-input embeddings (syllable grapheme, prosodic weight, intrinsic vs. positional heaviness, relative position).
  * Evaluates against held-out test words and saves weights to `trained_model/bilstm_grouping_model.pt`. Includes an interactive tester cell at the bottom.

* **Baseline ML Sequence Model (Notebook):**
  * Open and run [`syllable_grouping_training.ipynb`](syllable_grouping_training.ipynb).
  * Trains a feature-based sequence labeler (context-window N-grams + Logistic Regression) on `data.csv`.

* **Regenerate Dataset for ML:**
  ```bash
  python3 prepare_data_csv.py
  ```
  Extracts features and BIO labels from `corpus.json` to generate [`data.csv`](data.csv).

---

## Model Performance Comparison

| Approach | Model / Method | Exact Word-Sequence Match |
| :--- | :--- | :--- |
| **Algorithmic** | Rule-Based Right-to-Left Pairing | ~57.3% |
| **Machine Learning** | Context-Window Logistic Regression | ~90.4% |
| **Deep Learning** | Multi-Input Deep BiLSTM (`.pt`) | **~96.9%** |

---

## Repository Structure

```text
├── corpus.json                         # 13,000+ annotated Hindi word scansion dataset
├── data.csv                            # Syllable-level feature dataset with BIO labels
├── deformed.txt                        # Log of corrupted/malformed words in raw corpus
├── Pronunciation_Consonants_Amisha.txt # Pronunciation dictionary reference
├── Text to Matra Rules.pdf             # Reference documentation on matra & prosody rules
│
├── hindi_scansion_algorithm.py         # Full rule-based scansion & grouping algorithm
├── prosody_rules.py                    # Core rule definitions for syllabification & weights
├── evaluate_on_corpus.py               # Benchmark script for evaluating prosody rules
├── prepare_data_csv.py                 # Preprocessing script to create data.csv
├── hindi_prosody_pipeline.py           # Pipeline script with detailed diagnostics
├── hindi_prosody_pipeline_compact.py   # Compact standalone version of the prosody pipeline
│
├── hindi_scansion_interactive.ipynb    # Interactive CLI/Notebook for testing words with ML
├── syllable_grouping_training.ipynb    # Training notebook for baseline ML sequence model
├── syllable_grouping_bilstm.ipynb      # Training & evaluation notebook for Deep BiLSTM
│
└── trained_model/
    ├── bilstm_grouping_model.pt        # Pretrained Deep BiLSTM PyTorch weights
    └── syllable_grouping_model.joblib  # Pretrained baseline ML model
```
