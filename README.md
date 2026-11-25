# Computational Poetics: Hindi Metrical Scansion & Syllable Grouping

Computational analysis and automatic metrical scansion (Laghu–Guru weight assignment and bracket grouping) for Hindi text using updated classical *Chandas* prosody rules (**R1–R5**), compound/prefix boundary resolution, and Machine Learning / Deep Learning sequence models.

---

## Overview

The scansion pipeline operates in two core stages:
1. **Stage 1 — Syllabification & Weight Assignment (Laghu / Guru):**
   - Segments Devanagari words into aksharas (syllables), including independent *candra* vowels (`ऑ`, `ऍ`) and trailing nasal/visarga marks.
   - Applies classical Sanskrit/Hindi prosodic rules (short/long *mātrās*, *anusvāra*, *candrabindu*, *visarga*, *saṁyoga* conjunct cluster closure, and the single-consonant `+ ह` exemption) to assign **Laghu (`1`)** or **Guru (`2`)**.
   - Handles morpheme-level **compound and prefix boundaries (R4)** so conjuncts at the start of a second constituent do not artificially lengthen syllables across the boundary.
   - Achieves **99.69%** weight accuracy on Sir's corrected `क`-section and **99.94%** on the uncorrected consonants corpus.
2. **Stage 2 — Bracket Grouping:**
   - Groups syllables into bracketed prosodic units (e.g., `[[2, 1], [1, 1]]`).
   - With the updated **bare-`a` grouping rule (R5)** and compound splitting (**R4**), deterministic rule-based grouping reaches **95.29%** grouping accuracy (**94.99%** end-to-end exact match) on the clean corrected `क`-section and **93.70%** (**93.65%** end-to-end) on the uncorrected corpus.
   - Machine Learning and Deep BiLSTM sequence models trained on the earlier corpus (`corpus.json` / `data.csv`) also explore data-driven sequence labeling.

---

## Updated Prosody Rules (`hindi_prosody_pipeline.py`)

`hindi_prosody_pipeline.py` implements rules **R1–R5** with data hygiene and leave-one-out dictionary evaluation:

- **R1 — Long *Candra* Vowels:** `ॉ` is classified under `LONG_MATRAS`, with independent vowels `ऑ` (long) and `ऍ` (short).
- **R2 — Visarga Colon Normalization:** ASCII `:` is normalized to Devanagari visarga `ः` before syllabification.
- **R3 — Single-Consonant + `ह` Exemption:** The `म्ह`/`न्ह`/`ल्ह` non-bumping exception applies only when `lead_conjunct == 1` (doubled consonants before `ह` close the preceding syllable).
- **R4 — Compound & Prefix Splitting:** Splits words via `approved_splits`, `PREFIX_LIST`, or leave-one-out dictionary + `extra_lexicon.json` lookup (excluding `COMPOUND_BLACKLIST`), scanning constituents independently so weights reset at morpheme boundaries.
- **R5 — Bare-`a` Bracket Grouping:** Right-to-left pass where heavy (`2`) syllables, independent vowels, and vowel-marked lights stand alone, while an unmarked light (*bare-a*) joins its left neighbor.

### Rule-Based Evaluation (R1–R5, Leave-One-Out)

Run with `python hindi_prosody_pipeline.py`:

| Subset | Words (*N*) | Weight Accuracy | Grouping Accuracy *(weights matched)* | End-to-End Exact Match |
| :--- | :---: | :---: | :---: | :---: |
| **Clean `क`-section** (`sir_corrected`) | 958 | 955 / 958 (**99.69%**) | 910 / 955 (**95.29%**) | 910 / 958 (**94.99%**) |
| **Uncorrected rest** | 12,139 | 12,132 / 12,139 (**99.94%**) | 11,368 / 12,132 (**93.70%**) | 11,368 / 12,139 (**93.65%**) |

---

## Generating Swar, Vyanjan & Compound Outputs

All generated scansion files use the project's one-line JSON dictionary format, guaranteeing `len(word chunks) == len(scansion brackets)` and `"".join(word chunks) == key`:

```json
{
  "कार्यक्रम": {"word": ["कार्य", "क्रम"], "scansion": [[2, 1], [1, 1]]},
  "कुख्यात": {"word": ["कु", "ख्यात"], "scansion": [[1], [2, 1]]}
}
```

### 1. Swar & Unified Lexicon Generator
```bash
python generate_swar_outputs.py
```
Extracts, cleans, and scans words from `स्वर व्यंजन Word List.docx` into:
- `scansion_output_swar.txt` — **3,564** Swar entries sorted in Hindi dictionary (*Varṇamālā*) collation order.
- `scansion_output_swar_doc_order.txt` — **3,564** Swar entries preserved in original `.docx` order.
- `scansion_output_full_swar_vyanjan.txt` — **16,789** combined entries (**3,564** Swar + **13,225** Vyanjan from `Corrected_scansion_output_oneline_Consonants.txt`).

### 2. Compound & Prefix Generator (Swar + Vyanjan)
```bash
python generate_compound_outputs.py
```
Identifies compound and prefixed words across `scansion_output_full_swar_vyanjan.txt`, evaluates constituent boundaries independently, and writes:
- `scansion_output_compounds_required.txt` — **339** entries (**149** Swar, **190** Vyanjan) that strictly require compound generation (weight bump avoided, boundary chunk merge prevented, hyphenated, or approved `क` splits).
- `scansion_output_compounds_swar_vyanjan.txt` — **1,636** identified compound and prefixed entries scanned with constituent boundaries.
- `compound_candidates_swar_vyanjan.tsv` — Tab-separated comparison table (parts, split type, impact, whole vs. split weights/scansion, chunks, and notes).

### 3. Algorithmic & Evaluation Utilities
- `python hindi_prosody_pipeline.py` — Runs the R1–R5 pipeline on `Pronunciation_Consonants_Amisha.txt` and `Corrected_scansion_output_oneline_Consonants.txt`.
- `python generate_compound_candidates.py` — Generates `compound_candidates.tsv` for the `क`/`क़` section.
- `python compare_corrections.py` — Compares original vs. Sir-corrected consonant annotations.
- `python hindi_scansion_algorithm.py corpus.json` / `python evaluate_on_corpus.py corpus.json` — Earlier rule-based corpus evaluators.

---

## Machine Learning & Deep Learning Pipelines

If you want to train, evaluate, or interact with the ML models trained on `corpus.json` / `data.csv`:

* **Interactive Word Tester (Notebook):**
  * Open and run [`hindi_scansion_interactive.ipynb`](hindi_scansion_interactive.ipynb).
  * Loads [`trained_model/syllable_grouping_model.joblib`](trained_model/syllable_grouping_model.joblib) and provides an interactive prompt to inspect syllables, weights, heaviness type, and predicted bracket groupings.

* **Deep Bidirectional LSTM (BiLSTM) Model (Notebook):**
  * Open and run [`syllable_grouping_bilstm.ipynb`](syllable_grouping_bilstm.ipynb).
  * Trains a 2-layer PyTorch BiLSTM with multi-input embeddings and saves weights to [`trained_model/bilstm_grouping_model.pt`](trained_model/bilstm_grouping_model.pt).

* **Baseline ML Sequence Model (Notebook):**
  * Open and run [`syllable_grouping_training.ipynb`](syllable_grouping_training.ipynb).
  * Trains a feature-based sequence labeler (context-window N-grams + Logistic Regression) on [`data.csv`](data.csv).

* **Regenerate Dataset for ML:**
  ```bash
  python prepare_data_csv.py
  ```

### Earlier Benchmark Comparison (on `corpus.json` prior to R4/R5)

| Approach | Model / Method | Exact Word-Sequence Match |
| :--- | :--- | :--- |
| **Algorithmic (Legacy)** | Earlier Right-to-Left Pairing | ~57.3% |
| **Machine Learning** | Context-Window Logistic Regression | ~90.4% |
| **Deep Learning** | Multi-Input Deep BiLSTM (`.pt`) | ~96.9% |

---

## Repository Structure

```text
├── Pronunciation_Consonants_Amisha.txt              # Original consonants scansion corpus
├── Corrected_scansion_output_oneline_Consonants.txt # Consonants corpus with corrected क/क़ section
├── स्वर व्यंजन Word List.docx                       # Swar (अ–औ) source word list
├── extra_lexicon.json                               # Auxiliary morpheme entries for R4 splits
├── Text to Matra Rules.pdf                          # Reference prosody rule documentation
├── corpus.json                                      # 13,000+ annotated word dataset (ML pipeline)
├── data.csv                                         # Syllable-level feature dataset with BIO labels
├── deformed.txt                                     # Log of corrupted/malformed words in raw corpus
│
├── hindi_prosody_pipeline.py                        # Main R1–R5 rule pipeline & evaluation
├── hindi_prosody_pipeline_compact.py                # Compact standalone prosody pipeline
├── generate_swar_outputs.py                         # Swar & unified Swar+Vyanjan output generator
├── generate_compound_outputs.py                     # Swar+Vyanjan compound/prefix output generator
├── generate_compound_candidates.py                  # क-section compound candidates TSV generator
├── compare_corrections.py                           # Diff/analysis of original vs. corrected data
├── hindi_scansion_algorithm.py                      # Earlier rule-based scansion & grouping script
├── prosody_rules.py                                 # Earlier rule definitions
├── evaluate_on_corpus.py                            # Corpus weight evaluator
├── prepare_data_csv.py                              # Feature/BIO extractor for data.csv
│
├── scansion_output_swar.txt                         # 3,564 Swar words (dictionary order)
├── scansion_output_swar_doc_order.txt               # 3,564 Swar words (document order)
├── scansion_output_full_swar_vyanjan.txt            # 16,789 unified Swar + Vyanjan words
├── scansion_output_compounds_required.txt           # 339 words requiring compound generation
├── scansion_output_compounds_swar_vyanjan.txt       # 1,636 compound/prefixed words
├── compound_candidates.tsv                          # क-section compound candidate review sheet
├── compound_candidates_swar_vyanjan.tsv             # Full Swar+Vyanjan compound review sheet
│
├── hindi_scansion_interactive.ipynb                 # Interactive ML word tester notebook
├── syllable_grouping_training.ipynb                 # Baseline ML training notebook
├── syllable_grouping_bilstm.ipynb                   # Deep BiLSTM training & evaluation notebook
└── trained_model/
    ├── bilstm_grouping_model.pt                     # Pretrained Deep BiLSTM PyTorch weights
    └── syllable_grouping_model.joblib               # Pretrained baseline ML model
```
