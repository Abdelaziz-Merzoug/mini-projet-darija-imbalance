# Sentiment Analysis on Algerian Dialect (Darija) — Class Imbalance Management

> **Gestion du Desequilibre de Classes pour l'Analyse de Sentiments en Dialecte Algerien**

---

## Project Identity

| Field              | Value                                                    |
|--------------------|----------------------------------------------------------|
| **University**     | USDB Blida 1 — Departement Informatique                  |
| **Program**        | Master 1 Data Science & NLP — Semestre 2                 |
| **Module**         | Machine Learning                                         |
| **Student**        | Abdelaziz Merzoug (solo project)                         |
| **Supervisor**     | Dr. Soraya Cheriguene                                    |
| **Period**         | 08 March — 26 April 2026                                 |
| **Language**       | French (comments, analysis) / Python (code)              |

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Key Concepts — For Beginners](#2-key-concepts--for-beginners)
3. [Dataset — TWIFL Corpus](#3-dataset--twifl-corpus)
4. [Model — DziriBERT](#4-model--dziribert)
5. [The Problem: Class Imbalance](#5-the-problem-class-imbalance)
6. [The Four Strategies Tested](#6-the-four-strategies-tested)
7. [Results Summary](#7-results-summary)
8. [Project Structure](#8-project-structure)
9. [Notebooks — What Each One Does](#9-notebooks--what-each-one-does)
10. [Experimental Protocol](#10-experimental-protocol)
11. [Metrics Explained](#11-metrics-explained)
12. [Installation](#12-installation)
13. [How to Reproduce Results](#13-how-to-reproduce-results)
14. [Global Constants — Never Change These](#14-global-constants--never-change-these)
15. [Full Results Table](#15-full-results-table)
16. [Key Findings](#16-key-findings)
17. [References](#17-references)

---

## 1. What This Project Does

This project tackles a classic machine learning problem: **what happens when your training data has far more examples of some categories than others?**

We classify Algerian tweets (in "Darija" — a mix of Arabic, French, and Arabizi) into three sentiment categories:

- **Positive** (happy, praise, support)
- **Negative** (criticism, anger, complaint)
- **Neutral** (factual, ambiguous, informational)

The challenge: Positive tweets are almost **4 times more common** than Neutral tweets in our training data. A naive model simply learns to predict "Positive" most of the time and still looks accurate — but it fails badly on Neutral content.

We test **three strategies** to fix this imbalance, plus a **bonus hybrid strategy**, using the specialized Algerian Arabic language model DziriBERT.

---

## 2. Key Concepts — For Beginners

### What is "Darija"?
Darija (الدارجة) is the Algerian spoken dialect. It is a mix of:
- **Arabic script** (classical Arabic words)
- **French words** (colonial heritage, education system)
- **Arabizi** (Arabic written in Latin letters, e.g. "chokran" instead of "شكرا")

This makes standard Arabic NLP tools fail badly — they were never trained on this mix.

### What is a "Pre-trained Language Model"?
Think of it like an expert who already read millions of texts. We "fine-tune" (teach) this expert on our specific task (sentiment analysis) rather than training from zero. DziriBERT is an expert who specifically read Algerian texts.

### What is "Class Imbalance"?
Imagine training a spam filter where 95% of emails are normal and only 5% are spam. A model that labels everything as "not spam" achieves 95% accuracy but catches zero spam. This is the trap. Our Neutral class has the same problem.

### What is "F1-macro"?
Instead of accuracy (which rewards predicting the majority class), F1-macro averages the F1 score of each class equally. A model that only predicts "Positive" would get F1-macro ≈ 0.26, not 0.52. This is our primary metric.

### What is "Back-Translation"?
Take a tweet in Darija → translate to French → translate back to Arabic. You get a new version of the same tweet with different wording but the same meaning. This gives us more training examples for the rare Neutral class.

---

## 3. Dataset — TWIFL Corpus

| Property               | Value                                                        |
|------------------------|--------------------------------------------------------------|
| **Name**               | TWIFL (Twitter Algerian Corpus)                              |
| **Source**             | https://huggingface.co/datasets/arbml/Twifil                 |
| **Total tweets**       | 6,000                                                        |
| **Text column**        | `Post`                                                       |
| **Label column**       | `Polarity Class`                                             |
| **Language column**    | `lang`                                                       |
| **Classes**            | Positive / Negative / Neutral                                |
| **Raw distribution**   | Positive: 2,864 — Negative: 1,773 — Neutral: 1,363          |
| **Raw imbalance**      | 2.10:1 (Positive vs Neutral)                                 |
| **After preprocessing**| 4,916 tweets remain (1,075 duplicates + 9 empty removed)     |
| **Post-prep imbalance**| **3.93:1** (worsened — Neutral lost 52.6% of its examples)  |
| **Emoji rate**         | 21.6% of tweets contain at least one emoji                   |
| **Code-switching**     | 38.1% mix Arabic and French/Latin characters                 |
| **Language breakdown** | ar=2,462 / und=1,244 / fr=1,110 / en=733 (30 unique codes)  |

### Why did preprocessing worsen the imbalance?
Neutral tweets tend to be shorter and contain more noise (empty content, symbols). After removing duplicates and near-empty tweets, Neutral lost 717 examples (52.6%) while Positive only lost 323 (11.3%). This is an important structural finding documented explicitly in the project.

### Data split (created ONCE, never regenerated)

| Set        | Size  | Positive | Negative | Neutral |
|------------|-------|----------|----------|---------|
| **Train**  | 3,440 | 1,778    | 1,210    | 452     |
| **Val**    | 738   | 382      | 259      | 97      |
| **Test**   | 738   | 381      | 260      | 97      |

Split method: **stratified 70/15/15** using a two-step procedure (see [Section 10](#10-experimental-protocol)).
The split indices are saved in `data/split_indices.json` and loaded by all notebooks. **Never delete this file.**

---

## 4. Model — DziriBERT

**DziriBERT** (`alger-ia/dziribert`) is the only publicly available language model pre-trained specifically on the Algerian dialect.

| Property              | Value                                      |
|-----------------------|--------------------------------------------|
| **Architecture**      | Bidirectional BERT encoder                 |
| **[CLS] dimension**   | 768                                        |
| **Vocabulary size**   | 50,000 tokens (verified from tokenizer)    |
| **Pre-training data** | Algerian tweets (Arabic, Arabizi, mixed)   |
| **Parameters**        | ~124 million                               |
| **HuggingFace ID**    | `alger-ia/dziribert`                       |

### Two usage modes in this project

```python
# Mode 1 — Fine-tuning for classification (NB02, NB03, NB05)
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    'alger-ia/dziribert', num_labels=3
)

# Mode 2 — Feature extraction / embedding generation (NB04)
from transformers import AutoModel
model = AutoModel.from_pretrained('alger-ia/dziribert')
# Returns 768-dimensional [CLS] vectors — no classification head
```

> **Critical rule:** DziriBERT must be reloaded from `from_pretrained` before every new experiment. Never fine-tune on top of a previously fine-tuned model — this would make comparisons unfair.

---

## 5. The Problem: Class Imbalance

After preprocessing, the training set has this distribution:

```
Positive  ████████████████████████████████████████  1,778  (51.7%)
Negative  ████████████████████████                  1,210  (35.2%)
Neutral   ██████                                      452  (13.1%)
```

**Imbalance ratio: 3.93:1** (Positive vs Neutral)

A naive model learns to mostly predict "Positive" because:
- It gets rewarded 52% of the time just for guessing "Positive"
- Accuracy looks fine (0.72) but F1-Neutral collapses (0.56)

All strategies in this project are designed to make the model pay more attention to the underrepresented Neutral class.

---

## 6. The Four Strategies Tested

### Strategy 0 — Baseline (NB02)
Fine-tune DziriBERT on the raw imbalanced data. No correction whatsoever.
This is the reference score that all other strategies must beat.

---

### Strategy 1 — Loss Function Modification (NB03)
Instead of treating all prediction errors equally, we modify the loss function to penalize errors on minority classes more heavily.

**Variant A — Class Weighting:**
```
weight = total_examples / (num_classes × examples_per_class)
```
Results in: Positive=0.64, Negative=0.95, Neutral=2.54
The model is penalized 2.54× more for missing a Neutral tweet.

**Variant B1 — Focal Loss γ=1:**
Reduces the contribution of easy examples to the loss. Well-classified examples barely affect training; the model focuses on hard (minority) cases.

**Variant B2 — Focal Loss γ=2:**
Same idea with stronger focus. γ=2 means examples predicted with >90% confidence contribute near zero to the loss.

**Variant C — Class Weighting + Focal Loss γ=2:**
Combines both corrections (investigated as a bonus — see results for the "double correction" problem).

---

### Strategy 2 — Embedding-Space Rebalancing (NB04)
Rather than working with raw text, we extract the 768-dimensional [CLS] vectors from DziriBERT for every training tweet. We then create synthetic new vectors in this mathematical space.

**Why embedding space?**
Darija with code-switching and Arabizi makes coherent text interpolation impossible. In 768-dimensional continuous space, interpolating between two same-class vectors produces a semantically valid new representation.

**Variant 2A — SMOTE Full Balance:**
Generate synthetic Neutral and Negative vectors until all three classes have equal counts (1,778 each).

**Variant 2B — SMOTE Partial Balance:**
More conservative: multiply Neutral by 1.5 (452 → 678) and cap Negative at 1,778. Avoids over-correction.

**Variant 2C — ADASYN:**
Adaptive method: generates more synthetic points near decision boundaries (where the model is confused) and fewer in easy zones.

A light MLP (Multi-Layer Perceptron) classifier is trained on the rebalanced embeddings. DziriBERT itself is not fine-tuned again.

---

### Strategy 3 — Back-Translation Data Augmentation (NB05)
Generates new Neutral training tweets using a two-step translation pipeline:

```
Darija tweet (Arabic script)
        ↓  Helsinki-NLP/opus-mt-ar-fr
French intermediate translation
        ↓  Helsinki-NLP/opus-mt-fr-ar
New Arabic paraphrase (MSA-flavored)
```

**Quality filter:** Only paraphrases with cosine similarity between **0.50 and 0.85** (measured by DziriBERT embeddings) are kept:
- Below 0.50 → meaning changed too much (possible sentiment flip) → rejected
- Above 0.85 → too similar to the original → no diversity added → rejected
- Between 0.50 and 0.85 → ideal balance of diversity and faithfulness → kept

**Result:** 214 valid paraphrases from 452 original Neutral tweets (47.3% survival rate).

**Important limitation — MSA Drift:**
Helsinki-NLP was trained on Modern Standard Arabic (UN texts, news, subtitles). Algerian Darija is almost absent from its training data. The paraphrases are grammatically correct Arabic but sound "formal" — like a news anchor, not an Algerian on Twitter. This drift is documented and analyzed in the project.

**Augmentation rates tested:** +10%, +20%, +50%, +75%, +100% of the Neutral class.

---

### Bonus — Hybrid Strategy (NB06)

**Option A — SMOTE partial + Weighted MLP:**
Combines Strategy 2 rebalancing with class-weighted loss in the MLP. Result: negative — demonstrates "double correction interference".

**Option B — 5-point BT ratio curve:**
Maps F1-macro as a function of augmentation rate across all 5 ratios to find the optimal point empirically.

---

## 7. Results Summary

**Best configuration: Back-Translation +20% — F1-macro = 0.6909 (+0.0104 vs Baseline)**

The only strategy to surpass the Baseline among all 11 tested configurations.

| Rank | Configuration         | F1-macro | F1-Neutral | vs Baseline |
|------|-----------------------|----------|------------|-------------|
| 1    | **BT +20%**           | **0.6909** | **0.5833** | **+0.0104** |
| 2    | BT +10% (bonus)       | 0.6884   | 0.5814     | +0.0078     |
| 3    | **Baseline**          | **0.6805** | **0.5596** | **—**       |
| 4    | Focal Loss (γ=2)      | 0.6794   | 0.5543     | -0.0011     |
| 5    | Focal Loss (γ=1)      | 0.6784   | 0.5525     | -0.0021     |
| 6    | BT +50/75/100%        | 0.6715   | 0.5376     | -0.0091     |
| 7    | CW + Focal Loss       | 0.6683   | 0.5446     | -0.0123     |
| 8    | ADASYN                | 0.6617   | 0.5495     | -0.0188     |
| 9    | SMOTE Partial         | 0.6470   | 0.5275     | -0.0335     |
| 10   | Class Weighting       | 0.6386   | 0.5134     | -0.0419     |
| 11   | SMOTE Full            | 0.6355   | 0.5000     | -0.0450     |
| 12   | Bonus A (SMOTE+CW)    | 0.6197   | 0.4876     | -0.0608     |

> Note: BT +50%, +75%, +100% all produce **identical results** because the paraphrase pool is exhausted at 214 samples for any rate ≥ 50%.

---

## 8. Project Structure

```
mini_projet_darija/
│
├── data/
│   ├── split_indices.json          ← CRITICAL: 70/15/15 split indices — DO NOT DELETE
│   ├── bt_paraphrases_neutral.json ← 214 back-translated paraphrases (cosine-filtered)
│   └── bt_annotated_examples.json  ← 3 annotated examples with cosine scores
│
├── models/
│   └── baseline_dziribert/         ← Fine-tuned baseline model (also on HuggingFace Hub)
│
├── results/
│   │
│   ├── ── Strategy metrics (14 JSON files) ──
│   ├── baseline_metrics.json       ← NB02: Baseline reference
│   ├── cw_metrics.json             ← NB03: Class Weighting
│   ├── fl_g1_metrics.json          ← NB03: Focal Loss γ=1
│   ├── fl_g2_metrics.json          ← NB03: Focal Loss γ=2
│   ├── cw_fl_metrics.json          ← NB03: CW + Focal Loss γ=2
│   ├── smote_full_metrics.json     ← NB04: SMOTE full balance
│   ├── smote_partial_metrics.json  ← NB04: SMOTE partial balance
│   ├── adasyn_metrics.json         ← NB04: ADASYN adaptive sampling
│   ├── bt_20pct_metrics.json       ← NB05: Back-Translation +20%
│   ├── bt_50pct_metrics.json       ← NB05: Back-Translation +50%
│   ├── bt_100pct_metrics.json      ← NB05: Back-Translation +100%
│   ├── bt_10pct_metrics.json       ← NB05 BONUS: BT +10%
│   ├── bt_75pct_metrics.json       ← NB05 BONUS: BT +75%
│   ├── bt_5ratios_summary.json     ← NB05 BONUS: 5-point BT curve summary
│   ├── bonus_smote_cw_metrics.json ← NB06 BONUS: SMOTE partial + Weighted MLP
│   │
│   ├── ── Comparison tables (CSV) ──
│   ├── evaluation_finale_comparatif.csv    ← Main 11-configuration table
│   ├── evaluation_finale_avec_bonus.csv    ← Extended 14-configuration table (with bonus)
│   ├── strategie1_comparatif.csv           ← Strategy 1 comparison
│   ├── strategie2_comparatif.csv           ← Strategy 2 comparison
│   └── strategie3_comparatif.csv           ← Strategy 3 comparison
│
├── figures/  (47 PNG files — generated by notebooks)
│   │
│   ├── ── EDA figures (NB01) ──
│   ├── class_distribution_bar.png
│   ├── class_distribution_pie.png
│   ├── lang_distribution.png
│   ├── tweet_length_chars_histogram.png
│   ├── tweet_length_words_boxplot.png
│   ├── wordcloud_positive.png
│   ├── wordcloud_negative.png
│   ├── wordcloud_neutral.png
│   └── emoji_proportion.png
│   │
│   ├── ── Baseline figures (NB02) ──
│   ├── preprocessing_impact.png
│   ├── baseline_training_curves.png
│   └── baseline_confusion_matrix.png
│   │
│   ├── ── Strategy 1 figures (NB03) ──
│   ├── cw_confusion_matrix.png
│   ├── fl_g1_confusion_matrix.png
│   ├── fl_g2_confusion_matrix.png
│   ├── cw_fl_confusion_matrix.png
│   ├── strategie1_f1_macro_comparison.png
│   ├── strategie1_f1_per_class.png
│   ├── strategie1_cm_comparison.png
│   └── strategie1_auc_pr_per_class.png
│   │
│   ├── ── Strategy 2 figures (NB04) ──
│   ├── tsne_before_rebalancing.png
│   ├── tsne_after_smote_full.png
│   ├── tsne_after_smote_partial.png
│   ├── tsne_after_adasyn.png
│   ├── smote_full_confusion_matrix.png
│   ├── smote_partial_confusion_matrix.png
│   ├── adasyn_confusion_matrix.png
│   ├── strategie2_f1_macro_comparison.png
│   └── strategie2_f1_per_class.png
│   │
│   ├── ── Strategy 3 figures (NB05) ──
│   ├── bt_cosine_distribution.png
│   ├── bt_10pct_confusion_matrix.png
│   ├── bt_20pct_confusion_matrix.png
│   ├── bt_50pct_confusion_matrix.png
│   ├── bt_75pct_confusion_matrix.png
│   ├── bt_100pct_confusion_matrix.png
│   ├── strategie3_f1_macro_comparison.png
│   ├── strategie3_f1_per_class.png
│   ├── bonus_bt_ratio_curve.png
│   ├── bonus_smote_cw_confusion_matrix.png
│   └── bonus_bt_analysis.png
│   │
│   └── ── Final evaluation figures (NB06) ──
│       ├── bonus_final_comparison.png
│       ├── finale_f1_macro_all.png
│       ├── finale_f1_per_class_heatmap.png
│       ├── finale_radar_top3.png
│       ├── finale_strategy_comparison.png
│       ├── finale_neutral_focus.png
│       └── finale_accuracy_vs_f1macro.png
│
├── 01_EDA.ipynb                         ← Exploratory data analysis
├── 02_Preprocessing_Baseline.ipynb      ← Data cleaning + baseline model
├── 03_Strategie1_Loss_Functions.ipynb   ← Loss function modifications
├── 04_Strategie2_SMOTE_ADASYN.ipynb     ← Embedding-space rebalancing
├── 05_Strategie3_BackTranslation.ipynb  ← Back-translation augmentation
├── 06_Evaluation_Finale.ipynb           ← Final comparison + bonus strategies
├── requirements.txt                     ← Python dependencies
└── README.md                            ← This file
```

---

## 9. Notebooks — What Each One Does

### NB01 — Exploratory Data Analysis (`01_EDA.ipynb`)
**Platform:** Google Colab CPU (no GPU needed) | **Duration:** ~10 minutes

What it does:
- Plots the class distribution (bar chart + pie chart)
- Analyzes which languages (`lang` column) appear in the corpus
- Shows tweet length distributions (character histogram, word count boxplot)
- Measures emoji rate and code-switching rate
- Generates Arabic word clouds per class (using `arabic_reshaper` + `python-bidi` for correct RTL rendering)
- Saves a statistics table as CSV

Outputs: 9 PNG figures + `results/eda_statistics_table.csv`

---

### NB02 — Preprocessing & Baseline (`02_Preprocessing_Baseline.ipynb`)
**Platform:** Google Colab T4 GPU | **Duration:** ~60 minutes

What it does:
- Removes URLs, @mentions, normalizes repeated characters, converts Eastern Arabic numerals (٠١٢ → 012)
- Converts emojis to text descriptions (so DziriBERT can read them)
- Removes 1,075 duplicate tweets and 9 near-empty tweets
- Creates the **stratified 70/15/15 split** and saves indices to `data/split_indices.json`
- Fine-tunes DziriBERT on the raw (imbalanced) training set — this is the Baseline
- Saves the model to HuggingFace Hub (private repository)

Outputs: `data/split_indices.json` + `results/baseline_metrics.json` + confusion matrix + training curve

**Baseline results:**

| Metric      | Value  |
|-------------|--------|
| F1-macro    | 0.6805 |
| F1-Positive | 0.7806 |
| F1-Negative | 0.7014 |
| F1-Neutral  | 0.5596 |
| AUC-PR      | 0.7534 |
| G-mean      | 0.7558 |
| Accuracy    | 0.7249 |

---

### NB03 — Strategy 1: Loss Functions (`03_Strategie1_Loss_Functions.ipynb`)
**Platform:** Google Colab T4 GPU | **Duration:** ~3 hours (4 independent runs)

What it does:
- Implements `WeightedLossTrainer` (custom HuggingFace Trainer with class-weighted CrossEntropyLoss)
- Implements `FocalLoss` with configurable γ parameter
- Runs 4 completely separate fine-tuning experiments (DziriBERT reloaded from scratch before each)
- Generates comparison charts and confusion matrices

**Important:** DziriBERT is reloaded via `from_pretrained('alger-ia/dziribert')` at the start of each variant. This is verified by the "MISSING/UNEXPECTED keys" load report printed each time.

Outputs: 4 JSON metric files + 4 confusion matrices + 4 comparison figures

---

### NB04 — Strategy 2: Embedding Rebalancing (`04_Strategie2_SMOTE_ADASYN.ipynb`)
**Platform:** Kaggle P100 (embedding extraction ~20 min) | **Duration:** ~1.5 hours

What it does:
- Loads DziriBERT in feature extraction mode (`AutoModel`, no classification head)
- Extracts 768-dimensional [CLS] vectors for **train set only** (val and test are never rebalanced)
- Saves embeddings immediately to Drive: `data/train_embeddings.npy`
- Applies SMOTE/ADASYN to the embedding matrix
- Trains a PyTorch MLP on the rebalanced embeddings
- Generates t-SNE visualizations (2D projections) before and after rebalancing

Outputs: 3 JSON metric files + 4 t-SNE figures + 3 confusion matrices

---

### NB05 — Strategy 3: Back-Translation (`05_Strategie3_BackTranslation.ipynb`)
**Platform:** Google Colab T4 GPU | **Duration:** ~3 hours (5 fine-tuning runs)

What it does:
- Translates all 452 Neutral training tweets: Arabic → French (OPUS-MT ar-fr) → Arabic (OPUS-MT fr-ar)
- Wraps every translation call in `try/except` (Arabizi tweets crash the translator without protection)
- Filters paraphrases by cosine similarity [0.50, 0.85] → retains 214 valid paraphrases
- Saves 3 annotated examples with French intermediate + cosine score
- Builds augmented training sets for 5 rates: +10%, +20%, +50%, +75%, +100%
- Fine-tunes DziriBERT from scratch on each augmented set
- Documents pool exhaustion: rates ≥ +50% all use the same 214 paraphrases

Outputs: 5 JSON metric files + `bt_paraphrases_neutral.json` + `bt_5ratios_summary.json` + 5 confusion matrices + cosine distribution figure

---

### NB06 — Final Evaluation (`06_Evaluation_Finale.ipynb`)
**Platform:** Any GPU (or CPU — no fine-tuning in this notebook) | **Duration:** ~30 minutes

What it does:
- Loads all 11 pre-computed metric JSON files
- Builds and saves the 11-configuration comparative table
- Runs **Bonus Option A**: SMOTE partial + Weighted MLP (hybrid strategy)
- Runs **Bonus Option B**: plots F1-macro vs BT ratio curve across all 5 ratios
- Generates 6 final synthesis figures (radar plot, heatmap, strategy comparison, etc.)
- Verifies that BT+50% = BT+100% (pool exhaustion consistency check)

Outputs: `evaluation_finale_comparatif.csv` + `evaluation_finale_avec_bonus.csv` + 6 synthesis figures

---

## 10. Experimental Protocol

### Fixed Hyperparameters (locked for all experiments)

| Parameter      | Value          | Why locked                                          |
|----------------|----------------|-----------------------------------------------------|
| `epochs`       | 5              | Consistent training budget across all variants      |
| `learning_rate`| 2e-5           | Standard BERT fine-tuning rate                      |
| `batch_size`   | 16             | Fits T4 VRAM (15.6 GB) for 128-token sequences      |
| `optimizer`    | AdamW          | `optim='adamw_torch'` in TrainingArguments          |
| `seed`         | 42             | Reproducibility — applied everywhere                |

Seeds are set at **all levels**: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`, `os.environ['PYTHONHASHSEED']='42'`, `torch.backends.cudnn.deterministic=True`.

### Correct 70/15/15 Split Procedure

**The wrong way** (produces 72.25/12.75/15, not 70/15/15):
```python
# WRONG — do not do this
X_train, X_test = train_test_split(X, test_size=0.15)   # → 85/15
X_train, X_val  = train_test_split(X_train, test_size=0.15)  # → 72.25/12.75/15
```

**The correct way** (used in NB02):
```python
# STEP 1: Carve out exactly 15% as test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# STEP 2: From the remaining 85%, take 17.65% as validation
# Calculation: 15% of total = 15/85 = 17.647...% of the 85% remainder
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
)
```

This gives exact proportions: **Train=3440 (70.0%) / Val=738 (15.0%) / Test=738 (15.0%)**.

### Immutable Rules

| Rule | Reason |
|------|--------|
| Test set is NEVER modified, filtered, or augmented | It must reflect real-world distribution |
| Validation set is NEVER rebalanced | Only for early stopping and hyperparameter selection |
| DziriBERT reloaded from scratch before each new experiment | Guarantees fair comparison |
| `split_indices.json` loaded (never regenerated) in NB03–NB06 | Guarantees all experiments use identical test samples |
| French words are NEVER removed from text | Code-switching is a core Darija linguistic feature, not noise |
| No stemming or lemmatization | DziriBERT's tokenizer handles this internally |

---

## 11. Metrics Explained

All six metrics below are computed for **every single experiment** using a shared `evaluate_model()` function defined once in NB02 and copied to all subsequent notebooks.

| Metric | What it measures | Why we use it |
|--------|-----------------|---------------|
| **F1-macro** | Average F1 across all 3 classes (unweighted) | PRIMARY metric — not fooled by imbalance |
| **F1 per class** | F1 for Positive, Negative, Neutral separately | Diagnoses which class benefits or suffers |
| **Precision per class** | Of all "Neutral" predictions, how many are correct? | Measures false alarm rate |
| **Recall per class** | Of all true Neutral tweets, how many did we find? | Measures missed detection rate |
| **AUC-PR macro** | Area under the Precision-Recall curve, averaged across classes | Better than AUC-ROC on imbalanced data |
| **G-mean** | Geometric mean of per-class recalls | Penalizes any class being completely ignored |
| **Accuracy** | % of all predictions correct | Computed but **NOT used as primary metric** — misleading on imbalanced data |

**Why accuracy is misleading:**
A model that always predicts "Positive" on our test set (381 Positive, 260 Negative, 97 Neutral = 738 total) would achieve Accuracy = 381/738 = **51.6%** but F1-macro ≈ **0.22**. Accuracy rewards majority-class bias.

**AUC-PR implementation:**
```python
from sklearn.metrics import average_precision_score
auc_pr = average_precision_score(y_true_binarized, y_proba, average='macro')
# Uses softmax probability outputs, NOT binary predictions
```

**G-mean implementation:**
```python
from imblearn.metrics import geometric_mean_score
g_mean = geometric_mean_score(y_true, y_pred, average='macro')
```

---

## 12. Installation

### Requirements

- Python >= 3.10
- GPU strongly recommended (CUDA) for NB02–NB06 — CPU will work but be very slow
- Google Drive account for artifact persistence across Colab sessions
- HuggingFace account (token needed for `push_to_hub` in NB02 only)

### Local installation

```bash
git clone <your-repo-url>
cd mini_projet_darija
pip install -r requirements.txt
```

### On Google Colab / Kaggle

```python
!pip install -r requirements.txt
```

### GPU-specific PyTorch (recommended)

The `requirements.txt` lists `torch>=2.1.0` (CPU generic). For GPU:

```bash
# For CUDA 11.8
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

On Colab, PyTorch with GPU support is **already pre-installed** — no manual step needed.

### HuggingFace Token Setup

Always use Colab Secrets — **never hardcode a token**:

1. Go to https://huggingface.co/settings/tokens → create a token
2. In Colab: click the key icon (🔑) in the left panel → add secret named `HF_TOKEN`
3. In your notebook:

```python
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')

from huggingface_hub import login
login(token=HF_TOKEN)
```

---

## 13. How to Reproduce Results

### Step 0 — One-time Google Drive setup

```python
from google.colab import drive
drive.mount('/content/drive')

import os
BASE = '/content/drive/MyDrive/mini_projet_darija'
for folder in ['data', 'models', 'results', 'figures', 'logs']:
    os.makedirs(f'{BASE}/{folder}', exist_ok=True)
print("Drive folders created.")
```

### Step 1 — Run notebooks in this exact order

| # | Notebook | Platform | GPU | Est. Time | Prerequisite |
|---|----------|----------|-----|-----------|--------------|
| 1 | `01_EDA.ipynb` | Colab CPU | None | 10 min | None |
| 2 | `02_Preprocessing_Baseline.ipynb` | Colab T4 | T4 | 60 min | None |
| 3 | `03_Strategie1_Loss_Functions.ipynb` | Colab T4 | T4 | 3 h | `data/split_indices.json` |
| 4 | `04_Strategie2_SMOTE_ADASYN.ipynb` | Kaggle P100 | P100 | 1.5 h | `data/split_indices.json` |
| 5 | `05_Strategie3_BackTranslation.ipynb` | Colab T4 | T4 | 3 h | `data/split_indices.json` |
| 6 | `06_Evaluation_Finale.ipynb` | Any GPU | Any | 30 min | All 11+ metric JSON files |

### Step 2 — The critical artifact

`data/split_indices.json` is generated **once** by NB02 and contains the exact indices of the train/val/test samples. All subsequent notebooks load this file to use identical data splits. **Deleting or regenerating this file invalidates all comparisons.**

### Step 3 — Intermediate Drive saves

Each notebook saves intermediate results to Google Drive after every ~45-minute fine-tuning run. If Colab disconnects, you can resume without losing progress by loading from Drive at the start of the next session.

---

## 14. Global Constants — Never Change These

These constants are defined at the top of **every notebook** and must remain identical across all 6 notebooks:

```python
# Column names — verified against the real TWIFL dataset
TEXT_COL   = 'Post'           # The tweet text column
LABEL_COL  = 'Polarity Class' # The sentiment label column
LANG_COL   = 'lang'           # The language detection column

# Reproducibility
SEED = 42                     # Used EVERYWHERE — splits, SMOTE, ADASYN, PyTorch

# Label encoding (class index mapping)
LABEL_MAP   = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
LABEL_NAMES = ['Positive', 'Negative', 'Neutral']

# Model
MODEL_NAME = 'alger-ia/dziribert'

# Locked hyperparameters
EPOCHS     = 5
LR         = 2e-5
BATCH_SIZE = 16

# Drive path (Colab)
BASE = '/content/drive/MyDrive/mini_projet_darija'
```

---

## 15. Full Results Table

### Main 11-configuration comparison (primary metric: F1-macro)

| Configuration         | F1-macro | F1-Pos | F1-Neg | F1-Neu | AUC-PR | G-mean | Accuracy | vs Baseline |
|-----------------------|----------|--------|--------|--------|--------|--------|----------|-------------|
| Baseline              | 0.6805   | 0.7806 | 0.7014 | 0.5596 | 0.7534 | 0.7558 | 0.7249   | —           |
| Class Weighting       | 0.6386   | 0.7141 | 0.6882 | 0.5134 | 0.7571 | 0.7510 | 0.6694   | -0.0419     |
| Focal Loss (γ=1)      | 0.6784   | 0.7614 | 0.7213 | 0.5525 | 0.7687 | 0.7540 | 0.7209   | -0.0021     |
| Focal Loss (γ=2)      | 0.6794   | 0.7605 | 0.7233 | 0.5543 | 0.7647 | 0.7564 | 0.7209   | -0.0011     |
| CW + Focal Loss       | 0.6683   | 0.7533 | 0.7068 | 0.5446 | 0.7460 | 0.7576 | 0.7060   | -0.0122     |
| SMOTE Full Balance    | 0.6355   | 0.7334 | 0.6730 | 0.5000 | 0.6888 | 0.7186 | 0.6829   | -0.0450     |
| SMOTE Partial Balance | 0.6470   | 0.7336 | 0.6801 | 0.5275 | 0.6943 | 0.7282 | 0.6883   | -0.0335     |
| ADASYN                | 0.6617   | 0.7599 | 0.6757 | 0.5495 | 0.7085 | 0.7367 | 0.7046   | -0.0188     |
| **BT +20%**           |**0.6909**| 0.7786 | 0.7108 |**0.5833**|0.7652|**0.7618**| 0.7304 |**+0.0104**|
| BT +50%               | 0.6715   | 0.7752 | 0.7016 | 0.5376 | 0.7663 | 0.7484 | 0.7195   | -0.0090     |
| BT +100%              | 0.6715   | 0.7752 | 0.7016 | 0.5376 | 0.7663 | 0.7484 | 0.7195   | -0.0090     |

### Bonus — BT 5-ratio curve

| BT Rate | F1-macro | F1-Neutral | Paraphrases Used | Pool Status |
|---------|----------|------------|------------------|-------------|
| +10%    | 0.6884   | 0.5814     | 45               | OK          |
| +20%    | **0.6909** | **0.5833** | 90             | OK          |
| +50%    | 0.6715   | 0.5376     | 214              | **EXHAUSTED** |
| +75%    | 0.6715   | 0.5376     | 214              | **EXHAUSTED** |
| +100%   | 0.6715   | 0.5376     | 214              | **EXHAUSTED** |

The pool caps at 214 because only 47.3% of the 452 Neutral tweets produce paraphrases passing the cosine filter [0.50, 0.85]. Rates ≥ +50% all use the same 214 paraphrases and produce identical results.

### Bonus — Hybrid strategy results (Option A)

| Metric    | Value  | vs Baseline |
|-----------|--------|-------------|
| F1-macro  | 0.6197 | -0.0608     |
| AUC-PR    | 0.7311 | -0.0223     |
| G-mean    | 0.7314 | -0.0244     |

**Conclusion:** Combining SMOTE (already a correction) with class-weighted MLP (a second correction) causes "double correction interference." The model over-corrects toward Neutral, hurting Positive precision (-0.14) without recovering enough Neutral F1. This negative result is scientifically valid — it illustrates that stacking two imbalance corrections on a moderate 3.93:1 ratio amplifies rather than cancels the problem.

---

## 16. Key Findings

### 1. Back-Translation +20% is the only winning strategy
A gain of +0.0104 on F1-macro (0.6909 vs 0.6805). Providing new textual examples with diverse vocabulary allows DziriBERT to update its internal representations — something loss-function tricks and frozen-embedding interpolation cannot achieve.

### 2. Class Weighting degrades on moderate imbalance
The 3.93:1 post-preprocessing ratio produces aggressive weights (Neutral × 2.54). The model over-predicts Neutral: recall rises from 0.557 to 0.691 but precision crashes from 0.563 to 0.409, resulting in *worse* F1-Neutral (0.513 vs 0.560 baseline). This is the over-correction trap.

### 3. SMOTE/ADASYN on frozen embeddings cannot beat full fine-tuning
All three embedding-space variants score below baseline. Interpolating in R⁷⁶⁸ creates vectors that DziriBERT never produced — the transformer never updates its representations. ADASYN scores best (0.6617) by concentrating generation near decision boundaries, but the frozen-encoder bottleneck limits its ceiling.

### 4. BT pool exhaustion is a structural finding, not a bug
The cosine filter [0.50, 0.85] rejects 53% of candidate paraphrases: below threshold = meaning changed (sentiment flip risk), above threshold = too similar to original (no diversity added). The 214-sample cap reflects a real linguistic limit: Darija Neutral tweets are resistant to back-translation via MSA-trained OPUS-MT.

### 5. Increasing augmentation beyond +20% hurts performance
At +20%: 90 MSA paraphrases represent 2.5% of the training set — a tolerable signal. At +50%: 214 MSA paraphrases represent 5.9% — enough to bias DziriBERT toward formal Arabic, creating a domain mismatch at test time (100% Darija tweets).

### 6. Accuracy is a misleading metric on this corpus
The gap between accuracy and F1-macro ranges from +0.04 (BT+20%) to +0.05 (SMOTE Full). SMOTE Full reports Accuracy=0.683 but F1-macro=0.635 — a 4.8-point gap that hides the model's failure on Neutral (F1-Neutral=0.500 exactly).

---

## 17. References

```
[1]  A. Dossou et al., "DziriBERT: a Pre-trained Language Model for the
     Algerian Dialect," AfricaNLP, 2022.

[2]  Y. Boutaleb et al., "TWIFL: An Algerian Corpus and Annotation Platform,"
     LREC-COLING, 2024.

[3]  N. V. Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique,"
     Journal of Artificial Intelligence Research, vol. 16, pp. 321–357, 2002.

[4]  T. Y. Lin et al., "Focal Loss for Dense Object Detection,"
     IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2980–2988.

[5]  H. He et al., "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced
     Learning," IEEE International Joint Conference on Neural Networks (IJCNN),
     2008, pp. 1322–1328.

[6]  S. Edunov et al., "Understanding Back-Translation at Scale,"
     Proceedings of EMNLP, 2018, pp. 489–500.

[7]  J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers
     for Language Understanding," Proceedings of NAACL-HLT, 2019.

[8]  G. Haixiang et al., "Learning from class-imbalanced data: Review of methods
     and applications," Expert Systems with Applications, vol. 73, 2017.

[9]  J. Tiedemann and S. Thottingal, "OPUS-MT — Building open translation
     services for the World," Proceedings of EAMT, 2020.

[10] J. Davis and M. Goadrich, "The relationship between Precision-Recall and
     ROC curves," Proceedings of ICML, 2006.
```

---

*Project completed as part of the Machine Learning module — Master 1 Data Science & NLP,
USDB Blida 1, under the supervision of Dr. Soraya Cheriguene. April 2026.*
