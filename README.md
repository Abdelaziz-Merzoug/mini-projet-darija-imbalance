# Gestion du Desequilibre de Classes pour l'Analyse de Sentiments en Dialecte Algerien

**Universite :** USDB Blida 1 — Departement Informatique  
**Formation :** Master 1 Data Science & NLP — Semestre 2  
**Module :** Machine Learning  
**Etudiant :** Abdelaziz Merzoug  
**Encadrante :** Dr. Soraya Cheriguene  
**Periode :** 08 mars – 26 avril 2026

---

## Description du projet

Ce projet etudie les strategies de gestion du desequilibre de classes pour la
classification de sentiments en trois categories (Positif, Negatif, Neutre) sur
le corpus TWIFL de tweets algeriens en dialecte Darija (melange d'arabe, de
francais et d'Arabizi).

Le modele utilise est **DziriBERT** (`alger-ia/dziribert`), le seul modele
pre-entraine sur le dialecte algerien disponible publiquement. Quatre strategies
de reequilibrage sont comparees a l'aide du F1-macro comme metrique primaire.

---

## Structure du projet

```
mini_projet_darija/
|
|-- data/
|   |-- split_indices.json          # Indices de split 70/15/15 — NE PAS MODIFIER
|   |-- bt_paraphrases_neutral.json # Paraphrases filtrees par retro-traduction
|   `-- bt_annotated_examples.json  # 3 exemples annotes avec scores cosinus
|
|-- models/
|   `-- baseline_dziribert/         # Modele de reference fine-tune
|
|-- results/
|   |-- baseline_metrics.json       # NB02 — reference
|   |-- cw_metrics.json             # NB03 — Ponderation des classes
|   |-- fl_g1_metrics.json          # NB03 — Focal Loss gamma=1
|   |-- fl_g2_metrics.json          # NB03 — Focal Loss gamma=2
|   |-- cw_fl_metrics.json          # NB03 — CW + Focal Loss
|   |-- smote_full_metrics.json     # NB04 — SMOTE equilibrage total
|   |-- smote_partial_metrics.json  # NB04 — SMOTE equilibrage partiel
|   |-- adasyn_metrics.json         # NB04 — ADASYN
|   |-- bt_20pct_metrics.json       # NB05 — Retro-traduction +20%
|   |-- bt_50pct_metrics.json       # NB05 — Retro-traduction +50%
|   |-- bt_100pct_metrics.json      # NB05 — Retro-traduction +100%
|   |-- evaluation_finale_comparatif.csv
|   |-- strategie1_comparatif.csv
|   |-- strategie2_comparatif.csv
|   `-- strategie3_comparatif.csv
|
|-- figures/
|   |-- class_distribution_bar.png
|   |-- class_distribution_pie.png
|   |-- lang_distribution.png
|   |-- tweet_length_chars_histogram.png
|   |-- tweet_length_words_boxplot.png
|   |-- wordcloud_positive.png
|   |-- wordcloud_negative.png
|   |-- wordcloud_neutral.png
|   |-- emoji_proportion.png
|   |-- baseline_training_curves.png
|   |-- baseline_confusion_matrix.png
|   |-- strategie1_f1_macro_comparison.png
|   |-- strategie1_f1_per_class.png
|   |-- strategie1_cm_comparison.png
|   |-- tsne_before_rebalancing.png
|   |-- tsne_after_smote_full.png
|   |-- tsne_after_smote_partial.png
|   |-- tsne_after_adasyn.png
|   |-- strategie2_f1_macro_comparison.png
|   |-- strategie2_f1_per_class.png
|   |-- bt_cosine_distribution.png
|   |-- strategie3_f1_macro_comparison.png
|   |-- strategie3_f1_per_class.png
|   |-- finale_f1_macro_all.png
|   |-- finale_f1_per_class_heatmap.png
|   |-- finale_radar_top3.png
|   |-- finale_strategy_comparison.png
|   |-- finale_neutral_focus.png
|   `-- finale_accuracy_vs_f1macro.png
|
|-- notebooks/
|   |-- 01_EDA.ipynb
|   |-- 02_Preprocessing_Baseline.ipynb
|   |-- 03_Strategie1_Loss_Functions.ipynb
|   |-- 04_Strategie2_SMOTE_ADASYN.ipynb
|   |-- 05_Strategie3_BackTranslation.ipynb
|   `-- 06_Evaluation_Finale.ipynb
|
|-- requirements.txt
`-- README.md
```

---

## Corpus — TWIFL

| Propriete           | Valeur                                       |
|---------------------|----------------------------------------------|
| Source              | https://huggingface.co/datasets/arbml/Twifil |
| Taille              | 6 000 tweets algeriens                       |
| Colonne texte       | `Post`                                       |
| Colonne cible       | `Polarity Class`                             |
| Classes             | Positive / Negative / Neutral                |
| Distribution        | Positive: 2864 / Negative: 1773 / Neutral: 1363 |
| Ratio desequilibre  | 2,10:1 (modere)                              |
| Code-switching      | 38,1 % des tweets melangent arabe et francais |
| Taux d'emojis       | 21,6 %                                       |

---

## Modele

**DziriBERT** (`alger-ia/dziribert`)

- Architecture : BERT bidirectionnel
- Dimension [CLS] : 768
- Vocabulaire : 50 000 tokens
- Pre-entraine exclusivement sur le dialecte algerien

---

## Protocole experimental

### Hyperparametres fixes (identiques pour toutes les experiences)

| Parametre     | Valeur        |
|---------------|---------------|
| epochs        | 5             |
| learning_rate | 2e-5          |
| batch_size    | 16            |
| optimizer     | AdamW         |
| seed          | 42 (partout)  |

### Split des donnees

- Methode : stratifie, seed=42
- Proportions : 70 % train / 15 % validation / 15 % test
- Indices sauvegardes dans `data/split_indices.json` des la creation
- **Tous les notebooks NB03–NB06 chargent ce fichier — aucun re-split**

### Metriques (toutes calculees pour chaque experience)

| Metrique             | Justification                                           |
|----------------------|---------------------------------------------------------|
| **F1-macro**         | Metrique primaire — non sensible au desequilibre        |
| F1 par classe        | Diagnostic de la classe Neutre (minoritaire)            |
| Precision par classe | Qualite des predictions positives                       |
| Rappel par classe    | Couverture des vrais positifs                           |
| AUC-PR macro         | Aire sous la courbe Precision-Rappel (pas AUC-ROC)      |
| G-mean               | Moyenne geometrique des taux de rappel par classe       |
| Matrice de confusion | Visualisation des erreurs de classification             |
| Accuracy             | Calcule mais non presente comme metrique principale     |

---

## Notebooks — Ordre d'execution

### NB01 — Analyse exploratoire des donnees (EDA)

- **Plateforme :** Colab CPU (pas de GPU requis)
- **Duree :** ~10 minutes
- **Taches :**
  - Distribution des classes (graphiques barre et camembert)
  - Distribution des langues (`lang`)
  - Analyse de la longueur des tweets (histogramme, boxplot)
  - Taux d'emojis et code-switching
  - Nuages de mots par classe (arabic_reshaper + python-bidi)
  - Tableau statistique sauvegarde en CSV
- **Fichiers produits :** 9 figures PNG + `eda_statistics_table.csv`

---

### NB02 — Preprocessing et Baseline

- **Plateforme :** Google Colab T4 GPU
- **Duree :** ~60 minutes (5 epochs)
- **Taches :**
  - Preprocessing : deduplication, filtrage de langue, normalisation
  - Split 70/15/15 avec sauvegarde de `split_indices.json`
  - Fine-tuning DziriBERT sans reequilibrage (reference)
  - Calcul de toutes les metriques sur le jeu de test
- **Fichiers produits :** `split_indices.json` + `baseline_metrics.json`

**Resultats Baseline :**

| Metrique    | Valeur |
|-------------|--------|
| F1-macro    | 0.6805 |
| F1 Positive | 0.7806 |
| F1 Negative | 0.7014 |
| F1 Neutral  | 0.5596 |
| AUC-PR      | 0.7534 |
| G-mean      | 0.7558 |

---

### NB03 — Strategie 1 : Modification des fonctions de perte

- **Plateforme :** Google Colab T4 GPU
- **Duree :** ~3 heures (4 runs independants)
- **Variantes testees :**
  - A : Ponderation des classes (CrossEntropyLoss avec poids)
  - B1 : Focal Loss gamma=1 (sans poids de classe)
  - B2 : Focal Loss gamma=2 (sans poids de classe)
  - C : Ponderation + Focal Loss gamma=2 (combinaison)
- **Principe :** DziriBERT recharge depuis `alger-ia/dziribert` avant chaque variante
- **Fichiers produits :** `cw_metrics.json` + `fl_g1_metrics.json` + `fl_g2_metrics.json` + `cw_fl_metrics.json`

---

### NB04 — Strategie 2 : Reequilibrage dans l'espace des embeddings

- **Plateforme :** Kaggle P100 (extraction ~20 min pour 3 438 tweets train)
- **Taches :**
  - Extraction des vecteurs [CLS] 768D sur le train uniquement
  - SMOTE equilibrage total (classes egales)
  - SMOTE equilibrage partiel (minorites x 1,5)
  - ADASYN (adaptatif — zones frontiere)
  - Visualisation t-SNE avant/apres reequilibrage (4 figures)
  - Classificateur MLP sur embeddings reequilibres
- **Fichiers produits :** 3 fichiers metrics JSON + 4 figures t-SNE

---

### NB05 — Strategie 3 : Augmentation par retro-traduction

- **Plateforme :** Lightning AI A10G (Helsinki-NLP trop lent sur T4)
- **Pipeline :**
  1. Darija → Francais via `Helsinki-NLP/opus-mt-ar-fr`
  2. Francais → Arabe via `Helsinki-NLP/opus-mt-fr-ar`
  3. Filtrage par similarite cosinus [0,50 ; 0,85]
  4. Re-fine-tuning DziriBERT depuis le debut sur chaque jeu augmente
- **Taux testes :** +20% / +50% / +100% (appliques uniquement sur le train)
- **Fichiers produits :** 3 fichiers metrics JSON + `bt_paraphrases_neutral.json` + `bt_annotated_examples.json`

---

### NB06 — Evaluation finale

- **Plateforme :** Tout GPU
- **Taches :**
  - Chargement des 11 fichiers de metriques
  - Construction du tableau comparatif complet (11 configurations)
  - Visualisations de synthese (radar, heatmap, courbes)
  - Identification de la meilleure configuration
- **Fichiers produits :** `evaluation_finale_comparatif.csv` + 6 figures de synthese

---

## Resultats — Tableau comparatif final

| Configuration         | F1-macro | F1-Pos | F1-Neg | F1-Neu | AUC-PR | G-mean | Accuracy |
|-----------------------|----------|--------|--------|--------|--------|--------|----------|
| Baseline              | 0.6805   | 0.7806 | 0.7014 | 0.5596 | 0.7534 | 0.7558 | 0.7249   |
| Class Weighting       | 0.6386   | 0.7141 | 0.6882 | 0.5134 | 0.7571 | 0.7510 | 0.6694   |
| Focal Loss (gamma=1)  | 0.6784   | 0.7614 | 0.7213 | 0.5525 | 0.7687 | 0.7540 | 0.7209   |
| Focal Loss (gamma=2)  | 0.6794   | 0.7605 | 0.7233 | 0.5543 | 0.7647 | 0.7564 | 0.7209   |
| CW + Focal Loss       | 0.6683   | 0.7533 | 0.7068 | 0.5446 | 0.7460 | 0.7576 | 0.7060   |
| SMOTE Full Balance    | 0.6355   | 0.7334 | 0.6730 | 0.5000 | 0.6888 | 0.7186 | 0.6829   |
| SMOTE Partial Balance | 0.6470   | 0.7336 | 0.6801 | 0.5275 | 0.6943 | 0.7282 | 0.6883   |
| ADASYN                | 0.6617   | 0.7599 | 0.6757 | 0.5495 | 0.7085 | 0.7367 | 0.7046   |
| **BT +20%**           |**0.6909**|0.7786  | 0.7108 |**0.5833**|0.7652|**0.7618**|**0.7304**|
| BT +50%               | 0.6715   | 0.7752 | 0.7016 | 0.5376 | 0.7663 | 0.7484 | 0.7195   |
| BT +100%              | 0.6715   | 0.7752 | 0.7016 | 0.5376 | 0.7663 | 0.7484 | 0.7195   |

**Meilleure configuration : Back-Translation +20% (F1-macro = 0.6909, delta = +0.0104 vs Baseline)**

Seule configuration a surpasser la reference parmi les 10 strategies testees.

---

## Observations principales

1. **Focal Loss (gamma=2)** est la seule variante de la Strategie 1 a depasser
   le Baseline (+0.0003 sur F1-macro), mais l'amelioration est marginale.

2. **La Ponderation des classes** degrade les performances sur un desequilibre
   modere de 3,93:1 (post-preprocessing) : la sur-correction penalise la classe
   Neutre (F1-Neutral : -0.0462 vs Baseline).

3. **SMOTE et ADASYN** sur embeddings DziriBERT n'ameliorent pas le Baseline.
   L'interpolation lineaire en espace de plongement produit des representations
   qui se situent en dehors de la distribution des tweets reels.

4. **La retro-traduction a +20%** est la seule strategie efficace. Les taux
   superieurs (+50%, +100%) introduisent trop de derive MSA (arabe standard)
   par rapport au dialecte Darija et degradent les performances.

5. **La classe Neutre reste la plus difficile** dans toutes les configurations
   (F1-Neutral maximal : 0.5833 avec BT+20%), confirmant la difficulte
   intrinseque de cette classe dans le dialecte algerien.

---

## Installation

### Prerequis

- Python >= 3.10
- Acces GPU recommande (CUDA) pour les notebooks NB02 a NB06
- Compte Google Drive pour la persistence des artefacts
- Compte Hugging Face (token requis pour `push_to_hub`)

### Installation locale

```bash
git clone <url-du-depot>
cd mini_projet_darija
pip install -r requirements.txt
```

### Sur Google Colab

```python
!pip install -r requirements.txt
```

### Configuration du token Hugging Face (Colab Secrets)

```python
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')
```

Ne jamais coder le token en dur dans un notebook.

---

## Reproduction des resultats

### Etape 0 — Preparer Google Drive (une seule fois)

```python
from google.colab import drive
drive.mount('/content/drive')

import os
BASE = '/content/drive/MyDrive/mini_projet_darija'
for d in ['data', 'models', 'results', 'figures', 'notebooks']:
    os.makedirs(f'{BASE}/{d}', exist_ok=True)
```

### Etape 1 — Executer les notebooks dans l'ordre

| Ordre | Notebook                             | Plateforme           | Prerequis                  |
|-------|--------------------------------------|----------------------|----------------------------|
| 1     | `01_EDA.ipynb`                       | Colab CPU            | Aucun                      |
| 2     | `02_Preprocessing_Baseline.ipynb`    | Colab T4             | Aucun                      |
| 3     | `03_Strategie1_Loss_Functions.ipynb` | Colab T4             | `split_indices.json`       |
| 4     | `04_Strategie2_SMOTE_ADASYN.ipynb`   | Kaggle P100          | `split_indices.json`       |
| 5     | `05_Strategie3_BackTranslation.ipynb`| Lightning AI A10G    | `split_indices.json`       |
| 6     | `06_Evaluation_Finale.ipynb`         | Tout GPU             | 11 fichiers metrics JSON   |

### Artefact critique

`data/split_indices.json` est genere une seule fois par NB02.
Tous les notebooks NB03–NB06 le chargent au demarrage.
**Ne jamais supprimer ce fichier — ne jamais re-splitter les donnees.**

---

## Constantes globales

Definies au debut de chaque notebook et invariantes :

```python
TEXT_COL  = 'Post'
LABEL_COL = 'Polarity Class'
LANG_COL  = 'lang'
SEED      = 42
LABEL_MAP = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
BASE      = '/content/drive/MyDrive/mini_projet_darija'
MODEL_ID  = 'alger-ia/dziribert'
```

---

## References

```
[1]  A. Dossou et al., "DziriBERT: a Pre-trained Language Model for the
     Algerian Dialect," AfricaNLP, 2022.

[2]  Y. Boutaleb et al., "TWIFL: An Algerian Corpus and Annotation Platform,"
     LREC-COLING, 2024.

[3]  N. V. Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique,"
     JAIR, vol. 16, pp. 321-357, 2002.

[4]  T. Y. Lin et al., "Focal Loss for Dense Object Detection,"
     IEEE ICCV, 2017, pp. 2980-2988.

[5]  H. He et al., "ADASYN: Adaptive Synthetic Sampling Approach,"
     IEEE IJCNN, 2008, pp. 1322-1328.

[6]  S. Edunov et al., "Understanding Back-Translation at Scale,"
     EMNLP, 2018, pp. 489-500.

[7]  J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers,"
     NAACL-HLT, 2019.

[8]  G. Haixiang et al., "Learning from class-imbalanced data,"
     Expert Systems with Applications, vol. 73, 2017.

[9]  J. Tiedemann and S. Thottingal, "OPUS-MT — Building open translation
     services," EAMT, 2020.

[10] J. Davis and M. Goadrich, "The relationship between Precision-Recall
     and ROC curves," ICML, 2006.
```

---

*Projet realise dans le cadre du module Machine Learning — Master 1 DS & NLP,
USDB Blida 1, sous la supervision de Dr. Soraya Cheriguene.*
