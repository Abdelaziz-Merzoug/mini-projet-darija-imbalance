# GUIDE PARFAIT FINAL — Version Définitive

## Gestion du Déséquilibre de Classes pour l'Analyse de Sentiments en Dialecte Algérien

**Université USDB — Département Informatique — Master 1 DS & NLP — Semestre 2**
**Module :** Machine Learning | **Enseignante :** Dr. Soraya Cheriguene
**Distribution :** 08 Mars 2026 | **Soutenance :** 26 Avril 2026 | **Durée :** 6 semaines

---

> **Ce guide est la synthèse définitive** de l'analyse croisée entre l'Énoncé officiel (source de vérité absolue), le GUIDE_PARFAIT v2.0, et le mini_projet_guide_prompts. Chaque instruction a été vérifiée ligne par ligne contre l'Énoncé. Les éléments marqués ⛔ sont des contraintes dures dont la violation entraîne une pénalisation. Les éléments marqués ⚠️ sont des pièges courants. Les éléments marqués ✅ sont des bonnes pratiques qui maximisent les points.

---

## TABLE DES MATIÈRES

1. [Contraintes Absolues (extraites de l'Énoncé)](#1-contraintes-absolues)
2. [Grille de Notation — Stratégie Point par Point](#2-grille-de-notation)
3. [Master Prompt (à coller une seule fois)](#3-master-prompt)
4. [Semaine 1 — Setup + EDA](#semaine-1)
5. [Semaine 2 — Prétraitement + Baseline](#semaine-2)
6. [Semaine 3 — Stratégie 1 : Fonctions de Perte](#semaine-3)
7. [Semaine 4 — Stratégie 2 : SMOTE / ADASYN](#semaine-4)
8. [Semaine 5 — Stratégie 3 : Back-Translation](#semaine-5)
9. [Semaine 6 — Évaluation Finale + Rapport + Soutenance](#semaine-6)
10. [Bonus — Stratégie Hybride (+2 pts)](#bonus)
11. [Références IEEE Pré-Formatées](#references)
12. [Questions Probables en Soutenance](#soutenance-qa)
13. [Checklist Finale (J-1)](#checklist)
14. [Recommandations Modèles IA & GPU](#recommandations)

---

## 1. CONTRAINTES ABSOLUES (extraites mot-à-mot de l'Énoncé) {#1-contraintes-absolues}

### 1.1 Corpus

| Paramètre | Valeur exacte | Référence Énoncé |
|---|---|---|
| Dataset | arbml/Twifil (HuggingFace) | §2 |
| Taille | 6 000 tweets | §2 |
| Langue | Dialecte algérien (arabe, Arabizi, code-switching arabe-français) | §2 |
| Variable cible | `Polarity Class` | §2 |
| Classes | Positive, Negative, Neutral (exactement 3) | §2 |
| Colonnes utilisées | Texte + Polarity Class UNIQUEMENT | §2 : "Les autres colonnes ne sont pas utilisées" |

### 1.2 Modèle

| Paramètre | Valeur exacte | Référence Énoncé |
|---|---|---|
| Modèle obligatoire | alger-ia/dziribert | §4 |
| Architecture | BERT bidirectionnel | §4 |
| Dimension [CLS] | 768 | §4 |
| Vocabulaire | 50 000 tokens | §4 |
| ⛔ Autre modèle | INTERDIT sauf pour le bonus | §4 : "Aucun autre modèle de base n'est autorisé, sauf pour le bonus" |

### 1.3 Protocole Expérimental

| Paramètre | Valeur exacte | Référence Énoncé |
|---|---|---|
| Train | 70% | §5.1 |
| Validation | 15% | §5.1 |
| Test | 15% | §5.1 |
| random_state | 42 (PARTOUT) | §5.1 + §5.2 |
| Stratification | OUI obligatoire | §5.1 |
| ⛔ Test set | JAMAIS modifié, JAMAIS rééquilibré | §5.1 |
| ⛔ Rééquilibrage | UNIQUEMENT sur le train set | §5.1 |
| Epochs | 5 | §5.2 |
| Learning rate | 2e-5 | §5.2 |
| Batch size | 16 | §5.2 |
| Seed | 42 | §5.2 |
| Optimiseur | AdamW | §5.2 |
| ⛔ Écart au protocole | Doit être justifié par écrit sous peine de pénalisation | §5 |

### 1.4 Métriques Obligatoires

| Métrique | Description | Référence |
|---|---|---|
| ⛔ F1-macro | Métrique PRINCIPALE | §5.3 |
| F1 par classe | Positive, Negative, Neutral séparément | §5.3 |
| Precision / Rappel | Par classe séparément | §5.3 |
| AUC-PR | Aire sous courbe Precision-Rappel | §5.3 |
| G-mean | Racine moyenne géométrique des rappels | §5.3 |
| Matrice de confusion | Tableau prédictions vs vraies classes | §5.3 |
| ⛔ Accuracy | INTERDIT comme métrique principale. Mentionner uniquement pour illustrer le problème | §5.3 |

### 1.5 Livrables

| Livrable | Format | Échéance |
|---|---|---|
| Code Python | Notebooks Jupyter commentés + requirements.txt + README sur GitHub | J-1 avant soutenance |
| Rapport | PDF, 15-20 pages (hors annexes) | J-1 avant soutenance |
| Présentation | PowerPoint, ≤ 15 slides | Jour J |
| Soutenance orale | 10 min exposé + 5 min questions | Jour J |

---

## 2. GRILLE DE NOTATION — STRATÉGIE POINT PAR POINT {#2-grille-de-notation}

| Critère | Points | Ce que l'enseignante évalue | Comment maximiser |
|---|---|---|---|
| EDA | 1.5 | Distribution, visualisations, analyse linguistique, conclusions chiffrées | Tableau obligatoire rempli + WordClouds + validation Polarity Class + ratio déséquilibre chiffré |
| Preprocessing | 1.5 | Choix justifiés, adaptation au corpus TWIFL, analyse impact | Justifier CHAQUE décision (emojis, tweets courts, und) dans markdown cells |
| Baseline | 2.0 | Fine-tuning DziriBERT, résultats cohérents, métriques correctes | Toutes métriques + matrice confusion commentée + analyse du biais vers classe majoritaire |
| Stratégie 1 | 2.0 | Class Weighting + Focal Loss + combinaison | 3 variantes complètes + réponses aux 4 questions d'analyse |
| Stratégie 2 | 2.0 | Extraction embeddings + variantes comparées | SMOTE total ET partiel + ADASYN + t-SNE + réponses aux 4 questions |
| Stratégie 3 | 1.5 | Implémentation, filtrage, réévaluation, analyse paraphrases | 3 taux testés + exemples concrets + % filtre calculé + analyse dérive darija→MSA |
| Rapport + Discussion | 2.0 | Réponses aux questions, spécificités darija, perspectives | Répondre aux 4 questions de CHAQUE stratégie + discussion critique des limites |
| Qualité code | 1.0 | Test set intact, métriques imposées, tableau comparatif complet, structure, commentaires, reproductibilité | requirements.txt + README + seed=42 partout + notebooks exécutables bout en bout |
| Soutenance | 1.5 | Clarté, maîtrise du sujet, réponses aux questions | Préparer les 7 questions probables (voir Section 12) |
| **TOTAL** | **15** | | |
| Bonus | +2 | Stratégie hybride originale, documentée et analysée | CW+SMOTE ou analyse ratios (voir Section 10) |

---

## 3. MASTER PROMPT — À COLLER EN PREMIER (UNE SEULE FOIS) {#3-master-prompt}

> Collez ce prompt au tout début de votre session IA. Il établit le contexte complet. Après confirmation de l'IA, utilisez les prompts hebdomadaires dans l'ordre.

```
Tu es mon assistant expert en Machine Learning, NLP et Traitement du Langage
pour le dialecte algérien. Je vais te présenter mon mini-projet universitaire.

=== CONTEXTE DU PROJET ===
Université : USDB — Département Informatique — Master 1 DS & NLP
Module : Machine Learning | Enseignante : Dr. Soraya Cheriguene
Durée : 08 Mars au 26 Avril 2026 | Travail : Binôme ou monôme

=== CORPUS : TWIFL ===
Source : https://huggingface.co/datasets/arbml/Twifil
6 000 tweets algériens annotés (arabe, Arabizi, code-switching arabe-français)
Tâche : Classification 3 classes (Positive / Negative / Neutral)
Variable cible : Polarity Class
Valeurs attendues : UNIQUEMENT Positive, Negative, Neutral
Colonnes à ignorer : Emotion, User Age, et toutes autres colonnes non-texte.

=== MODÈLE OBLIGATOIRE : DziriBERT ===
HuggingFace : alger-ia/dziribert | Architecture : BERT bidirectionnel
Vocabulaire : 50 000 tokens | Embeddings [CLS] : dimension 768
AUCUN autre modèle de base autorisé (sauf bonus documenté)

=== PROTOCOLE OBLIGATOIRE (tout écart = pénalisation) ===
SPLIT : Train 70% / Val 15% / Test 15% — random_state=42 — stratifié
SPLIT TECHNIQUE : D'abord 85/15 (test), puis le 85% en 82.35/17.65 (train/val)
  → test_size=0.15 puis test_size=0.1765 pour obtenir exactement 70/15/15
VAL SET : Utilisé uniquement pour hyperparamètres — JAMAIS rééquilibré
TEST SET : JAMAIS modifié, JAMAIS rééquilibré, créé UNE seule fois
HYPERPARAMÈTRES IMPOSÉS : epochs=5, lr=2e-5, batch_size=16, seed=42, AdamW
MÉTRIQUES : F1-macro (PRINCIPALE), F1/classe, Precision/Rappel/classe,
  AUC-PR, G-mean, Matrice de confusion
INTERDIT : Accuracy comme métrique principale

=== 4 CONFIGURATIONS À IMPLÉMENTER ===
0. BASELINE : Fine-tuning DziriBERT sur données brutes déséquilibrées

1. STRATÉGIE 1 — Modification de la fonction de perte :
   A) Class Weighting — poids = total_exemples/(nb_classes × nb_exemples_classe)
   B) Focal Loss — tester gamma=1 ET gamma=2 séparément
   C) Combinaison Class Weighting + Focal Loss

2. STRATÉGIE 2 — Rééquilibrage dans l'espace des embeddings [CLS] 768D :
   A) SMOTE — tester équilibre total ET partiel
   B) ADASYN
   → Classifieur MLP ou SVM sur embeddings rééquilibrés
   → Évaluer sur test embeddings (jamais rééquilibrés)

3. STRATÉGIE 3 — Back-Translation :
   Darija→Français→Arabe via Helsinki-NLP (HuggingFace)
   Filtrage par similarité cosinus [0.5, 0.85]
   Tester 3 taux : +20%, +50%, +100% sur la classe minoritaire
   ATTENTION : gérer les erreurs de traduction (try/except) sur tweets Arabizi

=== BONUS (+2 pts) ===
Stratégie hybride originale : Class Weighting + SMOTE sur embeddings
OU analyse comparative de différents ratios d'augmentation
Doit être documentée et analysée dans le rapport.

=== LIVRABLES ===
Code Python : 6 Notebooks Jupyter commentés + requirements.txt + README (GitHub)
Rapport PDF : 15-20 pages (hors annexes) — Structure imposée (8 sections)
Présentation : ≤ 15 slides PowerPoint
Soutenance : 10 min exposé + 5 min questions

=== STRUCTURE DU RAPPORT (imposée par l'Énoncé) ===
1. Introduction et problématique (1-2 pages)
2. État de l'art (2-3 pages)
3. Présentation du corpus TWIFL — EDA, statistiques, nettoyage (2-3 pages)
4. Méthodologie — description de chaque stratégie (3-4 pages)
5. Résultats — tableau comparatif, matrices, courbes PR (3-4 pages)
6. Discussion — analyse critique, limites, spécificités darija (2-3 pages)
7. Conclusion et perspectives (1 page)
8. Références bibliographiques (IEEE ou APA)

=== QUESTIONS D'ANALYSE OBLIGATOIRES ===
Pour chaque stratégie, l'Énoncé demande de répondre à des questions précises
dans le rapport. Je te les fournirai semaine par semaine.

Confirme ta compréhension et donne-moi :
1. Les points-clés à ne pas oublier pour chaque stratégie
2. Les pièges techniques courants à éviter pour ce corpus
3. Un planning de travail réaliste sur 6 semaines
```

---

## SEMAINE 1 · 08-15 Mars 2026 {#semaine-1}
### Setup Environnement + Analyse Exploratoire (EDA)

**Points visés : 1.5 / 1.5 (EDA)**
**GPU nécessaire : Non (CPU suffit)**
**Durée estimée : 3-4h**
**Livrable : `01_EDA.ipynb` + `figures/` sauvegardées en PNG**

---

#### PROMPT SEMAINE 1

```
Nous commençons la SEMAINE 1 du projet.
Objectif : Setup complet + Analyse Exploratoire complète du corpus TWIFL.

=== TÂCHE 0 : VALIDATION DE LA COLONNE CIBLE [CRITIQUE] ===
AVANT toute analyse, valider la colonne Polarity Class.
L'Énoncé exige (§2.3) : "vérifier l'absence de valeurs inattendues".

Code obligatoire :
from datasets import load_dataset
import pandas as pd

dataset = load_dataset('arbml/Twifil')
df = dataset['train'].to_pandas()

# Validation critique
print('Colonnes disponibles :', df.columns.tolist())
print('Types :', df.dtypes)
print('Shape :', df.shape)
print()

# Vérifier les valeurs de Polarity Class
unique_vals = set(df['Polarity Class'].unique())
expected = {'Positive', 'Negative', 'Neutral'}
print('Valeurs uniques :', unique_vals)
print('Valeurs manquantes :', df['Polarity Class'].isna().sum())

if unique_vals != expected:
    unexpected = unique_vals - expected
    missing = expected - unique_vals
    print(f'⚠️ ATTENTION — Valeurs inattendues : {unexpected}')
    print(f'⚠️ ATTENTION — Valeurs manquantes : {missing}')
    # DOCUMENTER dans le rapport si des anomalies sont trouvées
else:
    print('✅ Colonne Polarity Class valide : exactement 3 classes attendues')

=== TÂCHE 1 : SETUP ENVIRONNEMENT ===
Génère le code complet pour installer toutes les dépendances :

# requirements.txt (à créer dès maintenant)
transformers>=4.36.0
datasets>=2.16.0
torch>=2.1.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
pandas>=2.1.0
numpy>=1.24.0
matplotlib>=3.8.0
seaborn>=0.13.0
wordcloud>=1.9.0
emoji>=2.8.0
arabic-reshaper>=3.0.0
python-bidi>=0.4.2

# Installation dans le notebook :
!pip install transformers datasets torch scikit-learn imbalanced-learn \
    pandas numpy matplotlib seaborn wordcloud emoji \
    arabic_reshaper python-bidi

=== TÂCHE 2 : ANALYSE DE LA VARIABLE CIBLE (Énoncé §1.1) ===
L'Énoncé demande EXACTEMENT ces 4 éléments :
1. Calculer le nombre d'exemples par classe (Positive / Negative / Neutral)
2. Calculer le ratio de déséquilibre : classe_majoritaire / classe_minoritaire
3. Tracer un bar chart ET un pie chart de la distribution
4. Conclure : corpus fortement (>5:1), modérément (3:1–5:1), ou faiblement (<3:1) déséquilibré

Génère le code complet pour ces 4 points. Chaque graphique doit être
sauvegardé en PNG dans un dossier figures/ :
    figures/class_distribution_bar.png
    figures/class_distribution_pie.png

=== TÂCHE 3 : ANALYSE LINGUISTIQUE (Énoncé §1.2) ===
L'Énoncé demande EXACTEMENT ces 5 analyses :
1. Distribution de longueur des tweets (en MOTS et en CARACTÈRES) — PAR CLASSE
   → Boxplot + histogramme séparé par classe (2 graphiques minimum)
2. Proportion de tweets en arabe / français / Arabizi / langue indéterminée
   (utiliser la colonne 'lang' si elle existe dans le dataset)
3. Proportion de tweets contenant des emojis
   (utiliser la librairie emoji : emoji.emoji_count(text) > 0)
4. Proportion de tweets avec code-switching
   (mélange de caractères arabes + mots en latin/français dans le même tweet)
5. Nuage de mots (WordCloud) SÉPARÉ pour CHAQUE classe de sentiment
   (3 WordClouds : Positive, Negative, Neutral)
   → Utiliser arabic_reshaper + python-bidi pour afficher l'arabe correctement

Sauvegarder TOUS les graphiques dans figures/ :
    figures/tweet_length_words_boxplot.png
    figures/tweet_length_chars_histogram.png
    figures/lang_distribution.png
    figures/emoji_proportion.png
    figures/wordcloud_positive.png
    figures/wordcloud_negative.png
    figures/wordcloud_neutral.png

=== TÂCHE 4 : TABLEAU RÉCAPITULATIF OBLIGATOIRE (Énoncé §1.3) ===
L'Énoncé exige CE TABLEAU EXACT rempli avec les vraies valeurs :

| Classe   | Nb exemples | % du corpus | Moy. mots | Moy. caractères |
|----------|-------------|-------------|-----------|-----------------|
| Positive | ...         | ...         | ...       | ...             |
| Negative | ...         | ...         | ...       | ...             |
| Neutral  | ...         | ...         | ...       | ...             |
| TOTAL    | 6 000       | 100%        | ...       | ...             |

Ce tableau doit apparaître DANS LE RAPPORT tel quel.
Génère le code pour calculer ces valeurs exactes et afficher le tableau.

=== TÂCHE 5 : CONCLUSIONS ÉCRITES ===
À la fin du notebook, ajouter une cellule Markdown avec :
1. Le niveau de déséquilibre observé (avec ratio chiffré)
2. La classe majoritaire et la classe minoritaire identifiées
3. Les caractéristiques linguistiques notables du corpus
4. Les problèmes de qualité de données observés (tweets vides, doublons, etc.)
5. Les implications pour la modélisation (anticipation du biais de la baseline)

=== LIVRABLES SEMAINE 1 ===
- Notebook : 01_EDA.ipynb (toutes les cellules exécutées, résultats visibles)
- Dossier : figures/ (tous les PNG sauvegardés)
- Fichier : README.md initial sur GitHub
- Conclusions écrites dans le notebook
```

#### ⚠️ PIÈGES SEMAINE 1

1. **La colonne texte peut avoir un nom différent** — Explorer `df.columns` AVANT de coder. Le nom peut être "text", "Text", "tweet", etc.
2. **WordCloud arabe** — Sans `arabic_reshaper` + `bidi`, les mots arabes apparaissent inversés et déconnectés. Tester le rendu.
3. **Colonne 'lang'** — Peut ne pas exister dans toutes les versions du dataset. Prévoir un fallback avec détection automatique (`langdetect` ou heuristique sur les caractères).
4. **Code-switching** — La détection n'est pas triviale. Une heuristique simple : vérifier si le tweet contient à la fois des caractères arabes (\\u0600-\\u06FF) et des caractères latins (a-zA-Z de plus de 2 caractères consécutifs).

---

## SEMAINE 2 · 15-22 Mars 2026 {#semaine-2}
### Prétraitement Darija + Modèle Baseline

**Points visés : 1.5 (Preprocessing) + 2.0 (Baseline) = 3.5 / 15**
**GPU nécessaire : OUI (T4 minimum) — Fine-tuning ~30-45 min**
**Durée estimée : 5-6h**
**Livrable : `02_Preprocessing_Baseline.ipynb` + modèle sauvegardé + `results/baseline_metrics.json`**

---

#### PROMPT SEMAINE 2

```
Nous commençons la SEMAINE 2 du projet.
Objectif : Pipeline de prétraitement adapté au darija + Entraînement du modèle Baseline.

=== PARTIE A : PREPROCESSING DARIJA ===

RAPPEL CRUCIAL DE L'ÉNONCÉ (§2, Note importante) :
"Les étapes de prétraitement sont des PROPOSITIONS. Vous êtes libres d'en
ajouter, d'en supprimer ou de les modifier. TOUT CHOIX doit être JUSTIFIÉ
dans votre rapport."

--- TÂCHE 1 : NETTOYAGE DE BASE (Énoncé §2.1) ---
Génère le code pour :
1. Supprimer les URLs : re.sub(r'https?://\S+|www\.\S+', '', text)
2. Supprimer les mentions : re.sub(r'@\w+', '', text)
3. Traitement des emojis — CHOISIR et JUSTIFIER :
   Option A : Supprimer tous les emojis
   Option B : Convertir les emojis en texte descriptif (emoji.demojize())
   → JUSTIFIER le choix dans une cellule markdown. Argument possible :
     les emojis portent une information sentimentale (😊 = positif, 😡 = négatif)
     donc Option B préserve cette information pour le modèle.
4. Supprimer les doublons stricts : df.drop_duplicates(subset='<colonne_texte>')
5. Supprimer les tweets vides ou sans texte utile (après nettoyage)

Pour CHAQUE étape, afficher un exemple avant/après ET le nombre de tweets
affectés. Ceci est nécessaire pour le rapport.

--- TÂCHE 2 : NORMALISATION DARIJA (Énoncé §2.2) ---
1. Normaliser les lettres arabes répétées :
   re.sub(r'(.)\1{2,}', r'\1\1', text)
   Exemple : مممزيان → مزيان (garder max 2 répétitions)
2. Normaliser les chiffres arabes-indiens → occidentaux si présents :
   text.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789'))
3. ⛔ NE PAS supprimer les mots français — le code-switching est une
   caractéristique linguistique du darija, PAS du bruit
4. ⛔ NE PAS faire de stemming ni de lemmatisation — DziriBERT tokenise seul
   (l'Énoncé est explicite sur ce point)

--- TÂCHE 3 : CAS SPÉCIAUX DU CORPUS TWIFL (Énoncé §2.3) ---
L'Énoncé demande SPÉCIFIQUEMENT de traiter ces 3 cas :

1. Tweets de langue 'und' (indéterminée) :
   → L'Énoncé dit "les conserver et analyser leur contenu"
   → Afficher 5-10 exemples et commenter dans le notebook
   → JUSTIFIER la décision de les garder dans le rapport

2. Tweets très courts (1-2 mots) :
   → L'Énoncé dit "décider si on les conserve ou les filtre (JUSTIFIER)"
   → Compter combien il y en a
   → Afficher des exemples
   → Prendre une décision ET écrire la justification

3. Colonne Polarity Class :
   → "vérifier l'absence de valeurs inattendues"
   → Déjà fait en Semaine 1, mais re-vérifier après nettoyage

--- TÂCHE 4 : VÉRIFICATION POST-NETTOYAGE ---
print(f'Tweets AVANT nettoyage : {len(df_original)}')
print(f'Tweets APRÈS nettoyage : {len(df_clean)}')
print(f'Tweets supprimés : {len(df_original) - len(df_clean)}')
print()
# Tableau détaillé des suppressions :
print('Détail des suppressions :')
print(f'  URLs/mentions : {n_urls}')
print(f'  Doublons : {n_duplicates}')
print(f'  Tweets vides : {n_empty}')
print(f'  Tweets très courts (si supprimés) : {n_short}')
print()
# VÉRIFIER que la distribution de classes est préservée :
print('Distribution AVANT :', df_original['Polarity Class'].value_counts().to_dict())
print('Distribution APRÈS :', df_clean['Polarity Class'].value_counts().to_dict())
# Si le nettoyage a déséquilibré davantage, le noter

--- TÂCHE 5 : SPLIT UNIQUE (Énoncé §2.4 + §5.1) ---
⛔ CRITIQUE : Ce split est fait UNE SEULE FOIS. Il ne sera PLUS JAMAIS modifié.

from sklearn.model_selection import train_test_split
import numpy as np

SEED = 42
np.random.seed(SEED)

X = df_clean['<colonne_texte>']
y = df_clean['Polarity Class']

# ÉTAPE 1 : Séparer le test set (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y
)

# ÉTAPE 2 : Séparer train (70%) et val (15%) à partir des 85% restants
# 15% du total = 17.65% de 85% → test_size ≈ 0.1765
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=SEED, stratify=y_temp
)

print(f'Train : {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)')
print(f'Val   : {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)')
print(f'Test  : {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)')
print()
# Vérifier la stratification dans chaque split :
for name, labels in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    dist = labels.value_counts(normalize=True)
    print(f'{name} : {dist.to_dict()}')

# SAUVEGARDER les splits pour réutilisation exacte dans toutes les semaines :
import json
splits = {
    'train_indices': X_train.index.tolist(),
    'val_indices': X_val.index.tolist(),
    'test_indices': X_test.index.tolist()
}
with open('data/split_indices.json', 'w') as f:
    json.dump(splits, f)

=== PARTIE B : MODÈLE BASELINE (Énoncé §ÉTAPE 3) ===

L'Énoncé dit explicitement :
"Le modèle baseline est entraîné sur les données brutes déséquilibrées,
sans AUCUNE technique de rééquilibrage."

--- TÂCHE 6 : FINE-TUNING DziriBERT ---
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
import torch

# Configuration EXACTE — ⛔ NE PAS MODIFIER
SEED = 42
EPOCHS = 5
LR = 2e-5
BATCH_SIZE = 16

torch.manual_seed(SEED)
np.random.seed(SEED)

model_name = 'alger-ia/dziribert'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3
)

# Encoder les labels : Positive=0, Negative=1, Neutral=2
label_map = {'Positive': 0, 'Negative': 1, 'Neutral': 2}

# Tokeniser tous les splits
def tokenize_data(texts, labels, tokenizer, max_length=128):
    encodings = tokenizer(
        list(texts), truncation=True, padding=True, max_length=max_length
    )
    encodings['labels'] = [label_map[l] for l in labels]
    return encodings  # Convertir en Dataset PyTorch

# Training arguments
training_args = TrainingArguments(
    output_dir='./models/baseline',
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    seed=SEED,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',  # PAS accuracy
    logging_dir='./logs/baseline',
)

# ⛔ PAS de class_weight, PAS de resampling, PAS de Focal Loss
# Fine-tuner sur le train set DÉSÉQUILIBRÉ tel quel

--- TÂCHE 7 : ÉVALUATION COMPLÈTE SUR TEST SET ---
⛔ Calculer TOUTES les métriques exigées par l'Énoncé (§5.3).
Créer une FONCTION RÉUTILISABLE (elle sera appelée pour CHAQUE stratégie) :

from sklearn.metrics import (
    f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
    average_precision_score
)
from imblearn.metrics import geometric_mean_score
import numpy as np

def evaluate_model(y_true, y_pred, y_proba, class_names=['Positive','Negative','Neutral']):
    """
    Fonction d'évaluation COMPLÈTE — réutilisée pour CHAQUE expérience.
    y_true : labels vrais (indices 0,1,2)
    y_pred : labels prédits (indices 0,1,2)
    y_proba : probabilités (shape N×3) — nécessaire pour AUC-PR
    """
    results = {}

    # 1. F1-macro (MÉTRIQUE PRINCIPALE)
    results['f1_macro'] = f1_score(y_true, y_pred, average='macro')

    # 2. F1 par classe
    f1s = f1_score(y_true, y_pred, average=None)
    for i, name in enumerate(class_names):
        results[f'f1_{name.lower()}'] = f1s[i]

    # 3. Precision et Rappel par classe
    prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i, name in enumerate(class_names):
        results[f'precision_{name.lower()}'] = prec[i]
        results[f'recall_{name.lower()}'] = rec[i]

    # 4. AUC-PR (Average Precision Score, macro)
    # Nécessite one-hot encoding des labels + probabilités
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    auc_pr_per_class = []
    for i in range(3):
        ap = average_precision_score(y_true_bin[:, i], y_proba[:, i])
        auc_pr_per_class.append(ap)
    results['auc_pr_macro'] = np.mean(auc_pr_per_class)

    # 5. G-mean
    results['g_mean'] = geometric_mean_score(y_true, y_pred, average='macro')

    # 6. Matrice de confusion
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    # 7. Rapport complet
    results['classification_report'] = classification_report(
        y_true, y_pred, target_names=class_names
    )

    return results

# Appeler cette fonction et SAUVEGARDER les résultats baseline :
baseline_results = evaluate_model(y_true, y_pred, y_proba)

import json
with open('results/baseline_metrics.json', 'w') as f:
    json.dump({k: v for k, v in baseline_results.items()
               if k != 'classification_report'}, f, indent=2)

# Afficher le rapport complet :
print(baseline_results['classification_report'])

# Afficher la matrice de confusion en heatmap :
import seaborn as sns
import matplotlib.pyplot as plt
cm = np.array(baseline_results['confusion_matrix'])
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Positive','Negative','Neutral'],
            yticklabels=['Positive','Negative','Neutral'], ax=ax)
ax.set_xlabel('Prédit')
ax.set_ylabel('Vrai')
ax.set_title('Matrice de Confusion — Baseline DziriBERT')
plt.savefig('figures/baseline_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

--- TÂCHE 8 : ANALYSE DU BIAIS (Énoncé §ÉTAPE 3) ---
L'Énoncé dit : "Attendez-vous à ce que la baseline soit biaisée vers la classe
majoritaire. C'est NORMAL et ATTENDU. Votre travail est de MONTRER ce biais
et de le corriger."

Répondre dans une cellule markdown :
1. Quelle classe a le meilleur F1 ? Le pire F1 ? Pourquoi ?
2. Le modèle prédit-il massivement la classe majoritaire ? (matrice confusion)
3. Quel est le F1-macro baseline ? → C'est votre SCORE DE RÉFÉRENCE à battre
4. Calculer aussi l'accuracy pour ILLUSTRER le problème :
   "L'accuracy est de X% mais le F1-macro n'est que de Y%, ce qui montre que
   l'accuracy masque les mauvaises performances sur les classes minoritaires."

# Sauvegarder le modèle pour persistance entre sessions :
model.save_pretrained('models/baseline_dziribert')
tokenizer.save_pretrained('models/baseline_dziribert')
# Sur HuggingFace Hub (optionnel mais recommandé pour Colab) :
# model.push_to_hub('votre_username/baseline-dziribert-twifl', private=True)

=== LIVRABLES SEMAINE 2 ===
- Notebook : 02_Preprocessing_Baseline.ipynb
- Modèle : models/baseline_dziribert/
- Fichier : results/baseline_metrics.json
- Fichier : data/split_indices.json (pour reproductibilité)
- Figure : figures/baseline_confusion_matrix.png
- Justifications de preprocessing en cellules markdown
```

#### ⚠️ PIÈGES SEMAINE 2

1. **Le ratio 0.1765** — Si vous faites `train_test_split(X, y, test_size=0.15)` puis `train_test_split(X_temp, y_temp, test_size=0.15)`, vous obtiendrez 72.25/12.75/15, PAS 70/15/15. Il faut `test_size=15/85 ≈ 0.1765` pour le second split.
2. **Label encoding cohérent** — Utilisez le MÊME `label_map` partout (Positive=0, Negative=1, Neutral=2). Si vous changez l'ordre, toutes les métriques par classe seront fausses.
3. **Sauvegarde Colab** — Colab déconnecte après ~12h. Sauvegardez le modèle sur Google Drive ou HuggingFace Hub à la fin de chaque entraînement.
4. **VRAM DziriBERT** — DziriBERT + batch_size=16 → ~10-12 GB VRAM. T4 (16 GB) suffit. Si OOM, réduire à batch_size=8 mais JUSTIFIER dans le rapport.
5. **AUC-PR** — Nécessite les PROBABILITÉS (softmax), pas seulement les prédictions. Récupérer `y_proba` depuis les logits du modèle.

---

## SEMAINE 3 · 22-29 Mars 2026 {#semaine-3}
### Stratégie 1 — Modification de la Fonction de Perte

**Points visés : 2.0 / 2.0**
**GPU nécessaire : OUI (4 runs × ~45 min = ~3h GPU)**
**Durée estimée : 6-8h total**
**Livrable : `03_Strategie1_Loss_Functions.ipynb`**

---

#### PROMPT SEMAINE 3

```
Nous commençons la SEMAINE 3 du projet.
Objectif : Implémenter les 3 variantes de modification de la perte.

RAPPEL CRUCIAL :
- On NE modifie PAS les données. Le train set reste déséquilibré.
- On change UNIQUEMENT la fonction de perte.
- MÊMES hyperparamètres que la baseline (epochs=5, lr=2e-5, batch=16, seed=42, AdamW).
- Le test set est IDENTIQUE à celui de la Semaine 2 (jamais modifié).
- Utiliser la MÊME fonction evaluate_model() créée en Semaine 2.

=== VARIANTE A : CLASS WEIGHTING (Énoncé §4.1) ===

L'Énoncé impose la formule EXACTE :
    poids = total_exemples / (nb_classes × nb_exemples_de_la_classe)

import torch
import numpy as np

# Calcul des poids avec la formule IMPOSÉE
total = len(y_train)
nb_classes = 3
class_counts = y_train.value_counts()
# Exemple : si Positive=2800, Negative=700, Neutral=700
# poids_Positive = 4200 / (3 × 2800) = 0.5
# poids_Negative = 4200 / (3 × 700)  = 2.0
# poids_Neutral  = 4200 / (3 × 700)  = 2.0

weights = {}
for cls in ['Positive', 'Negative', 'Neutral']:
    weights[cls] = total / (nb_classes * class_counts[cls])
    print(f'Poids {cls} : {weights[cls]:.4f}')

# Convertir en tenseur PyTorch (même ordre que label_map : Pos=0, Neg=1, Neu=2)
w_tensor = torch.tensor(
    [weights['Positive'], weights['Negative'], weights['Neutral']],
    dtype=torch.float
).to(device)

# Créer un CustomTrainer qui utilise CrossEntropyLoss avec les poids
from transformers import Trainer

class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Fine-tuner DziriBERT avec WeightedLossTrainer
# ⛔ MÊMES hyperparamètres que baseline
# Évaluer sur test set → sauvegarder dans results/cw_metrics.json

=== VARIANTE B : FOCAL LOSS (Énoncé §4.2) ===

L'Énoncé exige de tester DEUX valeurs : gamma=1 ET gamma=2

import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    """
    Focal Loss pour classification multi-classes.
    Réduit la contribution des exemples faciles (forte confiance)
    pour concentrer l'apprentissage sur les exemples difficiles.

    Quand gamma=0 → identique à CrossEntropy standard
    Quand gamma=1 → réduction modérée des exemples faciles
    Quand gamma=2 → forte concentration sur exemples difficiles
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # None = pas de class weighting
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.alpha, reduction='none'
        )
        pt = torch.exp(-ce_loss)  # probabilité de la classe correcte
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# CustomTrainer pour Focal Loss
class FocalLossTrainer(Trainer):
    def __init__(self, focal_loss_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = focal_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.focal_loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Expérience B1 : Focal Loss gamma=1 (SANS class weighting)
fl_g1 = FocalLoss(gamma=1.0, alpha=None)
# → Fine-tuner + évaluer → results/fl_g1_metrics.json

# Expérience B2 : Focal Loss gamma=2 (SANS class weighting)
fl_g2 = FocalLoss(gamma=2.0, alpha=None)
# → Fine-tuner + évaluer → results/fl_g2_metrics.json

# ⚠️ IMPORTANT : Recharger le modèle from scratch AVANT chaque expérience
# (ne pas continuer l'entraînement d'un modèle déjà fine-tuné)
model = AutoModelForSequenceClassification.from_pretrained(
    'alger-ia/dziribert', num_labels=3
)

=== VARIANTE C : COMBINAISON CW + FOCAL LOSS (Énoncé §4.3) ===

L'Énoncé dit : "appliquer les poids de classe ET la Focal Loss en même temps"
→ alpha = vecteur des poids de classe calculé en Variante A

fl_cw = FocalLoss(gamma=2.0, alpha=w_tensor)
# → Fine-tuner + évaluer → results/cw_fl_metrics.json

=== TABLEAU COMPARATIF (Énoncé §4.4) ===
Afficher ce tableau avec les VRAIS résultats :

| Configuration      | F1-macro | F1 Pos | F1 Neg | F1 Neu | AUC-PR | G-mean |
|--------------------|----------|--------|--------|--------|--------|--------|
| Baseline           | ...      | ...    | ...    | ...    | ...    | ...    |
| Class Weighting    | ...      | ...    | ...    | ...    | ...    | ...    |
| Focal Loss (γ=1)   | ...      | ...    | ...    | ...    | ...    | ...    |
| Focal Loss (γ=2)   | ...      | ...    | ...    | ...    | ...    | ...    |
| CW + Focal Loss    | ...      | ...    | ...    | ...    | ...    | ...    |

=== ANALYSE OBLIGATOIRE — 4 QUESTIONS (Énoncé §4.4) ===
L'Énoncé exige des RÉPONSES ÉCRITES à ces 4 questions dans le rapport :

1. "Le F1 de la classe minoritaire a-t-il augmenté par rapport à la Baseline ?
    De combien ?"
2. "Y a-t-il un compromis ? Le F1 de la classe majoritaire a-t-il baissé
    quand celui de la minoritaire a augmenté ?"
3. "Quel gamma donne les meilleurs résultats sur TWIFL ? Pouvez-vous
    expliquer pourquoi ?"
4. "Class Weighting ou Focal Loss : quelle variante est la plus efficace
    sur ce corpus ?"

→ Rédiger les réponses dans une cellule Markdown à la fin du notebook.
→ Ces réponses seront reprises dans la section Discussion du rapport.

=== VISUALISATIONS ===
1. Bar chart : F1-macro pour les 5 configurations (baseline + 4 variantes)
2. Matrices de confusion côte à côte : Baseline vs meilleure variante
3. (Optionnel) Courbes de loss par epoch pour chaque variante

=== LIVRABLES SEMAINE 3 ===
- Notebook : 03_Strategie1_Loss_Functions.ipynb
- Fichiers : results/cw_metrics.json, results/fl_g1_metrics.json,
  results/fl_g2_metrics.json, results/cw_fl_metrics.json
- Figures : matrices de confusion pour chaque variante
- Réponses aux 4 questions dans le notebook
```

#### ⚠️ PIÈGES SEMAINE 3

1. **Recharger le modèle** — AVANT chaque expérience, recharger DziriBERT from scratch. Si vous entraînez CW après la baseline sans recharger, vous faites du continual learning, pas un entraînement indépendant.
2. **Device mismatch** — `w_tensor` doit être sur le même device (GPU/CPU) que le modèle. Utiliser `.to(model.device)`.
3. **Focal Loss gradient** — La Focal Loss peut donner des gradients très petits si gamma est trop élevé. Avec gamma=2, vérifier que le modèle apprend encore (la loss doit diminuer).
4. **L'Énoncé ne demande PAS gamma=0** — Ne pas tester gamma=0 (c'est juste la CrossEntropy standard, équivalente à la baseline).

---

## SEMAINE 4 · 29 Mars - 5 Avril 2026 {#semaine-4}
### Stratégie 2 — SMOTE / ADASYN sur Embeddings

**Points visés : 2.0 / 2.0**
**GPU nécessaire : OUI (extraction embeddings ~15-20 min)**
**Durée estimée : 5-7h**
**Livrable : `04_Strategie2_SMOTE_ADASYN.ipynb` + embeddings .npy**

---

#### PROMPT SEMAINE 4

```
Nous commençons la SEMAINE 4 du projet.
Objectif : Rééquilibrer dans l'espace des embeddings [CLS] de DziriBERT.

RAPPEL CRUCIAL DE L'ÉNONCÉ (§5) :
"On ne travaille PAS directement sur le texte brut. On travaille dans
l'espace des embeddings de DziriBERT."
"Chaque tweet est représenté par un vecteur numérique de dimension 768
(le vecteur [CLS])."

=== TÂCHE 1 : EXTRACTION DES EMBEDDINGS [CLS] (Énoncé §5.1) ===

L'Énoncé exige :
- Charger DziriBERT en mode extraction de features (SANS tête de classification)
- Pour chaque tweet du train set → récupérer le vecteur [CLS]
- Stocker dans matrice (N, 768)
- ⛔ "Ne JAMAIS extraire les embeddings du val set ou du test set
  pour le rééquilibrage"
  → Vous POUVEZ extraire ceux du test set pour ÉVALUATION,
  mais JAMAIS les inclure dans le rééquilibrage SMOTE/ADASYN.

from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

model_name = 'alger-ia/dziribert'
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder = AutoModel.from_pretrained(model_name)  # PAS ForSequenceClassification
encoder.eval()
encoder.to(device)  # GPU obligatoire — CPU trop lent

def get_cls_embeddings(texts, tokenizer, encoder, batch_size=32, device='cuda'):
    """
    Extrait les vecteurs [CLS] de DziriBERT pour une liste de textes.
    Retourne un numpy array de forme (N, 768).
    IMPORTANT : Traiter par batch pour éviter les problèmes de mémoire.
    """
    all_embeddings = []
    encoder.eval()
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i+batch_size])
        inputs = tokenizer(
            batch, return_tensors='pt', padding=True,
            truncation=True, max_length=128
        ).to(device)
        with torch.no_grad():
            outputs = encoder(**inputs)
        # [CLS] = premier token = index 0
        cls_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_vectors)
        if (i // batch_size) % 10 == 0:
            print(f'  Batch {i//batch_size}/{len(texts)//batch_size}...')
    return np.vstack(all_embeddings)

# Extraire pour le TRAIN set (pour rééquilibrage)
print('Extraction embeddings TRAIN...')
train_embeddings = get_cls_embeddings(X_train.tolist(), tokenizer, encoder)
print(f'Shape train : {train_embeddings.shape}')  # (N_train, 768)

# Extraire pour le TEST set (pour évaluation UNIQUEMENT)
print('Extraction embeddings TEST...')
test_embeddings = get_cls_embeddings(X_test.tolist(), tokenizer, encoder)
print(f'Shape test : {test_embeddings.shape}')  # (N_test, 768)

# Extraire pour le VAL set (pour vérifier overfitting)
print('Extraction embeddings VAL...')
val_embeddings = get_cls_embeddings(X_val.tolist(), tokenizer, encoder)

# Encoder les labels en numérique
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_labels = le.fit_transform(y_train)
test_labels = le.transform(y_test)
val_labels = le.transform(y_val)

# Sauvegarder pour réutilisation :
np.save('data/train_embeddings.npy', train_embeddings)
np.save('data/train_labels.npy', train_labels)
np.save('data/test_embeddings.npy', test_embeddings)
np.save('data/test_labels.npy', test_labels)
np.save('data/val_embeddings.npy', val_embeddings)
np.save('data/val_labels.npy', val_labels)

=== TÂCHE 2 : VARIANTE A — SMOTE (Énoncé §5.2) ===

L'Énoncé demande : "Tester différents niveaux de rééquilibrage :
                     équilibre TOTAL ou PARTIEL"

from imblearn.over_sampling import SMOTE
from collections import Counter

print('Distribution avant SMOTE :', Counter(train_labels))

# --- TEST 1 : Équilibre TOTAL ---
smote_full = SMOTE(random_state=42, k_neighbors=5)
X_smote_full, y_smote_full = smote_full.fit_resample(train_embeddings, train_labels)
print(f'SMOTE total — Avant: {Counter(train_labels)} → Après: {Counter(y_smote_full)}')

# --- TEST 2 : Équilibre PARTIEL ---
# Les classes minoritaires sont augmentées à 70% de la classe majoritaire
majority_count = Counter(train_labels).most_common(1)[0][1]
target_partial = {
    cls: max(count, int(majority_count * 0.7))
    for cls, count in Counter(train_labels).items()
}
smote_partial = SMOTE(random_state=42, sampling_strategy=target_partial)
X_smote_partial, y_smote_partial = smote_partial.fit_resample(train_embeddings, train_labels)
print(f'SMOTE partiel — Après: {Counter(y_smote_partial)}')

# --- Entraîner un classifieur sur les embeddings rééquilibrés ---
L'Énoncé dit : "Entraîner un classifieur (MLP ou SVM)"

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# MLP sur SMOTE total
mlp_smote_full = MLPClassifier(
    hidden_layer_sizes=(256, 128), random_state=42, max_iter=300,
    early_stopping=True, validation_fraction=0.1
)
mlp_smote_full.fit(X_smote_full, y_smote_full)
y_pred_smote_full = mlp_smote_full.predict(test_embeddings)
y_proba_smote_full = mlp_smote_full.predict_proba(test_embeddings)
results_smote_full = evaluate_model(test_labels, y_pred_smote_full, y_proba_smote_full)

# MLP sur SMOTE partiel
mlp_smote_partial = MLPClassifier(
    hidden_layer_sizes=(256, 128), random_state=42, max_iter=300,
    early_stopping=True, validation_fraction=0.1
)
mlp_smote_partial.fit(X_smote_partial, y_smote_partial)
y_pred_smote_partial = mlp_smote_partial.predict(test_embeddings)
y_proba_smote_partial = mlp_smote_partial.predict_proba(test_embeddings)
results_smote_partial = evaluate_model(test_labels, y_pred_smote_partial, y_proba_smote_partial)

# (Optionnel mais recommandé) SVM aussi
svm_smote = SVC(kernel='rbf', random_state=42, probability=True)
svm_smote.fit(X_smote_full, y_smote_full)
y_pred_svm = svm_smote.predict(test_embeddings)
y_proba_svm = svm_smote.predict_proba(test_embeddings)
results_svm = evaluate_model(test_labels, y_pred_svm, y_proba_svm)

=== TÂCHE 3 : VARIANTE B — ADASYN (Énoncé §5.3) ===

L'Énoncé dit : "ADASYN concentre l'effort de génération sur les zones
de décision les plus difficiles"

from imblearn.over_sampling import ADASYN

adasyn = ADASYN(random_state=42, n_neighbors=5)
try:
    X_adasyn, y_adasyn = adasyn.fit_resample(train_embeddings, train_labels)
    print(f'ADASYN — Après: {Counter(y_adasyn)}')
except ValueError as e:
    # ADASYN peut échouer si une classe minoritaire n'a pas assez de voisins
    print(f'⚠️ ADASYN a échoué : {e}')
    print('→ Essayer avec n_neighbors=3')
    adasyn = ADASYN(random_state=42, n_neighbors=3)
    X_adasyn, y_adasyn = adasyn.fit_resample(train_embeddings, train_labels)

# Entraîner MLP sur ADASYN
mlp_adasyn = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=42, max_iter=300)
mlp_adasyn.fit(X_adasyn, y_adasyn)
y_pred_adasyn = mlp_adasyn.predict(test_embeddings)
y_proba_adasyn = mlp_adasyn.predict_proba(test_embeddings)
results_adasyn = evaluate_model(test_labels, y_pred_adasyn, y_proba_adasyn)

=== TÂCHE 4 : VISUALISATION t-SNE ===
Visualiser les embeddings AVANT et APRÈS SMOTE pour montrer l'impact.

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# AVANT rééquilibrage (subset pour rapidité)
n_vis = min(500, len(train_embeddings))
tsne_before = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d_before = tsne_before.fit_transform(train_embeddings[:n_vis])

# APRÈS SMOTE
tsne_after = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d_after = tsne_after.fit_transform(X_smote_full[:n_vis + (len(X_smote_full) - len(train_embeddings))])

# Créer figure côte-à-côte et sauvegarder
# figures/tsne_before_after_smote.png

=== ANALYSE OBLIGATOIRE — 4 QUESTIONS (Énoncé §5.4) ===
1. "Travailler dans l'espace des embeddings est-il plus efficace que de
    travailler sur le texte brut ?"
2. "Le rééquilibrage par embeddings améliore-t-il le F1-macro plus que
    la Stratégie 1 ?"
3. "Y a-t-il un risque de sur-apprentissage visible sur les courbes de
    validation ?" → Comparer F1 sur val_embeddings vs test_embeddings
4. "Quelle est l'influence du niveau de rééquilibrage choisi
    (total vs partiel) ?"

=== LIVRABLES SEMAINE 4 ===
- Notebook : 04_Strategie2_SMOTE_ADASYN.ipynb
- Fichiers : data/train_embeddings.npy, data/test_embeddings.npy
- Visualisation : figures/tsne_before_after_smote.png
- Tableau comparatif mis à jour (toutes les stratégies jusqu'ici)
- Réponses aux 4 questions dans le notebook
```

#### ⚠️ PIÈGES SEMAINE 4

1. **AutoModel vs AutoModelForSequenceClassification** — Pour extraire les embeddings, utilisez `AutoModel` (sans tête de classification). `AutoModelForSequenceClassification` n'expose pas directement le `last_hidden_state`.
2. **ADASYN peut échouer** — Si une classe minoritaire a très peu d'exemples, ADASYN échoue car il ne trouve pas assez de voisins. Prévoir un try/except et baisser `n_neighbors`.
3. **t-SNE reproductibilité** — t-SNE est stochastique. Utiliser `random_state=42` pour la reproductibilité.
4. **MLP vs SVM** — SVM avec kernel RBF est lent sur 768 dimensions. Tester d'abord MLP, puis SVM si le temps le permet.
5. **Comparaison équitable** — Cette stratégie utilise un classifieur classique (MLP/SVM) au lieu de fine-tuner DziriBERT. La comparaison avec les Stratégies 1 et 3 (qui fine-tunent DziriBERT) n'est pas strictement directe. Mentionner cette limite dans la Discussion du rapport.

---

## SEMAINE 5 · 5-12 Avril 2026 {#semaine-5}
### Stratégie 3 — Back-Translation

**Points visés : 1.5 / 1.5**
**GPU nécessaire : OUI (traduction + fine-tuning)**
**Durée estimée : 4-6h**
**Livrable : `05_Strategie3_BackTranslation.ipynb`**

---

#### PROMPT SEMAINE 5

```
Nous commençons la SEMAINE 5 du projet.
Objectif : Augmentation de données par Back-Translation sur la classe minoritaire.

L'Énoncé décrit 4 phases précises (§6.1). Les suivre dans l'ordre.

=== PHASE 1 : SÉLECTION (Énoncé §6.1) ===
"Identifier tous les tweets appartenant à la classe minoritaire dans le train set.
C'est UNIQUEMENT sur cette classe que la Back-Translation sera appliquée."

minority_class = y_train.value_counts().idxmin()
minority_texts = X_train[y_train == minority_class].tolist()
print(f'Classe minoritaire : {minority_class}')
print(f'Nombre de tweets : {len(minority_texts)}')

=== PHASE 2 : TRADUCTION DARIJA → FRANÇAIS (Énoncé §6.1) ===
"Traduire chaque tweet sélectionné de l'arabe/darija vers le français
en utilisant un système de traduction automatique (Helsinki-NLP sur HuggingFace)"

⚠️ L'Énoncé prévient : "les tweets en Arabizi peuvent poser problème car les
systèmes de traduction les reconnaissent mal."

from transformers import pipeline
import torch

translator_ar_fr = pipeline(
    'translation',
    model='Helsinki-NLP/opus-mt-ar-fr',
    device=0 if torch.cuda.is_available() else -1
)

def translate_to_french(tweets, batch_size=16):
    """Traduction arabe/darija → français avec gestion complète des erreurs."""
    translated = []
    failed_count = 0
    for i in range(0, len(tweets), batch_size):
        batch = tweets[i:i+batch_size]
        try:
            results = translator_ar_fr(batch, max_length=256, truncation=True)
            translated.extend([r['translation_text'] for r in results])
        except Exception as e:
            print(f'[WARNING] Batch {i//batch_size} échoué : {e}')
            # Placeholders pour maintenir l'alignement avec les originaux
            translated.extend(['[TRADUCTION_ECHOUEE]'] * len(batch))
            failed_count += len(batch)
    print(f'Traductions réussies : {len(tweets) - failed_count}/{len(tweets)}')
    print(f'Traductions échouées : {failed_count}/{len(tweets)}')
    return translated

french_translations = translate_to_french(minority_texts)

# Filtrer les traductions échouées
valid_pairs = [
    (orig, fr) for orig, fr in zip(minority_texts, french_translations)
    if fr != '[TRADUCTION_ECHOUEE]' and len(fr.strip()) > 5
]
print(f'Paires valides : {len(valid_pairs)} / {len(minority_texts)}')

# Afficher 10 exemples (original → français) pour le rapport
for i, (orig, fr) in enumerate(valid_pairs[:10]):
    print(f'\n--- Exemple {i+1} ---')
    print(f'  Original : {orig}')
    print(f'  Français : {fr}')

=== PHASE 3 : RETRADUCTION FRANÇAIS → ARABE (Énoncé §6.1) ===

translator_fr_ar = pipeline(
    'translation',
    model='Helsinki-NLP/opus-mt-fr-ar',
    device=0 if torch.cuda.is_available() else -1
)

def translate_back_to_arabic(french_texts, batch_size=16):
    """Retraduction français → arabe avec gestion des erreurs."""
    back_translated = []
    failed_count = 0
    for i in range(0, len(french_texts), batch_size):
        batch = french_texts[i:i+batch_size]
        try:
            results = translator_fr_ar(batch, max_length=256, truncation=True)
            back_translated.extend([r['translation_text'] for r in results])
        except Exception as e:
            print(f'[WARNING] Batch {i//batch_size} échoué : {e}')
            back_translated.extend(['[RETRADUCTION_ECHOUEE]'] * len(batch))
            failed_count += len(batch)
    print(f'Retraductions réussies : {len(french_texts) - failed_count}/{len(french_texts)}')
    return back_translated

originals_valid = [p[0] for p in valid_pairs]
french_valid = [p[1] for p in valid_pairs]
paraphrases_ar = translate_back_to_arabic(french_valid)

# Afficher 10 exemples du pipeline complet
for i in range(min(10, len(originals_valid))):
    print(f'\n--- Exemple {i+1} ---')
    print(f'  Original  : {originals_valid[i]}')
    print(f'  Français  : {french_valid[i]}')
    print(f'  Paraphrase: {paraphrases_ar[i]}')

=== PHASE 4 : FILTRAGE PAR SIMILARITÉ COSINUS (Énoncé §6.1) ===

L'Énoncé est PRÉCIS sur les seuils :
"Conserver UNIQUEMENT les paraphrases avec une similarité comprise entre 0.5 et 0.85"
- Trop similaire (> 0.85) → pas d'apport nouveau
- Trop différente (< 0.5) → le sens a été perdu

from sklearn.metrics.pairwise import cosine_similarity

def filter_paraphrases(originals, paraphrases, tokenizer, encoder,
                       min_sim=0.5, max_sim=0.85):
    """
    Filtre les paraphrases par similarité cosinus avec DziriBERT.
    Retourne les paraphrases conservées + statistiques.
    """
    kept, rejected_similar, rejected_different = [], [], []
    similarities = []

    for orig, para in zip(originals, paraphrases):
        if para in ['[TRADUCTION_ECHOUEE]', '[RETRADUCTION_ECHOUEE]']:
            continue
        if len(para.strip()) < 3:
            continue

        emb_orig = get_cls_embeddings([orig], tokenizer, encoder)
        emb_para = get_cls_embeddings([para], tokenizer, encoder)
        sim = cosine_similarity(emb_orig, emb_para)[0][0]

        if min_sim <= sim <= max_sim:
            kept.append({'original': orig, 'paraphrase': para, 'similarity': sim})
        elif sim > max_sim:
            rejected_similar.append({'original': orig, 'paraphrase': para, 'similarity': sim})
        else:
            rejected_different.append({'original': orig, 'paraphrase': para, 'similarity': sim})

    total = len(kept) + len(rejected_similar) + len(rejected_different)
    print(f'\n=== RÉSULTATS DU FILTRAGE ===')
    print(f'Total paraphrases traitées : {total}')
    print(f'Conservées [0.5-0.85]      : {len(kept)} ({len(kept)/total*100:.1f}%)')
    print(f'Rejetées (trop similaires)  : {len(rejected_similar)} ({len(rejected_similar)/total*100:.1f}%)')
    print(f'Rejetées (sens perdu)       : {len(rejected_different)} ({len(rejected_different)/total*100:.1f}%)')

    return kept, rejected_similar, rejected_different

kept, rej_sim, rej_diff = filter_paraphrases(
    originals_valid, paraphrases_ar, tokenizer, encoder
)

# Montrer des exemples de CHAQUE catégorie (pour le rapport)
print('\n--- Exemples CONSERVÉS ---')
for ex in kept[:3]:
    print(f'  Original: {ex["original"]}')
    print(f'  Paraphrase: {ex["paraphrase"]}')
    print(f'  Similarité: {ex["similarity"]:.3f}\n')

print('--- Exemples REJETÉS (trop similaires) ---')
for ex in rej_sim[:2]:
    print(f'  Similarité: {ex["similarity"]:.3f}')

print('--- Exemples REJETÉS (sens perdu) ---')
for ex in rej_diff[:2]:
    print(f'  Similarité: {ex["similarity"]:.3f}')

=== ÉTAPE 5 : INJECTION ET RÉÉVALUATION — 3 TAUX (Énoncé §6.2) ===

L'Énoncé exige de tester : +20%, +50%, +100%

kept_paraphrases = [k['paraphrase'] for k in kept]
N_orig = len(minority_texts)

augmentation_rates = {
    '20pct': 0.20,
    '50pct': 0.50,
    '100pct': 1.00
}

for rate_name, rate in augmentation_rates.items():
    n_add = min(int(N_orig * rate), len(kept_paraphrases))
    print(f'\n=== Back-Translation +{int(rate*100)}% ===')
    print(f'Paraphrases demandées : {int(N_orig * rate)}')
    print(f'Paraphrases disponibles : {len(kept_paraphrases)}')
    print(f'Paraphrases ajoutées : {n_add}')

    if n_add == 0:
        print('⚠️ Pas assez de paraphrases — documenter dans le rapport')
        continue

    paraphrases_to_add = kept_paraphrases[:n_add]

    # Construire train set augmenté
    import pandas as pd
    X_aug = pd.concat([
        X_train,
        pd.Series(paraphrases_to_add)
    ]).reset_index(drop=True)

    y_aug = pd.concat([
        y_train,
        pd.Series([minority_class] * n_add)
    ]).reset_index(drop=True)

    print(f'Train set augmenté : {len(X_aug)} (original : {len(X_train)})')
    print(f'Distribution : {y_aug.value_counts().to_dict()}')

    # Re-fine-tuner DziriBERT sur le train set augmenté
    # ⛔ MÊMES hyperparamètres : epochs=5, lr=2e-5, batch=16, seed=42
    # ⛔ Recharger le modèle from scratch AVANT chaque run
    model = AutoModelForSequenceClassification.from_pretrained(
        'alger-ia/dziribert', num_labels=3
    )
    # ... (même code de fine-tuning que la baseline)

    # Évaluer sur le TEST SET FIGÉ (jamais modifié)
    # Sauvegarder : results/bt_{rate_name}_metrics.json

=== ANALYSE OBLIGATOIRE — 4 QUESTIONS (Énoncé §6.3) ===
1. "Quel pourcentage des paraphrases générées passent le filtre [0.5-0.85] ?"
2. "Les paraphrases générées sont-elles en darija ou en arabe standard ?
    Discutez cette dérive."
    → Helsinki-NLP est entraîné sur arabe standard (MSA).
    → Les paraphrases seront probablement en MSA, pas en darija.
    → C'est une LIMITE IMPORTANTE à discuter dans le rapport.
3. "Le taux d'augmentation influence-t-il les performances ?
    Quel taux est optimal ?"
4. "Avez-vous observé des paraphrases où le sentiment a changé lors de la
    traduction ? Donnez des exemples."
    → Chercher des cas où un tweet négatif donne une paraphrase neutre/positive

=== LIVRABLES SEMAINE 5 ===
- Notebook : 05_Strategie3_BackTranslation.ipynb
- Fichier : data/augmented_train_100pct.json
- 3+ exemples annotés (original → français → paraphrase + similarité cosinus)
- Statistiques de filtrage (% conservé / rejeté)
- Réponses aux 4 questions dans le notebook
```

#### ⚠️ PIÈGES SEMAINE 5

1. **Helsinki-NLP produit du MSA, pas du darija** — C'est un fait attendu et prévisible. Ne pas le présenter comme un bug mais comme une limite à analyser. L'Énoncé demande explicitement de "discuter cette dérive".
2. **Pas assez de paraphrases après filtrage** — Si seulement 30% passent le filtre [0.5, 0.85], vous n'aurez peut-être pas assez pour atteindre +100%. Documenter le taux réel atteint.
3. **Arabizi → traduction échouée** — Les tweets en Arabizi (ex: "wch rak khouya") seront mal traduits par Helsinki-NLP car ce n'est ni de l'arabe ni du français. Le try/except est essentiel.
4. **Extraction embeddings pour le filtrage** — Vous avez besoin de DziriBERT SANS tête de classification (AutoModel) pour calculer la similarité cosinus. C'est le même encoder que la Semaine 4.

---

## SEMAINE 6 · 12-26 Avril 2026 {#semaine-6}
### Évaluation Finale + Rapport + Soutenance

**Points visés : 2.0 (Rapport/Discussion) + 1.0 (Code) + 1.5 (Soutenance) = 4.5 / 15**
**Durée estimée : 10-15h (rédaction + préparation)**
**Livrable : `06_Evaluation_Finale.ipynb` + Rapport PDF + Slides PowerPoint**

---

#### PROMPT SEMAINE 6

```
Nous commençons la SEMAINE 6 — dernière semaine.
Objectif : Consolider résultats + Rédiger rapport + Préparer soutenance.

=== TÂCHE 1 : TABLEAU COMPARATIF FINAL (Énoncé §7) ===
L'Énoncé exige CE TABLEAU EXACT rempli avec vos résultats réels :

| Configuration              | F1-macro | F1 Pos | F1 Neg | F1 Neu | AUC-PR | G-mean |
|---------------------------|----------|--------|--------|--------|--------|--------|
| Baseline (données brutes)  |          |        |        |        |        |        |
| Class Weighting            |          |        |        |        |        |        |
| Focal Loss (gamma=1)       |          |        |        |        |        |        |
| Focal Loss (gamma=2)       |          |        |        |        |        |        |
| CW + Focal Loss            |          |        |        |        |        |        |
| SMOTE total + MLP          |          |        |        |        |        |        |
| SMOTE partiel + MLP        |          |        |        |        |        |        |
| ADASYN + MLP               |          |        |        |        |        |        |
| Back-Translation (+20%)    |          |        |        |        |        |        |
| Back-Translation (+50%)    |          |        |        |        |        |        |
| Back-Translation (+100%)   |          |        |        |        |        |        |

Créer aussi ces visualisations :
1. Grouped bar chart : F1-macro pour TOUTES les configurations
2. Grouped bar chart : F1 par classe pour toutes les configurations
3. Matrices de confusion : Baseline vs meilleure stratégie (côte à côte)
4. (Si possible) Courbes Precision-Rappel pour les 3 meilleures stratégies

Sauvegarder le tableau en CSV : results/tableau_comparatif_final.csv

=== TÂCHE 2 : RAPPORT PDF — 15-20 PAGES (Énoncé §8) ===
Structure EXACTE imposée par l'Énoncé :

1. INTRODUCTION ET PROBLÉMATIQUE (1-2 pages)
   - Pourquoi le déséquilibre de classes est un problème en ML/NLP
   - Défis spécifiques aux langues peu dotées (darija algérien)
   - Objectifs de ce travail

2. ÉTAT DE L'ART (2-3 pages)
   - Déséquilibre de classes en NLP : approches data-level vs algorithm-level
   - TALN pour le dialecte algérien : travaux existants
   - DziriBERT : architecture, entraînement, capacités
   - Méthodes utilisées : SMOTE, ADASYN, Focal Loss, Back-Translation
   - Références : utiliser les 10 références IEEE de la Section 11

3. PRÉSENTATION DU CORPUS TWIFL (2-3 pages)
   - Description du dataset et source
   - Résultats EDA : distribution des classes, ratio déséquilibre
   - Analyse linguistique : langues, emojis, code-switching
   - Tableau de statistiques obligatoire (Énoncé §1.3)
   - Visualisations (WordClouds, distributions)
   - Observations sur la qualité des données

4. MÉTHODOLOGIE (3-4 pages)
   - Pipeline de prétraitement (avec justification de chaque choix)
   - Protocole expérimental (split, hyperparamètres, métriques)
   - Description de la Baseline
   - Stratégie 1 : Class Weighting + Focal Loss + Combinaison
   - Stratégie 2 : Extraction embeddings + SMOTE + ADASYN
   - Stratégie 3 : Back-Translation pipeline complet

5. RÉSULTATS (3-4 pages)
   - Tableau comparatif final (avec commentaire de chaque ligne)
   - Matrices de confusion (au moins Baseline vs meilleure stratégie)
   - Courbes Precision-Rappel
   - Analyse par classe : quelle stratégie bénéficie à quelle classe

6. DISCUSSION (2-3 pages)
   ⚠️ C'est ici que les 2 points "Discussion et analyse critique" se jouent.
   Répondre aux questions de CHAQUE stratégie (§4.4, §5.4, §6.3) :
   - Stratégie 1 : 4 questions (voir Semaine 3)
   - Stratégie 2 : 4 questions (voir Semaine 4)
   - Stratégie 3 : 4 questions (voir Semaine 5)
   + Analyse globale :
   - Quelle stratégie fonctionne le mieux sur TWIFL et pourquoi ?
   - Les spécificités du darija compliquent-elles certaines approches ?
   - Limites de chaque stratégie
   - Que feriez-vous différemment ?

7. CONCLUSION ET PERSPECTIVES (1 page)
   - Bilan : meilleure stratégie identifiée
   - Perspectives : modèles multilingues, augmentation de données plus avancée,
     corpus plus large, etc.

8. RÉFÉRENCES BIBLIOGRAPHIQUES (format IEEE)
   → Utiliser les références pré-formatées de la Section 11 de ce guide

=== TÂCHE 3 : PLAN DES 15 SLIDES ===
Slide 1 : Titre + noms + Dr. Soraya Cheriguene + date
Slide 2 : Problématique (déséquilibre de classes + spécificités darija)
Slide 3 : Corpus TWIFL (distribution + chiffres-clés)
Slide 4 : Architecture DziriBERT + protocole expérimental
Slide 5 : Résultats Baseline — matrice confusion + biais démontré
Slide 6 : Stratégie 1 — Class Weighting (résultats vs baseline)
Slide 7 : Stratégie 1 — Focal Loss γ=1 vs γ=2
Slide 8 : Stratégie 2 — Principe embeddings [CLS] + t-SNE
Slide 9 : Stratégie 2 — SMOTE vs ADASYN (résultats)
Slide 10 : Stratégie 3 — Pipeline Back-Translation + exemples
Slide 11 : Stratégie 3 — Impact des 3 taux d'augmentation
Slide 12 : Tableau comparatif final (TOUTES les stratégies)
Slide 13 : Discussion — Quelle stratégie gagne et pourquoi ?
Slide 14 : Limites et perspectives
Slide 15 : Conclusion + remerciements

=== TÂCHE 4 : CODE CLEANUP ===
- requirements.txt complet avec versions exactes
- README.md avec : description, installation, structure du repo, exécution
- Vérifier que random_state=42 est PARTOUT
- Vérifier que le test set n'a JAMAIS été modifié
- Vérifier que l'accuracy n'est JAMAIS présentée comme métrique principale
- Structure recommandée du repo GitHub :
  mini-projet-darija/
  ├── README.md
  ├── requirements.txt
  ├── notebooks/
  │   ├── 01_EDA.ipynb
  │   ├── 02_Preprocessing_Baseline.ipynb
  │   ├── 03_Strategie1_Loss_Functions.ipynb
  │   ├── 04_Strategie2_SMOTE_ADASYN.ipynb
  │   ├── 05_Strategie3_BackTranslation.ipynb
  │   └── 06_Evaluation_Finale.ipynb
  ├── data/
  │   ├── split_indices.json
  │   ├── train_embeddings.npy
  │   └── test_embeddings.npy
  ├── models/
  │   └── baseline_dziribert/
  ├── results/
  │   ├── baseline_metrics.json
  │   ├── cw_metrics.json
  │   └── tableau_comparatif_final.csv
  ├── figures/
  │   ├── class_distribution_bar.png
  │   ├── baseline_confusion_matrix.png
  │   └── ...
  └── rapport/
      ├── rapport_final.pdf
      └── soutenance.pptx
```

---

## BONUS — Stratégie Hybride (+2 pts) {#bonus}

L'Énoncé (§9, Bonus) : "Les groupes qui implémentent une stratégie hybride originale peuvent obtenir +2 pts. La contribution doit être clairement documentée et analysée dans le rapport."

#### PROMPT BONUS

```
=== BONUS : STRATÉGIE HYBRIDE (+2 pts) ===

OPTION A (RECOMMANDÉE) : Class Weighting + SMOTE sur Embeddings
Cette option combine une approche data-level (SMOTE) et algorithm-level (CW).

1. Extraire les embeddings [CLS] du train set (déjà fait en Semaine 4)
2. Appliquer SMOTE (équilibre partiel) sur les embeddings
3. Sur les embeddings rééquilibrés, entraîner un MLP avec
   class_weighted loss (ou utiliser sample_weight dans .fit())
4. Évaluer sur test embeddings
5. Comparer avec SMOTE seul et CW seul

Analyse pour le rapport :
- La combinaison surpasse-t-elle chaque stratégie individuellement ?
- Y a-t-il un risque de "double correction" (sur-corriger le déséquilibre) ?

OPTION B : Analyse Comparative des Ratios d'Augmentation
1. Tester 5+ ratios de rééquilibrage pour la meilleure stratégie :
   ratios = [0.10, 0.20, 0.30, 0.50, 0.75, 1.00]
2. Pour chaque ratio, appliquer la stratégie et mesurer le F1-macro
3. Tracer la courbe F1-macro = f(ratio)
4. Identifier le ratio optimal
5. Discuter : existe-t-il un point de sur-augmentation ?

=== INTÉGRATION AU RAPPORT (obligatoire pour les 2 pts) ===
Le bonus DOIT apparaître dans :
- Section Méthodologie : description de la stratégie hybride
- Section Résultats : résultats dans le tableau comparatif
- Section Discussion : analyse critique
```

---

## RÉFÉRENCES IEEE PRÉ-FORMATÉES {#references}

Prêtes à copier dans votre rapport :

```
[1] A. Dossou et al., "DziriBERT: a Pre-trained Language Model for the Algerian
    Dialect," in Proceedings of the 3rd Workshop on African Natural Language
    Processing (AfricaNLP), 2022.

[2] Y. Boutaleb et al., "TWIFL: An Algerian Corpus and an Annotation Platform
    for Opinion and Emotion Analysis," in Proc. LREC-COLING, 2024.

[3] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE:
    Synthetic Minority Over-sampling Technique," Journal of Artificial
    Intelligence Research, vol. 16, pp. 321–357, 2002.

[4] T. Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, "Focal Loss for
    Dense Object Detection," in Proc. IEEE ICCV, 2017, pp. 2980–2988.

[5] H. He, Y. Bai, E. A. Garcia, and S. Li, "ADASYN: Adaptive Synthetic
    Sampling Approach for Imbalanced Learning," in Proc. IEEE IJCNN, 2008,
    pp. 1322–1328.

[6] S. Edunov, M. Ott, M. Auli, and D. Grangier, "Understanding
    Back-Translation at Scale," in Proc. EMNLP, 2018, pp. 489–500.

[7] J. Devlin, M. W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding," in Proc.
    NAACL-HLT, 2019, pp. 4171–4186.

[8] G. Haixiang et al., "Learning from class-imbalanced data: Review of methods
    and applications," Expert Systems with Applications, vol. 73, pp. 220–239,
    2017.

[9] J. Tiedemann and S. Thottingal, "OPUS-MT — Building open translation
    services for the World," in Proc. EAMT, 2020.

[10] J. Davis and M. Goadrich, "The relationship between Precision-Recall and
     ROC curves," in Proc. ICML, 2006, pp. 233–240.
```

---

## QUESTIONS PROBABLES EN SOUTENANCE (1.5 pts) {#soutenance-qa}

Préparer des réponses claires pour ces questions :

**Q1 : Pourquoi ne pas utiliser l'accuracy comme métrique principale ?**
> Sur des données déséquilibrées, un modèle qui prédit toujours la classe majoritaire obtient une accuracy élevée (ex: 65%) sans rien apprendre. Le F1-macro pénalise ce comportement car il fait la moyenne non-pondérée du F1 de chaque classe.

**Q2 : Quelle est la différence entre SMOTE et ADASYN ?**
> SMOTE génère des points synthétiques uniformément entre des voisins de la classe minoritaire. ADASYN adapte la densité de génération : il génère plus d'exemples dans les zones frontalières (difficiles) et moins dans les zones bien séparées.

**Q3 : Pourquoi le test set ne doit jamais être rééquilibré ?**
> Le test set doit refléter la distribution réelle des données en production. Si on le rééquilibre, on évalue le modèle sur une distribution artificielle qui ne correspond pas à la réalité. Les métriques seraient faussement optimistes.

**Q4 : Pourquoi Helsinki-NLP produit de l'arabe standard et non du darija ?**
> Helsinki-NLP (OPUS-MT) est entraîné principalement sur des corpus parallèles en arabe standard (MSA) : textes de l'ONU, articles de presse, sous-titres. Le darija algérien n'est quasiment pas représenté dans les données d'entraînement. La retraduction "standardise" donc le dialecte.

**Q5 : Qu'est-ce que le paramètre gamma dans la Focal Loss ?**
> Gamma contrôle l'intensité de la réduction de la contribution des exemples faciles. Avec gamma=0, c'est une CrossEntropy standard. Avec gamma=2, les exemples classifiés avec >90% de confiance contribuent presque 0 à la perte, forçant le modèle à se concentrer sur les cas difficiles (souvent les classes minoritaires).

**Q6 : Comment avez-vous choisi le seuil [0.5, 0.85] pour le filtrage cosinus ?**
> C'est le seuil imposé par l'Énoncé. La logique : en dessous de 0.5, le sens a trop changé lors de la traduction (risque de flip de sentiment). Au-dessus de 0.85, la paraphrase est trop proche de l'original et n'apporte pas de diversité au modèle.

**Q7 : Pourquoi travailler dans l'espace des embeddings plutôt que sur le texte brut ?**
> Le darija est une langue complexe avec du code-switching et de l'Arabizi. Générer du texte cohérent par interpolation est impossible. Dans l'espace des embeddings (768 dimensions continues), l'interpolation linéaire entre deux vecteurs de même classe produit un vecteur sémantiquement cohérent, car les embeddings DziriBERT sont entraînés pour capturer la sémantique.

---

## CHECKLIST FINALE — J-1 AVANT SOUTENANCE {#checklist}

### Contraintes Dures (⛔ si violée = pénalisation)
- [ ] Test set intact : JAMAIS modifié, JAMAIS rééquilibré
- [ ] Val set : JAMAIS rééquilibré (utilisé uniquement pour hyperparamètres)
- [ ] random_state=42 PARTOUT dans le code
- [ ] Modèle : UNIQUEMENT DziriBERT (alger-ia/dziribert)
- [ ] Hyperparamètres : epochs=5, lr=2e-5, batch_size=16, seed=42, AdamW
- [ ] Accuracy JAMAIS présentée comme métrique principale
- [ ] Polarity Class validée (pas de valeurs inattendues)

### Métriques (⛔ si manquante)
- [ ] F1-macro calculé pour CHAQUE configuration
- [ ] F1 par classe (Positive, Negative, Neutral) pour chaque configuration
- [ ] Precision et Rappel par classe
- [ ] AUC-PR calculé
- [ ] G-mean calculé
- [ ] Matrice de confusion affichée pour chaque configuration

### Stratégies (vérifier complétude)
- [ ] Baseline : fine-tuning sur données brutes + métriques + analyse biais
- [ ] Class Weighting : formule correcte + métriques
- [ ] Focal Loss gamma=1 : métriques
- [ ] Focal Loss gamma=2 : métriques
- [ ] CW + Focal Loss : métriques
- [ ] SMOTE total : métriques
- [ ] SMOTE partiel : métriques
- [ ] ADASYN : métriques
- [ ] Back-Translation +20% : métriques
- [ ] Back-Translation +50% : métriques
- [ ] Back-Translation +100% : métriques
- [ ] Tableau comparatif COMPLET avec toutes les configurations

### Rapport (vérifier structure)
- [ ] Introduction (1-2 pages)
- [ ] État de l'art (2-3 pages)
- [ ] Corpus TWIFL (2-3 pages) avec EDA + tableau statistiques
- [ ] Méthodologie (3-4 pages) avec description de chaque stratégie
- [ ] Résultats (3-4 pages) avec tableau + matrices + courbes
- [ ] Discussion (2-3 pages) avec réponses aux 12 questions d'analyse
- [ ] Conclusion (1 page)
- [ ] Références (IEEE ou APA)
- [ ] Total : 15-20 pages (hors annexes)

### Code et Livrables
- [ ] 6 notebooks Jupyter commentés et exécutables bout en bout
- [ ] requirements.txt avec versions exactes
- [ ] README.md avec instructions de reproduction
- [ ] Dépôt GitHub organisé
- [ ] Présentation ≤ 15 slides
- [ ] Rapport PDF

### Bonus (si implémenté)
- [ ] Stratégie hybride documentée dans Méthodologie
- [ ] Résultats dans le tableau comparatif
- [ ] Analyse dans la Discussion

---

## RECOMMANDATIONS MODÈLES IA & GPU {#recommandations}

### Pour Coller les Prompts et Générer le Code

| Modèle | Force | Meilleur pour | Accès |
|---|---|---|---|
| **Claude Sonnet 4.6** | Code Python/PyTorch excellent, explications détaillées, correction d'erreurs | Prompts S1-S6, débogage, rédaction rapport | claude.ai (gratuit limité) |
| **Claude Opus 4.6** | Raisonnement le plus avancé, analyse complexe | Analyse critique, Discussion du rapport, questions difficiles | claude.ai (Pro) |
| **Gemini 2.5 Pro** | Contexte très long, intégration Colab | Code ML complexe S3-S5, utilisation directe dans Colab | gemini.google.com (gratuit) |
| **DeepSeek-V3** | Code fort, gratuit, pas de géo-bloc | Alternative si limites Claude/Gemini atteintes | platform.deepseek.com |
| **GitHub Copilot** | Autocomplétion temps réel VS Code | Écrire le code directement | Gratuit (GitHub Student Pack) |

### Workflow Recommandé
1. Coller le Master Prompt dans Claude ou Gemini
2. Coller le prompt hebdomadaire
3. Copier le code généré dans Google Colab
4. Exécuter sur GPU
5. Si erreur → copier le traceback et demander à l'IA de corriger

### Plateformes GPU

| Plateforme | GPU | Quand l'utiliser | Limite |
|---|---|---|---|
| **Google Colab** | T4 16GB | Fine-tuning DziriBERT (S2-S5) | ~12h/session |
| **Kaggle Notebooks** | P100 16GB | Extraction embeddings (S4), runs longs | 30h/semaine |
| **Lightning AI** | A10G 24GB | Back-Translation + runs longs (S5) | 22h/mois |

### Persistance entre Sessions
- Sauvegarder les modèles sur HuggingFace Hub (`model.push_to_hub(...)`)
- Sauvegarder les embeddings .npy sur Google Drive
- Sauvegarder les splits dans `data/split_indices.json`

---

*Ce guide a été construit par analyse croisée exhaustive de l'Énoncé officiel, du GUIDE_PARFAIT v2.0, et du mini_projet_guide_prompts. Chaque instruction a été vérifiée contre les exigences de l'Énoncé. Les éléments manquants dans les deux guides ont été ajoutés. Les erreurs ont été corrigées. Ce document est la version la plus complète et la plus alignée possible avec les attentes de l'enseignante.*
