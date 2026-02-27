# htr_active_learning.py

Active learning pour l'HTR : identifie les styles d'écriture difficiles dans
un jeu de test transcrit, puis classe un stock de pages non transcrites par
ordre d'utilité pour l'entraînement.

---

## Principe

```
Jeu de test (ALTO + images)
        │
        ▼
  [analyze] ── Clustering par style ──► Quels styles sont mal reconnus ?
        │
        ▼
  cluster_model.pkl + cer_per_page.json
        │
        ▼
  [rank] ── Stock d'images brutes ──► Dans quel ordre les transcrire ?
        │
        ▼
  ranked_pages.txt + rapport HTML
```

**Étape 1 — `analyze`** :
- Binarise chaque page (Otsu), extrait des features visuelles (histogramme + géométrie des lignes)
- Regroupe les pages par style d'écriture similaire (PCA + k-means, k optimal automatique)
- Calcule le CER de chaque page avec `ketos test`
- Produit un rapport HTML interactif + deux fichiers intermédiaires

**Étape 2 — `rank`** :
- Binarise les images brutes du stock (Otsu), les segmente avec `kraken segment`
- Projette chaque image dans l'espace PCA calculé à l'étape 1
- Score d'utilité = CER moyen du cluster le plus proche + bonus diversité
- Produit une liste ordonnée des pages à transcrire en priorité

---

## Installation

```bash
pip install -r requirements.txt
```

Télécharger le modèle de segmentation kraken par défaut :

```bash
# Depuis https://github.com/mittagessen/kraken/blob/main/kraken/blla.mlmodel
# Placer blla.mlmodel dans le répertoire de travail
```

---

## Usage

### Étape 1 — Analyser le jeu de test

```bash
python htr_active_learning.py analyze \
    --test_txt   test.txt \
    --model      fondue_gd_fr_v3.mlmodel \
    --output_dir analysis/
```

**Fichiers produits dans `analysis/` :**

| Fichier | Description |
|---|---|
| `rapport_clusters.html` | Rapport interactif (scatter PCA, CER par cluster, silhouette) |
| `cluster_model.pkl` | Modèle PCA/clusters — requis pour `rank` |
| `cer_per_page.json` | CER détaillé par page — requis pour `rank` |

**Options :**

| Option | Défaut | Description |
|---|---|---|
| `--test_txt` | — | Fichier `.txt` listant les fichiers ALTO du jeu de test (un par ligne) |
| `--model` | — | Modèle ketos (`.mlmodel`) utilisé pour `ketos test` |
| `--output_dir` | `analysis/` | Dossier de sortie |
| `--k_max` | `10` | Nombre maximum de clusters à tester |
| `--skip_cer` | false | Sauter le calcul du CER (plus rapide, pas de rapport de performance) |
| `--use_kraken` | false | Features extraites du modèle ketos (plus lent, plus précis) |

---

### Étape 2 — Classer le stock d'images à transcrire

```bash
python htr_active_learning.py rank \
    --images_dir stock/ \
    --output_dir analysis/ \
    --top_n      50
```

`--output_dir` doit être le même que pour `analyze` : le script y cherche
automatiquement `cluster_model.pkl` et `cer_per_page.json`.

**Fichiers produits :**

| Fichier | Description |
|---|---|
| `rapport_recommandations.html` | Rapport interactif avec projection PCA et tableau de classement |
| `ranked_pages.txt` | Liste ordonnée des chemins d'images (une par ligne, du plus utile au moins utile) |
| `segmented/` | Fichiers JSON de segmentation kraken |
| `segmented/binarized/` | Images binarisées (PNG) |

**Options :**

| Option | Défaut | Description |
|---|---|---|
| `--images_dir` | — | Dossier contenant les images brutes à classer |
| `--output_dir` | `analysis/` | Dossier contenant `cluster_model.pkl` et `cer_per_page.json` |
| `--cluster_model` | `<output_dir>/cluster_model.pkl` | Chemin explicite vers le `.pkl` |
| `--cer_json` | `<output_dir>/cer_per_page.json` | Chemin explicite vers le JSON de CER |
| `--top_n` | `50` | Nombre de pages à retenir dans le classement |
| `--diversity` | `0.3` | Coefficient de diversité `0..1` — `0` = cibler uniquement les clusters difficiles, `1` = maximiser la diversité des styles |
| `--seg_model` | `blla.mlmodel` | Modèle de segmentation kraken |
| `--skip_segment` | false | Ne pas segmenter, utiliser les JSON existants dans `images_dir` |

---

## Le coefficient de diversité

Sans diversité (`--diversity 0`), le classement recommanderait des dizaines
de pages au style quasi-identique appartenant au cluster le plus difficile.

Avec diversité, un bonus est accordé aux pages éloignées des pages déjà
sélectionnées dans l'espace PCA, ce qui garantit une couverture plus large
des styles difficiles.

Valeurs conseillées : `0.2`–`0.5`. Au-delà, la diversité l'emporte sur la
performance et des pages de clusters faciles peuvent remonter dans le classement.

---

## Format des fichiers d'entrée

### `test.txt` (et fichiers similaires)

Un chemin absolu vers un fichier ALTO par ligne. L'image correspondante doit
se trouver dans le même dossier avec le même nom de fichier (`.jpg`, `.jpeg`,
`.png`, `.tif` ou `.tiff`).

```
/data/nestor/test/page_001.xml
/data/nestor/test/page_002.xml
...
```

### Images du stock (`--images_dir`)

Dossier plat contenant les images brutes (`.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`).
Pas besoin de fichiers ALTO — la segmentation est faite automatiquement.

---

## Dépendances

- `kraken` — segmentation et test HTR (installé séparément via pip)
- `Pillow` — traitement d'image
- `scikit-learn` — PCA, k-means, silhouette score
- `numpy` — calcul numérique

Voir `requirements.txt` pour les versions exactes.

---

## Exemple de workflow complet

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Analyser le jeu de test (avec CER, ~quelques minutes)
python htr_active_learning.py analyze \
    --test_txt   test.txt \
    --model      fondue_gd_fr_v3.mlmodel \
    --output_dir analysis/

# 3. Ouvrir le rapport pour inspecter les clusters
open analysis/rapport_clusters.html

# 4. Classer le stock (segmentation automatique)
python htr_active_learning.py rank \
    --images_dir /data/stock/ \
    --output_dir analysis/ \
    --top_n 100 \
    --diversity 0.3

# 5. Consulter les recommandations
open analysis/rapport_recommandations.html
cat analysis/ranked_pages.txt | head -20

# 6. Optionnel : analyser sans CER (rapide, pour tester)
python htr_active_learning.py analyze \
    --test_txt test.txt \
    --model fondue_gd_fr_v3.mlmodel \
    --skip_cer

# 7. Optionnel : utiliser un modèle de segmentation custom
python htr_active_learning.py rank \
    --images_dir /data/stock/ \
    --output_dir analysis/ \
    --seg_model mon_modele_seg.mlmodel
```
