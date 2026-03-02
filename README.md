# htr_active_learning.py

Active learning for HTR: identifies difficult handwriting styles in a
transcribed test set, then ranks an untranscribed image stock by order of
usefulness for training.

---

## Rationale

The model performs poorly on certain handwriting styles. The goal is to find,
in a stock of untranscribed pages, those that most resemble these difficult
styles — so they can be transcribed first and used to improve training data
where it matters most.

```
Test set (ALTO + transcribed images)
        │
        ▼
  [analyze] → clustering by handwriting style + CER per page
        │
        ▼
  cluster_model.pkl    ← PCA reference space
  cer_per_page.json    ← measured CER per page
        │
        ▼
  [rank] → untranscribed images projected into PCA space
        │
        ▼
  ranked_pages.txt     ← pages to transcribe first
  rapport_recommandations.html
```

**Step 1 — `analyze`** :
- Binarizes each page (Otsu thresholding, pure numpy), extracts visual features
  (greyscale histogram + line geometry)
- Groups pages by similar handwriting style using PCA + k-means;
  the number of clusters k is chosen automatically (best silhouette score over k=2..k_max)
- Computes the real CER for each page using `ketos test`
- Produces an interactive HTML report + two intermediate files for `rank`

**Step 2 — `rank`** :
- Binarizes stock images (same Otsu pipeline), segments them with
  `kraken segment -bl` to extract line geometry
- Projects each image into the PCA space computed in step 1
- Assigns each image an **estimated CER** = mean CER of the nearest reference
  cluster in PCA space. This is not a measured CER (there is no reference
  transcription) but a proxy estimate: "this page looks like the pages in
  cluster X, which had a mean CER of Y%"
- Ranks by descending score with a diversity bonus to avoid recommending
  dozens of near-identical pages
- Produces an ordered list + HTML report

---

## Installation

```bash
pip install -r requirements.txt
```

`kraken` must be installed in the existing environment. Download the default
segmentation model (`blla.mlmodel`) from:
https://github.com/mittagessen/kraken/blob/main/kraken/blla.mlmodel

---

## Usage

### Step 1 — Analyze the test set

```bash
python htr_active_learning.py analyze \
    --test_txt   test.txt \
    --model      fondue_gd_fr_v3.mlmodel \
    --output_dir analysis/
```

`test.txt` must list **ALTO (XML)** files, one per line. The corresponding
image must be in the same folder with the same filename
(`.jpg`, `.jpeg`, `.png`, `.tif` or `.tiff`).

```
./test_2/page_001.xml
./test_2/page_002.xml
```

If `test.txt` lists images (`.jpg`), convert it first:
```bash
sed 's/\.\(jpg\|jpeg\|png\|tif\|tiff\)$/.xml/' test.txt > test.txt
```

**Files produced in `analysis/`:**

| File | Description |
|---|---|
| `rapport_clusters.html` | Interactive report: PCA scatter, CER per cluster, silhouette/elbow curves |
| `cluster_model.pkl` | PCA space + clusters — required for `rank` |
| `cer_per_page.json` | Measured CER per page — required for `rank` |

**Options:**

| Option | Default | Description |
|---|---|---|
| `--test_txt` | — | File listing the ALTO files of the test set |
| `--model` | — | ketos model (`.mlmodel`) used for `ketos test` |
| `--output_dir` | `analysis/` | Output directory |
| `--k_max` | `10` | Maximum number of clusters to test |
| `--skip_cer` | false | Skip CER computation (faster, for testing clustering only) |
| `--use_kraken` | false | Extract features from the ketos model rather than the image (slower) |

---

### Step 2 — Rank the untranscribed stock

```bash
python htr_active_learning.py rank \
    --images_dir stock/ \
    --output_dir analysis/ \
    --top_n      50
```

`--output_dir` must be the same as for `analyze`: the script automatically
loads `cluster_model.pkl` and `cer_per_page.json` from there.

**Files produced:**

| File | Description |
|---|---|
| `rapport_recommandations.html` | Interactive report: PCA projection + ranking table |
| `ranked_pages.txt` | Image paths ordered from most to least useful |
| `segmented/` | kraken segmentation JSON files |
| `segmented/binarized/` | Binarized images (PNG) |

**Options:**

| Option | Default | Description |
|---|---|---|
| `--images_dir` | — | Folder containing the raw images to rank |
| `--output_dir` | `analysis/` | Folder containing `cluster_model.pkl` and `cer_per_page.json` |
| `--cluster_model` | `<output_dir>/cluster_model.pkl` | Explicit path to the `.pkl` |
| `--cer_json` | `<output_dir>/cer_per_page.json` | Explicit path to the CER JSON |
| `--top_n` | `50` | Number of pages to retain in the ranking |
| `--diversity` | `0.3` | Diversity coefficient `0..1` |
| `--seg_model` | `blla.mlmodel` | kraken segmentation model |
| `--skip_segment` | false | Skip segmentation, use existing JSON files in `images_dir` |

---

## CER in the `rank` report

The `rapport_recommandations.html` report displays an **estimated CER**, not a
measured one. Each stock image is projected into the PCA space of the
transcribed pages and matched to the nearest cluster. The CER shown is the
mean CER of that cluster, as measured during the `analyze` step.

This is a proxy estimate: "this image looks like the pages in cluster 3,
which had a mean CER of 45.9%".

---

## The diversity coefficient

With `--diversity 0`, the ranking would recommend dozens of near-identical
pages from the most difficult cluster. With diversity enabled, a bonus is
given to pages that are far from already-selected pages in PCA space,
ensuring broader coverage of difficult styles.

Recommended values: `0.2`–`0.5`. Beyond that, diversity overrides performance
and pages from easy clusters may rise in the ranking.

---

## Full workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Check that test.txt lists XML files (not images)
head -3 test.txt   # should show .xml paths

# 3. Analyze the test set
python htr_active_learning.py analyze \
    --test_txt   test.txt \
    --model      fondue_gd_fr_v3.mlmodel \
    --output_dir analysis/

# 4. Open the clustering report
open analysis/rapport_clusters.html

# 5. Rank the stock
python htr_active_learning.py rank \
    --images_dir /data/stock/ \
    --output_dir analysis/ \
    --top_n 100

# 6. Browse the recommendations
open analysis/rapport_recommandations.html
head -20 analysis/ranked_pages.txt
```

---

## License

This project is licensed under the **GNU General Public License v3.0**.
See the [LICENSE](LICENSE) file for details, or visit
https://www.gnu.org/licenses/gpl-3.0.html.
