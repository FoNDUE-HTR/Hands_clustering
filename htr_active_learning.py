"""
htr_active_learning.py
======================
Active learning pour l'HTR : identifie les styles d'écriture difficiles
dans un jeu de test transcrit, puis classe un stock de pages non transcrites
par ordre d'utilité pour l'entraînement.

Deux sous-commandes :

  analyze  — Analyse le jeu de test :
               1. Extrait les features visuelles de chaque page (binarisation Otsu)
               2. Clustering automatique (PCA + k-means, k optimal par silhouette)
               3. Calcule le CER par page avec ketos test
               4. Produit un rapport HTML + cluster_model.pkl + cer_per_page.json

  rank     — Classe un stock de pages non transcrites :
               1. Segmente les images brutes avec kraken (binarisation Otsu maison)
               2. Extrait leurs features, les projette dans l'espace PCA du jeu de test
               3. Score d'utilité = CER moyen du cluster le plus proche + bonus diversité
               4. Produit ranked_pages.txt + rapport HTML

Exemples d'utilisation
----------------------
  # Étape 1 : analyser le jeu de test
  python htr_active_learning.py analyze \\
      --test_txt  test.txt \\
      --model     fondue_gd_fr_v3.mlmodel \\
      --output_dir analysis/

  # Étape 2 : classer le stock
  python htr_active_learning.py rank \\
      --images_dir stock/ \\
      --output_dir analysis/ \\
      --top_n      50

  # Options avancées
  python htr_active_learning.py analyze --k_max 15 --skip_cer --use_kraken
  python htr_active_learning.py rank --diversity 0.5 --seg_model blla.mlmodel --skip_segment
"""

import argparse
import json
import pickle
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from html import escape as he
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    sys.exit("❌ Pillow manquant : pip install Pillow")

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit("❌ scikit-learn manquant : pip install scikit-learn")

try:
    import torch
    from kraken.lib import models as kraken_models
    KRAKEN_AVAILABLE = True
except ImportError:
    KRAKEN_AVAILABLE = False

IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']

CLUSTER_COLORS = [
    '#e63946', '#2a9d8f', '#e9c46a', '#457b9d', '#f4a261',
    '#6d6875', '#80b918', '#f72585', '#4361ee', '#fb8500',
]
CER_STOPS = [(0, '#2dc653'), (10, '#f7b731'), (25, '#ff6b35'), (50, '#c0392b')]


# ═════════════════════════════════════════════════════════════════════════════
# Utilitaires communs
# ═════════════════════════════════════════════════════════════════════════════

def _otsu_threshold(arr: np.ndarray) -> int:
    """Seuil de binarisation Otsu, pur numpy."""
    hist = np.bincount(arr.flatten(), minlength=256).astype(float)
    total = hist.sum()
    if total == 0:
        return 128
    sum_total = np.dot(np.arange(256), hist)
    sum_b = w_b = max_var = 0.0
    threshold = 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / w_b
        mean_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (mean_b - mean_f) ** 2
        if var > max_var:
            max_var, threshold = var, t
    return threshold


def binarize(img_path: Path) -> np.ndarray:
    """Charge une image en niveaux de gris et la binarise (Otsu)."""
    arr = np.array(Image.open(img_path).convert('L'))
    t = _otsu_threshold(arr)
    return (arr < t).astype(np.uint8) * 255


def _lerp_color(c0: str, c1: str, t: float) -> str:
    def h(s): return [int(s[i:i+2], 16) for i in (1, 3, 5)]
    r0, g0, b0 = h(c0); r1, g1, b1 = h(c1)
    return '#{:02x}{:02x}{:02x}'.format(
        int(r0+(r1-r0)*t), int(g0+(g1-g0)*t), int(b0+(b1-b0)*t))


def cer_to_color(cer) -> str:
    if cer is None:
        return '#888888'
    for i in range(len(CER_STOPS) - 1):
        t0, c0 = CER_STOPS[i]; t1, c1 = CER_STOPS[i+1]
        if t0 <= cer <= t1:
            return _lerp_color(c0, c1, (cer - t0) / (t1 - t0))
    return CER_STOPS[-1][1]


# ═════════════════════════════════════════════════════════════════════════════
# Extraction de features visuelles
# ═════════════════════════════════════════════════════════════════════════════

def features_from_image(img_path: Path, lines: list = None) -> np.ndarray:
    """
    Features basées sur l'image binarisée :
      - histogramme niveaux de gris (64 bins)
      - densité d'encre
      - ratio h/w moyen des lignes, écart-type
    Total : 67 dimensions.
    """
    try:
        arr = binarize(img_path)
    except Exception:
        return np.zeros(67)

    hist, _ = np.histogram(arr.flatten(), bins=64, range=(0, 256))
    hist = hist.astype(float) / (hist.sum() + 1e-9)
    ink  = (arr < 128).mean()

    ratios = []
    for line in (lines or []):
        poly = line.get('polygon', '')
        if not poly:
            continue
        try:
            pts = list(map(int, poly.split()))
            xs = pts[0::2]; ys = pts[1::2]
            w = max(xs) - min(xs); h = max(ys) - min(ys)
            if w > 0:
                ratios.append(h / w)
        except Exception:
            continue

    return np.concatenate([hist, [ink,
                                  np.mean(ratios) if ratios else 0.0,
                                  np.std(ratios)  if ratios else 0.0]])


def features_from_kraken(img_path: Path, model_path: str) -> np.ndarray:
    """
    Features extraites de l'avant-dernière couche du modèle ketos
    via un forward hook. Fallback sur features_from_image si échec.
    """
    try:
        nn  = kraken_models.load_any(model_path)
        net = nn.nn
        feats = []

        def hook(_, __, out):
            feats.append(out.detach().cpu().numpy().mean(axis=-1).mean(axis=-1).flatten())

        layers = list(net.children())
        if len(layers) < 2:
            return features_from_image(img_path)
        handle = layers[-2].register_forward_hook(hook)

        arr  = binarize(img_path).astype(np.float32) / 255.0
        t    = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            try: net(t)
            except Exception: pass
        handle.remove()
        return feats[0] if feats else features_from_image(img_path)
    except Exception as e:
        print(f"  ⚠️  features_kraken échoué ({img_path.name}) : {e} → fallback")
        return features_from_image(img_path)


# ═════════════════════════════════════════════════════════════════════════════
# Parsing ALTO (jeu de test transcrit)
# ═════════════════════════════════════════════════════════════════════════════

def parse_alto(xml_path: Path) -> list:
    """Retourne les lignes d'un fichier ALTO (polygones + texte)."""
    try:
        root = ET.parse(xml_path).getroot()
        ns   = root.tag.split('}')[0] + '}' if '}' in root.tag else ''
        lines = []
        for tl in root.iter(f'{ns}TextLine'):
            pe = tl.find(f'{ns}Shape/{ns}Polygon')
            ce = tl.find(f'.//{ns}String')
            lines.append({
                'polygon': pe.get('POINTS', '') if pe is not None else '',
                'text':    ce.get('CONTENT', '') if ce is not None else '',
            })
        return lines
    except Exception:
        return []


# ═════════════════════════════════════════════════════════════════════════════
# Parsing JSON de segmentation kraken (pages non transcrites)
# ═════════════════════════════════════════════════════════════════════════════

def parse_seg_json(json_path: Path) -> list:
    """Retourne les lignes d'un JSON de segmentation kraken."""
    try:
        data = json.loads(json_path.read_text())
        lines = []
        for line in data.get('lines', []):
            boundary = line.get('boundary', [])
            if boundary:
                pts = ' '.join(f'{int(p[0])} {int(p[1])}' for p in boundary)
                lines.append({'polygon': pts})
        return lines
    except Exception:
        return []


# ═════════════════════════════════════════════════════════════════════════════
# Segmentation avec kraken
# ═════════════════════════════════════════════════════════════════════════════

def segment_image(img_path: Path, seg_dir: Path, seg_model: str) -> Path | None:
    """
    Binarise l'image (Otsu maison) puis appelle :
      kraken -a -i <bin.png> <out.json> -n segment -bl [-m <model>]
    Retourne le chemin du JSON, ou None si échec.
    """
    json_path = seg_dir / f'{img_path.stem}.json'
    if json_path.exists():
        return json_path

    # Binarisation maison → fichier temporaire
    bin_dir = seg_dir / 'binarized'
    bin_dir.mkdir(exist_ok=True)
    bin_path = bin_dir / f'{img_path.stem}_bin.png'
    if not bin_path.exists():
        try:
            Image.fromarray(binarize(img_path)).save(bin_path)
        except Exception as e:
            print(f"  ⚠️  Binarisation échouée ({img_path.name}) : {e}")
            bin_path = img_path  # fallback

    cmd = ['kraken', '-a', '-i', str(bin_path), str(json_path), '-n', 'segment', '-bl']
    if seg_model:
        cmd += ['-m', seg_model]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return json_path if json_path.exists() else None
    except Exception as e:
        print(f"  ⚠️  Segmentation échouée ({img_path.name}) : {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# CER par page (ketos test)
# ═════════════════════════════════════════════════════════════════════════════

def _parse_accuracy(output: str, label: str):
    m = re.search(rf'([\d.]+)%\s+{re.escape(label)}', output)
    return float(m.group(1)) if m else None


def compute_cer_per_page(pages: list, model_path: str) -> dict:
    """
    Lance `ketos test` sur chaque page individuellement.
    Retourne {stem: {cer, ca, wa}} où cer = 100 - ca.
    """
    results = {}
    for i, (xml_path, _) in enumerate(pages):
        stem = xml_path.stem
        print(f'  [{i+1}/{len(pages)}] {stem}...', end='\r')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(str(xml_path) + '\n')
            tmp = f.name
        try:
            r = subprocess.run(
                ['ketos', 'test', '-f', 'alto', '-m', model_path, '-e', tmp],
                capture_output=True, text=True, timeout=120)
            out = r.stdout + r.stderr
            ca  = _parse_accuracy(out, 'Character Accuracy')
            wa  = _parse_accuracy(out, 'Word Accuracy')
            results[stem] = {
                'ca':  round(ca, 2)       if ca is not None else None,
                'wa':  round(wa, 2)       if wa is not None else None,
                'cer': round(100-ca, 2)   if ca is not None else None,
                'wer': round(100-wa, 2)   if wa is not None else None,
            }
        except Exception as e:
            print(f'\n  ⚠️  {stem} : {e}')
            results[stem] = {'ca': None, 'wa': None, 'cer': None, 'wer': None}
        finally:
            Path(tmp).unlink(missing_ok=True)
    print()
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Clustering (PCA + k-means, k optimal automatique)
# ═════════════════════════════════════════════════════════════════════════════

def find_best_k(features_dict: dict, k_max: int):
    """
    Teste k=2..k_max et choisit le meilleur silhouette score.
    Retourne (stems, labels, X_2d, explained, best_sil, k_scores).
    """
    stems = list(features_dict.keys())
    X     = np.array([features_dict[s] for s in stems])
    n     = len(stems)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_pca = min(32, X_scaled.shape[1], n - 1)
    X_pca = PCA(n_components=n_pca, random_state=42).fit_transform(X_scaled)

    pca2  = PCA(n_components=min(2, n-1), random_state=42).fit(X_scaled)
    X_2d  = pca2.transform(X_scaled)
    if X_2d.shape[1] == 1:
        X_2d = np.column_stack([X_2d, np.zeros(n)])
    explained = pca2.explained_variance_ratio_ * 100

    k_max_eff = min(k_max, n - 1)
    k_scores  = []
    best_k, best_sil, best_labels = 2, -1.0, None

    print(f'  Test k=2..{k_max_eff}...')
    for k in range(2, k_max_eff + 1):
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        sil    = round(silhouette_score(X_pca, labels), 4) if len(set(labels)) > 1 else 0.0
        k_scores.append((k, sil, round(km.inertia_, 2)))
        print(f'    k={k:2d}  silhouette={sil:.4f}  inertie={km.inertia_:.0f}')
        if sil > best_sil:
            best_sil, best_k, best_labels = sil, k, labels

    print(f'  → k optimal : {best_k}  (silhouette={best_sil:.4f})')
    return stems, best_labels, X_2d, explained, round(best_sil, 4), k_scores


# ═════════════════════════════════════════════════════════════════════════════
# Projection dans l'espace PCA de référence (pour rank)
# ═════════════════════════════════════════════════════════════════════════════

def project_into_reference(ref_features: dict, new_features: dict):
    """
    Ajuste scaler + PCA sur les données de référence,
    transforme les nouvelles pages dans cet espace.
    Retourne (ref_stems, X_ref_pca, X_ref_2d, new_stems, X_new_pca, X_new_2d).
    """
    ref_stems = list(ref_features.keys())
    new_stems = list(new_features.keys())
    X_ref     = np.array([ref_features[s] for s in ref_stems])
    X_new     = np.array([new_features[s] for s in new_stems])

    scaler = StandardScaler().fit(X_ref)
    Xr_sc  = scaler.transform(X_ref)
    Xn_sc  = scaler.transform(X_new)

    n_pca    = min(32, Xr_sc.shape[1], len(ref_stems) - 1)
    pca_full = PCA(n_components=n_pca, random_state=42).fit(Xr_sc)
    Xr_pca   = pca_full.transform(Xr_sc)
    Xn_pca   = pca_full.transform(Xn_sc)

    pca2   = PCA(n_components=min(2, len(ref_stems)-1), random_state=42).fit(Xr_sc)
    Xr_2d  = pca2.transform(Xr_sc)
    Xn_2d  = pca2.transform(Xn_sc)
    if Xr_2d.shape[1] == 1:
        Xr_2d = np.column_stack([Xr_2d, np.zeros(len(ref_stems))])
        Xn_2d = np.column_stack([Xn_2d, np.zeros(len(new_stems))])

    return ref_stems, Xr_pca, Xr_2d, new_stems, Xn_pca, Xn_2d


# ═════════════════════════════════════════════════════════════════════════════
# Score d'utilité (pour rank)
# ═════════════════════════════════════════════════════════════════════════════

def score_candidates(new_stems, X_new, ref_stems, X_ref, ref_labels,
                     cluster_cer: dict, diversity: float) -> list:
    """
    Score d'utilité pour chaque page candidate :
      score = CER_moyen_cluster_proche + diversity * distance_min_aux_déjà_sélectionnées
    Retourne liste triée par score décroissant.
    """
    n_clusters = max(ref_labels) + 1
    centroids  = np.array([
        X_ref[[i for i, l in enumerate(ref_labels) if l == k]].mean(axis=0)
        if any(l == k for l in ref_labels) else np.zeros(X_ref.shape[1])
        for k in range(n_clusters)
    ])

    raw = []
    for i, stem in enumerate(new_stems):
        dists = np.linalg.norm(centroids - X_new[i], axis=1)
        k     = int(np.argmin(dists))
        raw.append({
            'stem':          stem,
            'cluster':       k,
            'dist_centroid': round(float(dists[k]), 4),
            'cluster_cer':   cluster_cer.get(k, {}).get('cer_mean', 0.0),
        })

    selected_vecs = []
    scored = []
    for item in sorted(raw, key=lambda x: -x['cluster_cer']):
        i  = new_stems.index(item['stem'])
        dv = min((np.linalg.norm(X_new[i] - sv) for sv in selected_vecs), default=0.0)
        item['diversity_bonus'] = round(diversity * dv, 4)
        item['final_score']     = round(item['cluster_cer'] + item['diversity_bonus'], 4)
        scored.append(item)
        selected_vecs.append(X_new[i])

    return sorted(scored, key=lambda x: -x['final_score'])


# ═════════════════════════════════════════════════════════════════════════════
# Rapport HTML — analyze
# ═════════════════════════════════════════════════════════════════════════════

CSS_BASE = """
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
  :root {
    --bg:#0f1117; --surface:#1a1d27; --border:#2a2d3a;
    --accent:#00d4aa; --accent2:#ff6b35; --text:#e8eaf0; --muted:#7a7f9a;
    --mono:'IBM Plex Mono',monospace; --sans:'IBM Plex Sans',sans-serif;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--text); font-family:var(--sans); font-weight:300; line-height:1.6; }
  header { padding:2.5rem 4rem 1.5rem; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:1rem; }
  header h1 { font-family:var(--mono); font-size:1.3rem; font-weight:600; color:var(--accent); }
  header p  { font-size:0.78rem; color:var(--muted); font-family:var(--mono); }
  .meta { display:flex; gap:2rem; flex-wrap:wrap; padding:1.2rem 4rem; border-bottom:1px solid var(--border); background:var(--surface); }
  .mi { display:flex; flex-direction:column; gap:0.15rem; }
  .mi label { font-size:0.68rem; color:var(--muted); font-family:var(--mono); text-transform:uppercase; letter-spacing:0.08em; }
  .mi value { font-size:1.05rem; font-family:var(--mono); font-weight:600; }
  .mi value.warn { color:var(--accent2); }
  main { padding:2rem 4rem; display:grid; gap:2rem; }
  .stitle { font-family:var(--mono); font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; color:var(--muted); margin-bottom:0.8rem; padding-bottom:0.4rem; border-bottom:1px solid var(--border); }
  .card { background:var(--surface); border:1px solid var(--border); border-radius:4px; padding:1.4rem; }
  .g2 { display:grid; grid-template-columns:1fr 1fr; gap:1.4rem; }
  .canvas-wrap { width:100%; aspect-ratio:1; background:#13161f; border-radius:2px; overflow:hidden; }
  canvas { width:100%; height:100%; }
  .tooltip { position:fixed; background:#22253a; border:1px solid var(--border); border-radius:3px; padding:0.5rem 0.8rem; font-family:var(--mono); font-size:0.7rem; pointer-events:none; z-index:100; display:none; max-width:230px; line-height:1.9; }
  table { width:100%; border-collapse:collapse; font-family:var(--mono); font-size:0.76rem; }
  th { text-align:left; padding:0.45rem 0.7rem; font-size:0.63rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); border-bottom:1px solid var(--border); }
  td { padding:0.45rem 0.7rem; border-bottom:1px solid #1e2130; vertical-align:middle; }
  tr:hover td { background:#1e2130; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; }
  .trow { display:flex; gap:0.4rem; margin-bottom:0.8rem; flex-wrap:wrap; }
  .tbtn { font-family:var(--mono); font-size:0.68rem; padding:0.25rem 0.7rem; border:1px solid var(--border); border-radius:2px; background:transparent; color:var(--muted); cursor:pointer; transition:all 0.12s; text-transform:uppercase; }
  .tbtn.active { background:var(--accent); color:var(--bg); border-color:var(--accent); font-weight:600; }
  .legend { display:flex; gap:1.2rem; flex-wrap:wrap; margin-top:0.7rem; font-size:0.7rem; font-family:var(--mono); color:var(--muted); }
  .li { display:flex; align-items:center; gap:5px; }
  .target { background:linear-gradient(135deg,#1a0f0a,#2a1408); border:1px solid #ff6b3540; border-left:3px solid var(--accent2); border-radius:4px; padding:1rem 1.3rem; font-family:var(--mono); font-size:0.78rem; line-height:2; margin-top:1.2rem; }
  .target strong { color:var(--accent2); font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; display:block; margin-bottom:0.2rem; }
  .ptags { display:flex; flex-wrap:wrap; gap:0.25rem; margin-top:0.4rem; }
  .ptag { font-size:0.62rem; padding:0.12rem 0.45rem; border-radius:2px; background:#1e2130; border:1px solid var(--border); color:var(--muted); }
  .bar-wrap { display:flex; align-items:center; gap:7px; }
  .bar { height:5px; border-radius:3px; }
  @media(max-width:900px){.g2{grid-template-columns:1fr}header,.meta,main{padding-left:1.5rem;padding-right:1.5rem}}
"""

JS_LINECHART = """
function drawLineChart(id, data, xk, yk, color, label, hx) {
  const c = document.getElementById(id);
  if (!c || !data.length) return;
  const dpr=window.devicePixelRatio||1, rect=c.parentElement.getBoundingClientRect();
  c.width=rect.width*dpr; c.height=180*dpr;
  c.style.width=rect.width+'px'; c.style.height='180px';
  const ctx=c.getContext('2d'); ctx.scale(dpr,dpr);
  const W=rect.width, H=180, p={t:18,r:16,b:26,l:42};
  const xs=data.map(d=>d[xk]), ys=data.map(d=>d[yk]);
  const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys),yr=y1-y0||1;
  const tx=x=>p.l+(x-x0)/(x1-x0+1e-9)*(W-p.l-p.r);
  const ty=y=>H-p.b-(y-y0)/yr*(H-p.t-p.b);
  ctx.strokeStyle='#1e2130'; ctx.lineWidth=1;
  for(let i=0;i<=4;i++){
    const y=p.t+i*(H-p.t-p.b)/4;
    ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(W-p.r,y);ctx.stroke();
    ctx.fillStyle='#555';ctx.font='8px IBM Plex Mono,monospace';
    ctx.fillText((y1-i*yr/4).toFixed(3),2,y+3);
  }
  if(hx!==undefined){
    ctx.strokeStyle='#00d4aa40';ctx.lineWidth=14;
    ctx.beginPath();ctx.moveTo(tx(hx),p.t);ctx.lineTo(tx(hx),H-p.b);ctx.stroke();
  }
  ctx.strokeStyle=color;ctx.lineWidth=2;ctx.lineJoin='round';
  ctx.beginPath();
  data.forEach((d,i)=>{i===0?ctx.moveTo(tx(d[xk]),ty(d[yk])):ctx.lineTo(tx(d[xk]),ty(d[yk]));});
  ctx.stroke();
  data.forEach(d=>{
    ctx.beginPath();ctx.arc(tx(d[xk]),ty(d[yk]),4,0,Math.PI*2);
    ctx.fillStyle=d[xk]===hx?'#00d4aa':color+'99';ctx.fill();
    ctx.fillStyle='#777';ctx.font='8px IBM Plex Mono,monospace';
    ctx.fillText(d[xk],tx(d[xk])-3,H-8);
  });
  ctx.fillStyle='#555';ctx.font='9px IBM Plex Mono,monospace';ctx.fillText(label,p.l,11);
}
"""


def build_analyze_report(stems, labels, X_2d, explained, silhouette,
                         cer_results, n_clusters, k_scores,
                         model_path, feature_mode, output_path: Path):

    cluster_stats = {}
    for k in range(n_clusters):
        idx  = [i for i, l in enumerate(labels) if l == k]
        ss   = [stems[i] for i in idx]
        cers = [cer_results.get(s, {}).get('cer') for s in ss]
        cv   = [c for c in cers if c is not None]
        cluster_stats[k] = {
            'stems':    ss, 'count': len(ss),
            'cer_mean': round(np.mean(cv), 2) if cv else None,
            'cer_min':  round(np.min(cv),  2) if cv else None,
            'cer_max':  round(np.max(cv),  2) if cv else None,
        }

    pts = []
    for i, s in enumerate(stems):
        cr = cer_results.get(s, {})
        pts.append({'x': round(float(X_2d[i,0]),4), 'y': round(float(X_2d[i,1]),4),
                    'cluster': int(labels[i]), 'stem': s,
                    'cer': cr.get('cer'), 'ca': cr.get('ca'), 'wa': cr.get('wa'),
                    'cc': CLUSTER_COLORS[int(labels[i]) % len(CLUSTER_COLORS)]})

    all_cers   = [cer_results.get(s, {}).get('cer') for s in stems]
    valid_cers = [c for c in all_cers if c is not None]
    global_cer = round(np.mean(valid_cers), 2) if valid_cers else 'N/A'
    worst3     = [str(k) for k, _ in sorted(
        [(k, v['cer_mean'] or 0) for k, v in cluster_stats.items()],
        key=lambda x: -x[1])[:3]]

    pj = json.dumps(pts)
    sj = json.dumps(cluster_stats)
    cj = json.dumps(CLUSTER_COLORS)
    kj = json.dumps(k_scores or [])
    ex = round(float(explained[0]), 1)
    ey = round(float(explained[1]), 1)

    html = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<title>Analyse clusters HTR</title>
<style>{CSS_BASE}</style></head><body>
<header>
  <div><h1>Analyse clusters — jeu de test HTR</h1>
  <p>Regroupement par style d'écriture × performance du modèle</p></div>
  <p>Modèle : <strong>{he(model_path)}</strong>&nbsp;|&nbsp;Features : {he(feature_mode)}</p>
</header>
<div class="meta">
  <div class="mi"><label>Pages</label><value>{len(stems)}</value></div>
  <div class="mi"><label>Clusters</label><value>{n_clusters}</value></div>
  <div class="mi"><label>CER global</label>
    <value class="{'warn' if isinstance(global_cer,float) and global_cer>15 else ''}">{global_cer}{'%' if global_cer!='N/A' else ''}</value></div>
  <div class="mi"><label>Silhouette</label><value>{silhouette}</value></div>
  <div class="mi"><label>Variance PCA</label><value>{ex}% + {ey}%</value></div>
</div>
<div id="tt" class="tooltip"></div>
<main>

<div><div class="stitle">Projection PCA des styles d'écriture</div>
<div class="g2">
  <div class="card">
    <div class="trow">
      <button class="tbtn active" onclick="setMode('cluster',this)">Par cluster</button>
      <button class="tbtn" onclick="setMode('cer',this)">Par CER</button>
    </div>
    <div class="canvas-wrap"><canvas id="scatter"></canvas></div>
    <div class="legend" id="legend"></div>
  </div>
  <div class="card">
    <div class="stitle">Statistiques par cluster</div>
    <table><thead><tr><th>Cluster</th><th>Pages</th><th>CER moy.</th><th>min / max</th></tr></thead>
    <tbody id="ctable"></tbody></table>
    <div class="target">
      <strong>🎯 Clusters à cibler en priorité</strong>
      CER moyen le plus élevé — générer des données synthétiques pour ces styles.
      <div class="ptags" id="tpages"></div>
    </div>
  </div>
</div></div>

<div><div class="stitle">Sélection automatique du nombre de clusters</div>
<div class="card"><div class="g2">
  <div><div class="stitle">Silhouette score par k</div><canvas id="sil"></canvas></div>
  <div><div class="stitle">Inertie (coude)</div><canvas id="elbow"></canvas></div>
</div></div></div>

<div><div class="stitle">Détail par page</div>
<div class="card"><table>
  <thead><tr><th>Page</th><th>Cluster</th><th>CER</th><th>CA</th><th>WA</th></tr></thead>
  <tbody id="ptable"></tbody>
</table></div></div>

</main>
<script>
const PTS={pj},STATS={sj},COLORS={cj},WORST=[{','.join(worst3)}],KS={kj};
const EX={ex},EY={ey},NK={n_clusters};
const canvas=document.getElementById('scatter'),ctx=canvas.getContext('2d');
const tt=document.getElementById('tt');
let mode='cluster';
const xs=PTS.map(p=>p.x),ys=PTS.map(p=>p.y);
const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys),pad=0.08;
function tc(x,y){{
  const W=canvas.width/devicePixelRatio,H=canvas.height/devicePixelRatio;
  return[pad*W+(x-x0)/(x1-x0+1e-9)*W*(1-2*pad),H-(pad*H+(y-y0)/(y1-y0+1e-9)*H*(1-2*pad))];
}}
function cerCol(c){{
  if(c==null)return'#888';
  const s=[[0,'#2dc653'],[10,'#f7b731'],[25,'#ff6b35'],[50,'#c0392b']];
  for(let i=0;i<s.length-1;i++){{
    if(c>=s[i][0]&&c<=s[i+1][0]){{
      const r=(c-s[i][0])/(s[i+1][0]-s[i][0]),lc=(a,b,t)=>{{
        const h=v=>[parseInt(v.slice(1,3),16),parseInt(v.slice(3,5),16),parseInt(v.slice(5,7),16)];
        const[r0,g0,b0]=h(a),[r1,g1,b1]=h(b);
        return`rgb(${{Math.round(r0+(r1-r0)*t)}},${{Math.round(g0+(g1-g0)*t)}},${{Math.round(b0+(b1-b0)*t)}})`;
      }};return lc(s[i][1],s[i+1][1],r);
    }}
  }}return s[s.length-1][1];
}}
function draw(){{
  const dpr=devicePixelRatio||1,rect=canvas.parentElement.getBoundingClientRect();
  canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
  ctx.scale(dpr,dpr);canvas.style.width=rect.width+'px';canvas.style.height=rect.height+'px';
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle='#444';ctx.font='9px IBM Plex Mono,monospace';
  ctx.fillText(`PC1 (${{EX}}%)`,8,canvas.height/dpr-7);
  ctx.save();ctx.translate(11,canvas.height/dpr/2);ctx.rotate(-Math.PI/2);
  ctx.fillText(`PC2 (${{EY}}%)`,0,0);ctx.restore();
  PTS.forEach(p=>{{
    const[cx,cy]=tc(p.x,p.y),col=mode==='cluster'?COLORS[p.cluster%COLORS.length]:cerCol(p.cer);
    ctx.beginPath();ctx.arc(cx,cy,6,0,Math.PI*2);
    ctx.fillStyle=col+'cc';ctx.fill();ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.stroke();
  }});
}}
function buildLegend(){{
  const el=document.getElementById('legend');
  if(mode==='cluster'){{
    el.innerHTML=Object.keys(STATS).map(k=>`<span class="li"><span class="dot" style="background:${{COLORS[k%COLORS.length]}}"></span>Cluster ${{k}} (${{STATS[k].count}} p.)</span>`).join('');
  }}else{{
    el.innerHTML=`<span class="li"><span class="dot" style="background:#2dc653"></span>&lt;10%</span>
    <span class="li"><span class="dot" style="background:#f7b731"></span>10–25%</span>
    <span class="li"><span class="dot" style="background:#ff6b35"></span>25–50%</span>
    <span class="li"><span class="dot" style="background:#c0392b"></span>&gt;50%</span>`;
  }}
}}
function buildCTable(){{
  document.getElementById('ctable').innerHTML=Object.entries(STATS)
    .sort((a,b)=>(b[1].cer_mean||0)-(a[1].cer_mean||0))
    .map(([k,s])=>{{
      const w=WORST.includes(parseInt(k));
      return`<tr${{w?' style="background:#1a0f0a"':''}}><td><span class="dot" style="background:${{COLORS[k%COLORS.length]}}"></span>Cluster ${{k}}${{w?' 🎯':''}}</td>
      <td>${{s.count}}</td>
      <td style="color:${{w?'#ff6b35':'inherit'}};font-weight:${{w?600:300}}">${{s.cer_mean!=null?s.cer_mean+'%':'N/A'}}</td>
      <td style="color:var(--muted)">${{s.cer_min!=null?s.cer_min+'% – '+s.cer_max+'%':'—'}}</td></tr>`;
    }}).join('');
}}
function buildTPages(){{
  const pages=[];
  WORST.forEach(k=>{{(STATS[k]?.stems||[]).forEach(s=>pages.push({{s,k}}));}});
  pages.sort((a,b)=>{{const pa=PTS.find(p=>p.stem===a.s),pb=PTS.find(p=>p.stem===b.s);return(pb?.cer||0)-(pa?.cer||0);}});
  document.getElementById('tpages').innerHTML=pages.map(p=>{{
    const pt=PTS.find(x=>x.stem===p.s);
    return`<span class="ptag" style="border-color:${{COLORS[p.k%COLORS.length]}}40">${{he(p.s)}} <span style="color:#ff6b35">${{pt?.cer!=null?pt.cer+'%':'?'}}</span></span>`;
  }}).join('');
}}
function buildPTable(){{
  document.getElementById('ptable').innerHTML=[...PTS].sort((a,b)=>(b.cer||0)-(a.cer||0)).map(p=>{{
    const bw=p.cer!=null?Math.min(100,p.cer*2):0;
    return`<tr><td>${{he(p.stem)}}</td>
    <td><span class="dot" style="background:${{COLORS[p.cluster%COLORS.length]}}"></span>Cluster ${{p.cluster}}</td>
    <td><div class="bar-wrap"><div class="bar" style="width:${{bw}}px;background:${{cerCol(p.cer)}}"></div><span>${{p.cer!=null?p.cer+'%':'N/A'}}</span></div></td>
    <td>${{p.ca!=null?p.ca+'%':'—'}}</td><td>${{p.wa!=null?p.wa+'%':'—'}}</td></tr>`;
  }}).join('');
}}
function he(s){{return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}}
canvas.addEventListener('mousemove',e=>{{
  const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  let cl=null,md=18;
  PTS.forEach(p=>{{const[cx,cy]=tc(p.x,p.y),d=Math.hypot(cx-mx,cy-my);if(d<md){{md=d;cl=p;}}}});
  if(cl){{tt.style.display='block';tt.style.left=(e.clientX+13)+'px';tt.style.top=(e.clientY-8)+'px';
    tt.innerHTML=`<strong>${{he(cl.stem)}}</strong><br>Cluster : ${{cl.cluster}}<br>CER : ${{cl.cer!=null?cl.cer+'%':'N/A'}}<br>CA : ${{cl.ca!=null?cl.ca+'%':'N/A'}}<br>WA : ${{cl.wa!=null?cl.wa+'%':'N/A'}}`;
  }}else tt.style.display='none';
}});
canvas.addEventListener('mouseleave',()=>tt.style.display='none');
function setMode(m,btn){{
  mode=m;document.querySelectorAll('.tbtn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');draw();buildLegend();
}}
{JS_LINECHART}
function drawCharts(){{
  if(KS.length){{
    drawLineChart('sil',  KS.map(([k,s])=>(({{k,s}}))),'k','s','#00d4aa','Silhouette',NK);
    drawLineChart('elbow',KS.map(([k,,i])=>(({{k,i}}))),'k','i','#ff6b35','Inertie',   NK);
  }}
}}
window.addEventListener('load',()=>{{draw();buildLegend();buildCTable();buildTPages();buildPTable();drawCharts();}});
window.addEventListener('resize',()=>{{draw();drawCharts();}});
</script></body></html>"""

    output_path.write_text(html, encoding='utf-8')
    print(f'✅ Rapport → {output_path}')


# ═════════════════════════════════════════════════════════════════════════════
# Rapport HTML — rank
# ═════════════════════════════════════════════════════════════════════════════

def build_rank_report(ranked, ref_stems, ref_labels, X_ref_2d,
                      new_stems, X_new_2d, cluster_cer,
                      top_n, output_path: Path):

    top_set  = {r['stem'] for r in ranked[:top_n]}
    rank_map = {r['stem']: r for r in ranked}

    ref_pts = [{'x': round(float(X_ref_2d[i,0]),4), 'y': round(float(X_ref_2d[i,1]),4),
                'stem': ref_stems[i], 'cluster': int(ref_labels[i]),
                'color': CLUSTER_COLORS[int(ref_labels[i]) % len(CLUSTER_COLORS)],
                'type': 'ref'} for i in range(len(ref_stems))]

    new_pts = []
    for i, s in enumerate(new_stems):
        info = rank_map.get(s, {})
        k    = info.get('cluster', 0)
        new_pts.append({'x': round(float(X_new_2d[i,0]),4), 'y': round(float(X_new_2d[i,1]),4),
                        'stem': s, 'cluster': k, 'in_top': s in top_set,
                        'score': info.get('final_score',0), 'cer': info.get('cluster_cer',0),
                        'color': CLUSTER_COLORS[k % len(CLUSTER_COLORS)], 'type': 'new'})

    rows = ''
    for rank, item in enumerate(ranked[:top_n], 1):
        cc = CLUSTER_COLORS[item['cluster'] % len(CLUSTER_COLORS)]
        ec = cer_to_color(item['cluster_cer'])
        rows += (f'<tr><td style="color:var(--muted)">#{rank}</td>'
                 f'<td>{he(item["stem"])}</td>'
                 f'<td><span class="dot" style="background:{cc}"></span>Cluster {item["cluster"]}</td>'
                 f'<td style="color:{ec};font-weight:600">{item["cluster_cer"]}%</td>'
                 f'<td style="color:var(--muted)">{item["dist_centroid"]}</td>'
                 f'<td style="color:var(--accent)">{item["final_score"]}</td></tr>')

    rj  = json.dumps(ref_pts)
    nj  = json.dumps(new_pts)
    cj  = json.dumps(CLUSTER_COLORS)
    cdj = json.dumps({str(k): v for k, v in cluster_cer.items()})

    html = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<title>Pages recommandées — HTR</title>
<style>{CSS_BASE}</style></head><body>
<header>
  <div><h1>Pages recommandées à transcrire</h1>
  <p>Classées par utilité pour l'entraînement — proximité aux clusters problématiques</p></div>
</header>
<div class="meta">
  <div class="mi"><label>Candidates</label><value>{len(new_stems)}</value></div>
  <div class="mi"><label>Recommandées</label><value style="color:var(--accent)">{top_n}</value></div>
  <div class="mi"><label>Référence</label><value>{len(ref_stems)}</value></div>
  <div class="mi"><label>Clusters</label><value>{len(cluster_cer)}</value></div>
</div>
<div id="tt" class="tooltip"></div>
<main>

<div><div class="stitle">Projection PCA — référence + candidates</div>
<div class="card">
  <div class="trow">
    <button class="tbtn active" onclick="setShow('all',this)">Toutes</button>
    <button class="tbtn" onclick="setShow('top',this)">Top {top_n}</button>
    <button class="tbtn" onclick="setShow('ref',this)">Référence</button>
  </div>
  <div class="canvas-wrap"><canvas id="scatter"></canvas></div>
  <div class="legend" id="legend"></div>
</div></div>

<div><div class="stitle">Top {top_n} pages recommandées</div>
<div class="card"><table>
  <thead><tr><th>#</th><th>Page</th><th>Cluster</th><th>CER cluster</th><th>Dist. centroïde</th><th>Score</th></tr></thead>
  <tbody>{rows}</tbody>
</table></div></div>

</main>
<script>
const RP={rj},NP={nj},CL={cj},CD={cdj},TN={top_n};
const canvas=document.getElementById('scatter'),ctx=canvas.getContext('2d'),tt=document.getElementById('tt');
let show='all';
const all=[...RP,...NP],xs=all.map(p=>p.x),ys=all.map(p=>p.y);
const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys),pad=0.08;
function tc(x,y){{
  const W=canvas.width/devicePixelRatio,H=canvas.height/devicePixelRatio;
  return[pad*W+(x-x0)/(x1-x0+1e-9)*W*(1-2*pad),H-(pad*H+(y-y0)/(y1-y0+1e-9)*H*(1-2*pad))];
}}
function draw(){{
  const dpr=devicePixelRatio||1,rect=canvas.parentElement.getBoundingClientRect();
  canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
  ctx.scale(dpr,dpr);canvas.style.width=rect.width+'px';canvas.style.height=rect.height+'px';
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(show!=='top')RP.forEach(p=>{{
    const[cx,cy]=tc(p.x,p.y);
    ctx.beginPath();ctx.arc(cx,cy,4,0,Math.PI*2);
    ctx.fillStyle=p.color+'55';ctx.fill();ctx.strokeStyle=p.color+'88';ctx.lineWidth=1;ctx.stroke();
  }});
  if(show!=='ref')NP.forEach(p=>{{
    if(show==='top'&&!p.in_top)return;
    const[cx,cy]=tc(p.x,p.y),top=p.in_top;
    ctx.beginPath();ctx.arc(cx,cy,top?7:4,0,Math.PI*2);
    ctx.fillStyle=top?p.color+'ee':p.color+'44';ctx.fill();
    if(top){{ctx.strokeStyle=p.color;ctx.lineWidth=2;ctx.stroke();
      ctx.beginPath();ctx.arc(cx,cy,10,0,Math.PI*2);ctx.strokeStyle=p.color+'44';ctx.lineWidth=1;ctx.stroke();}}
  }});
}}
function buildLegend(){{
  const ks=[...new Set(all.map(p=>p.cluster))].sort();
  document.getElementById('legend').innerHTML=ks.map(k=>`<span class="li"><span class="dot" style="background:${{CL[k%CL.length]}}"></span>Cluster ${{k}} (CER ${{CD[k]?.cer_mean??'?'}}%)</span>`).join('')+
  `<span class="li" style="margin-left:0.8rem">⬤ <span style="font-size:0.62em;margin-left:3px">Référence</span></span>
   <span class="li">● <span style="font-size:0.62em;margin-left:3px">Top ${{TN}}</span></span>`;
}}
canvas.addEventListener('mousemove',e=>{{
  const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  let cl=null,md=15;
  all.forEach(p=>{{const[cx,cy]=tc(p.x,p.y),d=Math.hypot(cx-mx,cy-my);if(d<md){{md=d;cl=p;}}}});
  if(cl){{tt.style.display='block';tt.style.left=(e.clientX+13)+'px';tt.style.top=(e.clientY-8)+'px';
    tt.innerHTML=`<strong>${{he(cl.stem)}}</strong><br>Type : ${{cl.type==='ref'?'Référence':'Candidat'}}<br>Cluster : ${{cl.cluster}}${{cl.type==='new'?'<br>Score : '+cl.score+'<br>CER cluster : '+cl.cer+'%':''}}`
  }}else tt.style.display='none';
}});
canvas.addEventListener('mouseleave',()=>tt.style.display='none');
function he(s){{return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}}
function setShow(m,btn){{show=m;document.querySelectorAll('.tbtn').forEach(b=>b.classList.remove('active'));btn.classList.add('active');draw();}}
window.addEventListener('load',()=>{{draw();buildLegend();}});
window.addEventListener('resize',draw);
</script></body></html>"""

    output_path.write_text(html, encoding='utf-8')
    print(f'✅ Rapport → {output_path}')


# ═════════════════════════════════════════════════════════════════════════════
# Sous-commande : analyze
# ═════════════════════════════════════════════════════════════════════════════

def cmd_analyze(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Charger les pages
    xml_paths = [Path(l.strip()) for l in Path(args.test_txt).read_text().splitlines() if l.strip()]
    pages = []
    for xp in xml_paths:
        for ext in IMAGE_EXTS:
            ip = xp.with_suffix(ext)
            if ip.exists():
                pages.append((xp, ip)); break
        else:
            print(f'  ⚠️  Image non trouvée pour {xp.name}')
    print(f'✅ {len(pages)} pages chargées')
    if not pages:
        sys.exit('❌ Aucune page trouvée')

    # Features
    use_kraken   = args.use_kraken and KRAKEN_AVAILABLE
    feature_mode = 'kraken' if use_kraken else 'histogram+geometry'
    print(f'📐 Extraction features ({feature_mode})...')
    feats = {}
    for i, (xp, ip) in enumerate(pages):
        print(f'  [{i+1}/{len(pages)}] {ip.name}', end='\r')
        lines = parse_alto(xp)
        feats[xp.stem] = features_from_kraken(ip, args.model) if use_kraken \
                          else features_from_image(ip, lines)
    print(f'\n✅ Features extraites')

    # Clustering
    print('🔵 Clustering...')
    stems, labels, X_2d, explained, sil, k_scores = find_best_k(feats, args.k_max)
    n_clusters = len(set(labels))

    # Sauvegarder le modèle pour rank
    model_data = {
        'stems': stems, 'labels': labels.tolist(),
        'features': {s: feats[s].tolist() for s in stems},
        'k_scores': k_scores, 'n_clusters': n_clusters,
        'feature_mode': feature_mode, 'model_path': str(args.model),
    }
    pkl = out / 'cluster_model.pkl'
    pkl.write_bytes(pickle.dumps(model_data))
    print(f'✅ Modèle sauvegardé → {pkl}')

    # CER par page
    cer_results = {}
    if not args.skip_cer:
        print(f'📊 Calcul CER ({len(pages)} pages)...')
        cer_results = compute_cer_per_page(pages, args.model)
        cer_json = out / 'cer_per_page.json'
        cer_json.write_text(json.dumps(cer_results, indent=2, ensure_ascii=False))
        print(f'✅ CER sauvegardés → {cer_json}')
    else:
        print('⏭️  CER ignoré (--skip_cer)')

    # Rapport
    print('📄 Génération du rapport...')
    build_analyze_report(stems, labels, X_2d, explained, sil,
                         cer_results, n_clusters, k_scores,
                         args.model, feature_mode,
                         out / 'rapport_clusters.html')

    # Résumé terminal
    print(f'\n{"="*55}\nRÉSUMÉ PAR CLUSTER\n{"="*55}')
    for k in range(n_clusters):
        idx  = [i for i, l in enumerate(labels) if l == k]
        ss   = [stems[i] for i in idx]
        cv   = [cer_results.get(s, {}).get('cer') for s in ss]
        cv   = [c for c in cv if c is not None]
        cstr = f'{np.mean(cv):.1f}% (min {np.min(cv):.1f}%, max {np.max(cv):.1f}%)' if cv else 'N/A'
        print(f'  Cluster {k} : {len(ss)} pages | CER moy : {cstr}')
    print(f'{"="*55}')


# ═════════════════════════════════════════════════════════════════════════════
# Sous-commande : rank
# ═════════════════════════════════════════════════════════════════════════════

def cmd_rank(args):
    out       = Path(args.output_dir)
    img_dir   = Path(args.images_dir)
    seg_dir   = out / 'segmented'
    out.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(exist_ok=True)

    # Charger le modèle de clustering
    model_data   = pickle.loads(Path(args.cluster_model).read_bytes())
    ref_stems    = model_data['stems']
    ref_labels   = np.array(model_data['labels'])
    ref_features = {s: np.array(v) for s, v in model_data['features'].items()}
    n_clusters   = model_data['n_clusters']
    print(f'✅ {len(ref_stems)} pages de référence, {n_clusters} clusters')

    # CER moyen par cluster
    cer_per_page = json.loads(Path(args.cer_json).read_text())
    cluster_cer  = {}
    for k in range(n_clusters):
        idx  = [i for i, l in enumerate(ref_labels) if l == k]
        ss   = [ref_stems[i] for i in idx]
        cv   = [cer_per_page.get(s, {}).get('cer') for s in ss]
        cv   = [c for c in cv if c is not None]
        cluster_cer[k] = {'cer_mean': round(np.mean(cv), 2) if cv else 0.0, 'count': len(ss)}
        print(f'  Cluster {k} : CER moy = {cluster_cer[k]["cer_mean"]}%')

    # Trouver les images candidates
    candidates = sorted({p for ext in IMAGE_EXTS for p in img_dir.glob(f'*{ext}')})
    print(f'✅ {len(candidates)} images candidates')
    if not candidates:
        sys.exit('❌ Aucune image trouvée dans ' + str(img_dir))

    # Segmentation + features
    print(f'{"⏭️  Segmentation ignorée" if args.skip_segment else "✂️  Segmentation + extraction features"}...')
    new_feats = {}
    for i, ip in enumerate(candidates):
        print(f'  [{i+1}/{len(candidates)}] {ip.name}', end='\r')
        lines = []
        if not args.skip_segment:
            jp = segment_image(ip, seg_dir, args.seg_model)
            if jp: lines = parse_seg_json(jp)
        else:
            jp = img_dir / f'{ip.stem}.json'
            if jp.exists(): lines = parse_seg_json(jp)
        new_feats[ip.stem] = features_from_image(ip, lines)
    print(f'\n✅ Features extraites ({len(new_feats)} pages)')

    # Projection + scoring
    print('🔵 Projection dans l\'espace PCA de référence...')
    ref_stems_out, Xr_pca, Xr_2d, new_stems, Xn_pca, Xn_2d = \
        project_into_reference(ref_features, new_feats)

    print('📊 Calcul des scores d\'utilité...')
    ranked = score_candidates(new_stems, Xn_pca, list(ref_stems_out),
                              Xr_pca, ref_labels, cluster_cer, args.diversity)

    # Sauvegarder ranked_pages.txt
    txt_path = out / 'ranked_pages.txt'
    with open(txt_path, 'w') as f:
        for item in ranked[:args.top_n]:
            for ext in IMAGE_EXTS:
                ip = img_dir / f'{item["stem"]}{ext}'
                if ip.exists():
                    f.write(str(ip) + '\n'); break
    print(f'✅ Liste ordonnée → {txt_path}')

    # Rapport HTML
    print('📄 Génération du rapport...')
    build_rank_report(ranked, ref_stems_out, ref_labels, Xr_2d,
                      new_stems, Xn_2d, cluster_cer,
                      args.top_n, out / 'rapport_recommandations.html')

    # Résumé terminal
    print(f'\n{"="*55}\nTOP {args.top_n} PAGES RECOMMANDÉES\n{"="*55}')
    for rank, item in enumerate(ranked[:min(10, args.top_n)], 1):
        print(f'  #{rank:2d} {item["stem"]:<40} cluster={item["cluster"]} '
              f'CER={item["cluster_cer"]}% score={item["final_score"]}')
    if args.top_n > 10:
        print(f'  … ({args.top_n-10} autres dans {txt_path})')
    print(f'{"="*55}')


# ═════════════════════════════════════════════════════════════════════════════
# Point d'entrée
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog='htr_active_learning.py',
        description='Active learning HTR — analyse clusters + classement pages à transcrire',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python htr_active_learning.py analyze \\
      --test_txt test.txt --model fondue_gd_fr_v3.mlmodel --output_dir analysis/

  python htr_active_learning.py rank \\
      --images_dir stock/ --output_dir analysis/ --top_n 50
""")
    sub = parser.add_subparsers(dest='cmd', required=True)

    # ── analyze ──────────────────────────────────────────────────────────────
    pa = sub.add_parser('analyze', help='Analyse le jeu de test transcrit')
    pa.add_argument('--test_txt',   required=True,
                    help='Fichier .txt listant les ALTO du jeu de test')
    pa.add_argument('--model',      required=True,
                    help='Modèle ketos (.mlmodel) pour ketos test')
    pa.add_argument('--output_dir', default='analysis',
                    help='Dossier de sortie (défaut : analysis/)')
    pa.add_argument('--k_max',      type=int, default=10,
                    help='Nombre max de clusters à tester (défaut : 10)')
    pa.add_argument('--skip_cer',   action='store_true',
                    help='Ne pas calculer le CER (plus rapide)')
    pa.add_argument('--use_kraken', action='store_true',
                    help='Features via le modèle kraken (plus lent, plus précis)')

    # ── rank ─────────────────────────────────────────────────────────────────
    pr = sub.add_parser('rank', help='Classe un stock de pages non transcrites')
    pr.add_argument('--images_dir',    required=True,
                    help='Dossier contenant les images brutes')
    pr.add_argument('--output_dir',    default='analysis',
                    help='Dossier contenant cluster_model.pkl et cer_per_page.json (défaut : analysis/)')
    pr.add_argument('--cluster_model', default=None,
                    help='Chemin vers cluster_model.pkl (défaut : <output_dir>/cluster_model.pkl)')
    pr.add_argument('--cer_json',      default=None,
                    help='Chemin vers cer_per_page.json (défaut : <output_dir>/cer_per_page.json)')
    pr.add_argument('--top_n',         type=int, default=50,
                    help='Nombre de pages à retenir (défaut : 50)')
    pr.add_argument('--diversity',     type=float, default=0.3,
                    help='Coefficient de diversité 0..1 (défaut : 0.3)')
    pr.add_argument('--seg_model',     default='blla.mlmodel',
                    help='Modèle de segmentation kraken (défaut : blla.mlmodel)')
    pr.add_argument('--skip_segment',  action='store_true',
                    help='Ne pas segmenter, utiliser les JSON existants dans images_dir')

    args = parser.parse_args()

    # Résoudre les chemins par défaut pour rank
    if args.cmd == 'rank':
        if args.cluster_model is None:
            args.cluster_model = str(Path(args.output_dir) / 'cluster_model.pkl')
        if args.cer_json is None:
            args.cer_json = str(Path(args.output_dir) / 'cer_per_page.json')

    if args.cmd == 'analyze':
        cmd_analyze(args)
    else:
        cmd_rank(args)


if __name__ == '__main__':
    main()
