#!/usr/bin/env python3
"""
Regenerate fig4_bootstrap_ci_table.png from scores_cache_v2.json.

Fixes stale/estimated CI rows for NeuralODE, RawNet2 FT, RawNet2 ZS.
Includes NeuralODE (AUC=92.18%) using confirmed scores from cache.

Input:  interpretability_figures/comprehensive/figures/scores_cache_v2.json
Output: papers/figures/fig4_bootstrap_ci_table.png
        papers/figures/bootstrap_ci_results.json  (raw numbers for paper)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import roc_auc_score, roc_curve

rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

PROJECT  = Path('/home/ms4726/project_pi_ev6/ms4726/Data-Management-DDAA-KAN')
CACHE    = PROJECT / 'interpretability_figures' / 'comprehensive' / 'figures' / 'scores_cache_v2.json'
OUT_DIR  = PROJECT / 'papers' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Display order — must match paper Table
MODEL_ORDER = ['PINN', 'KAN', 'AASIST', 'NeuralODE', 'Transformer', 'RawNet2 FT', 'RawNet2 ZS', 'AASIST ZS']

# Map cache keys → display names
CACHE_KEY_MAP = {
    'PINN':         'PINN',
    'KAN':          'KAN',
    'AASIST':       'AASIST',
    'NeuralODE':    'NeuralODE',
    'Transformer':  'Transformer',
    'RawNet2 FT':   'RawNet2 FT',
    'RawNet2 ZS':   'RawNet2 ZS',
    'AASIST ZS':    'AASIST ZS',
}

N_BOOT = 1000


def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def bootstrap_ci(scores, labels, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    auc_vals, eer_vals = [], []
    for _ in range(n_boot):
        idx = rng.choice(len(scores), len(scores), replace=True)
        sb, lb = scores[idx], labels[idx]
        if lb.sum() == 0 or lb.sum() == len(lb):
            continue
        auc_vals.append(roc_auc_score(lb, sb))
        eer_vals.append(compute_eer(sb, lb) * 100)

    auc_arr = np.array(auc_vals)
    eer_arr = np.array(eer_vals)
    return {
        'auc_mean':    float(np.mean(auc_arr)),
        'auc_lo':      float(np.percentile(auc_arr, 2.5)),
        'auc_hi':      float(np.percentile(auc_arr, 97.5)),
        'eer_mean':    float(np.mean(eer_arr)),
        'eer_lo':      float(np.percentile(eer_arr, 2.5)),
        'eer_hi':      float(np.percentile(eer_arr, 97.5)),
        'n_boot':      len(auc_vals),
    }


def render_table(rows):
    """Render bootstrap CI results as a matplotlib table figure."""
    col_labels = ['Model', 'AUC mean', 'AUC 95% CI', 'EER mean (%)', 'EER 95% CI (%)']
    table_data = []
    for r in rows:
        table_data.append([
            r['model'],
            f"{r['auc_mean']:.4f}",
            f"[{r['auc_lo']:.4f}, {r['auc_hi']:.4f}]",
            f"{r['eer_mean']:.2f}",
            f"[{r['eer_lo']:.2f}, {r['eer_hi']:.2f}]",
        ])

    fig, ax = plt.subplots(figsize=(12, len(rows) * 0.6 + 1.5))
    ax.axis('off')

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    # Header styling
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#2C3E50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    # Alternating row colors
    for i in range(1, len(rows) + 1):
        color = '#F2F2F2' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(color)

    ax.set_title(
        f'Bootstrap Confidence Intervals (95%, n={N_BOOT} resamples)\n'
        'AUC-ROC and EER on DDAA test set (61,803 samples)',
        fontsize=12, fontweight='bold', pad=12
    )

    out = OUT_DIR / 'fig4_bootstrap_ci_table.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def main():
    print(f'Loading cache: {CACHE}')
    if not CACHE.exists():
        print(f'[ERROR] Cache not found: {CACHE}')
        return

    with open(CACHE) as f:
        cache = json.load(f)

    labels = np.array(cache['_meta']['labels'])
    print(f'Cache loaded. n={len(labels)}, keys={[k for k in cache if not k.startswith("_")]}')

    rows = []
    results_json = {}

    for display_name in MODEL_ORDER:
        # Try exact key, then case-insensitive search
        cache_key = None
        for ck, dn in CACHE_KEY_MAP.items():
            if dn == display_name and ck in cache:
                cache_key = ck
                break

        if cache_key is None:
            # Try direct match
            if display_name in cache:
                cache_key = display_name

        if cache_key is None:
            print(f'[SKIP] {display_name} not found in cache')
            continue

        scores = np.array(cache[cache_key]['scores'])

        # Align lengths — cache may have 62,303 entries (61,803 + 500 Bark appended)
        n = min(len(scores), len(labels))
        scores = scores[:n]
        labels_n = labels[:n]

        print(f'Computing bootstrap CIs for {display_name} (n={n})...')
        ci = bootstrap_ci(scores, labels_n)

        row = {'model': display_name, **ci}
        rows.append(row)
        results_json[display_name] = ci

        print(f'  {display_name}: AUC={ci["auc_mean"]:.4f} [{ci["auc_lo"]:.4f}, {ci["auc_hi"]:.4f}]'
              f'  EER={ci["eer_mean"]:.2f}% [{ci["eer_lo"]:.2f}, {ci["eer_hi"]:.2f}]')

    # Save raw JSON for paper
    out_json = OUT_DIR / 'bootstrap_ci_results.json'
    with open(out_json, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f'Raw results saved to {out_json}')

    # Render figure
    render_table(rows)
    print('Done.')


if __name__ == '__main__':
    main()
