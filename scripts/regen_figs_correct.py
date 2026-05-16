#!/usr/bin/env python3
"""
Regenerate fig3 (score distributions), fig4 (DET curves + bootstrap CI), fig5 (per-TTS EER)
using correct Best Models/ checkpoints for all 6 detectors.

Models:
  PINN FT       -> Best Models/pinn_best.pt         (AUC=97.6%)
  KAN FT        -> Best Models/kan_best.pt           (AUC=97.2%)
  AASIST FT     -> baseline_finetuned/aasist_custom_best.pt (AUC=95.1%)
  Transformer FT -> Best Models/transformer_hp_best.pt (AUC=89.0%)
  RawNet2 FT    -> baseline_finetuned/rawnet2_custom_best.pt (AUC=87.4%)
  RawNet2 ZS    -> LA/Baseline-RawNet2/pre_trained_DF_RawNet2.pth (AUC=43.1%)
"""

import os
import sys
import json
import importlib.util
import warnings
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
import torch
import torch.nn as nn
import librosa

warnings.filterwarnings('ignore')

rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})
sns.set_style('whitegrid')

PROJECT = Path('/gpfs/gibbs/project/lawrence_wilen/ms4726/Data-Management-DDAA-KAN')
OFFICIAL = PROJECT / 'official_dataset'
OUTDIR   = PROJECT / 'interpretability_figures' / 'comprehensive' / 'figures'
OUTDIR.mkdir(parents=True, exist_ok=True)

SR = 16000
MAX_LEN = 64600

MODEL_ORDER = ['PINN', 'KAN', 'AASIST', 'Transformer', 'RawNet2 FT', 'RawNet2 ZS']
MODEL_COLORS = {
    'PINN':         '#2E8B57',
    'KAN':          '#E87722',
    'AASIST':       '#1F77B4',
    'Transformer':  '#7B2D8B',
    'RawNet2 FT':   '#D62728',
    'RawNet2 ZS':   '#8C8C8C',
}

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def load_audio(path, sr=SR, max_len=MAX_LEN):
    try:
        wav, _ = librosa.load(str(path), sr=sr, mono=True)
        if len(wav) > max_len:
            wav = wav[:max_len]
        elif len(wav) < max_len:
            wav = np.pad(wav, (0, max_len - len(wav)))
        return torch.FloatTensor(wav)
    except Exception as e:
        print(f'  Warning: could not load {path}: {e}')
        return torch.zeros(max_len)


def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])


def compute_mintdcf(scores, labels, c_miss=1, c_fa=10, p_target=0.05):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    tdcf = c_miss * p_target * fnr + c_fa * (1 - p_target) * fpr
    return float(np.min(tdcf))


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def _import_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_all_models(device):
    sys.path.insert(0, str(PROJECT / 'scripts'))
    sys.path.insert(0, str(PROJECT))

    models = {}

    # ── PINN ──────────────────────────────────────────────────────────────────
    try:
        print('[PINN] loading...')
        from train_pinn import PINNDetector
        ck   = torch.load(PROJECT / 'Best Models' / 'pinn_best.pt', map_location=device)
        sd   = ck.get('model_state_dict', ck.get('state_dict', ck))
        m    = PINNDetector()
        m.load_state_dict(sd, strict=False)
        m.to(device).eval()
        models['PINN'] = m
        print('[PINN] OK')
    except Exception:
        traceback.print_exc()

    # ── KAN ───────────────────────────────────────────────────────────────────
    try:
        print('[KAN] loading...')
        from train_kan import KANDetector
        ck   = torch.load(PROJECT / 'Best Models' / 'kan_best.pt', map_location=device)
        sd   = ck.get('model_state_dict', ck.get('state_dict', ck))
        m    = KANDetector()
        m.load_state_dict(sd, strict=False)
        m.to(device).eval()
        models['KAN'] = m
        print('[KAN] OK')
    except Exception:
        traceback.print_exc()

    # ── Transformer ───────────────────────────────────────────────────────────
    try:
        print('[Transformer] loading...')
        from train import HighPerformanceDetector
        ck   = torch.load(PROJECT / 'Best Models' / 'transformer_hp_best.pt', map_location=device)
        sd   = ck.get('model_state_dict', ck.get('state_dict', ck))
        m    = HighPerformanceDetector()
        m.load_state_dict(sd, strict=False)
        m.to(device).eval()
        models['Transformer'] = m
        print('[Transformer] OK')
    except Exception:
        traceback.print_exc()

    # ── AASIST FT ─────────────────────────────────────────────────────────────
    try:
        print('[AASIST] loading...')
        aasist_dir = PROJECT / 'baselines' / 'aasist'
        sys.path.insert(0, str(aasist_dir))
        import json
        with open(aasist_dir / 'config' / 'AASIST.conf') as f:
            aasist_cfg = json.load(f)
        amod = _import_file('aasist_model', aasist_dir / 'models' / 'AASIST.py')
        m    = amod.Model(aasist_cfg['model_config']).to(device)
        sd   = torch.load(PROJECT / 'baseline_finetuned' / 'aasist_custom_best.pt', map_location=device)
        m.load_state_dict(sd, strict=False)
        m.eval()
        models['AASIST'] = m
        print('[AASIST] OK')
    except Exception:
        traceback.print_exc()

    # ── RawNet2 FT ────────────────────────────────────────────────────────────
    rawnet2_dir = PROJECT / 'baselines' / 'asvspoof2021' / 'LA' / 'Baseline-RawNet2'
    try:
        print('[RawNet2 FT] loading...')
        import yaml
        with open(rawnet2_dir / 'model_config_RawNet.yaml') as f:
            rn_cfg = yaml.safe_load(f)
        rmod = _import_file('rawnet2_model_ft', rawnet2_dir / 'model.py')
        m    = rmod.RawNet(rn_cfg['model'], device).to(device)
        sd   = torch.load(PROJECT / 'baseline_finetuned' / 'rawnet2_custom_best.pt', map_location=device)
        m.load_state_dict(sd, strict=False)
        m.eval()
        models['RawNet2 FT'] = m
        print('[RawNet2 FT] OK')
    except Exception:
        traceback.print_exc()

    # ── RawNet2 ZS (ASVspoof pre-trained weights only) ────────────────────────
    try:
        print('[RawNet2 ZS] loading...')
        import yaml
        with open(rawnet2_dir / 'model_config_RawNet.yaml') as f:
            rn_cfg = yaml.safe_load(f)
        rmod = _import_file('rawnet2_model_zs', rawnet2_dir / 'model.py')
        m    = rmod.RawNet(rn_cfg['model'], device).to(device)
        sd   = torch.load(rawnet2_dir / 'pre_trained_DF_RawNet2.pth', map_location=device)
        m.load_state_dict(sd, strict=False)
        m.eval()
        models['RawNet2 ZS'] = m
        print('[RawNet2 ZS] OK')
    except Exception:
        traceback.print_exc()

    return models


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def is_rawnet2(name):
    return 'RawNet2' in name

def run_inference(models, device, df, cache_path):
    if cache_path.exists():
        print('Loading cached scores...')
        with open(cache_path) as f:
            return json.load(f)

    results = {'_meta': {
        'labels':         df['bin_label'].tolist(),
        'generator_type': df['generator_type'].tolist(),
    }}

    batch_size = 32
    for model_name, model in models.items():
        print(f'Inference: {model_name} on {len(df)} samples...')
        all_scores = []

        for start in range(0, len(df), batch_size):
            batch_df = df.iloc[start:start + batch_size]
            wavs = []
            for _, row in batch_df.iterrows():
                # full_path is relative to OFFICIAL dataset root
                fp = OFFICIAL / row['full_path']
                if not fp.exists():
                    # fallback: try bark_test_set_final
                    bark_fp = PROJECT / 'bark_test_set_final' / Path(row['full_path']).name
                    fp = bark_fp if bark_fp.exists() else fp
                wavs.append(load_audio(fp))

            batch_tensor = torch.stack(wavs).to(device)

            with torch.no_grad():
                try:
                    out = model(batch_tensor)
                    if isinstance(out, (tuple, list)):
                        logits = out[0]
                    else:
                        logits = out

                    if is_rawnet2(model_name):
                        # RawNet2 outputs log_softmax — convert to prob of fake (class 1)
                        probs = torch.exp(logits)[:, 1]
                    elif logits.shape[-1] == 2:
                        probs = torch.softmax(logits, dim=-1)[:, 1]
                    else:
                        probs = torch.sigmoid(logits).squeeze(-1)

                    all_scores.extend(probs.cpu().numpy().tolist())
                except Exception as e:
                    print(f'  Batch error [{model_name}] step {start}: {e}')
                    all_scores.extend([0.5] * len(batch_df))

            if (start // batch_size) % 50 == 0:
                pct = 100 * start / len(df)
                print(f'  {model_name}: {pct:.0f}% ({start}/{len(df)})')

        results[model_name] = {'scores': all_scores}
        print(f'  {model_name} done. n={len(all_scores)}')

    print('Saving cache...')
    with open(cache_path, 'w') as f:
        json.dump(results, f)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Fig 3: Score Distributions
# ──────────────────────────────────────────────────────────────────────────────

def fig_score_dist(results, df):
    present = [m for m in MODEL_ORDER if m in results]
    ncols = 3
    nrows = (len(present) + 2) // 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    labels = np.array(results['_meta']['labels'])
    for i, mname in enumerate(present):
        ax = axes[i]
        scores = np.array(results[mname]['scores'])
        real_s = scores[labels == 0]
        fake_s = scores[labels == 1]

        eer, eer_thresh = compute_eer(scores, labels)
        ks_stat, ks_p   = stats.ks_2samp(real_s, fake_s)
        color = MODEL_COLORS.get(mname, '#333333')

        ax.hist(real_s, bins=60, alpha=0.6, color='#4472C4',
                label=f'Real (n={len(real_s)})', density=True)
        ax.hist(fake_s, bins=60, alpha=0.6, color='#C0392B',
                label=f'Fake (n={len(fake_s)})', density=True)
        ax.axvline(eer_thresh, color='k', linestyle='--', lw=1,
                   label=f'EER threshold ({eer_thresh:.3f})')

        ax.set_title(f'{mname}\nEER={eer*100:.2f}%', fontweight='bold', color=color)
        ax.set_xlabel('P(fake)')
        ax.set_ylabel('Density')
        ax.text(0.03, 0.97, f'KS={ks_stat:.3f}\np={ks_p:.2e}',
                transform=ax.transAxes, va='top', fontsize=8)
        ax.legend(fontsize=8)

    for j in range(len(present), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Score Distributions: Real vs Fake', fontsize=14, fontweight='bold')
    fig.tight_layout()
    out = OUTDIR / 'fig2_score_distributions.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ──────────────────────────────────────────────────────────────────────────────
# Fig 4a: DET Curves
# ──────────────────────────────────────────────────────────────────────────────

def fig_det_curves(results):
    from scipy.special import ndtri  # probit transform

    present = [m for m in MODEL_ORDER if m in results]
    labels  = np.array(results['_meta']['labels'])

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_facecolor('white')
    ax.grid(True, which='both', alpha=0.3)

    for mname in present:
        scores = np.array(results[mname]['scores'])
        fpr, tpr, _ = roc_curve(labels, scores)
        fnr = 1 - tpr
        # Clip to avoid ndtri(0) or ndtri(1)
        eps = 1e-6
        fpr_c = np.clip(fpr, eps, 1 - eps)
        fnr_c = np.clip(fnr, eps, 1 - eps)
        color = MODEL_COLORS.get(mname, '#333333')
        ax.plot(ndtri(fpr_c), ndtri(fnr_c), color=color, label=mname, lw=1.5)

        eer, _ = compute_eer(scores, labels)
        eer_z  = ndtri(eer)
        ax.scatter([eer_z], [eer_z], color=color, s=40, zorder=5)
        ax.annotate(f'EER={eer*100:.1f}%', (eer_z, eer_z),
                    textcoords='offset points', xytext=(6, -6),
                    fontsize=7, color=color)

    # EER diagonal line
    lim = ndtri(np.array([0.001, 0.999]))
    ax.plot(lim, lim, 'k--', lw=1, alpha=0.5, label='EER line')

    ticks_pct = [0.1, 0.5, 1, 2, 5, 10, 20, 40]
    tick_vals  = [ndtri(p / 100) for p in ticks_pct]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f'{p}%' for p in ticks_pct], fontsize=8)
    ax.set_yticks(tick_vals)
    ax.set_yticklabels([f'{p}%' for p in ticks_pct], fontsize=8)
    ax.set_xlim(ndtri(0.0005), ndtri(0.999))
    ax.set_ylim(ndtri(0.0005), ndtri(0.999))
    ax.set_xlabel('False Match Rate (FMR)')
    ax.set_ylabel('False Non-Match Rate (FNMR)')
    ax.set_title('Detection Error Tradeoff (DET) Curves', fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')

    out = OUTDIR / 'fig3_det_curves.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ──────────────────────────────────────────────────────────────────────────────
# Fig 4b: Bootstrap CI Table
# ──────────────────────────────────────────────────────────────────────────────

def fig_bootstrap_ci(results):
    present = [m for m in MODEL_ORDER if m in results]
    labels  = np.array(results['_meta']['labels'])
    rng     = np.random.default_rng(42)
    N_BOOT  = 1000

    rows = []
    for mname in present:
        scores = np.array(results[mname]['scores'])
        auc_vals, eer_vals = [], []
        for _ in range(N_BOOT):
            idx  = rng.choice(len(scores), len(scores), replace=True)
            sb, lb = scores[idx], labels[idx]
            if lb.sum() == 0 or lb.sum() == len(lb):
                continue
            auc_vals.append(roc_auc_score(lb, sb))
            eer_vals.append(compute_eer(sb, lb)[0] * 100)

        auc_arr = np.array(auc_vals)
        eer_arr = np.array(eer_vals)
        rows.append({
            'Model':       mname,
            'AUC_mean':    f'{np.mean(auc_arr):.4f}',
            'AUC_95CI':    f'[{np.percentile(auc_arr,2.5):.4f}, {np.percentile(auc_arr,97.5):.4f}]',
            'EER_mean_%':  f'{np.mean(eer_arr):.2f}',
            'EER_95CI_%':  f'[{np.percentile(eer_arr,2.5):.2f}, {np.percentile(eer_arr,97.5):.2f}]',
        })

    df_tbl = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, len(rows) * 0.55 + 1.2))
    ax.axis('off')
    tbl = ax.table(
        cellText  = df_tbl.values,
        colLabels = df_tbl.columns,
        cellLoc   = 'center',
        loc       = 'center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(col=list(range(len(df_tbl.columns))))

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#F2F3F4')
        cell.set_edgecolor('#BDC3C7')

    fig.tight_layout()
    out = OUTDIR / 'fig4_bootstrap_ci_table.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')
    df_tbl.to_csv(OUTDIR / 'fig4_bootstrap_ci_table.csv', index=False)


# ──────────────────────────────────────────────────────────────────────────────
# Fig 5: Per-TTS EER breakdown
# ──────────────────────────────────────────────────────────────────────────────

def fig_per_tts_eer(results, df):
    """EER per TTS system for fine-tuned models (exclude ZS as it collapses)."""
    ft_models = [m for m in ['PINN', 'KAN', 'AASIST', 'Transformer', 'RawNet2 FT']
                 if m in results]

    labels      = np.array(results['_meta']['labels'])
    gen_types   = np.array(results['_meta']['generator_type'])

    # Map generator_type strings → display label (TTS system)
    # generator_type was computed as: 'RVC', 'Edge-TTS', 'Bark', 'Unknown'/'Other' → 'Real'
    tts_systems = ['Edge-TTS', 'RVC', 'Bark']
    # Only synthetic samples
    data = {}
    for mname in ft_models:
        scores = np.array(results[mname]['scores'])
        row = {}
        for sys_name in tts_systems:
            # Mask: this generator type AND synthetic label
            mask = (gen_types == sys_name) & (labels == 1)
            if mask.sum() < 10:
                row[sys_name] = float('nan')
                continue
            # Also need real samples for EER computation → use all real + these fake
            real_mask = labels == 0
            idx_use   = np.where(real_mask | mask)[0]
            s_use, l_use = scores[idx_use], labels[idx_use]
            if l_use.sum() == 0:
                row[sys_name] = float('nan')
                continue
            eer, _ = compute_eer(s_use, l_use)
            row[sys_name] = eer * 100
        data[mname] = row

    df_eer = pd.DataFrame(data, index=tts_systems).T  # rows=models, cols=tts

    # Also compute overall EER for each model
    overall = {}
    for mname in ft_models:
        scores = np.array(results[mname]['scores'])
        eer, _ = compute_eer(scores, labels)
        overall[mname] = eer * 100
    df_eer['Overall'] = pd.Series(overall)

    print('Per-TTS EER (%):\n', df_eer.round(1))

    cols   = tts_systems + ['Overall']
    x      = np.arange(len(cols))
    width  = 0.8 / len(ft_models)
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, mname in enumerate(ft_models):
        vals   = [df_eer.loc[mname, c] if c in df_eer.columns else float('nan') for c in cols]
        offset = (i - len(ft_models) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=mname,
                        color=MODEL_COLORS.get(mname, '#888'), alpha=0.85)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f'{v:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(cols, fontsize=10)
    ax.set_ylabel('EER (%)')
    ax.set_title('Per-TTS-System EER Breakdown', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(30, df_eer.max().max() * 1.2) if not df_eer.empty else 30)
    fig.tight_layout()

    out = OUTDIR / 'fig5_per_tts_eer.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ──────────────────────────────────────────────────────────────────────────────
# Fig 11: min-tDCF Summary Table
# ──────────────────────────────────────────────────────────────────────────────

def fig_mintdcf_table(results):
    present = [m for m in MODEL_ORDER if m in results]
    labels  = np.array(results['_meta']['labels'])

    rows = []
    for mname in present:
        scores = np.array(results[mname]['scores'])
        auc    = roc_auc_score(labels, scores)
        eer, eer_thresh = compute_eer(scores, labels)
        mintdcf = compute_mintdcf(scores, labels)
        rows.append({
            'Model':        mname,
            'AUC':          f'{auc:.4f}',
            'EER (%)':      f'{eer * 100:.2f}',
            'EER Threshold': f'{eer_thresh:.4f}',
            'min-tDCF':     f'{mintdcf:.4f}',
        })

    df_tbl = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, len(rows) * 0.55 + 1.5))
    ax.axis('off')
    ax.set_title('min-tDCF and EER Summary\n(C_miss=1, C_fa=10, P_target=0.05)',
                 fontsize=11, pad=10)
    tbl = ax.table(
        cellText  = df_tbl.values,
        colLabels = df_tbl.columns,
        cellLoc   = 'center',
        loc       = 'center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(col=list(range(len(df_tbl.columns))))

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#F2F3F4')
        cell.set_edgecolor('#BDC3C7')

    fig.tight_layout()
    out = OUTDIR / 'fig11_mintdcf_table.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load test CSV ──────────────────────────────────────────────────────────
    test_csv = PROJECT / 'unified_dataset' / 'test.csv'
    df = pd.read_csv(test_csv)
    df['bin_label'] = (df['label'] == 'synthetic').astype(int)

    # ── Build generator_type from metadata ────────────────────────────────────
    print('Building generator labels...')
    meta_cache = {}
    for d in OFFICIAL.glob('processed_dataset_*'):
        m = d / 'metadata.csv'
        if m.exists():
            mdf = pd.read_csv(m)
            for _, r in mdf.iterrows():
                fname = r.get('filename', r.get('file', ''))
                meta_cache[Path(fname).name] = r.get('generator', 'unknown')

    def classify_gen(row):
        fname = Path(row['full_path']).name
        gen   = meta_cache.get(fname, 'unknown').lower()
        if 'bark' in gen:
            return 'Bark'
        if 'rvc' in gen:
            return 'RVC'
        if 'edge' in gen:
            return 'Edge-TTS'
        return 'unknown'

    df['generator_type'] = df.apply(classify_gen, axis=1)
    print('Generator type distribution:')
    print(df['generator_type'].value_counts())

    # ── Load models ────────────────────────────────────────────────────────────
    print('\nLoading models...')
    models = load_all_models(device)
    print(f'Loaded models: {list(models.keys())}')

    if not models:
        print('ERROR: No models loaded. Exiting.')
        sys.exit(1)

    # ── Run inference (cached) ─────────────────────────────────────────────────
    cache_path = OUTDIR / 'scores_cache_v2.json'
    results    = run_inference(models, device, df, cache_path)

    # ── Generate figures ───────────────────────────────────────────────────────
    print('\nGenerating fig3 (score distributions)...')
    fig_score_dist(results, df)

    print('\nGenerating fig4a (DET curves)...')
    fig_det_curves(results)

    print('\nGenerating fig4b (bootstrap CI table)...')
    fig_bootstrap_ci(results)

    print('\nGenerating fig5 (per-TTS EER)...')
    fig_per_tts_eer(results, df)

    print('\nGenerating fig11 (min-tDCF summary table)...')
    fig_mintdcf_table(results)

    print('\nAll done.')


if __name__ == '__main__':
    main()
