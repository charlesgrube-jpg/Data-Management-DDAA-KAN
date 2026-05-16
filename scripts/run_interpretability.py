#!/usr/bin/env python3
"""
run_interpretability.py — Architecture-Specific XAI for Deepfake Audio Detectors

Runs five tailored interpretability analyses and saves publication-quality figures:

  1. TRANSFORMER  — Attention Rollout (Abnar & Zuidema 2020) + per-head entropy
  2. KAN          — B-spline edge-function visualization + input-feature importance
  3. NEURAL ODE   — Latent-trajectory PCA, class separation over integration time,
                    trajectory curvature (real vs. fake)
  4. PINN         — Physics-residual distributions, temporal violation profile,
                    per-dimension deepfake fingerprint
  5. ALL MODELS   — Integrated Gradients cross-architecture comparison +
                    inter-model attribution correlation matrix

Usage (quickstart — uses the latest checkpoint for each model automatically):
    python scripts/run_interpretability.py \\
        --data_dir  unified_dataset \\
        --output_dir interpretability_figures \\
        --n_samples 64

Usage (explicit checkpoints):
    python scripts/run_interpretability.py \\
        --transformer_ckpt checkpoints/transformer_20260111_143414/best_model.pt \\
        --kan_ckpt         checkpoints/kan_20260111_143414/best_model.pt \\
        --ode_ckpt         checkpoints/neural_ode_20260111_143414/best_model.pt \\
        --pinn_ckpt        checkpoints/pinn_20260111_143414/best_model.pt \\
        --data_dir         unified_dataset \\
        --output_dir       interpretability_figures \\
        --n_samples        64
"""

# =============================================================================
# Stdlib / third-party imports
# =============================================================================
import argparse
import csv
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import matplotlib
matplotlib.use("Agg")          # headless – safe on HPC / SSH
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# =============================================================================
# Path setup — make the scripts/ directory importable
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# =============================================================================
# Import the four detector classes from their training scripts
# =============================================================================
try:
    from train_pinn import PINNDetector
    print("[Import] PINNDetector ✓")
except ImportError as e:
    PINNDetector = None
    print(f"[Import] PINNDetector FAILED: {e}")

try:
    from train_kan import KANDetector, KANLinear
    print("[Import] KANDetector ✓")
except ImportError as e:
    KANDetector = KANLinear = None
    print(f"[Import] KANDetector FAILED: {e}")

try:
    from train_ode import ODEDetector
    print("[Import] ODEDetector ✓")
except ImportError as e:
    ODEDetector = None
    print(f"[Import] ODEDetector FAILED: {e}")


# =============================================================================
# Helpers: checkpoint discovery & model loading
# =============================================================================

def _latest_checkpoint(prefix: str, ckpt_root: Path) -> Optional[Path]:
    """Return the best_model.pt from the most-recent run matching prefix."""
    runs = sorted(
        [d for d in ckpt_root.iterdir()
         if d.is_dir() and d.name.startswith(prefix)],
        key=lambda d: d.name, reverse=True
    )
    for run in runs:
        pt = run / "best_model.pt"
        if pt.exists():
            return pt
    return None


def load_model(model_type: str, ckpt_path: Path, device: torch.device) -> Optional[nn.Module]:
    """Instantiate the correct detector class and load weights from ckpt_path."""
    cls_map = {
        "transformer": PINNDetector,   # same backbone; no physics loss used at inference
        "pinn":        PINNDetector,
        "kan":         KANDetector,
        "neural_ode":  ODEDetector,
    }
    cls = cls_map.get(model_type)
    if cls is None:
        print(f"[Load] {model_type}: class not available — skipping.")
        return None

    print(f"[Load] {model_type}: instantiating {cls.__name__}...")
    model = cls()

    print(f"[Load] {model_type}: reading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    try:
        model.load_state_dict(state)
        print(f"[Load] {model_type}: weights loaded (strict) ✓")
    except RuntimeError as e:
        info = model.load_state_dict(state, strict=False)
        print(f"[Load] {model_type}: non-strict load — "
              f"missing {len(info.missing_keys)}, unexpected {len(info.unexpected_keys)}")

    model.to(device).eval()
    return model


# =============================================================================
# Dataset: load balanced real / fake batches from test CSV
# =============================================================================

SAMPLE_RATE = 16_000
MAX_LENGTH  = SAMPLE_RATE * 5   # 5 s


def _load_wav(path: str) -> torch.Tensor:
    """Load, mono-mix, resample to 16 kHz, pad/trim to 5 s, peak-normalize."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav.unsqueeze(0)).squeeze(0)
    if wav.shape[0] > MAX_LENGTH:
        wav = wav[:MAX_LENGTH]
    elif wav.shape[0] < MAX_LENGTH:
        wav = F.pad(wav, (0, MAX_LENGTH - wav.shape[0]))
    peak = wav.abs().max()
    if peak > 1e-6:
        wav = wav / peak
    return wav


def load_balanced_batch(
    csv_path: Path,
    n_per_class: int,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    """
    Read the test CSV and return two tensors:
        x_real : (n_per_class, T_audio)
        x_fake : (n_per_class, T_audio)
    plus the corresponding file paths for reference.
    Attempts multiple path resolution strategies to handle relocated datasets.
    """
    real_paths, fake_paths = [], []

    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))

    rng = random.Random(seed)
    rng.shuffle(rows)

    for row in rows:
        p = row.get("full_path", row.get("filename", "")).strip()
        label_str = row.get("label", "real")
        is_fake = label_str in ("synthetic", "1", 1)

        # Path resolution candidates
        for candidate in [
            Path(p),
            PROJECT_ROOT / p,
            PROJECT_ROOT / "official_dataset" / p,
            PROJECT_ROOT / "completed_correct_dataset" / p,
            csv_path.parent / p,
        ]:
            if candidate.exists():
                if is_fake and len(fake_paths) < n_per_class:
                    fake_paths.append(str(candidate))
                elif not is_fake and len(real_paths) < n_per_class:
                    real_paths.append(str(candidate))
                break

        if len(real_paths) >= n_per_class and len(fake_paths) >= n_per_class:
            break

    print(f"[Data] Real: {len(real_paths)} | Fake: {len(fake_paths)}")
    if not real_paths or not fake_paths:
        raise RuntimeError(
            "Could not resolve any audio paths from the CSV. "
            "Check --data_dir and that official_dataset/ is accessible."
        )

    x_real = torch.stack([_load_wav(p) for p in real_paths])
    x_fake = torch.stack([_load_wav(p) for p in fake_paths])
    return x_real, x_fake, real_paths, fake_paths


# =============================================================================
# ── METHOD 1: TRANSFORMER — Attention Rollout + Head Entropy ─────────────────
# =============================================================================

def _get_attn_maps(model: nn.Module, x: torch.Tensor, chunk_size: int = 8) -> Dict[int, torch.Tensor]:
    transformer = model.transformer
    all_cache = {}
    original_forwards = {}
    
    # Monkey-patch self_attn.forward to force need_weights=True
    for i, layer in enumerate(transformer.layers):
        original_forwards[i] = layer.self_attn.forward
        if hasattr(layer.self_attn, "average_attn_weights"):
            layer.self_attn.average_attn_weights = False
            
    try:
        for start_idx in range(0, x.shape[0], chunk_size):
            chunk_x = x[start_idx:start_idx + chunk_size]
            chunk_cache = {}
            
            for i, layer in enumerate(transformer.layers):
                def make_patched(idx, orig_fw):
                    def _patched(*args, **kwargs):
                        kwargs["need_weights"] = True
                        out = orig_fw(*args, **kwargs)
                        if out[1] is not None:
                            chunk_cache[idx] = out[1].detach().cpu()
                        return out
                    return _patched
                layer.self_attn.forward = make_patched(i, original_forwards[i])
                
            with torch.no_grad():
                model(chunk_x)
                
            # Accumulate cache
            for idx, act in chunk_cache.items():
                if idx not in all_cache:
                    all_cache[idx] = []
                all_cache[idx].append(act)
                
    finally:
        # Restore original forwards
        for i, layer in enumerate(transformer.layers):
            layer.self_attn.forward = original_forwards[i]
            if hasattr(layer.self_attn, "average_attn_weights"):
                layer.self_attn.average_attn_weights = True
                
    # Concatenate accumulated cache across batch dimension
    return {idx: torch.cat(acts, dim=0) for idx, acts in all_cache.items()}


def _rollout(
    attn_maps: Dict[int, torch.Tensor],
    discard_ratio: float = 0.5,
) -> torch.Tensor:
    """Compute Attention Rollout (Abnar & Zuidema 2020). Returns (B, T, T)."""
    fused = []
    for i in range(len(attn_maps)):
        A = attn_maps[i]
        if A.dim() == 4:          # (B, H, T, T)
            A = A.mean(dim=1)     # average heads → (B, T, T)
        fused.append(A)

    B, T, _ = fused[0].shape
    rollout = torch.eye(T).unsqueeze(0).expand(B, -1, -1).clone()

    for A in fused:
        if discard_ratio > 0:
            thresh = A.flatten(1).quantile(discard_ratio, dim=1).view(B, 1, 1)
            A = A * (A >= thresh).float()
        A_hat = A + torch.eye(T, device=A.device).unsqueeze(0)
        A_hat = A_hat / (A_hat.sum(dim=-1, keepdim=True) + 1e-8)
        rollout = torch.bmm(A_hat, rollout)
    return rollout  # (B, T, T)


def run_transformer_interpretability(
    model: nn.Module,
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    out_dir: Path,
    device: torch.device,
    frame_rate: float = 12.5,    # ≈ 50 Hz W2V2 frames / 4× downsample
):
    print("\n[Transformer] Running attention rollout + head entropy analysis...")
    x_real = x_real.to(device)
    x_fake = x_fake.to(device)

    attn_real = _get_attn_maps(model, x_real)
    attn_fake = _get_attn_maps(model, x_fake)

    n_layers = len(attn_real)
    first    = attn_real[0]
    n_heads  = first.shape[1] if first.dim() == 4 else 1
    per_head = (first.dim() == 4)

    rollout_real = _rollout(attn_real)                        # (B, T, T)
    rollout_fake = _rollout(attn_fake)
    imp_real = rollout_real.mean(dim=1).mean(0).numpy()       # (T,) avg importance
    imp_fake = rollout_fake.mean(dim=1).mean(0).numpy()
    T = len(imp_real)
    t_ax = np.arange(T) / frame_rate

    # ── Figure 1: Rollout importance + matrix ─────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        "Attention Rollout — Transformer Head\n"
        "Which temporal frames matter most for real vs. fake decisions?",
        fontsize=13, fontweight="bold",
    )

    for col, (imp, rollout, label, color, cmap) in enumerate([
        (imp_real, rollout_real, "Real", "#2196F3", "Blues"),
        (imp_fake, rollout_fake, "Fake", "#F44336", "Reds"),
    ]):
        # Row 0: importance curve
        ax = axes[0, col]
        ax.plot(t_ax, imp, color=color, lw=2.0)
        ax.fill_between(t_ax, imp, alpha=0.18, color=color)
        ax.set_title(f"Per-Frame Rollout Importance — {label}", fontsize=11)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Rollout score")
        ax.grid(True, alpha=0.4)
        ax.set_xlim(t_ax[0], t_ax[-1])

        # Row 1: rollout matrix for first sample
        ax = axes[1, col]
        im = ax.imshow(rollout[0].numpy(), aspect="auto", cmap=cmap,
                       interpolation="nearest", origin="upper")
        ax.set_title(f"Rollout Matrix — {label} (sample 0)", fontsize=11)
        ax.set_xlabel("Source frame (key)")
        ax.set_ylabel("Destination frame (query)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    p = out_dir / "transformer_attention_rollout.pdf"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  Saved → {p}")
    plt.close(fig)

    # ── Figure 2: Per-head entropy ─────────────────────────────────────────────
    eps = 1e-8
    def _entropy(attn_maps):
        out = {}
        for i, A in attn_maps.items():
            if A.dim() == 4:
                H = -(A * torch.log(A + eps)).sum(-1).mean(-1)  # (B, n_heads)
            else:
                H = -(A * torch.log(A + eps)).sum(-1).mean(-1).unsqueeze(1)
            out[i] = H
        return out

    ent_real = _entropy(attn_real)
    ent_fake = _entropy(attn_fake)

    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 5), sharey=True)
    if n_layers == 1:
        axes = [axes]
    fig.suptitle(
        "Per-Head Attention Entropy: Real vs. Fake\n"
        "(lower = more focused; bigger gap = more discriminative head)",
        fontsize=12, fontweight="bold",
    )
    x_pos = np.arange(n_heads)
    w = 0.38
    for l, ax in enumerate(axes):
        e_r = ent_real[l].mean(0).numpy()
        e_f = ent_fake[l].mean(0).numpy()
        ax.bar(x_pos - w / 2, e_r, w, label="Real", color="#2196F3", alpha=0.85)
        ax.bar(x_pos + w / 2, e_f, w, label="Fake", color="#F44336", alpha=0.85)
        # Annotate heads with large real/fake gap
        max_ent = max(e_r.max(), e_f.max()) + 1e-8
        for h in range(n_heads):
            gap = abs(e_f[h] - e_r[h])
            if gap > 0.06 * max_ent:
                ax.annotate(
                    f"Δ{gap:.2f}",
                    xy=(h, max(e_r[h], e_f[h]) + 0.01 * max_ent),
                    ha="center", fontsize=7, color="#333",
                )
        ax.set_title(f"Layer {l + 1}", fontsize=11)
        ax.set_xlabel("Head index")
        ax.set_xticks(x_pos)
        if l == 0:
            ax.set_ylabel("Entropy (nats)")
        ax.grid(True, alpha=0.4, axis="y")
        if l == n_layers - 1:
            ax.legend(fontsize=9)

    plt.tight_layout()
    p = out_dir / "transformer_head_entropy.pdf"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  Saved → {p}")
    plt.close(fig)

    # ── Figure 3: Raw attention map grid (first sample, all layers × heads) ───
    if per_head:
        fig, axes = plt.subplots(n_layers, n_heads,
                                 figsize=(2.2 * n_heads, 2.2 * n_layers),
                                 squeeze=False)
        fig.suptitle(
            "Raw Attention Maps — All Layers × Heads (Real, sample 0)\n"
            "Row = Layer, Column = Head",
            fontsize=11, fontweight="bold",
        )
        for l in range(n_layers):
            A = attn_real[l]           # (B, H, T, T)
            for h in range(n_heads):
                ax = axes[l, h]
                ax.imshow(A[0, h].numpy(), aspect="auto", cmap="viridis",
                          interpolation="nearest", vmin=0)
                ax.set_xticks([]); ax.set_yticks([])
                if l == 0:
                    ax.set_title(f"H{h}", fontsize=9)
                if h == 0:
                    ax.set_ylabel(f"L{l+1}", fontsize=9)
        plt.tight_layout()
        p = out_dir / "transformer_raw_attn_grid.pdf"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved → {p}")
        plt.close(fig)

    print("[Transformer] Done.")


# =============================================================================
# ── METHOD 2: KAN — B-Spline Visualization + Input-Feature Importance ────────
# =============================================================================

def _find_kan_layers(model: nn.Module) -> list:
    """Collect all KANLinear layers from model.kan_head."""
    if KANLinear is None:
        return []
    return [m for m in model.kan_head.modules() if isinstance(m, KANLinear)]


def _edge_importance(layer) -> torch.Tensor:
    """L2 norm of scaled spline weights + absolute base weight per edge.

    Returns (out_features, in_features) importance matrix.
    """
    spline_imp = layer.scaled_spline_weight.data.pow(2).sum(-1).sqrt()  # (out, in)
    base_imp   = layer.base_weight.data.abs()                           # (out, in)
    return spline_imp + base_imp


def _bspline_edge_fn(
    layer,
    in_idx: int,
    out_idx: int,
    n_pts: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_vals, base_vals, spline_vals, total_vals) for edge (in→out).

    Constructs a synthetic batch where only dimension in_idx varies so we can
    call layer.b_splines() — which expects (batch, in_features) — efficiently.
    """
    grid = layer.grid[in_idx]         # (n_knots_extended,)
    x_min = grid[layer.spline_order].item()
    x_max = grid[-(layer.spline_order + 1)].item()

    x_vals = torch.linspace(x_min, x_max, n_pts)

    # Build synthetic batch: all zeros except in_idx column
    x_batch = torch.zeros(n_pts, layer.in_features)
    x_batch[:, in_idx] = x_vals

    with torch.no_grad():
        bases = layer.b_splines(x_batch)   # (n_pts, in_features, n_basis)

    basis_at_in = bases[:, in_idx, :]                                  # (n_pts, n_basis)
    w_spline    = layer.scaled_spline_weight[out_idx, in_idx, :]       # (n_basis,)
    spline_vals = (basis_at_in * w_spline.unsqueeze(0)).sum(-1)        # (n_pts,)

    base_w    = layer.base_weight[out_idx, in_idx].item()
    base_vals = F.silu(x_vals) * base_w

    total_vals = base_vals + spline_vals
    return (
        x_vals.numpy(),
        base_vals.detach().numpy(),
        spline_vals.detach().numpy(),
        total_vals.detach().numpy(),
    )


def run_kan_interpretability(
    model: nn.Module,
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    out_dir: Path,
    device: torch.device,
    top_k: int = 12,
):
    print("\n[KAN] Running B-spline edge visualization + input-feature importance...")
    layers = _find_kan_layers(model)
    if not layers:
        print("  No KANLinear layers found — skipping.")
        return

    first_layer = layers[0]

    # ── Figure 1: Input-feature importance bar chart ───────────────────────────
    imp_matrix = _edge_importance(first_layer)           # (out, in)
    in_imp     = imp_matrix.sum(0).detach().cpu().numpy() # (in_features,)
    top_n      = min(40, len(in_imp))
    top_idx    = np.argsort(in_imp)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 1.0, top_n)[::-1])
    ax.bar(range(top_n), in_imp[top_idx], color=colors)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([f"d{i}" for i in top_idx], rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("W2V2 latent dimension fed into KAN head (after ASP)", fontsize=11)
    ax.set_ylabel("Edge importance  ||spline||₂ + |base|  (summed over outputs)", fontsize=11)
    ax.set_title(
        f"Top-{top_n} Input Features — KAN Layer 1\n"
        "Which Wav2Vec2 latent dimensions drive the deepfake decision?",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.4, axis="y")
    plt.tight_layout()
    p = out_dir / "kan_input_feature_importance.pdf"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  Saved → {p}")
    plt.close(fig)

    # ── Figure 2: Top-K spline edge functions ──────────────────────────────────
    flat_imp   = imp_matrix.flatten()
    top_k_flat = flat_imp.topk(top_k).indices
    out_idxs   = (top_k_flat // imp_matrix.shape[1]).cpu().numpy()
    in_idxs    = (top_k_flat %  imp_matrix.shape[1]).cpu().numpy()
    imp_vals   = flat_imp[top_k_flat].detach().cpu().numpy()

    ncols = 4
    nrows = math.ceil(top_k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows), squeeze=False)
    fig.suptitle(
        f"Top-{top_k} Learned B-Spline Edge Functions — KAN Layer 1\n"
        "Blue = spline component  |  Orange dashed = SiLU base  |  Black = total",
        fontsize=12, fontweight="bold",
    )
    imp_norm = plt.Normalize(imp_vals.min(), imp_vals.max())
    cmap_imp  = plt.cm.RdYlGn_r

    for rank, (o, i, imp_v) in enumerate(zip(out_idxs, in_idxs, imp_vals)):
        ax = axes[rank // ncols, rank % ncols]
        x_v, base_v, spline_v, total_v = _bspline_edge_fn(first_layer, int(i), int(o))
        ax.plot(x_v, spline_v, color="#2196F3", lw=1.5, label="Spline", alpha=0.85)
        ax.plot(x_v, base_v,   color="#FF9800", lw=1.5, ls="--", label="Base (SiLU×w)")
        ax.plot(x_v, total_v,  color="#212121", lw=2.0, label="Total")
        ax.axhline(0, color="gray", lw=0.5, ls=":")
        ax.set_title(f"#{rank+1}: in_{i}→out_{o}  (imp={imp_v:.3f})", fontsize=8, pad=3)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        # Color border by importance
        border_color = cmap_imp(imp_norm(imp_v))
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color); spine.set_linewidth(2.0)

    for ax in axes.flatten()[top_k:]:
        ax.set_visible(False)

    legend_elements = [
        plt.Line2D([0], [0], color="#2196F3", lw=2, label="Spline"),
        plt.Line2D([0], [0], color="#FF9800", lw=2, ls="--", label="Base (SiLU×w)"),
        plt.Line2D([0], [0], color="#212121", lw=2, label="Total"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    p = out_dir / "kan_top_spline_functions.pdf"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  Saved → {p}")
    plt.close(fig)

    # ── Figure 3: Importance heatmap (out × in) for first KAN layer ───────────
    imp_np = imp_matrix.detach().cpu().numpy()           # (out, in)
    fig, ax = plt.subplots(figsize=(min(20, imp_np.shape[1] * 0.15 + 2),
                                    max(3, imp_np.shape[0] * 0.5)))
    im = ax.imshow(imp_np, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Input dimension (W2V2 latent after ASP)", fontsize=11)
    ax.set_ylabel("Output neuron", fontsize=11)
    ax.set_title(
        "KAN Layer 1 — Edge Importance Heatmap\n"
        "Bright = high importance edge  ||spline||₂ + |base|",
        fontsize=12, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    p = out_dir / "kan_edge_importance_heatmap.pdf"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  Saved → {p}")
    plt.close(fig)

    print("[KAN] Done.")


# =============================================================================
# ── METHOD 3: NEURAL ODE — Latent Trajectory Analysis ────────────────────────
# =============================================================================

def _extract_ode_trajectories(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    chunk_size: int = 8,
) -> torch.Tensor:
    """Extract ODE trajectories by hooking model.ode_head in chunks."""
    all_trajs = []
    original_forward = model.ode_head.forward

    for start_idx in range(0, x.shape[0], chunk_size):
        chunk_x = x[start_idx:start_idx + chunk_size]
        traj_cache = {}

        def _patched(x_in):
            logits, z_traj = original_forward(x_in)
            traj_cache["z_traj"] = z_traj.detach().cpu()
            return logits, z_traj

        model.ode_head.forward = _patched
        with torch.no_grad():
            model(chunk_x.to(device))
            
        all_trajs.append(traj_cache["z_traj"])

    model.ode_head.forward = original_forward   # restore
    return torch.cat(all_trajs, dim=0)   # (B, N_STEPS, hidden_dim)


def run_ode_interpretability(
    model: nn.Module,
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    out_dir: Path,
    device: torch.device,
):
    from sklearn.decomposition import PCA
    from scipy import stats

    print("\n[ODE] Running latent-trajectory analysis...")

    traj_real = _extract_ode_trajectories(model, x_real, device)  # (B_r, N, H)
    traj_fake = _extract_ode_trajectories(model, x_fake, device)  # (B_f, N, H)

    B_r, N, H = traj_real.shape
    B_f = traj_fake.shape[0]
    t_axis = np.linspace(0, 1, N)

    # ── PCA to 2D ─────────────────────────────────────────────────────────────
    all_traj = torch.cat([traj_real, traj_fake], dim=0)            # (B_r+B_f, N, H)
    flat     = all_traj.reshape(-1, H).numpy()
    pca      = PCA(n_components=2)
    reduced  = pca.fit_transform(flat).reshape(B_r + B_f, N, 2)
    red_real = reduced[:B_r]
    red_fake = reduced[B_r:]
    evr      = pca.explained_variance_ratio_

    # ── Figure 1: Trajectory PCA ───────────────────────────────────────────────
    n_show = min(20, B_r, B_f)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    fig.suptitle(
        "Neural ODE Latent Trajectories: Real vs. Fake Audio\n"
        "(Each line = one audio sample's continuous path through latent space;\n"
        " ○ = start t=0,  ★ = end t=1,  colour encodes integration time)",
        fontsize=12, fontweight="bold",
    )
    xlabel = f"PC 1  ({evr[0]*100:.1f}% var)"
    ylabel = f"PC 2  ({evr[1]*100:.1f}% var)"

    for ax, (trajs, label, base_color, cmap_name) in zip(axes, [
        (red_real[:n_show], "Real",  "#2196F3", "Blues"),
        (red_fake[:n_show], "Fake",  "#F44336", "Reds"),
    ]):
        for traj in trajs:
            pts  = traj.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc   = LineCollection(segs, cmap=cmap_name,
                                  norm=plt.Normalize(0, 1),
                                  linewidth=1.5, alpha=0.55)
            lc.set_array(t_axis[:-1])
            ax.add_collection(lc)
        ax.scatter(trajs[:, 0, 0], trajs[:, 0, 1],
                   s=45, facecolors="white", edgecolors=base_color, lw=1.5,
                   zorder=5, label="t = 0")
        ax.scatter(trajs[:, -1, 0], trajs[:, -1, 1],
                   s=70, marker="*", c=base_color, zorder=5, label="t = 1")
        ax.set_title(f"{label}  (n={len(trajs)})", fontsize=11)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.autoscale()

    sm = plt.cm.ScalarMappable(cmap="Greys", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.55, pad=0.01)
    cbar.set_label("Integration time  t ∈ [0, 1]", fontsize=10)

    plt.tight_layout()
    p = out_dir / "ode_trajectory_pca.pdf"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  Saved → {p}")
    plt.close(fig)

    # ── Figure 2: Class separation (Fisher) over time ─────────────────────────
    all_traj_np = all_traj.numpy()
    labels_np   = np.array([0] * B_r + [1] * B_f)
    fisher      = np.zeros(N)
    for t in range(N):
        z_t    = all_traj_np[:, t, :]
        z_r    = z_t[labels_np == 0];  z_f = z_t[labels_np == 1]
        mu_r   = z_r.mean(0);          mu_f = z_f.mean(0)
        var_r  = z_r.var(0).mean();    var_f = z_f.var(0).mean()
        fisher[t] = np.sum((mu_f - mu_r) ** 2) / (var_r + var_f + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_axis, fisher, "o-", color="#4CAF50", lw=2.5, ms=6)
    ax.fill_between(t_axis, fisher, alpha=0.18, color="#4CAF50")
    ax.axhline(fisher[0], color="gray", ls=":", lw=1.5,
               label=f"t=0  (Fisher={fisher[0]:.1f})")
    ax.set_xlabel("ODE integration time  t", fontsize=12)
    ax.set_ylabel("Fisher class separation  ↑ more separable", fontsize=11)
    ax.set_title(
        "Does the Neural ODE Progressively Separate Real from Fake?\n"
        "Fisher criterion at each ODE time step",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    p = out_dir / "ode_class_separation.pdf"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  Saved → {p}")
    plt.close(fig)

    # ── Figure 3: Trajectory curvature distribution ───────────────────────────
    def _curvature(traj_tensor: torch.Tensor) -> np.ndarray:
        z   = traj_tensor.numpy()
        vel = z[:, 1:, :] - z[:, :-1, :]
        acc = vel[:, 1:, :] - vel[:, :-1, :]
        return (acc ** 2).sum(-1).mean(-1)          # (B,)

    curv_r = _curvature(traj_real)
    curv_f = _curvature(traj_fake)
    mwu    = stats.mannwhitneyu(curv_r, curv_f, alternative="two-sided")

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot([curv_r, curv_f], positions=[0, 1],
                          showmeans=True, showmedians=True)
    for pc, col in zip(parts["bodies"], ["#2196F3", "#F44336"]):
        pc.set_facecolor(col); pc.set_alpha(0.55)
    rng = np.random.default_rng(0)
    ax.scatter(rng.normal(0, 0.05, len(curv_r)), curv_r,
               alpha=0.45, s=18, color="#2196F3", label="Real")
    ax.scatter(rng.normal(1, 0.05, len(curv_f)), curv_f,
               alpha=0.45, s=18, color="#F44336", label="Fake")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Real", "Fake"], fontsize=13)
    ax.set_ylabel("Trajectory curvature  ||d²z/dt²||²  (mean over steps)", fontsize=11)
    ax.set_title(
        "Neural ODE Trajectory Curvature: Real vs. Fake\n"
        "(higher = more 'jittery' latent path)",
        fontsize=12, fontweight="bold",
    )
    ax.text(0.5, 0.93, f"Mann–Whitney  p = {mwu.pvalue:.3e}",
            transform=ax.transAxes, ha="center", fontsize=10,
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.85))
    ax.legend(fontsize=10); ax.grid(True, alpha=0.4, axis="y")
    plt.tight_layout()
    p = out_dir / "ode_trajectory_curvature.pdf"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  Saved → {p}")
    plt.close(fig)

    print(f"  Fisher at t=0: {fisher[0]:.2f}  →  t=1: {fisher[-1]:.2f}"
          f"  (ratio {fisher[-1] / (fisher[0] + 1e-8):.2f}×)")
    print(f"  Curvature  real={curv_r.mean():.4f}  fake={curv_f.mean():.4f}"
          f"  (MW p={mwu.pvalue:.3e})")
    print("[ODE] Done.")


# =============================================================================
# ── METHOD 4: PINN — Physics Residual Analysis ───────────────────────────────
# =============================================================================

def _extract_features_seq(model: nn.Module, x: torch.Tensor, device: torch.device, chunk_size: int = 8) -> torch.Tensor:
    """Hook model.transformer to capture the (B, T, D) sequence BEFORE ASP."""
    all_seqs = []

    for start_idx in range(0, x.shape[0], chunk_size):
        chunk_x = x[start_idx:start_idx + chunk_size]
        cache = {}

        def _hook(mod, inp, out):
            cache["seq"] = out.detach().cpu()

        handle = model.transformer.register_forward_hook(_hook)
        with torch.no_grad():
            model(chunk_x.to(device))
        handle.remove()
        
        all_seqs.append(cache["seq"])

    return torch.cat(all_seqs, dim=0)   # (B, T, D)


def _smoothness_residual(seq: torch.Tensor, order: int = 2) -> torch.Tensor:
    """Per-sample 2nd-order temporal derivative energy  E[||Δ²h||²]. Returns (B,)."""
    d = seq
    for _ in range(order):
        d = d[:, 1:, :] - d[:, :-1, :]
    return (d ** 2).mean(dim=(1, 2))   # (B,)


def run_pinn_interpretability(
    model: nn.Module,
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    out_dir: Path,
    device: torch.device,
    frame_rate: float = 12.5,
):
    from scipy.stats import gaussian_kde, ks_2samp
    from sklearn.metrics import roc_auc_score, roc_curve

    print("\n[PINN] Running physics-residual analysis...")

    seq_real = _extract_features_seq(model, x_real, device)   # (B_r, T, D)
    seq_fake = _extract_features_seq(model, x_fake, device)   # (B_f, T, D)

    res_r = _smoothness_residual(seq_real).numpy()   # (B_r,)
    res_f = _smoothness_residual(seq_fake).numpy()   # (B_f,)

    residuals = np.concatenate([res_r, res_f])
    labels    = np.array([0] * len(res_r) + [1] * len(res_f))
    ks_stat, ks_p = ks_2samp(res_r, res_f)
    auc            = roc_auc_score(labels, residuals)
    fpr, tpr, _    = roc_curve(labels, residuals)

    # ── Figure 1: Residual distribution + ROC ─────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "PINN Physics Residual: Acoustic Smoothness Violation as Deepfake Fingerprint\n"
        r"$r(x) = \mathbb{E}_{t,d}[(\Delta^2 h_t)^2]$  — does synthetic audio violate smooth-dynamics priors?",
        fontsize=12, fontweight="bold",
    )

    clip_99 = np.percentile(residuals, 99)
    bins    = np.linspace(residuals.min(), clip_99, 60)
    ax1.hist(res_r, bins=bins, density=True, alpha=0.55, color="#2196F3", label="Real")
    ax1.hist(res_f, bins=bins, density=True, alpha=0.55, color="#F44336", label="Fake")
    xr = np.linspace(bins[0], bins[-1], 200)
    ax1.plot(xr, gaussian_kde(res_r[res_r <= clip_99])(xr), color="#1565C0", lw=2)
    ax1.plot(xr, gaussian_kde(res_f[res_f <= clip_99])(xr), color="#B71C1C", lw=2)
    ax1.set_xlabel("Physics residual  r(x)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title(f"Residual Distribution\nKS stat = {ks_stat:.3f},  p = {ks_p:.2e}", fontsize=11)
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.4)

    ax2.plot(fpr, tpr, color="#4CAF50", lw=2.5, label=f"AUC = {auc:.3f}")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax2.fill_between(fpr, tpr, alpha=0.10, color="#4CAF50")
    ax2.set_xlabel("False Positive Rate", fontsize=11)
    ax2.set_ylabel("True Positive Rate", fontsize=11)
    ax2.set_title("ROC — Physics Residual as Standalone Classifier", fontsize=11)
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.4)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    plt.tight_layout()
    p = out_dir / "pinn_residual_distribution.pdf"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  Saved → {p}")
    plt.close(fig)

    # ── Figure 2: Temporal violation profile (when in time?) ──────────────────
    def _temporal_profile(seq: torch.Tensor) -> np.ndarray:
        d = seq
        for _ in range(2):
            d = d[:, 1:, :] - d[:, :-1, :]
        return (d ** 2).mean(dim=2).mean(dim=0).numpy()    # (T-2,)

    prof_r = _temporal_profile(seq_real)
    prof_f = _temporal_profile(seq_fake)
    T2     = len(prof_r)
    t_ax   = np.arange(T2) / frame_rate

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_ax, prof_r, color="#2196F3", lw=2, label="Real (avg)", alpha=0.9)
    ax.plot(t_ax, prof_f, color="#F44336", lw=2, label="Fake (avg)", alpha=0.9)
    ax.fill_between(t_ax, prof_r, prof_f,
                    where=(prof_f > prof_r), alpha=0.15, color="#F44336",
                    label="Fake > Real")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel(r"$\mathbb{E}_d[(\Delta^2 h_{t,d})^2]$  (avg over dims)", fontsize=11)
    ax.set_title(
        "Temporal Profile of Physics Violations: Real vs. Fake\n"
        "When (in time) does deepfake audio break acoustic-smoothness constraints?",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    p = out_dir / "pinn_temporal_violation_profile.pdf"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  Saved → {p}")
    plt.close(fig)

    # ── Figure 3: Per-dimension fingerprint (which latent dims are non-smooth?) ─
    def _dim_profile(seq: torch.Tensor) -> np.ndarray:
        d = seq
        for _ in range(2):
            d = d[:, 1:, :] - d[:, :-1, :]
        return (d ** 2).mean(dim=(0, 1)).numpy()    # (D,)

    dim_r   = _dim_profile(seq_real)
    dim_f   = _dim_profile(seq_fake)
    diff    = dim_f - dim_r                         # positive = fake violates more
    top_k_d = min(40, len(diff))
    top_dim = np.argsort(np.abs(diff))[::-1][:top_k_d]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        "Per-Dimension Physics Residual: Deepfake Fingerprint in W2V2 Latent Space\n"
        "Red = fake more non-smooth  |  Blue = real more non-smooth",
        fontsize=12, fontweight="bold",
    )
    D = len(diff)
    colors_all = np.where(diff > 0, "#F44336", "#2196F3")
    axes[0].bar(np.arange(D), diff, color=colors_all, alpha=0.75, linewidth=0)
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set_xlabel("W2V2 latent dimension (after secondary Transformer)", fontsize=11)
    axes[0].set_ylabel("Δresidual  (Fake − Real)", fontsize=11)
    axes[0].set_title("All dimensions", fontsize=10)
    axes[0].grid(True, alpha=0.3, axis="y")

    sorted_diff  = diff[top_dim]
    colors_top   = np.where(sorted_diff > 0, "#F44336", "#2196F3")
    axes[1].bar(range(top_k_d), sorted_diff, color=colors_top, alpha=0.85)
    axes[1].set_xticks(range(top_k_d))
    axes[1].set_xticklabels([f"d{d}" for d in top_dim],
                             rotation=60, ha="right", fontsize=8)
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set_ylabel("Δresidual  (Fake − Real)", fontsize=11)
    axes[1].set_title(f"Top-{top_k_d} most differentiating dimensions", fontsize=11)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    p = out_dir / "pinn_dimension_fingerprint.pdf"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  Saved → {p}")
    plt.close(fig)

    print(f"  Physics residual  real={res_r.mean():.4f} ± {res_r.std():.4f}"
          f"  fake={res_f.mean():.4f} ± {res_f.std():.4f}")
    print(f"  KS stat={ks_stat:.3f}  p={ks_p:.2e}  |  Residual-only AUC={auc:.3f}")
    print("[PINN] Done.")


# =============================================================================
# ── METHOD 5: INTEGRATED GRADIENTS — Cross-Architecture Comparison ────────────
# =============================================================================

def _integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int = 1,
    n_steps: int = 50,
) -> torch.Tensor:
    """Compute IG attributions (Sundararajan et al. 2017) on raw waveform input.

    Baseline = silence (zeros).  Returns (B, T_audio) attribution tensor.
    """
    baseline = torch.zeros_like(x)
    alphas   = torch.linspace(0, 1, n_steps, device=x.device)
    grad_acc = torch.zeros_like(x)
    model.eval()

    prev_grad = None
    for i, alpha in enumerate(alphas):
        x_interp = (baseline + alpha * (x - baseline)).requires_grad_(True)
        out       = model(x_interp)
        score     = out[:, target_class].sum()
        grad      = torch.autograd.grad(score, x_interp, create_graph=False)[0]

        if i == 0:
            prev_grad = grad.detach()
        else:
            # Trapezoidal rule
            delta_a  = alphas[i] - alphas[i - 1]
            grad_acc = grad_acc + (prev_grad + grad.detach()) / 2.0 * delta_a
            prev_grad = grad.detach()

    return grad_acc * (x - baseline)   # (B, T_audio)


def _frame_attrs(attrs: torch.Tensor, frame_len: int = 320) -> np.ndarray:
    """Aggregate sample-level IG to 20 ms frames via L1 pooling. Returns (B, F)."""
    B, T  = attrs.shape
    n_f   = T // frame_len
    chunk = attrs[:, :n_f * frame_len].reshape(B, n_f, frame_len)
    return chunk.abs().sum(-1).cpu().numpy()   # (B, F)


def run_ig_interpretability(
    models: Dict[str, nn.Module],
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    out_dir: Path,
    device: torch.device,
    n_steps: int = 50,
    frame_len: int = 320,   # 20 ms @ 16 kHz
):
    from scipy import stats as sp_stats

    print("\n[IG] Running Integrated Gradients for all available models...")

    # Compute for fake class (class 1) on a small subset to keep runtime tractable
    n_ig = min(8, x_real.shape[0], x_fake.shape[0])
    x_r  = x_real[:n_ig].to(device)
    x_f  = x_fake[:n_ig].to(device)

    all_attrs_real: Dict[str, np.ndarray] = {}
    all_attrs_fake: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        if model is None:
            continue
        print(f"  Computing IG for {name}  (n_steps={n_steps}, n={n_ig} per class)...")
        with torch.enable_grad():
            attr_r = _integrated_gradients(model, x_r, target_class=1, n_steps=n_steps)
            attr_f = _integrated_gradients(model, x_f, target_class=1, n_steps=n_steps)
        all_attrs_real[name] = _frame_attrs(attr_r, frame_len)
        all_attrs_fake[name] = _frame_attrs(attr_f, frame_len)
        print(f"    {name}: attribution shape {all_attrs_fake[name].shape}")

    if not all_attrs_fake:
        print("[IG] No models available — skipping.")
        return

    model_names = list(all_attrs_fake.keys())
    n_models    = len(model_names)
    SR          = 16_000
    n_frames    = list(all_attrs_fake.values())[0].shape[1]
    t_ax        = np.arange(n_frames) * frame_len / SR

    # ── Figure 1: Side-by-side IG attribution panels for one sample ───────────
    colors = {
        "Transformer": "#9C27B0",
        "KAN":         "#FF9800",
        "NeuralODE":   "#4CAF50",
        "PINN":        "#F44336",
    }
    default_colors = ["#9C27B0", "#FF9800", "#4CAF50", "#F44336", "#00BCD4"]

    for batch_cls, all_attrs, cls_label in [
        (x_r, all_attrs_real, "Real"),
        (x_f, all_attrs_fake, "Fake"),
    ]:
        n_rows = n_models + 1    # +1 for waveform
        fig, axes = plt.subplots(n_rows, 1,
                                 figsize=(16, 2.5 * n_rows),
                                 sharex=True)
        if n_rows == 1:
            axes = [axes]
        fig.suptitle(
            f"Integrated Gradients — {cls_label} Audio (sample 0)\n"
            "Positive bars = evidence for FAKE  |  Negative = evidence for REAL",
            fontsize=12, fontweight="bold",
        )
        sample = 0
        for ax_idx, (name, attrs) in enumerate(all_attrs.items()):
            ax   = axes[ax_idx]
            attr = attrs[sample]
            col  = colors.get(name, default_colors[ax_idx % len(default_colors)])
            bar_w = frame_len / SR * 0.9
            ax.bar(t_ax, np.maximum(attr, 0), width=bar_w,
                   color=col,    alpha=0.85, align="edge", label="→ Fake")
            ax.bar(t_ax, np.minimum(attr, 0), width=bar_w,
                   color="gray", alpha=0.45, align="edge", label="→ Real")
            ax.axhline(0, color="black", lw=0.5)
            ax.set_ylabel(f"{name}\nAttrib.", fontsize=8)
            ax.grid(True, alpha=0.3)
            if ax_idx == 0:
                ax.legend(loc="upper right", fontsize=8)

        # Waveform overlay
        ax  = axes[n_models]
        wav = batch_cls[sample].cpu().numpy()
        t_w = np.arange(len(wav)) / SR
        ax.plot(t_w, wav, color="#607D8B", lw=0.5, alpha=0.8)
        ax.set_ylabel("Waveform", fontsize=8)
        ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)", fontsize=12)

        plt.tight_layout()
        p = out_dir / f"ig_attribution_comparison_{cls_label.lower()}.pdf"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved → {p}")
        plt.close(fig)

    # ── Figure 2: Attribution mean profile per model (avg over samples) ────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    fig.suptitle(
        "Mean IG Attribution Profile per Architecture\n"
        "(averaged over all samples; shaded = ±1 std)",
        fontsize=12, fontweight="bold",
    )
    for ax, (all_attrs, cls_label) in zip(axes, [
        (all_attrs_real, "Real audio"),
        (all_attrs_fake, "Fake audio"),
    ]):
        for idx, (name, attrs) in enumerate(all_attrs.items()):
            col   = colors.get(name, default_colors[idx % len(default_colors)])
            mean  = attrs.mean(0)
            std   = attrs.std(0)
            ax.plot(t_ax, mean, color=col, lw=2, label=name)
            ax.fill_between(t_ax, mean - std, mean + std, alpha=0.12, color=col)
        ax.set_title(f"{cls_label}", fontsize=11)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("L1-pooled attribution (20 ms frames)", fontsize=10)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
        ax.set_xlim(t_ax[0], t_ax[-1])

    plt.tight_layout()
    p = out_dir / "ig_mean_profile_per_model.pdf"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  Saved → {p}")
    plt.close(fig)

    # ── Figure 3: Inter-model attribution correlation matrix ──────────────────
    if n_models > 1:
        pearson  = np.zeros((n_models, n_models))
        spearman = np.zeros((n_models, n_models))
        # Use fake attributions for correlation (the classification-critical class)
        attr_vecs = {n: all_attrs_fake[n].flatten() for n in model_names}
        for i, ni in enumerate(model_names):
            for j, nj in enumerate(model_names):
                if i == j:
                    pearson[i, j] = spearman[i, j] = 1.0
                else:
                    ai, aj = attr_vecs[ni], attr_vecs[nj]
                    pearson[i, j]  = np.corrcoef(ai, aj)[0, 1]
                    spearman[i, j] = sp_stats.spearmanr(ai, aj).correlation

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            "Attribution Correlation Between Architectures (fake audio)\n"
            "High corr = shared strategy  |  Low corr = architecture-specific behaviour",
            fontsize=12, fontweight="bold",
        )
        for ax, mat, title in [
            (axes[0], pearson,  "Pearson Correlation"),
            (axes[1], spearman, "Spearman Correlation"),
        ]:
            im = ax.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="equal")
            ax.set_xticks(range(n_models))
            ax.set_yticks(range(n_models))
            ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=11)
            ax.set_yticklabels(model_names, fontsize=11)
            ax.set_title(title, fontsize=11)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            for i in range(n_models):
                for j in range(n_models):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                            fontsize=11, fontweight="bold",
                            color="white" if abs(mat[i, j]) > 0.5 else "black")

        plt.tight_layout()
        p = out_dir / "ig_attribution_correlation_matrix.pdf"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved → {p}")
        plt.close(fig)

    print("[IG] Done.")


# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    CKPT = PROJECT_ROOT / "checkpoints"

    parser = argparse.ArgumentParser(
        description="Architecture-specific interpretability for deepfake audio detectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Checkpoints (auto-discover latest run if not specified)
    parser.add_argument("--transformer_ckpt", type=str, default=None,
                        help="Path to Transformer best_model.pt. "
                             "Auto-discovers latest run if omitted.")
    parser.add_argument("--kan_ckpt", type=str, default=None,
                        help="Path to KAN best_model.pt.")
    parser.add_argument("--ode_ckpt", type=str, default=None,
                        help="Path to Neural ODE best_model.pt.")
    parser.add_argument("--pinn_ckpt", type=str, default=None,
                        help="Path to PINN best_model.pt.")

    # Data
    parser.add_argument("--data_dir",   type=str, default="unified_dataset",
                        help="Directory containing test.csv (or val.csv).")
    parser.add_argument("--split",      type=str, default="test",
                        choices=["test", "val"])
    parser.add_argument("--n_samples",  type=int, default=64,
                        help="Samples per class to load (real + fake).")
    parser.add_argument("--seed",       type=int, default=42)

    # IG settings
    parser.add_argument("--ig_steps",   type=int, default=50,
                        help="Integration steps for Integrated Gradients.")

    # Output
    parser.add_argument("--output_dir", type=str, default="interpretability_figures")
    parser.add_argument("--device",     type=str, default="auto")

    # Skip flags (useful when only one model is ready)
    parser.add_argument("--skip_transformer", action="store_true")
    parser.add_argument("--skip_kan",         action="store_true")
    parser.add_argument("--skip_ode",         action="store_true")
    parser.add_argument("--skip_pinn",        action="store_true")
    parser.add_argument("--skip_ig",          action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"  Deepfake Audio Interpretability Suite")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Output] Figures will be saved to: {out_dir}\n")

    # ── Checkpoint resolution ─────────────────────────────────────────────────
    ckpt_paths = {
        "transformer": PROJECT_ROOT / "Best Models" / "transformer_hp_best.pt",
        "kan":         PROJECT_ROOT / "Best Models" / "kan_best.pt",
        "neural_ode":  PROJECT_ROOT / "Best Models" / "ode_best.pt",
        "pinn":        PROJECT_ROOT / "Best Models" / "pinn_best.pt",
    }
    
    # Override with arguments if provided
    if args.transformer_ckpt: ckpt_paths["transformer"] = Path(args.transformer_ckpt)
    if args.kan_ckpt: ckpt_paths["kan"] = Path(args.kan_ckpt)
    if args.ode_ckpt: ckpt_paths["neural_ode"] = Path(args.ode_ckpt)
    if args.pinn_ckpt: ckpt_paths["pinn"] = Path(args.pinn_ckpt)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n[Models] Loading checkpoints...")
    models = {}
    for mtype, ckpt_path in ckpt_paths.items():
        if ckpt_path is None:
            models[mtype] = None
        else:
            models[mtype] = load_model(mtype, ckpt_path, device)

    # ── Load data ─────────────────────────────────────────────────────────────
    data_dir = PROJECT_ROOT / args.data_dir
    csv_path = data_dir / f"{args.split}.csv"
    if not csv_path.exists():
        csv_path = data_dir / "val.csv"
        print(f"[Data] {args.split}.csv not found — falling back to val.csv")

    print(f"\n[Data] Loading {args.n_samples} real + {args.n_samples} fake samples "
          f"from {csv_path}...")
    x_real, x_fake, real_paths, fake_paths = load_balanced_batch(
        csv_path, n_per_class=args.n_samples, seed=args.seed
    )
    print(f"[Data] x_real: {x_real.shape}  x_fake: {x_fake.shape}\n")

    # ── Run interpretability analyses ─────────────────────────────────────────
    # 1. Transformer attention rollout
    if not args.skip_transformer and models.get("transformer"):
        run_transformer_interpretability(
            models["transformer"], x_real, x_fake, out_dir, device
        )

    # 2. KAN spline visualization
    if not args.skip_kan and models.get("kan"):
        run_kan_interpretability(
            models["kan"], x_real, x_fake, out_dir, device
        )

    # 3. Neural ODE trajectory analysis
    if not args.skip_ode and models.get("neural_ode"):
        run_ode_interpretability(
            models["neural_ode"], x_real, x_fake, out_dir, device
        )

    # 4. PINN physics residual analysis
    if not args.skip_pinn and models.get("pinn"):
        run_pinn_interpretability(
            models["pinn"], x_real, x_fake, out_dir, device
        )

    # 5. Cross-architecture Integrated Gradients
    if not args.skip_ig:
        # Only pass models that are available
        ig_models = {
            "Transformer": models.get("transformer"),
            "KAN":         models.get("kan"),
            "NeuralODE":   models.get("neural_ode"),
            "PINN":        models.get("pinn"),
        }
        ig_models = {k: v for k, v in ig_models.items() if v is not None}
        if ig_models:
            run_ig_interpretability(
                ig_models, x_real, x_fake, out_dir, device,
                n_steps=args.ig_steps,
            )
        else:
            print("[IG] No models loaded — skipping.")

    # ── Summary ───────────────────────────────────────────────────────────────
    saved = sorted(out_dir.glob("*.pdf"))
    print(f"\n{'='*60}")
    print(f"  Complete. {len(saved)} figures saved to {out_dir}/")
    print(f"{'='*60}")
    for f in saved:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
