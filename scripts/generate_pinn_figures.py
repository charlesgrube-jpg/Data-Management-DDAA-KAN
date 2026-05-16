
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Parameters from log
data = {
    'Epoch': list(range(1, 21)),
    'Train Loss': [0.4728, 0.3479, 0.3168, 0.3075, 0.2983, 0.2876, 0.1977, 0.1653, 0.1482, 0.1364, 0.1265, 0.1182, 0.1120, 0.1068, 0.1048, 0.1337, 0.1307, 0.1242, 0.1206, 0.1186],
    'Val Loss':   [0.3012, 0.2795, 0.2629, 0.2821, 0.2730, 0.1556, 0.2416, 0.2396, 0.2017, 0.3094, 0.2887, 0.2261, 0.2631, 0.2346, 0.2258, 0.2230, 0.3937, 0.1263, 0.1556, 0.2716],
    'Val Acc':    [0.8764, 0.8815, 0.8840, 0.8830, 0.8836, 0.9354, 0.9433, 0.9494, 0.9531, 0.9499, 0.9496, 0.9572, 0.9563, 0.9596, 0.9581, 0.9506, 0.9488, 0.9650, 0.9685, 0.9513],
    'Val F1':     [0.8557, 0.8627, 0.8660, 0.8639, 0.8646, 0.9159, 0.9204, 0.9306, 0.9352, 0.9305, 0.9301, 0.9421, 0.9398, 0.9447, 0.9422, 0.9318, 0.9284, 0.9536, 0.9576, 0.9320]
}

df = pd.DataFrame(data)

# Setup Style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('bmh')

plt.rcParams['font.family'] = 'sans-serif'

# Helper function
def save_plot(filename, title):
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Epoch', fontweight='bold')
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")

# 1. Loss Curve (Same as before)
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss', color='#3498db', linewidth=2.5, marker='o')
plt.plot(df['Epoch'], df['Val Loss'], label='Validation Loss', color='#e74c3c', linewidth=2.5, marker='s')
plt.axvline(x=6, color='gray', linestyle='--', alpha=0.7)
plt.text(6.2, 0.45, 'Encoder Unfrozen', rotation=0, color='#555555', fontweight='bold')
plt.ylabel('Loss (Cross-Entropy + Physics)', fontweight='bold')
save_plot('pinn_training_loss.png', 'PINN Training Dynamics: Physics Constraint + Wav2Vec2')

# 2. Test Accuracy Curve
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Val Acc'], label='Test Accuracy', color='#2ecc71', linewidth=2.5, marker='o')
best_acc = 0.9685
plt.plot(19, best_acc, marker='*', color='#f1c40f', markersize=15, markeredgecolor='black', zorder=10)
plt.text(19-3, best_acc-0.01, f'Best: {best_acc*100:.1f}%', fontweight='bold')
plt.axvline(x=6, color='gray', linestyle='--', alpha=0.7)
plt.ylabel('Accuracy', fontweight='bold')
save_plot('pinn_accuracy_curve.png', 'PINN Test Accuracy Over Epochs')

# 3. Test F1-Score Curve
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Val F1'], label='Test F1-Score', color='#9b59b6', linewidth=2.5, marker='^')
best_f1 = 0.9576
plt.plot(19, best_f1, marker='*', color='#f1c40f', markersize=15, markeredgecolor='black', zorder=10)
plt.text(19-3, best_f1-0.01, f'Best F1: {best_f1:.4f}', fontweight='bold')
plt.axvline(x=6, color='gray', linestyle='--', alpha=0.7)
plt.ylabel('F1 Score', fontweight='bold')
save_plot('pinn_f1_curve.png', 'PINN Test F1-Score Over Epochs')
