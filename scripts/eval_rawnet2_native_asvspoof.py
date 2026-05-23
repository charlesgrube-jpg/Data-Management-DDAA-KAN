#!/usr/bin/env python3
"""
Evaluate RawNet2 pre-trained (ASVspoof-native) weights on ASVspoof 2019 LA eval set.
Goal: fill the missing AUC cell in Table 3 (ASVspoof-native baselines row).

Checkpoint: baselines/asvspoof2021/LA/Baseline-RawNet2/pre_trained_DF_RawNet2.pth
Protocol:   asvspoof_data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt
Audio:      asvspoof_data/LA/ASVspoof2019_LA_eval/flac/

RawNet2 outputs log_softmax — use torch.exp(out)[:, 1] for P(fake).
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT     = Path('/home/ms4726/project_pi_ev6/ms4726/Data-Management-DDAA-KAN')
RAWNET2_DIR = PROJECT / 'baselines' / 'asvspoof2021' / 'LA' / 'Baseline-RawNet2'
CKPT        = RAWNET2_DIR / 'pre_trained_DF_RawNet2.pth'
CONFIG      = RAWNET2_DIR / 'model_config_RawNet.yaml'
MODEL_PY    = RAWNET2_DIR / 'model.py'

ASV_ROOT    = PROJECT / 'asvspoof_data'
EVAL_AUDIO  = ASV_ROOT / 'LA' / 'ASVspoof2019_LA_eval' / 'flac'
EVAL_PROTO  = ASV_ROOT / 'LA' / 'ASVspoof2019_LA_cm_protocols' / 'ASVspoof2019.LA.cm.eval.trl.txt'

OUT_JSON    = PROJECT / 'rawnet2_native_asvspoof_result.json'

SR      = 16000
MAX_LEN = 64600  # ~4.04 s at 16 kHz (standard ASVspoof length)


# ── Audio loading ──────────────────────────────────────────────────────────────
def load_audio(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != SR:
        wav = T.Resample(sr, SR)(wav.unsqueeze(0)).squeeze(0)
    if wav.shape[0] > MAX_LEN:
        wav = wav[:MAX_LEN]
    elif wav.shape[0] < MAX_LEN:
        import torch.nn.functional as F
        wav = F.pad(wav, (0, MAX_LEN - wav.shape[0]))
    return torch.nan_to_num(wav / (wav.abs().max() + 1e-6))


# ── Dataset ────────────────────────────────────────────────────────────────────
class ASVspoofEvalDataset(Dataset):
    """ASVspoof 2019 LA eval set loaded from protocol file."""

    def __init__(self, audio_dir: Path, protocol_file: Path):
        self.audio_dir = audio_dir
        self.items = []

        with open(protocol_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # Format: SPEAKER_ID FILE_ID - SYSTEM_ID LABEL
                utt_id = parts[1]
                label  = 0 if parts[4] == 'bonafide' else 1

                p = audio_dir / f'{utt_id}.flac'
                if not p.exists():
                    p = audio_dir / f'{utt_id}.wav'
                if p.exists():
                    self.items.append((p, label))

        real_n = sum(1 for _, l in self.items if l == 0)
        fake_n = sum(1 for _, l in self.items if l == 1)
        print(f'[Dataset] {len(self.items)} clips ({real_n} bonafide, {fake_n} spoof)')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        return load_audio(path), label


def collate_fn(batch):
    xs, ls = zip(*batch)
    return torch.stack(xs), torch.tensor(ls, dtype=torch.long)


# ── Model loading ──────────────────────────────────────────────────────────────
def load_rawnet2(config_path: Path, ckpt_path: Path, model_py: Path, device):
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    spec = importlib.util.spec_from_file_location('rawnet2_model', str(model_py))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    model = mod.RawNet(cfg['model'], device).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print(f'[RawNet2] loaded from {ckpt_path}')
    return model


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Validate paths
    for p, name in [(CKPT, 'RawNet2 checkpoint'), (CONFIG, 'RawNet2 config'),
                    (MODEL_PY, 'model.py'), (EVAL_AUDIO, 'eval audio dir'),
                    (EVAL_PROTO, 'eval protocol')]:
        if not p.exists():
            print(f'[ERROR] {name} not found: {p}')
            sys.exit(1)

    model = load_rawnet2(CONFIG, CKPT, MODEL_PY, device)

    dataset = ASVspoofEvalDataset(EVAL_AUDIO, EVAL_PROTO)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                         num_workers=4, collate_fn=collate_fn, pin_memory=True)

    all_scores, all_labels = [], []

    with torch.no_grad():
        for i, (wavs, labels) in enumerate(loader):
            wavs = wavs.to(device)
            out  = model(wavs)
            if isinstance(out, (tuple, list)):
                out = out[0]
            # RawNet2 outputs log_softmax — convert to P(fake)
            probs = torch.exp(out)[:, 1]
            all_scores.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

            if i % 20 == 0:
                print(f'  Batch {i}/{len(loader)} ({100*i/len(loader):.0f}%)')

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    auc = float(roc_auc_score(labels, scores))
    eer, eer_thresh = compute_eer(scores, labels)

    result = {
        'model':      'RawNet2 (ASVspoof 2019 native pretrained)',
        'eval_set':   'ASVspoof 2019 LA eval',
        'n_samples':  len(labels),
        'n_bonafide': int((labels == 0).sum()),
        'n_spoof':    int((labels == 1).sum()),
        'auc':        auc,
        'eer':        eer,
        'eer_thresh': eer_thresh,
    }

    print('\n' + '='*50)
    print(f'RawNet2 Native — ASVspoof 2019 LA eval')
    print(f'  AUC : {auc*100:.2f}%')
    print(f'  EER : {eer*100:.2f}%')
    print('='*50)

    with open(OUT_JSON, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Result saved to {OUT_JSON}')


if __name__ == '__main__':
    main()
