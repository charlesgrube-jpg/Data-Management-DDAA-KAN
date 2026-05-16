#!/usr/bin/env python3
"""
Approximate 95% confidence intervals for cross-generator AUC (Table 7 / tab:t2).

Uses the Hanley & McNeil (1982) analytic variance for the AUC when full score
logs are unavailable. Inputs must match the evaluation protocol: for each
generator, n_real = n_fake = 500 (Bark) or 2000 (MMS-TTS, SpeechT5).

Usage:
  python scripts/compute_crossarch_auc_cis.py \\
    --json new_generators/evaluation/crossarch_eval_complete.json \\
    --out   new_generators/evaluation/crossarch_auc_hm95ci.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    theta = float(auc)
    m, n = float(n_pos), float(n_neg)
    q1 = theta / (2.0 - theta)
    q2 = (2.0 * theta * theta) / (1.0 + theta)
    num = theta * (1.0 - theta) + (m - 1.0) * (q1 - theta * theta) + (n - 1.0) * (q2 - theta * theta)
    var = num / (m * n)
    return math.sqrt(max(var, 0.0))


def ci95(auc: float, n_pos: int, n_neg: int, z: float = 1.96) -> tuple[float, float]:
    se = hanley_mcneil_se(auc, n_pos, n_neg)
    lo = max(0.0, auc - z * se)
    hi = min(1.0, auc + z * se)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json",
        type=Path,
        default=Path("new_generators/evaluation/crossarch_eval_complete.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("new_generators/evaluation/crossarch_auc_hm95ci.json"),
    )
    args = ap.parse_args()

    sizes = {
        "Bark": (500, 500),
        "MMS-TTS": (2000, 2000),
        "SpeechT5": (2000, 2000),
    }
    data = json.loads(args.json.read_text())
    out: dict = {
        "_meta": {
            "method": "Hanley-McNeil-1982",
            "z": 1.96,
            "sizes": sizes,
            "note": (
                "Approximate AUC CIs when per-sample score vectors are unavailable; "
                "assumes balanced real/fake counts per generator (cross-arch eval logs). "
                "Prefer percentile bootstrap from stored scores when both classes are logged."
            ),
        }
    }

    for gen, models in data.items():
        if gen.startswith("_"):
            continue
        n_pos, n_neg = sizes[gen]
        out[gen] = {}
        for mname, stats in models.items():
            auc = float(stats["auc"])
            lo, hi = ci95(auc, n_pos, n_neg)
            out[gen][mname] = {
                "auc": auc,
                "auc_ci95_lo": round(lo, 4),
                "auc_ci95_hi": round(hi, 4),
                "auc_se_hm": round(hanley_mcneil_se(auc, n_pos, n_neg), 6),
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
