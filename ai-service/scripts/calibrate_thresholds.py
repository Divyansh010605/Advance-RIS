"""
Calibrate per-model, per-disease decision thresholds on local NIH ChestX-ray14 images.

Usage (from ai-service/):  python scripts/calibrate_thresholds.py [--max-images 2200] [--tta]
Writes configs/model_thresholds.json (used by MultiModelEnsemble) and prints per-model AUC.
Threshold = Youden-J optimum. Diseases with <MIN_POS positives keep the default threshold;
diseases where a model is not informative (AUC < MIN_AUC) get threshold 1.01 (model abstains).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.densenet import NIH_LABELS, build_densenet121  # noqa: E402
from models.ensemble import MultiModelEnsemble  # noqa: E402
from xai.image_processor import _transform  # noqa: E402

MIN_POS, MIN_AUC, DEFAULT_THR = 12, 0.55, 0.50


def auc_score(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def youden_threshold(y, s):
    best_j, best_t = -1, DEFAULT_THR
    for t in np.unique(np.quantile(s, np.linspace(0.02, 0.98, 97))):
        pred = s >= t
        tpr = (pred & (y == 1)).sum() / max(1, (y == 1).sum())
        fpr = (pred & (y == 0)).sum() / max(1, (y == 0).sum())
        if tpr - fpr > best_j:
            best_j, best_t = tpr - fpr, float(t)
    return float(np.clip(best_t, 0.02, 0.98))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=2200)
    ap.add_argument("--tta", action="store_true")
    a = ap.parse_args()

    img_dir = ROOT / "data" / "images"
    df = pd.read_csv(ROOT / "data" / "Data_Entry_2017_v2020.csv")
    df = df[df["Image Index"].isin({p.name for p in img_dir.glob("*.png")})].head(a.max_images)
    print(f"Calibrating on {len(df)} images")

    ens = MultiModelEnsemble(num_classes=15)
    ens.register_model("densenet121", build_densenet121(num_classes=15, pretrained=True))
    ens.load_from_checkpoints()
    names = list(ens.models)
    print("Models:", names)

    probs = {m: [] for m in names}
    Y = []
    for i, (_, row) in enumerate(df.iterrows()):
        x = _transform(Image.open(img_dir / row["Image Index"]).convert("RGB")).unsqueeze(0)
        out = ens.predict(x, use_tta=a.tta, tta_steps=2)
        for m in names:
            probs[m].append(out["per_model_probabilities"][m][0][:14])
        labels = set(row["Finding Labels"].split("|"))
        Y.append([int(d in labels) for d in NIH_LABELS[:14]])
        if i % 100 == 0:
            print(f"  {i}/{len(df)}", flush=True)

    Y = np.array(Y)
    result, report = {}, {}
    for m in names:
        P = np.array(probs[m])
        result[m], report[m] = {}, {}
        for j, d in enumerate(NIH_LABELS[:14]):
            y, s = Y[:, j], P[:, j]
            if y.sum() < MIN_POS or y.sum() == len(y):
                result[m][d], report[m][d] = DEFAULT_THR, None
                continue
            auc = auc_score(y, s)
            report[m][d] = round(auc, 3)
            result[m][d] = 1.01 if auc < MIN_AUC else round(youden_threshold(y, s), 3)
        valid = [v for v in report[m].values() if v is not None]
        print(f"{m}: mean AUC {np.mean(valid):.3f}  per-disease {report[m]}")

    out = {"n_images": len(df), "thresholds": result, "auc": report}
    (ROOT / "configs" / "model_thresholds.json").write_text(json.dumps(out, indent=2))
    print("Wrote configs/model_thresholds.json")


if __name__ == "__main__":
    main()
