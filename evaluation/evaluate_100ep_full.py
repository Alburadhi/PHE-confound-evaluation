"""
Reproduces: the per-case table behind Tables III-V
-> results/nnunet_100ep_percase.csv

Full-metric re-evaluation of the 100-epoch nnU-Net (epoch-matched to the
original PHE-SICH-CT-IDS benchmark, Ma & Li 2024) so it can serve as the
paper's primary per-case table. Uses the validation predictions already on
disk; no retraining and no new inference are performed here.

CHECKPOINT. Those predictions come from `nnUNetv2_train ... --val
--val_best`, i.e. the BEST checkpoint, selected on the same held-out fold
that is then reported. Their five-fold mean is 0.442. The corresponding
FINAL-checkpoint mean is 0.432; the 0.010 difference is the checkpoint
confound the paper measures in Sec. IV-E, and is recovered by
evaluation/extract_checkpoint_dice.py. An earlier version of this
docstring described the predictions as coming from training's built-in
validation, which would have been the final checkpoint. That was wrong.

Adds asd, precision, recall, pred_vol, gt_vol on top of the original
evaluate_100ep.py (which only computed dice/iou/hd95), matching the full
metric set evaluate_5fold_final.py computed for the (now secondary) 250-epoch
run, so every downstream section of the manuscript can be recomputed
consistently from one 100-epoch source CSV.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))
from paths import (RESULTS, SPLITS, NNUNET_RAW, NNUNET_RESULTS,
                   NNUNET_PREPROCESSED, DATASET, DATASET_442, TRAINER, require)
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import distance_transform_edt

RES = require(NNUNET_RESULTS / DATASET / TRAINER, "the nnU-Net results")
GT = require(NNUNET_RAW / DATASET / "labelsTr", "the reference masks")
eps = 1e-6

def hd95(p, g, sp):
    if not p.any() or not g.any(): return np.nan
    dg = distance_transform_edt(~g, sampling=sp); dp = distance_transform_edt(~p, sampling=sp)
    return float(np.percentile(np.concatenate([dg[p], dp[g]]), 95))

def asd(p, g, sp):
    if not p.any() or not g.any(): return np.nan
    dg = distance_transform_edt(~g, sampling=sp); dp = distance_transform_edt(~p, sampling=sp)
    return float((dg[p].sum() + dp[g].sum()) / (p.sum() + g.sum()))

rows = []
for fold in range(5):
    vdir = RES / f"fold_{fold}" / "validation"
    for pp in sorted(vdir.glob("PHE_*.nii.gz")):
        gp = GT / pp.name
        if not gp.exists(): continue
        pn = nib.load(str(pp)); gn = nib.load(str(gp)); sp = gn.header.get_zooms()[:3]
        voxel_ml = (sp[0]*sp[1]*sp[2]) / 1000.0
        pred = pn.get_fdata() > 0.5; gt = gn.get_fdata() > 0.5
        if not gt.any(): continue
        tp = float((pred & gt).sum()); fp = float((pred & ~gt).sum()); fn = float((~pred & gt).sum())
        rows.append({
            "case": pp.stem, "fold": fold,
            "dice": (2*tp+eps)/(2*tp+fp+fn+eps),
            "iou": (tp+eps)/(tp+fp+fn+eps),
            "recall": (tp+eps)/(tp+fn+eps),
            "precision": (tp+eps)/(tp+fp+eps),
            "hd95": hd95(pred, gt, sp),
            "asd": asd(pred, gt, sp),
            "pred_vol": pred.sum()*voxel_ml,
            "gt_vol": gt.sum()*voxel_ml,
        })

df = pd.DataFrame(rows)
print(f"Total cases evaluated: {len(df)}")

def boot(v, B=2000):
    v = np.array([x for x in v if not np.isnan(x)])
    rng = np.random.default_rng(42)
    bs = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(B)]
    return v.mean(), v.std(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

print("="*70)
print("  nnU-Net 100-EPOCH (PRIMARY, epoch-matched) — full metric set, n={}".format(len(df)))
print("="*70)
for m, unit in [("dice",""),("iou",""),("recall",""),("precision",""),("hd95","mm"),("asd","mm")]:
    mn, sd, lo, hi = boot(df[m].values)
    print(f"  {m:<12}{mn:>9.4f}{sd:>9.4f}   [{lo:.4f}, {hi:.4f}] {unit}")

fold_means = [df[df.fold==f]["dice"].mean() for f in range(5)]
print("\n  Per-fold Dice: " + " / ".join(f"{v:.3f}" for v in fold_means))
print(f"  Mean of fold-means: {np.mean(fold_means):.4f} +/- {np.std(fold_means):.4f}")

df.to_csv(RESULTS / "nnunet_100ep_percase.csv", index=False)
print("\nSaved results/nnunet_100ep_percase.csv")
