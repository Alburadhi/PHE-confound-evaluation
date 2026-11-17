"""
Reproduces: results/per_case.csv, the single per-patient table referenced by
Tables III-VI

Joins the two source tables on the case identifier:

    nnunet_100ep_percase.csv   segmentation metrics and volumes, best checkpoint
    intensity_scale_audit.csv  per-volume intensity audit (Sec. IV-F)

Nothing is recomputed; this is a join. Both source files are shipped unchanged
so every other script runs against its original input.

The case identifier differs between the two: the metrics table carries the
nnU-Net filename ("PHE_0006.nii") and the audit table the bare case
("PHE_0006"). The join normalises on the bare case and fails loudly if the two
sets do not match exactly, because a silent partial join here would attach the
wrong audit flag to a patient.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))
from paths import RESULTS

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

COLUMNS = ["case", "fold", "dice", "iou", "recall", "precision", "hd95",
           "asd", "pred_vol_ml", "gt_vol_ml",
           "hu_min", "hu_max", "hu_range", "n_unique", "audit_flag",
           "audit_reason"]


def main():
    m = pd.read_csv(RESULTS / "nnunet_100ep_percase.csv")
    a = pd.read_csv(RESULTS / "intensity_scale_audit.csv")

    m["case"] = m["case"].str.replace(".nii", "", regex=False)
    only_m = set(m["case"]) - set(a["case"])
    only_a = set(a["case"]) - set(m["case"])
    if only_m or only_a:
        raise SystemExit(
            f"FATAL: case sets differ. Only in metrics: {sorted(only_m)[:5]}; "
            f"only in audit: {sorted(only_a)[:5]}")

    df = m.merge(a, on="case", how="inner", validate="one_to_one")
    df = df.rename(columns={"pred_vol": "pred_vol_ml", "gt_vol": "gt_vol_ml",
                            "min": "hu_min", "max": "hu_max",
                            "range": "hu_range", "suspect": "audit_flag",
                            "reason": "audit_reason"})
    df = df[COLUMNS].sort_values("case")

    out = RESULTS / "per_case.csv"
    df.to_csv(out, index=False)

    flagged = int(df["audit_flag"].sum())
    print(f"{len(df)} patients joined, {len(df.columns)} columns")
    print(f"mean Dice {df['dice'].mean():.4f} (best checkpoint)")
    print(f"intensity audit flags {flagged} of {len(df)} volumes: "
          f"{', '.join(df.loc[df.audit_flag == 1, 'case'])}")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
