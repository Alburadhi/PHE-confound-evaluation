"""
Reproduces: Table VI and Fig. 7, the intensity audit (Sec. IV-F)
-> results/intensity_scale_audit.csv

Audit the intensity scale of every volume in the dataset.

WHY THIS SCRIPT EXISTS IN THIS FORM. A first version of this audit tested only
whether each volume's MINIMUM was near -1024 (air) and certified 114/115
volumes as "proper Hounsfield Units", finding a single 8-bit outlier
(PHE_0161, 0..255). That was wrong: seven further volumes (PHE_0160,
0162-0167) carry the same 8-bit quantization *shifted down by exactly 1024*,
which lands their minimum on a plausible air-HU value and slips straight past a
minimum-only test. The true affected count is 8/115 (7.0%), not 1/115.

That failure is structurally identical to an earlier bug in this project, where
a count-based check on the detection datasets passed because one missing
patient was coincidentally traded for another. In both cases the test looked
right, returned a clean result, and was blind along the one axis that mattered.
The fix in both cases is the same: test the property you actually care about,
from several independent angles, and make the check fail loudly.

A real diagnostic head CT should satisfy ALL of:
  * min near -1000 HU            (air outside the head)
  * max well above +300 HU       (bone, and usually metal/contrast)
  * thousands of distinct values (12-16 bit reconstruction)
An 8-bit-quantized volume fails the last two regardless of how it was shifted,
which is why max and unique-value count are both checked here and the range is
reported explicitly: an exact span of 255 with ~81-85 levels is the signature.

Output: results/intensity_scale_audit.csv, one row per case, with an explicit
`suspect` flag and the reason. Exits non-zero if any volume is unreadable.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))
from paths import (RESULTS, SPLITS, NNUNET_RAW, NNUNET_RESULTS,
                   NNUNET_PREPROCESSED, DATASET, DATASET_442, TRAINER, require)
import sys, csv
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import nibabel as nib

IMG = require(NNUNET_RAW / DATASET / "imagesTr", "the CT volumes")
OUT = RESULTS / "intensity_scale_audit.csv"

MAX_HU_FLOOR = 300      # bone; any real head CT exceeds this comfortably
MIN_UNIQUE = 500        # 12-bit+ reconstruction; 8-bit data has ~256 at most


def audit_one(path: Path) -> dict:
    a = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    mn, mx = float(a.min()), float(a.max())
    uniq = int(len(np.unique(a)))
    reasons = []
    if mx < MAX_HU_FLOOR:
        reasons.append(f"max {mx:.0f} < {MAX_HU_FLOOR} HU (no bone-density voxels)")
    if uniq < MIN_UNIQUE:
        reasons.append(f"only {uniq} distinct values (8-bit-like quantization)")
    if abs((mx - mn) - 255) < 1e-6:
        reasons.append("intensity span is exactly 255 (8-bit range)")
    return {
        "case": path.name.replace("_0000.nii.gz", ""),
        "min": round(mn, 1), "max": round(mx, 1),
        "range": round(mx - mn, 1), "n_unique": uniq,
        "suspect": int(bool(reasons)),
        "reason": "; ".join(reasons),
    }


if __name__ == "__main__":
    files = sorted(IMG.glob("*_0000.nii.gz"))
    if not files:
        raise SystemExit(f"FATAL: no volumes found in {IMG}")
    print(f"auditing {len(files)} volumes...")
    rows = [audit_one(f) for f in files]

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    bad = [r for r in rows if r["suspect"]]
    print(f"\n{'case':10s} {'min':>8s} {'max':>8s} {'range':>7s} {'uniq':>6s}  reason")
    for r in bad:
        print(f"{r['case']:10s} {r['min']:8.0f} {r['max']:8.0f} "
              f"{r['range']:7.0f} {r['n_unique']:6d}  {r['reason']}")
    print(f"\nSUSPECT: {len(bad)}/{len(rows)} = {len(bad)/len(rows)*100:.1f}%")
    print(f"wrote {OUT}")

    if bad:
        ids = sorted(int(r["case"].split("_")[1]) for r in bad)
        contiguous = ids == list(range(min(ids), max(ids) + 1))
        print(f"\naffected case IDs: {ids}")
        print(f"contiguous block: {contiguous}  "
              f"-- consistent with one acquisition/export batch, "
              f"not with random per-file corruption")
