"""
Reproduces: the per-case 4:4:2 results behind Table IV (Sec. IV-C)
-> results/per_case_442.csv

Each of the five 4:4:2 replicates trains on 46 patients and is evaluated on a
held-out 23. nnU-Net writes one summary.json per replicate; this script reads
those and flattens them into a single table of (replicate, case, dice), from
which the replicate means and the 0.390 +/- 0.025 of Table IV are recomputed.

The summary.json files are not redistributed: they embed absolute paths to the
machine they were produced on. Only the case identifier and the Dice are taken
here, so the released CSV carries no path information.

Every replicate uses the FINAL checkpoint. The test patients occupy nnU-Net's
validation slot, so selecting the best checkpoint would be selecting on the
very patients being reported -- the checkpoint confound the paper measures in
Sec. IV-E. See splits/README.md.

USAGE
    python extract_442_percase.py --trainer-dir <path>

where <path> holds fold_0 .. fold_4 for the 4:4:2 dataset, e.g.
    nnUNet_results/Dataset004_PHE442/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres
"""
import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "results" / "per_case_442.csv"
N_REPLICATES = 5
EXPECTED_TEST = 23


def case_id(ref: str) -> str:
    """Take the case identifier only; discard every directory component."""
    return Path(ref.replace("\\", "/")).name.replace(".nii.gz", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trainer-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()

    rows, means = [], []
    for r in range(N_REPLICATES):
        f = a.trainer_dir / f"fold_{r}" / "validation" / "summary.json"
        if not f.exists():
            raise SystemExit(f"FATAL: missing {f.parent.parent.name}/validation/"
                             f"summary.json")
        per = json.loads(f.read_text())["metric_per_case"]
        d = [(case_id(m["reference_file"]), m["metrics"]["1"]["Dice"])
             for m in per]
        if len(d) != EXPECTED_TEST:
            raise SystemExit(f"FATAL: replicate {r} has {len(d)} test cases, "
                             f"expected {EXPECTED_TEST}")
        for case, dice in sorted(d):
            rows.append({"replicate": r, "case": case,
                         "dice": f"{dice:.10f}"})
        m = st.mean(x[1] for x in d)
        means.append(m)
        print(f"replicate {r}: n={len(d)}  mean Dice {m:.4f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["replicate", "case", "dice"])
        w.writeheader()
        w.writerows(rows)

    mean, sd = st.mean(means), st.stdev(means)
    print(f"\nacross replicates: {mean:.4f} +/- {sd:.4f} "
          f"(range {min(means):.4f}-{max(means):.4f})")
    print(f"{len(rows)} rows -> {a.out.name}")


if __name__ == "__main__":
    main()
