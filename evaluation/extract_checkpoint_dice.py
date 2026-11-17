"""
Reproduces: the two five-fold means quoted in Sec. IV-E (0.432 final
checkpoint, 0.442 best checkpoint) -> results/fivefold_final_checkpoint.csv

WHY THIS SCRIPT EXISTS

The paper reports two five-fold means that differ by the checkpoint rule
alone, and treats that difference as one of the six measured confounds. The
per-case file shipped in results/ is the best-checkpoint run, so the
final-checkpoint mean cannot be recomputed from it. Rather than transcribe
five numbers by hand, this script recovers them from the nnU-Net training
logs, which are not redistributed.

HOW THE TWO VALIDATIONS ARE TOLD APART

nnU-Net writes one log per invocation. Training runs log every epoch and end
with a validation on the FINAL weights. A later `nnUNetv2_train ... --val
--val_best` run logs no epochs and validates the BEST checkpoint. So:

    final checkpoint  <- the log that contains training epochs
    best checkpoint   <- the earliest validation-only log

"Earliest" is resolved by parsing the timestamp out of the filename, not by
sorting the names: nnU-Net writes the month without zero padding, so
training_log_2026_10_26 sorts before training_log_2026_7_9 as a string.

A fold may legitimately hold more logs than these two -- an interrupted run
that was resumed, or a validation re-run. Logs with no "Mean Validation Dice"
line are ignored, and only the earliest validation-only log is taken, so a
later re-validation cannot displace the published figure.

USAGE
    python extract_checkpoint_dice.py --trainer-dir <path>

where <path> is the nnU-Net trainer directory holding fold_0 ... fold_4, e.g.
    nnUNet_results/Dataset001_PHE/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres
"""
import argparse
import csv
import re
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "results" / "fivefold_final_checkpoint.csv"
N_FOLDS = 5

STAMP = re.compile(r"training_log_(\d{4})_(\d{1,2})_(\d{1,2})_(\d{1,2})_"
                   r"(\d{1,2})_(\d{1,2})\.txt$")
DICE = re.compile(r"Mean Validation Dice:\s*([0-9.]+)")
EPOCH = re.compile(r"^\d{4}-\d{2}-\d{2}.*\bEpoch\s+\d+\s*$", re.M)


def parse_stamp(path: Path):
    m = STAMP.search(path.name)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups())
    return datetime(y, mo, d, h, mi, s)


def scan(log: Path):
    text = log.read_text(encoding="utf-8", errors="replace")
    dices = DICE.findall(text)
    if not dices:
        return None
    return {"path": log, "when": parse_stamp(log),
            "dice": float(dices[-1]),
            "n_epochs": len(EPOCH.findall(text))}


def fold_values(fold_dir: Path):
    logs = [r for r in (scan(p) for p in sorted(fold_dir.glob("training_log_*.txt")))
            if r and r["when"]]
    if not logs:
        raise SystemExit(f"FATAL: no usable logs in {fold_dir.name}")
    trained = [r for r in logs if r["n_epochs"] > 0]
    valonly = [r for r in logs if r["n_epochs"] == 0]
    if not trained:
        raise SystemExit(f"FATAL: {fold_dir.name} has no training log with a "
                         f"validation Dice; cannot recover the final checkpoint")
    if not valonly:
        raise SystemExit(f"FATAL: {fold_dir.name} has no validation-only log; "
                         f"cannot recover the best checkpoint")
    fin = max(trained, key=lambda r: r["when"])     # the completed run
    best = min(valonly, key=lambda r: r["when"])    # the first re-validation
    return fin, best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trainer-dir", required=True, type=Path,
                    help="nnU-Net trainer directory containing fold_0 .. fold_4")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()

    if not a.trainer_dir.is_dir():
        raise SystemExit(f"FATAL: not a directory: {a.trainer_dir}")

    rows = []
    for k in range(N_FOLDS):
        fd = a.trainer_dir / f"fold_{k}"
        if not fd.is_dir():
            raise SystemExit(f"FATAL: missing {fd.name}")
        fin, best = fold_values(fd)
        rows.append({"fold": k,
                     "dice_final_checkpoint": f"{fin['dice']:.10f}",
                     "dice_best_checkpoint": f"{best['dice']:.10f}",
                     # file names only; no directory component is recorded
                     "source_log_final": fin["path"].name,
                     "source_log_best": best["path"].name})
        print(f"fold {k}: final {fin['dice']:.4f} ({fin['n_epochs']} epochs "
              f"logged)   best {best['dice']:.4f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    mf = st.mean(float(r["dice_final_checkpoint"]) for r in rows)
    mb = st.mean(float(r["dice_best_checkpoint"]) for r in rows)
    print(f"\nfive-fold mean, final checkpoint : {mf:.4f}")
    print(f"five-fold mean, best  checkpoint : {mb:.4f}")
    print(f"checkpoint-selection difference  : {mb - mf:+.4f}")
    print(f"\nwrote {a.out.relative_to(a.out.parent.parent)}")


if __name__ == "__main__":
    main()
