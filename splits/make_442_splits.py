"""
Reproduces: the five 4:4:2 replicate splits (Sec. IV-C, Table IV)
-> splits/splits_442_replicates.json   [seed 20260926]

Build five independent 4:4:2 patient-level splits, replicating the protocol of
the published benchmark (Ma et al. 2024).

WHY THIS EXISTS

Reviewer point 7 is correct: our 0.442 and the benchmark's 0.351 were not
obtained under the same protocol, so the difference cannot be attributed to the
method. Two confounds, both real:

  * TRAINING SET SIZE. The benchmark splits 4:4:2, so it trains on 40% of the
    cohort (~46 patients). Patient-level 5-fold CV trains on 80% (~92). Twice
    the training data is on its own a plausible explanation for much of the gap.
  * METRIC DEFINITION. The benchmark's architectures are 2D slice-wise and the
    paper does not state whether Dice was averaged over slices or computed on
    reconstructed volumes. Ours is 3D volumetric.

This script addresses the first. It cannot address the second; that stays a
stated limitation.

DESIGN, AND WHY IT IS LEAK-FREE

Each replicate assigns patients 46/46/23 (train/unused/test):

    train  = 46 patients (40%)  -- matches the benchmark's training budget
    unused = 46 patients (40%)  -- the benchmark's validation split, which we
                                   deliberately do NOT train on, so our model
                                   never sees more data than theirs could have
    test   = 23 patients (20%)  -- matches the benchmark's test budget

nnU-Net's splits_final.json carries only `train` and `val`, and its `--val`
step predicts whatever is in `val`. We therefore put the TEST patients in
`val`, and evaluate with the FINAL checkpoint rather than the best one. Using
`--val_best` here would select the checkpoint on the same patients we then
report, which is exactly the optimistic bias this experiment exists to avoid.

Five replicates with different random patient assignments give a distribution
rather than a point estimate. That is the honest comparator for a single
published number, and it simultaneously measures how much a single 4:4:2 split
can move on this cohort.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))
from paths import (RESULTS, SPLITS, NNUNET_RAW, NNUNET_RESULTS,
                   NNUNET_PREPROCESSED, DATASET, DATASET_442, TRAINER, require)
import json
import random
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SRC = NNUNET_PREPROCESSED / DATASET
DST = NNUNET_PREPROCESSED / DATASET_442

N_REPLICATES = 5
SEED = 20260926           # fixed so the splits are reproducible
FRAC_TRAIN, FRAC_TEST = 0.40, 0.20


def main():
    if not SRC.exists():
        raise SystemExit(f"FATAL: {SRC} not found")

    # the 115 usable patients, taken from the canonical 5-fold file so that
    # this experiment runs on exactly the same cohort as everything else
    base = json.loads((SPLITS / "fivefold_splits.json").read_text())
    patients = sorted(set(base[0]["train"]) | set(base[0]["val"]))
    n = len(patients)
    n_train = round(n * FRAC_TRAIN)
    n_test = round(n * FRAC_TEST)
    print(f"cohort: {n} patients -> train {n_train} / unused "
          f"{n - n_train - n_test} / test {n_test}")
    if n != 115:
        print(f"  WARNING: expected 115 patients, found {n}")

    rng = random.Random(SEED)
    splits, record = [], []
    for r in range(N_REPLICATES):
        shuffled = patients[:]
        rng.shuffle(shuffled)
        train = sorted(shuffled[:n_train])
        test = sorted(shuffled[n_train:n_train + n_test])
        unused = sorted(shuffled[n_train + n_test:])
        assert not (set(train) & set(test)), "train/test overlap"
        assert len(train) + len(test) + len(unused) == n

        # `val` carries the TEST patients; see the module docstring.
        splits.append({"train": train, "val": test})
        record.append({"replicate": r, "train": train,
                       "test": test, "unused": unused})
        print(f"  replicate {r}: train {len(train)}, test {len(test)}, "
              f"unused {len(unused)}  (first test case {test[0]})")

    # distinct test sets, otherwise the replicates are not independent
    sigs = {tuple(s["val"]) for s in splits}
    if len(sigs) != N_REPLICATES:
        raise SystemExit("FATAL: duplicate test sets across replicates")

    if DST.exists():
        print(f"\n{DST.name} already exists; refusing to overwrite. "
              f"Delete it first if you mean to regenerate.")
        return
    print(f"\ncopying preprocessed data -> {DST.name} (the canonical "
          f"Dataset001 splits are left untouched)")
    shutil.copytree(SRC, DST)

    # CRITICAL. nnU-Net locates both the preprocessed folder and the results
    # folder from plans["dataset_name"], NOT from the directory it loaded the
    # plans out of. A verbatim copy therefore still says "Dataset001_PHE", and
    # training silently reads Dataset001's splits, finds Dataset001's existing
    # checkpoint, reports "Training done." in under a second, and overwrites
    # Dataset001's validation predictions. That happened on the first attempt.
    for name in ("nnUNetPlans.json", "nnUNetResEncUNetMPlans.json"):
        f = DST / name
        if not f.exists():
            continue
        plans = json.loads(f.read_text())
        plans["dataset_name"] = DST.name
        f.write_text(json.dumps(plans, indent=2))
        print(f"  patched dataset_name in {name} -> {DST.name}")

    (DST / "splits_final.json").write_text(json.dumps(splits, indent=1))
    out = SPLITS / "splits_442_replicates.json"
    out.write_text(json.dumps(record, indent=1))
    print(f"wrote {DST / 'splits_final.json'}")
    print(f"wrote {out}  (full record incl. the unused 40%)")


if __name__ == "__main__":
    main()
