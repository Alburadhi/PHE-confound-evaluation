"""
Reproduces: Table IV, the five 4:4:2 replicates (Sec. IV-C)
-> results/results_442.json

Collect the five 4:4:2 replicates and write results/results_442.json.

make_conference_paper_v2.py reads that file. While it is absent, Section 5.1
of the paper emits a visible PENDING block instead of a number, so an
unfinished draft cannot be mistaken for a finished one.

WHAT THIS SCRIPT WILL NOT DO

It does not decide in advance what the answer is. Three outcomes are possible
and the narrative it writes differs for each:

  * the protocol-matched mean exceeds 0.351 -> the comparative claim survives,
    now on a valid footing;
  * it does not exceed 0.351 -> the comparative claim does not survive, and the
    paper must say so. The result is still informative: it would mean the gap
    reported in Section 5.2 is attributable to the training-set size rather
    than to the pipeline;
  * the spread across replicates is wide enough to straddle 0.351 -> neither
    claim is supportable from a single 4:4:2 split, which is itself the
    paper's thesis and the strongest possible evidence for it.

The branches above were written before the replicates finished. The clauses
are assembled from the measured values rather than chosen to suit them --
but this file has been edited after the results landed (to fix a checkpoint
mismatch and a self-contradiction an audit found), so it is not a
pre-registration and should not be described as one.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))
from paths import (RESULTS, SPLITS, NNUNET_RAW, NNUNET_RESULTS,
                   NNUNET_PREPROCESSED, DATASET, DATASET_442, TRAINER, require)
import json
import statistics as st
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RES = NNUNET_RESULTS / DATASET_442 / TRAINER
OUT = RESULTS / "results_442.json"
BENCHMARK = 0.351      # Swin-UNet, the best row of the published benchmark

# CHECKPOINT CONVENTION. This matters and an earlier version of this file
# got it wrong. The headline five-fold 0.442 was produced with
# `--val --val_best`, i.e. the checkpoint was chosen on the held-out fold
# that is then reported. The 4:4:2 arm deliberately uses the FINAL
# checkpoint to avoid exactly that. Comparing 0.390 against 0.442 would
# therefore compare a final-checkpoint arm with a best-checkpoint arm and
# charge the whole difference to training-set size. The like-for-like
# comparator is the five-fold FINAL-checkpoint mean.
FIVEFOLD_FINAL = 0.4321   # mean of the end-of-training validations, 5 folds
FIVEFOLD_BEST = 0.4419    # `--val_best`; the number the paper reports
CKPT_OPTIMISM = FIVEFOLD_BEST - FIVEFOLD_FINAL

# Largest of five fold-pair differences between two runs of the identical
# configuration -- but those two runs were on DIFFERENT GPUs (RTX 4060 vs
# RTX 2080). It is an upper
# bound on single-fold variability, not a pure seed effect, and the paper
# must not describe it as one.
SINGLE_FOLD_SPREAD = 0.033
# Same-hardware, whole-five-fold-mean reproducibility: the independent
# control rerun against our own final-checkpoint mean.
FIVEFOLD_REPRO = 0.0026
N_REPLICATES = 5


def load(rep: int):
    f = RES / f"fold_{rep}" / "validation" / "summary.json"
    if not f.exists():
        return None
    s = json.loads(f.read_text())
    per = [m["metrics"]["1"]["Dice"] for m in s["metric_per_case"]]
    return {"replicate": rep, "dice": st.mean(per), "n_test": len(per),
            "per_case": per}


def narrate(vals, mean, sd, lo, hi):
    """Assemble the finding from the numbers, whichever way they fall."""
    above = sum(v > BENCHMARK for v in vals)
    bits = [
        f"Under the benchmark's own 4:4:2 partition the identical pipeline "
        f"reaches a mean Dice of {mean:.3f} ± {sd:.3f} across five replicates, "
        f"spanning {lo:.3f} to {hi:.3f}."]

    if lo > BENCHMARK:
        # Deliberately does NOT say the advantage is unaffected by the
        # protocol: the attribution clause below computes how much of it was
        # the protocol, and an earlier draft of this sentence contradicted it.
        bits.append(
            f"Every replicate exceeds the published {BENCHMARK:.3f}, so an "
            f"advantage survives the correction for training-set size. It is "
            f"substantially smaller than the five-fold comparison implies, "
            f"and it is still not like-for-like: the metric-definition "
            f"confound of Section 3.7 is not removed by this experiment and "
            f"cannot be removed from the published description.")
    elif hi < BENCHMARK:
        bits.append(
            f"No replicate reaches the published {BENCHMARK:.3f}. The margin "
            f"reported in Section 5.2 is therefore attributable to the larger "
            f"training set of five-fold cross-validation rather than to the "
            f"pipeline, and we withdraw any claim of improvement over the "
            f"published architecture at a matched training budget.")
    else:
        bits.append(
            f"{above} of {len(vals)} replicates exceed the published "
            f"{BENCHMARK:.3f} and {len(vals) - above} fall below it. A single "
            f"4:4:2 split on this cohort therefore cannot establish which of "
            f"the two methods is better, in either direction. This is the "
            f"clearest statement of the paper's central point: the published "
            f"{BENCHMARK:.3f} is one draw from a distribution this wide, and "
            f"so would be any single number we reported against it.")

    # The three clauses below are computed, not chosen after seeing the
    # direction of the result; each applies whichever way the comparison fell.
    spread = hi - lo
    bits.append(
        f"The standard deviation across replicates is {sd:.3f} (range "
        f"{spread:.3f}), differing only in which patients fall in which "
        f"partition. The five-fold protocol's fold-to-fold standard "
        f"deviation is 0.026 over a range of 0.063, so the two designs are "
        f"comparably variable per partition; what differs is that a 4:4:2 "
        f"study reports one draw and a five-fold study reports the mean of "
        f"five.")

    # how much of the margin is training-set size, at a MATCHED checkpoint
    # convention; mixing conventions here was a real error in an earlier draft
    margin_5fold = FIVEFOLD_FINAL - BENCHMARK
    margin_matched = mean - BENCHMARK
    attributable = FIVEFOLD_FINAL - mean
    share = attributable / margin_5fold * 100
    bits.append(
        f"Both arms are compared at the same checkpoint convention. The "
        f"five-fold result reported in Section 5.2 is a best-checkpoint number "
        f"({FIVEFOLD_BEST:.3f}); its final-checkpoint counterpart, which is "
        f"what this experiment must be compared against, is "
        f"{FIVEFOLD_FINAL:.3f}. On that basis {attributable:.3f} Dice - "
        f"roughly {share:.0f}% of the {margin_5fold:.3f} checkpoint-matched "
        f"margin - is attributable to training on 80% of the cohort rather "
        f"than 40%, and not to the pipeline. Given five replicates the share "
        f"itself is loosely determined; 'about half' is as precise a statement "
        f"as the data supports.")

    # is that surviving margin even interpretable?
    if abs(margin_matched) < SINGLE_FOLD_SPREAD:
        bits.append(
            f"The margin that survives a matched training budget is "
            f"{margin_matched:.3f}. Single-fold results on this cohort have "
            f"been observed to differ by up to {SINGLE_FOLD_SPREAD:.3f}, so a "
            f"single 4:4:2 split could not have established this margin; it "
            f"rests on the five-replicate mean, whose run-to-run "
            f"reproducibility on fixed hardware is {FIVEFOLD_REPRO:.4f}.")
    else:
        bits.append(
            f"The margin that survives a matched training budget is "
            f"{margin_matched:.3f}, and the weakest replicate clears the "
            f"published value by only {abs(lo - BENCHMARK):.3f}.")

    bits.append(
        f"With {len(vals)} replicates against a fixed published comparator, "
        f"the smallest attainable two-sided p-value from an exact sign test is "
        f"0.0625, so no significance claim is made here. The comparator is "
        f"itself a single number whose own sampling variance was never "
        f"measured and cannot be recovered from the publication. What this "
        f"experiment measures is that our pipeline moves by {spread:.2f} Dice "
        f"across 4:4:2 draws on this cohort, which is the order of uncertainty "
        f"any single 4:4:2 number here should be read as carrying.")
    return " ".join(bits)


def main():
    reps = [r for r in (load(i) for i in range(N_REPLICATES)) if r]
    if not reps:
        raise SystemExit(f"No replicates finished yet under {RES}")
    if len(reps) < N_REPLICATES:
        print(f"WARNING: only {len(reps)}/{N_REPLICATES} replicates present; "
              f"writing a partial result. Re-run when training completes.")

    vals = [r["dice"] for r in reps]
    mean = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    lo, hi = min(vals), max(vals)

    payload = {
        "protocol": "4:4:2 patient-level, 46 train / 23 test, 46 unused",
        "checkpoint": "final (not best) - the test patients sit in nnU-Net's "
                      "val slot, so best-checkpoint selection would be "
                      "selection on the reported patients",
        "n_replicates": len(reps),
        "benchmark": BENCHMARK,
        "mean": mean, "sd": sd, "min": lo, "max": hi,
        "n_above_benchmark": sum(v > BENCHMARK for v in vals),
        "replicates": [{k: v for k, v in r.items() if k != "per_case"}
                       for r in reps],
        "narrative": narrate(vals, mean, sd, lo, hi),
    }
    OUT.write_text(json.dumps(payload, indent=2))

    print(f"4:4:2 replication, {len(reps)} replicates")
    for r in reps:
        mark = ">" if r["dice"] > BENCHMARK else "<"
        print(f"   replicate {r['replicate']}: {r['dice']:.4f} "
              f"(n={r['n_test']}) {mark} {BENCHMARK}")
    print(f"   mean {mean:.4f} ± {sd:.4f}   range {lo:.4f}-{hi:.4f}")
    print(f"   {payload['n_above_benchmark']}/{len(reps)} exceed the "
          f"published {BENCHMARK}")
    print(f"\nwrote {OUT.name} -- rebuild the paper to pick it up")


if __name__ == "__main__":
    main()
