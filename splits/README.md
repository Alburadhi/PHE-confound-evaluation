# Splits

Patient-level splits used in the paper. Every experiment uses these files, so no patient appears in both the training and the test set of any fold.

- Five-fold cross-validation over the 115 usable patients (Sec. III).
- Five 4:4:2 replicate splits, each with the seed that generated it (Sec. IV-C, Table IV).

## Files

| File | Contents |
| --- | --- |
| `fivefold_splits.json` | The five-fold split, nnU-Net format: one object per fold with `train` (92 patients) and `val` (23). |
| `splits_442_replicates.json` | The five 4:4:2 replicates: one object each with `train` (46), `test` (23) and `unused` (46). |
| `make_442_splits.py` | Regenerates the replicates from `fivefold_splits.json`. |

## Seed

The five 4:4:2 replicates were generated with **seed 20260926**, set as `SEED` in `make_442_splits.py`. One `random.Random(20260926)` instance is drawn from across all five replicates in order, so replicate *k* depends on the draws before it and the five partitions are reproducible only as a set, by rerunning the script from the start.

## The 4:4:2 design

Each replicate matches the benchmark's published 4:4:2 ratio: 46 patients for training (40 %), 23 for test (20 %). The remaining 46 are left **unused** so that the training budget never exceeds what the benchmark's own protocol allows. The partition the benchmark actually used was never released, so the proportions are matched, not the patient assignment.

The test patients are placed in nnU-Net's validation slot, so the replicates are evaluated with the **final** checkpoint, not the best. Selecting the best checkpoint there would select on the very patients being reported — the confound the paper measures in Sec. IV-E.

The five test sets overlap: they share 3 to 7 of their 23 patients, and 80 of the 115 patients appear in at least one test set. The replicates are therefore distinct but not independent, which is stated as a limitation in the paper.
