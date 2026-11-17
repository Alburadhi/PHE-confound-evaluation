# Evaluation

Scripts that recompute the paper's tables and figures from the predictions and reference masks.

- Per-case Dice on reconstructed volumes, not slice-wise.
- Volume agreement: ICC(2,1), Bland–Altman bias and limits of agreement (Table III, Fig. 5).
- Size-stratified agreement at 2 mL of reference volume, with the 1 to 5 mL sensitivity analysis (Sec. IV-B).
- 95% confidence intervals by percentile bootstrap over patients, B = 2000.
- Intensity audit: per-volume HU range, maximum and distinct-value count (Sec. IV-F, Table VI, Fig. 7).

| Script | Produces | Needs |
| --- | --- | --- |
| `build_per_case.py` | `results/per_case.csv` | shipped CSVs only |
| `icc_confidence_intervals.py` | Table III, Sec. IV-B sweep → `results/icc_intervals.json` | shipped CSVs only |
| `bland_altman.py` | Fig. 5 and the Table III limits → `results/fig_bland_altman.png` | shipped CSVs only |
| `extract_checkpoint_dice.py` | Sec. IV-E → `results/fivefold_final_checkpoint.csv` | nnU-Net training logs |
| `extract_442_percase.py` | Table IV → `results/per_case_442.csv` | nnU-Net 4:4:2 output |
| `analyse_442.py` | Table IV → `results/results_442.json` | nnU-Net 4:4:2 output |
| `evaluate_100ep_full.py` | the per-case metrics table | predictions and reference masks |
| `intensity_scale_audit.py` | Table VI, Fig. 7 → `results/intensity_scale_audit.csv` | the CT volumes |

The three scripts in the first group reproduce every number in Table III without any data beyond this repository.

ICC(2,1) is two-way random effects, single measurement, absolute agreement. The bootstrap seed is 20261113; a normal-theory interval is not used because the reference volume distribution is strongly right-skewed (skewness 2.36).
