# Results

Per-case results for every experiment reported in the paper: one row per patient, with the patient ID, fold, Dice, predicted and reference PHE volume (mL), and the intensity-audit flag.

| File | Rows | Contents | Paper |
| --- | --- | --- | --- |
| `per_case.csv` | 115 | The joined table: patient ID, fold, Dice, IoU, recall, precision, HD95, ASD, predicted and reference volume (mL), HU min/max/range, distinct-value count, audit flag and reason. Built by `evaluation/build_per_case.py`. | Tables III–VI |
| `nnunet_100ep_percase.csv` | 115 | Segmentation metrics as produced by `evaluation/evaluate_100ep_full.py`. Best checkpoint; five-fold mean Dice 0.442. | Tables III–V |
| `intensity_scale_audit.csv` | 115 | Per-volume HU minimum, maximum, range and distinct-value count, with the suspect flag and the reason it was raised. | Table VI, Fig. 7 |
| `fivefold_final_checkpoint.csv` | 5 | Per-fold Dice under both checkpoint rules, recovered from the training logs by `evaluation/extract_checkpoint_dice.py`. Means: 0.432 final, 0.442 best. | Sec. IV-E |
| `per_case_442.csv` | 115 | Per-case Dice for the five 4:4:2 replicates (23 test patients each). Replicate means give 0.390 ± 0.025. | Table IV |
| `results_442.json` | 5 | Replicate means, spread, and the summary text for the 4:4:2 experiment. | Table IV |
| `icc_intervals.json` | — | ICC(2,1) with bootstrap confidence intervals, pooled and stratified, plus the 1–5 mL threshold sweep. | Table III, Sec. IV-B |
| `fig_bland_altman.png` | — | Produced by `evaluation/bland_altman.py`. | Fig. 5 |

Volumes are in millilitres. `audit_flag` is 1 for the eight volumes that are not Hounsfield-scaled, 0 otherwise. The two source CSVs are kept alongside the joined table so that every script runs against its original input.
