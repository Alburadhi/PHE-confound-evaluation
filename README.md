# A Confound Measurement Based Evaluation Approach for Perihematomal Edema Segmentation

Patient-level splits, evaluation code and per-case results for the paper accepted at ICCVIT 2026.

Ahmed Waleed Hatem Alburadhi<sup>1,2</sup>, Frank Kulwa<sup>1</sup>, Muhammad Hasnain<sup>1</sup>, Damilola Ogunsona<sup>1</sup>, Chen Li<sup>1,</sup>\*
<sup>1</sup> Microscopic Image and Medical Image Analysis Group, Northeastern University, Shenyang, China
<sup>2</sup> Al-Hakeem General Hospital, Najaf Health Directorate, Ministry of Health, Najaf, Iraq
\*Corresponding author: lichen@bmie.neu.edu.cn

## What is in this repository

| Folder | Contents | Paper |
| --- | --- | --- |
| `splits/` | Patient-level five-fold split, and the five 4:4:2 replicate splits with their seeds | Sec. III, Sec. IV-C, Table IV |
| `evaluation/` | Per-case Dice on reconstructed volumes, ICC(2,1), Bland–Altman limits of agreement, size-stratified agreement, bootstrap confidence intervals, and the intensity audit | Sec. IV-B, IV-F, Tables III and VI, Figs. 5 and 7 |
| `results/` | Per-case results for every experiment reported in the paper | Tables III–VI |
| `configs/` | nnU-Net plans and training settings | Sec. III |

## Data

The CT volumes and masks are not included. They belong to the PHE-SICH-CT-IDS dataset and must be obtained from its official release:

> D. Ma, C. Li, T. Du, et al., "PHE-SICH-CT-IDS: A benchmark CT image dataset for evaluation of semantic segmentation, object detection and radiomic feature extraction of perihematomal edema in spontaneous intracerebral hemorrhage," *Comput. Biol. Med.*, vol. 173, Art. no. 108342, 2024.

Of the 120 patients, 115 are usable and are the ones used here. Eight of the 115 volumes are not Hounsfield-scaled (Sec. IV-F of the paper). The intensity audit in `evaluation/` detects them from the image data alone, so the defect can be checked without our training code. The affected cases are PHE_0160 and PHE_0162–PHE_0167 (8-bit data shifted down by 1024) and PHE_0161 (unshifted 8-bit, 0–255); they are flagged in `results/per_case.csv`.

## Which checkpoint the results use

The paper reports two five-fold means that differ only by the checkpoint rule, and treats that difference as one of the measured confounds (Sec. IV-E):

- `results/nnunet_100ep_percase.csv` comes from the **best checkpoint**, five-fold mean Dice **0.442**. Every downstream analysis here uses it.
- The **final-checkpoint** mean is **0.432**. It is recovered from the nnU-Net training logs by `evaluation/extract_checkpoint_dice.py` into `results/fivefold_final_checkpoint.csv`.

The 4:4:2 replicates (Sec. IV-C) use the final checkpoint throughout, because their test patients occupy nnU-Net's validation slot.

## Setup

```
pip install -r requirements.txt
```

Scripts that read only the shipped CSV and JSON files need nothing further. To recompute from the CT volumes or from an nnU-Net output directory, edit `configs/paths.py` or set:

```
PHE_NNUNET_RAW           directory holding Dataset001_PHE/imagesTr and labelsTr
PHE_NNUNET_RESULTS       nnU-Net results directory
PHE_NNUNET_PREPROCESSED  nnU-Net preprocessed directory
```

## Reproducing the analyses

Run from the repository root. Each script states at the top which table or figure it produces.

**No data needed** — these run against the shipped results:

```
python evaluation/build_per_case.py              # results/per_case.csv
python evaluation/icc_confidence_intervals.py    # Table III, and the Sec. IV-B sweep
python evaluation/bland_altman.py                # Fig. 5, and the Table III limits
```

**An nnU-Net output directory is needed** — these read its logs or predictions:

```
python evaluation/extract_checkpoint_dice.py --trainer-dir <results>/Dataset001_PHE/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres
python evaluation/extract_442_percase.py     --trainer-dir <results>/Dataset004_PHE442/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres
python evaluation/analyse_442.py                 # Table IV
python evaluation/evaluate_100ep_full.py         # the per-case metrics table
```

**The CT volumes are needed**:

```
python evaluation/intensity_scale_audit.py       # Table VI, Fig. 7
```

**To regenerate the 4:4:2 splits** — this also creates the nnU-Net dataset directory the replicates train from, so it needs a preprocessed dataset:

```
python splits/make_442_splits.py
```

## Training settings

nnU-Net v2 (`nnunetv2` 2.8.1), configuration `3d_fullres`, trainer `nnUNetTrainer_100epochs`. The full plan is `configs/nnUNetPlans.json`; the values the paper quotes are:

| Setting | Value |
| --- | --- |
| Epochs | 100, matched to the benchmark of Ma et al. |
| Patch size | 16 × 320 × 320 |
| Target spacing | 5.0 × 0.488 × 0.488 mm |
| Batch size | 2 |
| Encoder | 7 stages, 32-64-128-256-320-320-320 |
| Anisotropy handling | first three stages pool in-plane only, (1, 3, 3) kernels |
| Normalization | CTNormalization |
| Optimizer | SGD with a polynomial learning-rate schedule |

Everything above is selected by nnU-Net's self-configuration from the dataset fingerprint rather than chosen by us (Sec. III).

Confidence intervals for the ICC estimates are percentile bootstrap intervals over patients with B = 2000 resamples, seed 20261113 (set in `evaluation/icc_confidence_intervals.py`). Fixing the seed makes them reproducible to the printed precision.

## Citation

If you use this code or these results, please cite the paper (see `CITATION.cff`). The citation will be completed with the IEEE proceedings details after publication.

## License

The code is released under the MIT License (`LICENSE`). The files in `splits/` and `results/` are released under CC BY 4.0. The PHE-SICH-CT-IDS images and masks remain under the terms of their original release.
