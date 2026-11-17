"""
The single place where paths are configured.

Everything the release ships -- splits, per-case results, audit output -- is
located relative to this repository, so those need no configuration. The two
things that are NOT shipped do: the CT volumes, and the nnU-Net output
directory. Point the two variables below at your own copies, or set the
matching environment variables, and every script in evaluation/ will find them.

    PHE_NNUNET_RAW      directory holding Dataset001_PHE/imagesTr and labelsTr
    PHE_NNUNET_RESULTS  nnU-Net results directory
    PHE_NNUNET_PREPROCESSED  nnU-Net preprocessed directory (splits only)

Scripts that only read the shipped CSV and JSON files run with no setup at all.
"""
import os
from pathlib import Path

# repository root: this file lives in <root>/configs/
REPO_ROOT = Path(__file__).resolve().parent.parent

SPLITS = REPO_ROOT / "splits"
RESULTS = REPO_ROOT / "results"
CONFIGS = REPO_ROOT / "configs"

# --- not shipped: set these to your own copies -----------------------------
# The dataset must be obtained from the PHE-SICH-CT-IDS release; see README.md.
NNUNET_RAW = Path(os.environ.get("PHE_NNUNET_RAW", REPO_ROOT / "data" / "nnUNet_raw"))
NNUNET_RESULTS = Path(os.environ.get("PHE_NNUNET_RESULTS",
                                     REPO_ROOT / "data" / "nnUNet_results"))
NNUNET_PREPROCESSED = Path(os.environ.get(
    "PHE_NNUNET_PREPROCESSED", REPO_ROOT / "data" / "nnUNet_preprocessed"))

DATASET = "Dataset001_PHE"
DATASET_442 = "Dataset004_PHE442"
TRAINER = "nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres"


def require(path: Path, what: str) -> Path:
    """Fail with an actionable message rather than a stack trace."""
    if not path.exists():
        raise SystemExit(
            f"FATAL: {what} not found at {path}\n"
            f"       This script needs data that is not part of this "
            f"repository.\n"
            f"       Set PHE_NNUNET_RAW / PHE_NNUNET_RESULTS, or edit "
            f"configs/paths.py.")
    return path
