"""
Reproduces: Fig. 5 and the limits of agreement in Table III (Sec. IV-B)
-> results/fig_bland_altman.png

Bland-Altman agreement between predicted and reference PHE volume, pooled and
within the two size strata. The computation is copied verbatim from the script
that produced the figure in the paper; only the output path differs.

ddof=1 (sample SD) throughout. This is not cosmetic: `diff` is a plain numpy
array, so a bare .std() defaults to ddof=0 and yields +/-7.44 where the paper
reports +/-7.47. The stratified limits are computed the same way, so all three
rows of Table III use one convention.

The scatter is visibly heteroscedastic -- spread grows with mean volume -- so
the pooled band understates disagreement for large lesions and overstates it
for small ones. That is the point of the stratified rows, and it is stated on
the figure itself so the band is never read as a constant-variance result.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))
from paths import RESULTS

sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

THRESHOLD = 2.0


def limits(pv, gv):
    """Bias and 95% limits of agreement, ddof=1."""
    diff = pv - gv
    return diff.mean(), 1.96 * diff.std(ddof=1)


def main():
    df = pd.read_csv(RESULTS / "nnunet_100ep_percase.csv")
    pv, gv = df["pred_vol"].values, df["gt_vol"].values
    mean_v = (pv + gv) / 2
    diff = pv - gv
    bias, loa = limits(pv, gv)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(mean_v, diff, alpha=0.5, c="#1f77b4")
    ax.axhline(bias, c="red", label=f"Bias {bias:+.2f} mL")
    ax.axhline(bias + loa, c="gray", ls="--")
    ax.axhline(bias - loa, c="gray", ls="--",
               label=f"95% LoA ±{loa:.2f} mL")
    ax.set_xlabel("Mean PHE volume (mL)")
    ax.set_ylabel("Pred − GT (mL)")
    ax.set_title("Bland–Altman: PHE Volume Agreement\n"
                 "(100-epoch primary model)", fontsize=11)
    ax.text(0.98, 0.03,
            "Variance increases with volume;\n"
            "pooled LoA assumes constant variance (see Sec. IV-B)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            style="italic", color="#444")
    ax.legend()
    plt.tight_layout()
    out = RESULTS / "fig_bland_altman.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"{'stratum':<22}{'n':>5}{'bias mL':>10}{'95% LoA mL':>14}")
    for name, sub in (("Pooled", df),
                      (f"Small (<{THRESHOLD:g} mL)", df[df.gt_vol < THRESHOLD]),
                      (f"Large (>={THRESHOLD:g} mL)", df[df.gt_vol >= THRESHOLD])):
        b, l = limits(sub["pred_vol"].values, sub["gt_vol"].values)
        print(f"{name:<22}{len(sub):>5}{b:>+10.2f}{l:>14.2f}")
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
