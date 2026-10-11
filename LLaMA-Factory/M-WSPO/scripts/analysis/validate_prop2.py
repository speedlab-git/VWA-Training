"""
Proposition 2 validation figure.

Compares logps/chosen_ml = log π(y_w | x, m_l) across training steps for:
  - m-WSPO (regression L_I active):   logps/chosen_ml should stabilise near weak target
  - Contrastive (L_I disabled = DPO): logps/chosen_ml should decrease monotonically

Usage:
    python scripts/analysis/validate_prop2.py \
        --mwspo_state  saves/llava_1_5_7b/m_wspo_v2_qwen3b/trainer_state.json \
        --contra_state saves/llava_1_5_7b/contrastive_prop2/trainer_state.json \
        --out          results/figures/prop2_validation
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ── Style (matches histogram.ipynb) ─────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

C_MWSPO  = "#2E6E62"   # teal  — m-WSPO
C_CONTRA = "#B5763A"   # amber — contrastive / DPO
C_WEAK   = "0.45"      # grey  — weak target


def _load(path: str) -> dict[str, list]:
    """Load trainer_state.json and extract step-level curves."""
    with open(path) as f:
        state = json.load(f)
    history = state["log_history"]

    steps, chosen_ml, weak_lp = [], [], []
    for entry in history:
        if "logps/chosen_ml" not in entry:
            continue
        steps.append(entry["step"])
        chosen_ml.append(entry["logps/chosen_ml"])
        if "weak_lp/chosen_ml" in entry:
            weak_lp.append(entry["weak_lp/chosen_ml"])
        else:
            weak_lp.append(float("nan"))

    return {"steps": steps, "chosen_ml": chosen_ml, "weak_lp": weak_lp}


def _smooth(y: list[float], w: int = 3) -> np.ndarray:
    """Simple centred moving average, same width at boundaries."""
    y = np.array(y, dtype=float)
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")


def make_figure(mwspo_path: str, contra_path: str, out: str) -> None:
    mw  = _load(mwspo_path)
    con = _load(contra_path)

    # Use epoch as x-axis if step counts differ between runs.
    # Both runs use the same log_steps cadence so steps are directly comparable.
    fig, ax = plt.subplots(figsize=(4.0, 2.6))

    # ── m-WSPO ──
    mw_steps = np.array(mw["steps"])
    mw_y     = np.array(mw["chosen_ml"])
    mw_weak  = np.array(mw["weak_lp"])
    ax.plot(mw_steps, mw_y, color=C_MWSPO, lw=1.5, zorder=5, label="m-WSPO (regression)")

    # Weak target (from m-WSPO run; should be same in contrastive run).
    if not np.all(np.isnan(mw_weak)):
        ax.plot(mw_steps, mw_weak, color=C_WEAK, lw=0.9,
                ls=(0, (4, 2.5)), zorder=3, label="Weak model target")

    # ── Contrastive ──
    con_steps = np.array(con["steps"])
    con_y     = np.array(con["chosen_ml"])
    ax.plot(con_steps, con_y, color=C_CONTRA, lw=1.5, zorder=5,
            label="Contrastive / DPO (no $L_I$)")

    # ── Annotations ──
    # Mark the stabilisation region for m-WSPO with a shaded band.
    if len(mw_y) >= 5:
        tail = mw_y[len(mw_y)//2:]   # second half of training
        band_lo, band_hi = tail.min() - 0.002, tail.max() + 0.002
        ax.axhspan(band_lo, band_hi, color=C_MWSPO, alpha=0.08, zorder=1)

    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$\log\,\pi(y_w \mid x,\,m_l)$")
    ax.set_title(r"Prop. 2: mismatched-image likelihood during training")
    ax.grid(axis="y", color="0.92", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Legend inside or outside depending on direction of curves.
    ax.legend(loc="lower left", fontsize=7.8)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path) + ".pdf")
    fig.savefig(str(out_path) + ".png")
    print(f"Saved → {out_path}.pdf / .png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mwspo_state",  default="saves/llava_1_5_7b/m_wspo_v2_qwen3b/trainer_state.json")
    parser.add_argument("--contra_state", default="saves/llava_1_5_7b/contrastive_prop2/trainer_state.json")
    parser.add_argument("--out",          default="results/figures/prop2_validation")
    args = parser.parse_args()

    make_figure(args.mwspo_state, args.contra_state, args.out)
