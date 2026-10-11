# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Empirical validation of Proposition 3 (m-WSPO fixed-point prediction).

Proposition 3 states that the trained strong-model preference margin Delta*
satisfies the fixed-point equation

    Delta* = hat_Delta + kappa * sigma(-beta * Delta* / gamma)          (*)

where hat_Delta is the *gamma-normalised* weak preference margin
hat_Delta_raw / gamma, beta is the DPO temperature, gamma is the WSPO
transfer coefficient, and kappa is a coupling constant.

Theory (from the first-order stationarity of the m-WSPO objective using
sum log-probabilities for both the WSPO and DPO components):

    Delta* = hat_Delta_norm                                    [WSPO only]
           + (2 * beta_dpo * beta) / gamma^2 * sigma(-beta * Delta*)  [DPO correction]
           + lambda_nll / gamma^2                              [NLL offset]

which matches (*) with kappa = 2*beta_dpo*beta/gamma^2 and the sigmoid
argument -beta*Delta* (since in our v2 recipe beta=gamma so -beta*Delta*
and -beta*Delta*/gamma differ by a factor of 1, but we keep them explicit
below to stay general).

Inputs:
  cache/rlhf_v_mwspo_weak_lp.parquet   -- weak pair shifts (already exists)
  cache/rlhf_v_mwspo_strong_lp.parquet -- strong pair shifts (from cache_strong_logprobs.sh)

Output:
  results/figures/prop3_validation.{pdf,png}

Usage:
  # After running cache_strong_logprobs.sh:
  python scripts/analysis/validate_prop3.py

  # Custom paths:
  python scripts/analysis/validate_prop3.py \
      --weak_cache  cache/rlhf_v_mwspo_weak_lp.parquet \
      --strong_cache cache/rlhf_v_mwspo_strong_lp.parquet \
      --outdir results/figures
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Hyperparameters (v2 recipe)
# ---------------------------------------------------------------------------
GAMMA = 0.1         # mwspo_gamma
BETA = 0.1          # mwspo_dpo_temperature
BETA_DPO = 1.0      # mwspo_beta_dpo (weight on L_DPO)
LAMBDA_NLL = 0.2    # mwspo_lambda_nll

# Theoretical coupling constants (derived from first-order stationarity;
# see docstring above).
KAPPA_DPO = 2 * BETA_DPO * BETA / GAMMA**2          # = 20 (DPO correction amplitude)
KAPPA_NLL = LAMBDA_NLL / GAMMA**2                    # = 20 (NLL constant offset)

# ---------------------------------------------------------------------------
# Plot style (ACL)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

COL_SCATTER = "#b7c9e2"
COL_BIN     = "#2e6e62"
COL_THEORY  = "#c0392b"
COL_WSPO    = "#7f8c8d"
GRID_KW     = dict(axis="both", color="0.90", lw=0.5, zorder=0)


# ---------------------------------------------------------------------------
# Fixed-point solver
# ---------------------------------------------------------------------------
def fp_solve(hat_delta_norm: float, kappa_dpo: float, kappa_nll: float, beta: float) -> float:
    r"""Numerically solve Delta* = hat_delta_norm + kappa_dpo*sigma(-beta*Delta*) + kappa_nll.

    The sigmoid argument uses ``-beta * Delta*``; when beta = gamma (our v2 recipe)
    this equals ``-beta * Delta* / gamma`` from the compact form in the paper.
    """
    def f(d: float) -> float:
        return d - hat_delta_norm - kappa_dpo / (1.0 + np.exp(beta * d)) - kappa_nll

    # Bracket: f(-1000) < 0 and f(+3000) > 0 are safe bounds.
    try:
        return brentq(f, -1000.0, 3000.0, xtol=1e-4)
    except ValueError:
        return np.nan


def theory_curve(
    hat_norm_grid: np.ndarray,
    kappa_dpo: float = KAPPA_DPO,
    kappa_nll: float = KAPPA_NLL,
    beta: float = BETA,
) -> np.ndarray:
    return np.array([fp_solve(h, kappa_dpo, kappa_nll, beta) for h in hat_norm_grid])


# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------
def binned_stats(
    x: np.ndarray, y: np.ndarray, n_bins: int = 20, q_lo: float = 3.0, q_hi: float = 97.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centre, bin_mean_y, bin_se_y) using equal-count bins."""
    lo, hi = np.percentile(x, q_lo), np.percentile(x, q_hi)
    mask = (x >= lo) & (x <= hi)
    x_m, y_m = x[mask], y[mask]
    edges = np.percentile(x_m, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    centres, means, ses = [], [], []
    for i in range(len(edges) - 1):
        sel = (x_m >= edges[i]) & (x_m < edges[i + 1])
        if sel.sum() < 5:
            continue
        centres.append(x_m[sel].mean())
        means.append(y_m[sel].mean())
        ses.append(y_m[sel].std() / np.sqrt(sel.sum()))
    return np.array(centres), np.array(means), np.array(ses)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_margins(
    weak_cache: str, strong_cache: str
) -> pd.DataFrame:
    """Merge weak and strong caches; compute per-example preference margins."""
    wdf = pd.read_parquet(weak_cache)
    sdf = pd.read_parquet(strong_cache)

    # Weak preference margin (raw mean-lp units)
    hat_sw = wdf["weak_aligned_lp_yw_mw"] - wdf["weak_ref_lp_yw_mw"]
    hat_sl = wdf["weak_aligned_lp_yl_mw"] - wdf["weak_ref_lp_yl_mw"]
    hat_delta_raw = hat_sw - hat_sl

    # Gamma-normalised weak margin (this is what Prop. 3's hat_Delta refers to)
    hat_delta_norm = hat_delta_raw / GAMMA

    # Strong model preference margin (policy vs reference, mean lp units)
    # NOTE: cache_strong_logprobs.sh puts the trained policy in "weak_aligned"
    #       and the strong reference in "weak_ref" -- same schema, relabelled.
    s_w = sdf["weak_aligned_lp_yw_mw"] - sdf["weak_ref_lp_yw_mw"]
    s_l = sdf["weak_aligned_lp_yl_mw"] - sdf["weak_ref_lp_yl_mw"]
    delta_star = s_w - s_l

    df = pd.DataFrame({
        "sample_id":      wdf["sample_id"].values,
        "hat_delta_raw":  hat_delta_raw.values,
        "hat_delta_norm": hat_delta_norm.values,
        "delta_star":     delta_star.values,
        # Convenience columns used in the figure
        "hat_sw":  hat_sw.values,
        "hat_sl":  hat_sl.values,
        "s_w":     s_w.values,
        "s_l":     s_l.values,
    })
    return df


def make_figure(df: pd.DataFrame, out_prefix: str) -> None:
    """Two-panel figure validating Proposition 3.

    Panel A — Main claim:
        x = hat_Delta (gamma-normalised weak margin)
        y = Delta* (trained strong model margin)
        Overlaid: identity y=x  [WSPO-only prediction]
                  fixed-point curve with DPO+NLL correction

    Panel B — DPO correction term:
        x = hat_Delta (gamma-normalised weak margin)
        y = Delta* − hat_Delta (deviation from WSPO-only prediction)
        Theory: kappa_dpo * sigma(−beta * Delta*(x)) + kappa_nll
        Expected: largest correction at small hat_Delta; decays as hat_Delta grows.
    """
    # Clip outliers for display (>1 % / <99 %)
    lo_h, hi_h = np.percentile(df["hat_delta_norm"], [1, 99])
    lo_d, hi_d = np.percentile(df["delta_star"],     [1, 99])
    mask = (
        df["hat_delta_norm"].between(lo_h, hi_h)
        & df["delta_star"].between(lo_d, hi_d)
    )
    plot_df = df[mask]

    hat  = plot_df["hat_delta_norm"].values
    dstr = plot_df["delta_star"].values
    correction = dstr - hat                 # y for panel B: Delta* − hat_Delta

    # Theory grid
    h_grid = np.linspace(lo_h, hi_h, 300)
    d_theory = theory_curve(h_grid)
    corr_theory = d_theory - h_grid         # Delta* − hat_Delta along the theory curve

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # ── Panel A ────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(hat, dstr, s=3, alpha=0.25, color=COL_SCATTER, rasterized=True, zorder=2)

    # Binned means
    bx, by, bs = binned_stats(hat, dstr, n_bins=18)
    ax.errorbar(bx, by, yerr=2 * bs, fmt="o", ms=4.5, color=COL_BIN,
                lw=1.0, capsize=2, zorder=4, label="Binned mean ±2 SE")

    # WSPO-only prediction (identity line)
    ax.plot(h_grid, h_grid, "--", color=COL_WSPO, lw=1.2, zorder=3,
            label=r"WSPO-only: $\Delta^*=\hat{\Delta}$")

    # Full fixed-point prediction
    valid = ~np.isnan(d_theory)
    ax.plot(h_grid[valid], d_theory[valid], "-", color=COL_THEORY, lw=1.5, zorder=5,
            label=r"Fixed-point (Prop.~3)")

    ax.set_xlabel(r"$\hat{\Delta} = \hat{\delta}_w / \gamma$ (normalised weak margin)")
    ax.set_ylabel(r"$\Delta^*$ (strong margin)")
    ax.set_title("(a) Strong vs.\ weak preference margin")
    ax.legend(fontsize=7.5)
    ax.grid(**GRID_KW)
    ax.set_axisbelow(True)

    # ── Panel B ────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.scatter(hat, correction, s=3, alpha=0.25, color=COL_SCATTER, rasterized=True, zorder=2)

    # Binned means of correction
    bx, by, bs = binned_stats(hat, correction, n_bins=18)
    ax.errorbar(bx, by, yerr=2 * bs, fmt="o", ms=4.5, color=COL_BIN,
                lw=1.0, capsize=2, zorder=4, label="Binned mean ±2 SE")

    # Theory correction curve
    valid = ~np.isnan(corr_theory)
    ax.plot(h_grid[valid], corr_theory[valid], "-", color=COL_THEORY, lw=1.5, zorder=5,
            label=r"$\kappa\,\sigma(-\beta\Delta^*/\gamma) + c_\text{NLL}$ (theory)")

    # DPO-only correction (without NLL constant) for reference
    dpo_corr_only = np.array([
        KAPPA_DPO / (1.0 + np.exp(BETA * fp_solve(h, KAPPA_DPO, 0.0, BETA)))
        if not np.isnan(fp_solve(h, KAPPA_DPO, 0.0, BETA)) else np.nan
        for h in h_grid
    ])
    ax.plot(h_grid[~np.isnan(dpo_corr_only)],
            dpo_corr_only[~np.isnan(dpo_corr_only)],
            ":", color=COL_THEORY, lw=1.0, zorder=3,
            label=r"DPO correction only ($\lambda_\text{NLL}=0$)")

    ax.axhline(KAPPA_NLL, color=COL_WSPO, lw=0.8, ls="--", zorder=2,
               label=rf"NLL offset $\lambda_\text{{NLL}}/\gamma^2={KAPPA_NLL:.0f}$")

    ax.set_xlabel(r"$\hat{\Delta}$ (normalised weak margin)")
    ax.set_ylabel(r"$\Delta^* - \hat{\Delta}$")
    ax.set_title("(b) DPO correction term")
    ax.legend(fontsize=7.0)
    ax.grid(**GRID_KW)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.5, w_pad=1.2)
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    fig.savefig(out_prefix + ".pdf", dpi=1500)
    fig.savefig(out_prefix + ".png", dpi=300)
    print(f"Saved {out_prefix}.pdf  /  {out_prefix}.png")
    return fig


def print_summary(df: pd.DataFrame) -> None:
    hat  = df["hat_delta_norm"].values
    dstr = df["delta_star"].values

    print("\n=== Proposition 3 validation summary ===")
    print(f"  N examples                      : {len(df)}")
    print(f"  hat_Delta (normalised)  mean±std: {hat.mean():.2f} ± {hat.std():.2f}")
    print(f"  Delta*                  mean±std: {dstr.mean():.2f} ± {dstr.std():.2f}")
    print(f"  Pearson r(hat_Delta, Delta*)    : {np.corrcoef(hat, dstr)[0,1]:.4f}")

    # How well does the WSPO-only prediction Delta* ≈ hat_Delta explain the data?
    wspo_resid = dstr - hat
    print(f"  WSPO residual (Delta*−hat_Delta) mean: {wspo_resid.mean():.2f}")
    print(f"  WSPO residual std               : {wspo_resid.std():.2f}")

    # Mean absolute error of fixed-point prediction
    d_pred = theory_curve(hat)
    valid = ~np.isnan(d_pred)
    mae = np.abs(dstr[valid] - d_pred[valid]).mean()
    mse = ((dstr[valid] - d_pred[valid]) ** 2).mean() ** 0.5
    print(f"  Fixed-point prediction MAE      : {mae:.3f}")
    print(f"  Fixed-point prediction RMSE     : {mse:.3f}")

    # Correlation between correction and hat_Delta
    # Theory: correction should DECREASE with hat_Delta (DPO sigmoid decays)
    r_corr = np.corrcoef(hat, dstr - hat)[0, 1]
    print(f"  r(hat_Delta, Delta*−hat_Delta)  : {r_corr:.4f}  (expect negative if DPO correction dominates)")
    print("=========================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weak_cache",
        default="cache/rlhf_v_mwspo_weak_lp.parquet",
        help="Path to the weak pair log-prob cache (default: %(default)s).",
    )
    parser.add_argument(
        "--strong_cache",
        default="cache/rlhf_v_mwspo_strong_lp.parquet",
        help="Path to the strong pair log-prob cache produced by cache_strong_logprobs.sh.",
    )
    parser.add_argument(
        "--outdir",
        default="results/figures",
        help="Output directory for PDF / PNG (default: %(default)s).",
    )
    args = parser.parse_args()

    out_prefix = os.path.join(args.outdir, "prop3_validation")

    print(f"Loading weak  cache : {args.weak_cache}")
    print(f"Loading strong cache: {args.strong_cache}")

    df = load_margins(args.weak_cache, args.strong_cache)
    print_summary(df)
    make_figure(df, out_prefix)


if __name__ == "__main__":
    main()
