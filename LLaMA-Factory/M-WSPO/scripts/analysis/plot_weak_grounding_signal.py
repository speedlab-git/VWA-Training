# Copyright 2026 the LlamaFactory team.
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

"""Diagnose whether the cached weak pair encodes visual grounding.

For each training example the cache stores mean token log-probs of the
aligned / reference weak models on (y_w, m_w), (y_w, m_l), (y_l, m_w).
This script computes the three weak shifts and the two margins:

    s_M = log π_r^w(y_w | x, m_w) - log π_ref^w(y_w | x, m_w)
    s_I = log π_r^w(y_w | x, m_l) - log π_ref^w(y_w | x, m_l)
    s_R = log π_r^w(y_l | x, m_w) - log π_ref^w(y_l | x, m_w)
    Δ_img  = s_M - s_I
    Δ_resp = s_M - s_R

Usage:
    python scripts/analysis/plot_weak_grounding_signal.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_CACHES = (
    ("3B teacher", Path("cache/rlhf_v_mwspo_weak_lp_qwen3b.parquet")),
    ("7B teacher", Path("cache/rlhf_v_mwspo_weak_lp.parquet")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help="Override default caches. Repeatable, e.g. --cache '7B teacher (NeXT-13B)=cache/rlhf_v_mwspo_weak_lp.parquet'.",
    )
    parser.add_argument(
        "--style",
        choices=("histogram", "ridgeline"),
        default="histogram",
        help="Main-paper 1x2 histograms (default) or the compact ridgeline.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/weak_grounding_hist_7b.png"),
    )
    parser.add_argument("--bins", type=int, default=36)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def resolve_caches(args: argparse.Namespace) -> tuple[tuple[str, Path], ...]:
    if not args.cache:
        return DEFAULT_CACHES
    parsed: list[tuple[str, Path]] = []
    for item in args.cache:
        if "=" not in item:
            raise SystemExit(f"--cache must be LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        parsed.append((label, Path(path)))
    return tuple(parsed)


def load_shifts(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {
        "weak_aligned_lp_yw_mw",
        "weak_ref_lp_yw_mw",
        "weak_aligned_lp_yw_ml",
        "weak_ref_lp_yw_ml",
        "weak_aligned_lp_yl_mw",
        "weak_ref_lp_yl_mw",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    out = pd.DataFrame(index=df.index)
    out["s_M"] = df["weak_aligned_lp_yw_mw"] - df["weak_ref_lp_yw_mw"]
    out["s_I"] = df["weak_aligned_lp_yw_ml"] - df["weak_ref_lp_yw_ml"]
    out["s_R"] = df["weak_aligned_lp_yl_mw"] - df["weak_ref_lp_yl_mw"]
    out["delta_img"] = out["s_M"] - out["s_I"]
    out["delta_resp"] = out["s_M"] - out["s_R"]
    if "sample_id" in df.columns:
        out["sample_id"] = df["sample_id"]
    return out


def summarize(name: str, shifts: pd.DataFrame) -> dict[str, float | str | int]:
    return {
        "weak_teacher": name,
        "n": int(len(shifts)),
        "E[s_M]": float(shifts["s_M"].mean()),
        "E[s_I]": float(shifts["s_I"].mean()),
        "E[s_R]": float(shifts["s_R"].mean()),
        "E[d_img]": float(shifts["delta_img"].mean()),
        "median[d_img]": float(shifts["delta_img"].median()),
        "pct_d_img>0": float((shifts["delta_img"] > 0).mean() * 100.0),
        "E[d_resp]": float(shifts["delta_resp"].mean()),
        "median[d_resp]": float(shifts["delta_resp"].median()),
        "pct_d_resp>0": float((shifts["delta_resp"] > 0).mean() * 100.0),
        "pct_both>0": float(((shifts["delta_img"] > 0) & (shifts["delta_resp"] > 0)).mean() * 100.0),
    }


def _clip_for_plot(values: pd.Series, lo: float = 1.0, hi: float = 99.0) -> pd.Series:
    return values.clip(values.quantile(lo / 100.0), values.quantile(hi / 100.0))


def plot_histograms(shifts: pd.DataFrame, output: Path, dpi: int, bins: int) -> None:
    """1x2 histogram of the two 7B-teacher margins. Y-axis is fraction of examples."""
    panels = (
        (
            shifts["delta_img"],
            r"$\Delta_{\mathrm{img}}^w=\hat{s}_M-\hat{s}_I$",
            r"$\Delta_{\mathrm{img}}^w$",
        ),
        (
            shifts["delta_resp"],
            r"$\Delta_{\mathrm{resp}}^w=\hat{s}_M-\hat{s}_R$",
            r"$\Delta_{\mathrm{resp}}^w$",
        ),
    )
    color = "#4C72B0"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))

    for ax, (values, xlabel, _short) in zip(axes, panels):
        lo, hi = values.quantile([0.01, 0.99])
        in_range = values.between(lo, hi)
        shown = values[in_range]
        weights = np.full(len(shown), 1.0 / len(values))
        ax.hist(
            shown,
            bins=bins,
            range=(float(lo), float(hi)),
            weights=weights,
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_xlim(float(lo), float(hi))
        ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
        pct = float((values > 0).mean() * 100.0)
        ax.text(
            0.97,
            0.92,
            f"{pct:.1f}% $>0$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
        )
        ax.set_xlabel(xlabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Fraction of examples")
    fig.subplots_adjust(wspace=0.18, left=0.10, right=0.98, top=0.96, bottom=0.22)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    print(f"Saved {output.with_suffix('.pdf')}")


def plot_figure(
    series: list[tuple[str, pd.DataFrame]],
    output: Path,
    dpi: int,
) -> None:
    """Compact four-row ridgeline of the two weak-alignment margins."""
    teacher_colors = {"3B": "#0072B2", "7B": "#D55E00"}
    rows: list[dict] = []
    for teacher_idx, (name, shifts) in enumerate(series):
        color = teacher_colors["7B"] if "7B" in name else teacher_colors["3B"]
        if "3B" not in name and "7B" not in name:
            color = ["#0072B2", "#D55E00"][teacher_idx % 2]
        short = name.replace(" teacher", "")
        rows.append(
            {
                "teacher": name,
                "signal": rf"{short}  Visual  $\hat{{s}}_M-\hat{{s}}_I$",
                "values": shifts["delta_img"],
                "pct": float((shifts["delta_img"] > 0).mean() * 100.0),
                "color": color,
            }
        )
        rows.append(
            {
                "teacher": name,
                "signal": rf"{short}  Response  $\hat{{s}}_M-\hat{{s}}_R$",
                "values": shifts["delta_resp"],
                "pct": float((shifts["delta_resp"] > 0).mean() * 100.0),
                "color": color,
            }
        )

    all_values = pd.concat([row["values"] for row in rows], ignore_index=True)
    xmin, xmax = np.quantile(all_values, [0.01, 0.99])
    pad = 0.08 * (xmax - xmin)
    xmin, xmax = xmin - pad, xmax + pad
    grid = np.linspace(xmin, xmax, 320)

    fig, ax = plt.subplots(figsize=(6.8, 1.05 + 0.62 * len(rows)))
    ridge_height = 0.82
    yticks = []
    yticklabels = []
    for i, row in enumerate(rows):
        y0 = len(rows) - 1 - i
        clipped = _clip_for_plot(row["values"])
        kde = gaussian_kde(clipped.to_numpy(), bw_method=0.18)
        density = kde(grid)
        density = density / density.max() * ridge_height
        visible = density > 0.015 * ridge_height
        ax.fill_between(grid, y0, y0 + density, where=visible, color=row["color"], alpha=0.40, linewidth=0, interpolate=True)
        ax.plot(grid[visible], (y0 + density)[visible], color=row["color"], linewidth=1.35, solid_capstyle="round")
        median = float(row["values"].median())
        ax.plot(median, y0, "o", color=row["color"], markersize=4.6, zorder=3)
        ax.text(
            1.01,
            y0 + 0.18,
            f"{row['pct']:.1f}% $>0$",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=9,
        )
        yticks.append(y0 + 0.22)
        yticklabels.append(row["signal"])

    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0, zorder=2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.25, len(rows) - 0.05)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Weak alignment margin")
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.30, right=0.82, top=0.96, bottom=0.18)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    print(f"Saved {output.with_suffix('.pdf')}")


def main() -> None:
    args = parse_args()
    series: list[tuple[str, pd.DataFrame]] = []
    rows: list[dict[str, float | str | int]] = []
    for name, path in resolve_caches(args):
        if not path.exists():
            print(f"Skipping missing cache: {path}")
            continue
        shifts = load_shifts(path)
        series.append((name, shifts))
        row = summarize(name, shifts)
        rows.append(row)
        print(f"=== {name}  n={row['n']} ===")
        print(
            f"  E[s_M]={row['E[s_M]']:+.4f}  E[s_I]={row['E[s_I]']:+.4f}  E[s_R]={row['E[s_R]']:+.4f}"
        )
        print(
            f"  E[Δ_img]={row['E[d_img]']:+.4f}  median={row['median[d_img]']:+.4f}  "
            f"%Δ_img>0={row['pct_d_img>0']:.1f}"
        )
        print(
            f"  E[Δ_resp]={row['E[d_resp]']:+.4f}  median={row['median[d_resp]']:+.4f}  "
            f"%Δ_resp>0={row['pct_d_resp>0']:.1f}  %both>0={row['pct_both>0']:.1f}"
        )

    if not series:
        raise SystemExit("No weak-logprob caches found.")

    if args.style == "histogram":
        chosen = next((item for item in series if "7B" in item[0]), series[0])
        print(f"Histogram uses: {chosen[0]}")
        plot_histograms(chosen[1], args.output, args.dpi, args.bins)
    else:
        plot_figure(series, args.output, args.dpi)
    table_path = args.output.with_name(args.output.stem + "_table.json")
    table_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved {table_path}")

    # Markdown table for easy paste into the paper / rebuttal.
    print()
    print("| Weak teacher | n | E[s_M] | E[s_I] | E[s_R] | E[Δ_img] | % Δ_img>0 | E[Δ_resp] | % Δ_resp>0 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['weak_teacher']} | {row['n']} | "
            f"{row['E[s_M]']:.3f} | {row['E[s_I]']:.3f} | {row['E[s_R]']:.3f} | "
            f"{row['E[d_img]']:.3f} | {row['pct_d_img>0']:.1f} | "
            f"{row['E[d_resp]']:.3f} | {row['pct_d_resp>0']:.1f} |"
        )


if __name__ == "__main__":
    main()
