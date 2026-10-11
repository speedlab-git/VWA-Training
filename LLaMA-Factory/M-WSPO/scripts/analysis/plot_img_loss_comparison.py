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

"""Plot regression and Bradley-Terry image-loss diagnostics from Trainer states."""

import argparse
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRICS = (
    "logps/chosen_ml",
    "weak_lp/chosen_ml",
    "shift/chosen_ml",
    "weak_shift/chosen_ml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare regression and Bradley-Terry image objectives using trainer_state.json logs."
    )
    parser.add_argument(
        "--regression-state",
        type=Path,
        default=Path("saves/llava_1_5_7b/ablation_regression_no_nll/trainer_state.json"),
        help="Trainer state from the regression run (no-NLL recommended).",
    )
    parser.add_argument(
        "--bt-state",
        type=Path,
        default=Path("saves/llava_1_5_7b/ablation_img_bt_no_nll/trainer_state.json"),
        help="Trainer state from the Bradley-Terry run (no-NLL recommended).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/regression_vs_bt_image_alignment_no_nll.png"),
        help="Output PNG path. A PDF with the same stem is also saved.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, list[float]]:
    with path.open(encoding="utf-8") as file:
        state = json.load(file)

    rows = [
        row
        for row in state.get("log_history", [])
        if "step" in row and all(metric in row for metric in METRICS)
    ]
    if not rows:
        raise ValueError(f"No rows containing all required metrics found in {path}")

    rows.sort(key=lambda row: row["step"])
    result = {"step": [row["step"] for row in rows]}
    result.update({metric: [row[metric] for row in rows] for metric in METRICS})
    print(f"Loaded {len(rows)} logged steps from {path}")
    return result


def plot_comparison(
    regression: dict[str, list[float]],
    bt: dict[str, list[float]],
    output: Path,
    dpi: int,
) -> None:
    colors = {"regression": "#0072B2", "bt": "#D55E00", "target": "#4D4D4D"}
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True, constrained_layout=True)

    axes[0].plot(
        regression["step"],
        regression["logps/chosen_ml"],
        color=colors["regression"],
        marker="o",
        markersize=3,
        linewidth=1.8,
        label="Regression",
    )
    axes[0].plot(
        bt["step"],
        bt["logps/chosen_ml"],
        color=colors["bt"],
        marker="s",
        markersize=3,
        linewidth=1.8,
        label="Bradley–Terry",
    )
    axes[0].plot(
        regression["step"],
        regression["weak_lp/chosen_ml"],
        color=colors["target"],
        linestyle="--",
        linewidth=1.6,
        label="Weak-model target",
    )
    axes[0].set_ylabel(r"Mean token log-probability")
    axes[0].set_title(r"(a) $\log \pi_\theta(y_w \mid x, m_l)$")
    axes[0].legend(frameon=False, ncol=3)

    axes[1].plot(
        regression["step"],
        regression["shift/chosen_ml"],
        color=colors["regression"],
        marker="o",
        markersize=3,
        linewidth=1.8,
        label="Regression",
    )
    axes[1].plot(
        bt["step"],
        bt["shift/chosen_ml"],
        color=colors["bt"],
        marker="s",
        markersize=3,
        linewidth=1.8,
        label="Bradley–Terry",
    )
    axes[1].plot(
        regression["step"],
        regression["weak_shift/chosen_ml"],
        color=colors["target"],
        linestyle="--",
        linewidth=1.6,
        label="Weak-model target",
    )
    axes[1].axhline(0.0, color="#999999", linewidth=0.8, zorder=0)
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel(r"Policy shift ($\gamma \log \frac{\pi}{\pi_{\rm ref}}$)")
    axes[1].set_title("(b) Policy shift on the corrupted-image pair")
    axes[1].legend(frameon=False, ncol=3)

    for axis in axes:
        axis.grid(alpha=0.22, linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    print(f"Saved {pdf_output}")


def main() -> None:
    args = parse_args()
    regression = load_metrics(args.regression_state)
    bt = load_metrics(args.bt_state)
    plot_comparison(regression, bt, args.output, args.dpi)


if __name__ == "__main__":
    main()
