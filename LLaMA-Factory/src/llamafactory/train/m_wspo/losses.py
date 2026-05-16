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

r"""m-WSPO loss math.

The four loss components from the paper are:

  Eq. 6  (L_WSPO)            : weak-to-strong transfer on (x, m_w, y_w).
  Eq. 8  (L_img_WSPO_winning): vision-oriented weak-to-strong on (x, m_l, y_w).
  Eq. 9  (L_img_WSPO_losing) : vision-oriented weak-to-strong on (x, m_w, y_l).
  Eq. 10 (L_DPO^m)           : multimodal DPO on (x, m_w, [y_w, y_l]).

Eqs. 6 / 8 / 9 use *length-normalized* mean log-prob (helper `mean_lp`); Eq. 10
uses *summed* log-prob (helper `sum_lp`). Mixing the two silently rescales the
DPO temperature -- see the §8 convention note in `agent.md`.

The WSPO loss math itself is ported from the original WSPO repo
(https://github.com/zwhong714/weak-to-strong-preference-optimization,
`compute_preference_loss_wspo` in `train/wspo/trainer.py`):

    losses = gamma * (pol - ref_pol) - (weak_aln - weak_ref)
    losses = losses ** 2 / valid_length

generalized to the (x, m, y) tuples used by m-WSPO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F


if TYPE_CHECKING:
    pass


@dataclass
class MWSPOHyperParams:
    r"""Lightweight, picklable view of the four m-WSPO hyperparameters."""

    gamma: float
    alpha: float
    beta_dpo: float
    dpo_temperature: float
    disable_wspo: bool = False
    disable_img_wspo: bool = False
    disable_dpo: bool = False


def split_chosen_rejected(t: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    r"""Split a (2N, ...) tensor produced by `PairwiseDataCollator` into
    (chosen=N, rejected=N) halves.
    """
    bsz = t.size(0) // 2
    return t[:bsz], t[bsz:]


def mean_lp_from_sum(logps_sum: "torch.Tensor", valid_length: "torch.Tensor") -> "torch.Tensor":
    r"""Convert summed log-probs to per-token *mean* log-probs (Eqs. 6, 8, 9).

    `valid_length` is clamped to >=1 to avoid division by zero on samples
    where every label is `IGNORE_INDEX` (shouldn't happen in practice).
    """
    return logps_sum / valid_length.clamp(min=1).to(logps_sum.dtype)


def loss_wspo_mm(
    strong_pol_lp: "torch.Tensor",
    strong_ref_lp: "torch.Tensor",
    weak_aln_lp: "torch.Tensor",
    weak_ref_lp: "torch.Tensor",
    gamma: float,
) -> "torch.Tensor":
    r"""Eq. 6 -- weak-to-strong transfer on (x, m_w, y_w).

    All inputs are per-example mean log-probs of y_w under the corresponding
    model conditioned on (x, m_w). Returns a scalar mean over the batch.
    """
    s = gamma * (strong_pol_lp - strong_ref_lp)
    w = weak_aln_lp - weak_ref_lp
    return ((s - w) ** 2).mean()


def loss_img_wspo_winning(
    strong_pol_lp_neg_yw: "torch.Tensor",
    strong_ref_lp_neg_yw: "torch.Tensor",
    weak_aln_lp_neg_yw: "torch.Tensor",
    weak_ref_lp_neg_yw: "torch.Tensor",
    gamma: float,
) -> "torch.Tensor":
    r"""Eq. 8 -- vision-oriented weak-to-strong on (x, m_l, y_w).

    Penalises the strong policy for *increasing* the probability of the
    winning response when conditioned on a *mismatched* image.
    """
    s = gamma * (strong_pol_lp_neg_yw - strong_ref_lp_neg_yw)
    w = weak_aln_lp_neg_yw - weak_ref_lp_neg_yw
    return ((w - s) ** 2).mean()


def loss_img_wspo_losing(
    strong_pol_lp_yl: "torch.Tensor",
    strong_ref_lp_yl: "torch.Tensor",
    weak_aln_lp_yl: "torch.Tensor",
    weak_ref_lp_yl: "torch.Tensor",
    gamma: float,
) -> "torch.Tensor":
    r"""Eq. 9 -- vision-oriented weak-to-strong on (x, m_w, y_l).

    Penalises the strong policy for *decreasing* the probability of the
    losing response when conditioned on the matched image (cross-modal
    contrast on the rejected branch).
    """
    s = gamma * (strong_pol_lp_yl - strong_ref_lp_yl)
    w = weak_aln_lp_yl - weak_ref_lp_yl
    return ((w - s) ** 2).mean()


def loss_dpo_mm(
    strong_pol_sum_yw: "torch.Tensor",
    strong_pol_sum_yl: "torch.Tensor",
    strong_ref_sum_yw: "torch.Tensor",
    strong_ref_sum_yl: "torch.Tensor",
    dpo_temperature: float,
) -> "torch.Tensor":
    r"""Eq. 10 -- standard DPO loss conditioned on the matched image m_w.

    NOTE: uses *summed* log-probs, not length-normalized -- this is the
    canonical DPO form and the temperature `dpo_temperature` is calibrated
    against summed log-ratios.
    """
    pi_w = strong_pol_sum_yw - strong_ref_sum_yw
    pi_l = strong_pol_sum_yl - strong_ref_sum_yl
    return -F.logsigmoid(dpo_temperature * (pi_w - pi_l)).mean()


def m_wspo_total_loss(
    *,
    # Eq. 6 inputs (mean log-probs on (x, m_w, y_w))
    strong_pol_mean_yw: "torch.Tensor",
    strong_ref_mean_yw: "torch.Tensor",
    weak_aln_mean_yw: "torch.Tensor",
    weak_ref_mean_yw: "torch.Tensor",
    # Eq. 8 inputs (mean log-probs on (x, m_l, y_w))
    strong_pol_mean_neg_yw: "torch.Tensor",
    strong_ref_mean_neg_yw: "torch.Tensor",
    weak_aln_mean_neg_yw: "torch.Tensor",
    weak_ref_mean_neg_yw: "torch.Tensor",
    # Eq. 9 inputs (mean log-probs on (x, m_w, y_l))
    strong_pol_mean_yl: "torch.Tensor",
    strong_ref_mean_yl: "torch.Tensor",
    weak_aln_mean_yl: "torch.Tensor",
    weak_ref_mean_yl: "torch.Tensor",
    # Eq. 10 inputs (sum log-probs on (x, m_w, [y_w, y_l]))
    strong_pol_sum_yw: "torch.Tensor",
    strong_pol_sum_yl: "torch.Tensor",
    strong_ref_sum_yw: "torch.Tensor",
    strong_ref_sum_yl: "torch.Tensor",
    hp: "MWSPOHyperParams",
) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
    r"""Assemble Eq. 11.

    Returns:
        total_loss: scalar tensor.
        metrics: dict of scalar component losses for logging.
    """
    zero = strong_pol_mean_yw.new_zeros(())

    L_wspo = (
        zero
        if hp.disable_wspo
        else loss_wspo_mm(strong_pol_mean_yw, strong_ref_mean_yw, weak_aln_mean_yw, weak_ref_mean_yw, hp.gamma)
    )

    if hp.disable_img_wspo:
        L_imgW = zero
        L_imgL = zero
    else:
        L_imgW = loss_img_wspo_winning(
            strong_pol_mean_neg_yw, strong_ref_mean_neg_yw, weak_aln_mean_neg_yw, weak_ref_mean_neg_yw, hp.gamma
        )
        L_imgL = loss_img_wspo_losing(
            strong_pol_mean_yl, strong_ref_mean_yl, weak_aln_mean_yl, weak_ref_mean_yl, hp.gamma
        )

    L_dpo = (
        zero
        if hp.disable_dpo
        else loss_dpo_mm(strong_pol_sum_yw, strong_pol_sum_yl, strong_ref_sum_yw, strong_ref_sum_yl, hp.dpo_temperature)
    )

    total = L_wspo + hp.alpha * (L_imgW + L_imgL) + hp.beta_dpo * L_dpo
    metrics = {
        "L_wspo": L_wspo.detach(),
        "L_img_w": L_imgW.detach(),
        "L_img_l": L_imgL.detach(),
        "L_dpo": L_dpo.detach(),
    }
    return total, metrics
