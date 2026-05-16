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

r"""Smoke tests for the m-WSPO loss math (`agent.md` §12, items 3 & 4).

These are deterministic, CPU-only unit tests; no checkpoints or dataset
required. They guard against the most common regressions in the loss math:

- §12.3: ``loss_wspo_mm`` with ``gamma=0`` AND ``weak_aln == weak_ref`` must
  collapse to zero. Catches sign / argument-order bugs.
- §12.4: ``loss_dpo_mm`` with ``strong_pol == strong_ref`` must equal
  ``log 2``. Catches the ``sum_lp`` vs ``mean_lp`` confusion that silently
  rescales the DPO temperature.
"""

from __future__ import annotations

import math

import pytest
import torch

from llamafactory.train.m_wspo.losses import (
    MWSPOHyperParams,
    loss_dpo_mm,
    loss_img_wspo_losing,
    loss_img_wspo_winning,
    loss_wspo_mm,
    m_wspo_total_loss,
    mean_lp_from_sum,
    split_chosen_rejected,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# ------------------------------- helpers -------------------------------


def _rand_lp(batch_size: int = 4) -> torch.Tensor:
    return torch.randn(batch_size, dtype=torch.float64)


# ------------------------------- §12.3 ---------------------------------


def test_loss_wspo_zero_gamma_and_equal_weak_pair_is_zero():
    """gamma=0 + weak_aln==weak_ref => (0 - 0)^2 = 0."""
    pol = _rand_lp()
    ref = _rand_lp()
    weak = _rand_lp()  # same tensor for aligned and ref => w = 0

    loss = loss_wspo_mm(strong_pol_lp=pol, strong_ref_lp=ref, weak_aln_lp=weak, weak_ref_lp=weak, gamma=0.0)
    assert torch.isclose(loss, torch.zeros_like(loss), atol=1e-12), loss


def test_loss_img_wspo_winning_zero_gamma_and_equal_weak_pair_is_zero():
    pol, ref, weak = _rand_lp(), _rand_lp(), _rand_lp()
    loss = loss_img_wspo_winning(pol, ref, weak, weak, gamma=0.0)
    assert torch.isclose(loss, torch.zeros_like(loss), atol=1e-12), loss


def test_loss_img_wspo_losing_zero_gamma_and_equal_weak_pair_is_zero():
    pol, ref, weak = _rand_lp(), _rand_lp(), _rand_lp()
    loss = loss_img_wspo_losing(pol, ref, weak, weak, gamma=0.0)
    assert torch.isclose(loss, torch.zeros_like(loss), atol=1e-12), loss


# ------------------------------- §12.4 ---------------------------------


def test_loss_dpo_mm_when_pol_equals_ref_is_log2():
    """strong_pol == strong_ref => pi_w = pi_l = 0 => -log sigma(0) = log 2."""
    sum_yw = _rand_lp()
    sum_yl = _rand_lp()
    loss = loss_dpo_mm(
        strong_pol_sum_yw=sum_yw,
        strong_pol_sum_yl=sum_yl,
        strong_ref_sum_yw=sum_yw,
        strong_ref_sum_yl=sum_yl,
        dpo_temperature=0.1,
    )
    expected = torch.tensor(math.log(2.0), dtype=loss.dtype)
    assert torch.isclose(loss, expected, atol=1e-10), (loss, expected)


def test_loss_dpo_mm_dpo_temperature_zero_is_log2():
    """temperature=0 => pi_w - pi_l = 0 => log 2 regardless of inputs."""
    pol_yw, pol_yl, ref_yw, ref_yl = _rand_lp(), _rand_lp(), _rand_lp(), _rand_lp()
    loss = loss_dpo_mm(pol_yw, pol_yl, ref_yw, ref_yl, dpo_temperature=0.0)
    expected = torch.tensor(math.log(2.0), dtype=loss.dtype)
    assert torch.isclose(loss, expected, atol=1e-10), (loss, expected)


# ------------------------------- helpers --------------------------------


def test_mean_lp_from_sum_clamps_zero_lengths():
    sums = torch.tensor([1.0, -2.0])
    lengths = torch.tensor([0, 5])  # 0 should be clamped to 1
    means = mean_lp_from_sum(sums, lengths)
    assert torch.allclose(means, torch.tensor([1.0, -2.0 / 5.0]), atol=1e-7)


def test_split_chosen_rejected():
    t = torch.arange(8, dtype=torch.float32)
    chosen, rejected = split_chosen_rejected(t)
    assert torch.equal(chosen, torch.arange(4, dtype=torch.float32))
    assert torch.equal(rejected, torch.arange(4, 8, dtype=torch.float32))


# ----------------------------- total loss -------------------------------


def test_total_loss_assembly_matches_eq_11():
    """Verify the assembled total exactly equals
    L_wspo + alpha*(L_imgW + L_imgL) + beta_dpo*L_dpo.
    """
    bsz = 3
    rand = lambda: torch.randn(bsz, dtype=torch.float64)  # noqa: E731

    pol_w, ref_w, wa_w, wr_w = rand(), rand(), rand(), rand()
    pol_neg, ref_neg, wa_neg, wr_neg = rand(), rand(), rand(), rand()
    pol_l, ref_l, wa_l, wr_l = rand(), rand(), rand(), rand()
    pol_sum_yw, pol_sum_yl, ref_sum_yw, ref_sum_yl = rand(), rand(), rand(), rand()
    hp = MWSPOHyperParams(gamma=0.1, alpha=0.5, beta_dpo=1.0, dpo_temperature=0.1)

    total, comps = m_wspo_total_loss(
        strong_pol_mean_yw=pol_w, strong_ref_mean_yw=ref_w, weak_aln_mean_yw=wa_w, weak_ref_mean_yw=wr_w,
        strong_pol_mean_neg_yw=pol_neg, strong_ref_mean_neg_yw=ref_neg, weak_aln_mean_neg_yw=wa_neg,
        weak_ref_mean_neg_yw=wr_neg,
        strong_pol_mean_yl=pol_l, strong_ref_mean_yl=ref_l, weak_aln_mean_yl=wa_l, weak_ref_mean_yl=wr_l,
        strong_pol_sum_yw=pol_sum_yw, strong_pol_sum_yl=pol_sum_yl,
        strong_ref_sum_yw=ref_sum_yw, strong_ref_sum_yl=ref_sum_yl,
        hp=hp,
    )

    expected = comps["L_wspo"] + hp.alpha * (comps["L_img_w"] + comps["L_img_l"]) + hp.beta_dpo * comps["L_dpo"]
    assert torch.isclose(total.detach(), expected, atol=1e-10), (total, expected)


def test_disable_flags_zero_components():
    rand = lambda: torch.randn(2, dtype=torch.float64)  # noqa: E731
    pol = rand()
    args = dict(
        strong_pol_mean_yw=pol, strong_ref_mean_yw=rand(), weak_aln_mean_yw=rand(), weak_ref_mean_yw=rand(),
        strong_pol_mean_neg_yw=rand(), strong_ref_mean_neg_yw=rand(),
        weak_aln_mean_neg_yw=rand(), weak_ref_mean_neg_yw=rand(),
        strong_pol_mean_yl=rand(), strong_ref_mean_yl=rand(), weak_aln_mean_yl=rand(), weak_ref_mean_yl=rand(),
        strong_pol_sum_yw=rand(), strong_pol_sum_yl=rand(),
        strong_ref_sum_yw=rand(), strong_ref_sum_yl=rand(),
    )
    hp_off = MWSPOHyperParams(
        gamma=0.1, alpha=0.5, beta_dpo=1.0, dpo_temperature=0.1,
        disable_wspo=True, disable_img_wspo=True, disable_dpo=True,
    )
    total, comps = m_wspo_total_loss(hp=hp_off, **args)
    assert float(total) == 0.0
    assert float(comps["L_wspo"]) == 0.0
    assert float(comps["L_img_w"]) == 0.0
    assert float(comps["L_img_l"]) == 0.0
    assert float(comps["L_dpo"]) == 0.0
