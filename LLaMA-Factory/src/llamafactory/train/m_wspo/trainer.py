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

r"""m-WSPO trainer.

Subclasses LLaMA-Factory's `CustomDPOTrainer` so we inherit its multimodal
batch handling, ref-model wrapping, optimizer/scheduler hooks, BAdam wiring,
and DeepSpeed integration; only the loss computation is replaced.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import torch
from trl.models.utils import prepare_deepspeed, prepare_fsdp
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..dpo.trainer import CustomDPOTrainer
from ..trainer_utils import get_batch_logps, nested_detach
from .losses import MWSPOHyperParams, m_wspo_total_loss, mean_lp_from_sum, split_chosen_rejected


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


# Columns produced by `scripts/m_wspo/cache_weak_logprobs.py` and consumed
# by the trainer-side cache lookup below. Ordering doesn't matter -- we
# index by name.
_CACHE_COLUMNS = (
    "weak_aligned_lp_yw_mw",
    "weak_ref_lp_yw_mw",
    "weak_aligned_lp_yw_ml",
    "weak_ref_lp_yw_ml",
    "weak_aligned_lp_yl_mw",
    "weak_ref_lp_yl_mw",
)


# Keys produced by the multimodal collator (for the negative-image pass).
# Their `neg_*` counterparts replace these fields when we do the m_l forward.
_NEG_SWAP_KEYS = (
    "input_ids",
    "attention_mask",
    "labels",
    "position_ids",
    "rope_deltas",
    "pixel_values",
    "pixel_values_videos",
    "pixel_attention_mask",
    "image_grid_thw",
    "video_grid_thw",
    "image_sizes",
    "image_bound",
    "tgt_sizes",
    "patch_attention_mask",
    "aspect_ratio_ids",
    "aspect_ratio_mask",
    "cross_attention_mask",
    "mm_token_type_ids",
    "token_type_ids",
    "second_per_grid_ts",
    "video_second_per_grid",
)


def _build_neg_batch(batch: dict[str, "torch.Tensor"]) -> dict[str, "torch.Tensor"]:
    r"""Construct the m_l batch by replacing image-derived fields with their
    `neg_*` counterparts and stripping any leftover `neg_*` keys.

    Keys not produced by the negative collator pass (e.g. some sample-level
    metadata) are inherited unchanged from the m_w batch.
    """
    neg_batch: dict[str, Any] = {}
    for key, value in batch.items():
        if key.startswith("neg_"):
            continue

        neg_key = f"neg_{key}"
        if neg_key in batch:
            neg_batch[key] = batch[neg_key]
        elif key in _NEG_SWAP_KEYS:
            neg_batch[key] = value
        else:
            neg_batch[key] = value

    return neg_batch


def _strip_neg_keys(batch: dict[str, "torch.Tensor"]) -> dict[str, "torch.Tensor"]:
    return {k: v for k, v in batch.items() if not k.startswith("neg_")}


class MWSPOTrainer(CustomDPOTrainer):
    r"""Trainer that optimizes the unified m-WSPO objective (Eq. 11)."""

    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        weak_ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        weak_aligned_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            ref_model=ref_model,
            finetuning_args=finetuning_args,
            processor=processor,
            disable_dropout=disable_dropout,
            **kwargs,
        )
        self.weak_ref_model = self._prepare_frozen_external(weak_ref_model)
        self.weak_aligned_model = self._prepare_frozen_external(weak_aligned_model)
        self.processor = processor
        self._mwspo_hp = MWSPOHyperParams(
            gamma=finetuning_args.mwspo_gamma,
            alpha=finetuning_args.mwspo_alpha,
            beta_dpo=finetuning_args.mwspo_beta_dpo,
            dpo_temperature=finetuning_args.mwspo_dpo_temperature,
            disable_wspo=finetuning_args.mwspo_disable_wspo,
            disable_img_wspo=finetuning_args.mwspo_disable_img_wspo,
            disable_dpo=finetuning_args.mwspo_disable_dpo,
        )
        self._mask_image_in_dpo = finetuning_args.mwspo_mask_image_in_dpo

        self._weak_cache: Optional[dict[int, dict[str, float]]] = None
        if finetuning_args.mwspo_cache_weak_logprobs:
            self._weak_cache = self._load_weak_cache(finetuning_args.mwspo_weak_logprob_cache)
            logger.info_rank0(
                f"Loaded weak-logprob cache with {len(self._weak_cache)} rows from "
                f"{finetuning_args.mwspo_weak_logprob_cache!r}."
            )
            if self.weak_ref_model is None and self.weak_aligned_model is None:
                logger.info_rank0(
                    "No weak models loaded; relying entirely on the cache. Cache misses will raise."
                )

    @staticmethod
    def _load_weak_cache(path: Optional[str]) -> dict[int, dict[str, float]]:
        r"""Load the weak-logprob parquet produced by
        `scripts/m_wspo/cache_weak_logprobs.py` into a {sample_id -> row} dict.
        """
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(
                f"`mwspo_weak_logprob_cache` is set but {path!r} does not exist. "
                "Run `python scripts/m_wspo/cache_weak_logprobs.py ...` first."
            )

        import pandas as pd  # noqa: WPS433

        df = pd.read_parquet(path)
        missing = {"sample_id", *_CACHE_COLUMNS} - set(df.columns)
        if missing:
            raise ValueError(f"Weak-logprob cache at {path!r} is missing columns: {sorted(missing)}.")

        cache: dict[int, dict[str, float]] = {}
        for row in df.itertuples(index=False):
            row_dict = row._asdict()
            sample_id = int(row_dict["sample_id"])
            cache[sample_id] = {col: float(row_dict[col]) for col in _CACHE_COLUMNS}

        return cache

    def _lookup_weak_cache(
        self, sample_ids: list[int], device: torch.device
    ) -> Optional[dict[str, "torch.Tensor"]]:
        r"""Return cached MEAN log-probs for `sample_ids` if all are present,
        otherwise return None (signalling fall-through to the live forward).
        """
        if self._weak_cache is None:
            return None

        gathered: dict[str, list[float]] = {col: [] for col in _CACHE_COLUMNS}
        for sid in sample_ids:
            row = self._weak_cache.get(sid)
            if row is None:
                logger.warning_rank0(
                    f"Weak-logprob cache missing sample_id={sid}; falling back to live weak forward."
                )
                return None

            for col in _CACHE_COLUMNS:
                gathered[col].append(row[col])

        return {col: torch.tensor(values, dtype=torch.float32, device=device) for col, values in gathered.items()}

    def _prepare_frozen_external(
        self, model: Optional[Union["PreTrainedModel", torch.nn.Module]]
    ) -> Optional[torch.nn.Module]:
        r"""Mirror `CustomDPOTrainer.__init__`'s ref-model preparation logic
        for an arbitrary frozen MLLM (used for the weak pair)."""
        if model is None:
            return None

        for p in model.parameters():
            p.requires_grad_(False)

        model.eval()

        if self.is_deepspeed_enabled:
            if not (getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)):
                model = prepare_deepspeed(model, self.accelerator)
        elif self.is_fsdp_enabled:
            if self.accelerator.is_fsdp2:
                from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                model = fsdp2_prepare_model(self.accelerator, model)
            else:
                model = prepare_fsdp(model, self.accelerator)
        else:
            model = self.accelerator.prepare_model(model, evaluation_mode=True)
            model.eval()

        return model

    def _logps_for_batch(
        self,
        model: "PreTrainedModel",
        batch: dict[str, "torch.Tensor"],
        is_no_grad: bool,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        r"""Forward `model` on a 2N pairwise batch and return (sum_logps, valid_length).

        `sum_logps` and `valid_length` are length-2N tensors; chosen and
        rejected halves can be split via `split_chosen_rejected`.
        """
        ctx = torch.no_grad() if is_no_grad else nullcontext()
        with ctx:
            # CustomDPOTrainer.concatenated_forward pops labels and runs the
            # forward; we replicate its body here so we can compute logps from
            # the same `get_batch_logps` helper without touching parent state.
            local_batch = nested_detach(batch, clone=True)
            labels = local_batch.pop("labels")
            outputs = model(**local_batch, return_dict=True, use_cache=False)
            logits = outputs.logits.to(torch.float32)
            sum_logps, valid_length = get_batch_logps(
                logits=logits, labels=labels, label_pad_token_id=IGNORE_INDEX, ld_alpha=None
            )

        return sum_logps, valid_length

    def _strong_ref_logps(
        self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"]
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        r"""Compute strong reference logps; if `self.ref_model is None` and
        we're using LoRA, fall back to the base model via `disable_adapter`."""
        if self.ref_model is None:
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
            ref_model = model
        else:
            ref_context = nullcontext()
            ref_model = self.ref_model

        with ref_context:
            return self._logps_for_batch(ref_model, batch, is_no_grad=True)

    @staticmethod
    def _zero_pixel_values(batch: dict[str, "torch.Tensor"]) -> dict[str, "torch.Tensor"]:
        r"""Sanity-ablation helper used by `mwspo_mask_image_in_dpo`."""
        masked: dict[str, Any] = {}
        for key, value in batch.items():
            if key in {"pixel_values", "pixel_values_videos"} and torch.is_tensor(value):
                masked[key] = torch.zeros_like(value)
            else:
                masked[key] = value

        return masked

    def _maybe_check_grid_consistency(
        self, pos_batch: dict[str, "torch.Tensor"], neg_batch: dict[str, "torch.Tensor"]
    ) -> None:
        r"""Soft check: warn once when m_w and m_l image grids diverge in
        ways that change input_ids length (Qwen2-VL / LLaVA-Next).

        We don't fail hard because differing grids are still mathematically
        well-defined; we only point users at `image_max_pixels` if they
        haven't capped the resolution.
        """
        if getattr(self, "_warned_grid_mismatch", False):
            return

        pos_len = pos_batch["input_ids"].shape[1] if "input_ids" in pos_batch else None
        neg_len = neg_batch["input_ids"].shape[1] if "input_ids" in neg_batch else None
        if pos_len is not None and neg_len is not None and pos_len != neg_len:
            logger.warning_rank0(
                f"m_w / m_l input_ids lengths differ ({pos_len} vs {neg_len}); this is fine but "
                "indicates variable image-token counts. For peak efficiency, set `image_max_pixels` "
                "to a single canonical resolution in your YAML."
            )
            self._warned_grid_mismatch = True

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
        r"""Compute the unified m-WSPO loss (Eq. 11) and per-component metrics."""
        hp = self._mwspo_hp
        sample_id_tensor = batch.get("sample_id")
        sample_ids: list[int] = sample_id_tensor.tolist() if sample_id_tensor is not None else []
        pos_batch = _strip_neg_keys(batch)
        pos_batch.pop("sample_id", None)
        neg_batch = _build_neg_batch(batch)
        neg_batch.pop("sample_id", None)
        self._maybe_check_grid_consistency(pos_batch, neg_batch)

        # ---- Strong policy ----
        pol_pos_sum, pol_pos_len = self._logps_for_batch(model, pos_batch, is_no_grad=False)
        if not hp.disable_img_wspo:
            pol_neg_sum, pol_neg_len = self._logps_for_batch(model, neg_batch, is_no_grad=False)
        else:
            pol_neg_sum = pol_neg_len = None

        # ---- Strong reference (frozen, or LoRA-disabled adapter) ----
        ref_pos_sum, _ = self._strong_ref_logps(model, pos_batch)
        if not hp.disable_img_wspo:
            ref_neg_sum, _ = self._strong_ref_logps(model, neg_batch)
        else:
            ref_neg_sum = None

        # ---- Weak pair: cache lookup or live forward ----
        cached = self._lookup_weak_cache(sample_ids, device=pol_pos_sum.device) if sample_ids else None
        if cached is not None:
            wa_mean_yw = cached["weak_aligned_lp_yw_mw"]
            wr_mean_yw = cached["weak_ref_lp_yw_mw"]
            wa_mean_yl = cached["weak_aligned_lp_yl_mw"]
            wr_mean_yl = cached["weak_ref_lp_yl_mw"]
            wa_mean_neg_yw = cached["weak_aligned_lp_yw_ml"]
            wr_mean_neg_yw = cached["weak_ref_lp_yw_ml"]
        else:
            if self.weak_aligned_model is None or self.weak_ref_model is None:
                raise RuntimeError(
                    "Weak-pair forward required but neither model is loaded and the cache "
                    "doesn't cover this batch. Either provide `mwspo_weak_*_model` so we can "
                    "fall back, or rebuild the cache to include all sample IDs."
                )

            wa_pos_sum, _ = self._logps_for_batch(self.weak_aligned_model, pos_batch, is_no_grad=True)
            wr_pos_sum, _ = self._logps_for_batch(self.weak_ref_model, pos_batch, is_no_grad=True)
            wa_pos_sum_yw, wa_pos_sum_yl = split_chosen_rejected(wa_pos_sum)
            wr_pos_sum_yw, wr_pos_sum_yl = split_chosen_rejected(wr_pos_sum)
            _live_pol_pos_len_yw, _live_pol_pos_len_yl = split_chosen_rejected(pol_pos_len)
            wa_mean_yw = mean_lp_from_sum(wa_pos_sum_yw, _live_pol_pos_len_yw)
            wa_mean_yl = mean_lp_from_sum(wa_pos_sum_yl, _live_pol_pos_len_yl)
            wr_mean_yw = mean_lp_from_sum(wr_pos_sum_yw, _live_pol_pos_len_yw)
            wr_mean_yl = mean_lp_from_sum(wr_pos_sum_yl, _live_pol_pos_len_yl)

            if not hp.disable_img_wspo:
                wa_neg_sum, _ = self._logps_for_batch(self.weak_aligned_model, neg_batch, is_no_grad=True)
                wr_neg_sum, _ = self._logps_for_batch(self.weak_ref_model, neg_batch, is_no_grad=True)
                wa_neg_sum_yw, _ = split_chosen_rejected(wa_neg_sum)
                wr_neg_sum_yw, _ = split_chosen_rejected(wr_neg_sum)
                _live_pol_neg_len_yw, _ = split_chosen_rejected(pol_neg_len)
                wa_mean_neg_yw = mean_lp_from_sum(wa_neg_sum_yw, _live_pol_neg_len_yw)
                wr_mean_neg_yw = mean_lp_from_sum(wr_neg_sum_yw, _live_pol_neg_len_yw)
            else:
                wa_mean_neg_yw = wr_mean_neg_yw = None

        # Split strong policy / reference into chosen vs rejected.
        pol_pos_sum_yw, pol_pos_sum_yl = split_chosen_rejected(pol_pos_sum)
        pol_pos_len_yw, pol_pos_len_yl = split_chosen_rejected(pol_pos_len)
        ref_pos_sum_yw, ref_pos_sum_yl = split_chosen_rejected(ref_pos_sum)

        pol_mean_yw = mean_lp_from_sum(pol_pos_sum_yw, pol_pos_len_yw)
        ref_mean_yw = mean_lp_from_sum(ref_pos_sum_yw, pol_pos_len_yw)
        pol_mean_yl = mean_lp_from_sum(pol_pos_sum_yl, pol_pos_len_yl)
        ref_mean_yl = mean_lp_from_sum(ref_pos_sum_yl, pol_pos_len_yl)

        if not hp.disable_img_wspo:
            pol_neg_sum_yw, _ = split_chosen_rejected(pol_neg_sum)
            pol_neg_len_yw, _ = split_chosen_rejected(pol_neg_len)
            ref_neg_sum_yw, _ = split_chosen_rejected(ref_neg_sum)
            pol_mean_neg_yw = mean_lp_from_sum(pol_neg_sum_yw, pol_neg_len_yw)
            ref_mean_neg_yw = mean_lp_from_sum(ref_neg_sum_yw, pol_neg_len_yw)
        else:
            pol_mean_neg_yw = pol_mean_yw
            ref_mean_neg_yw = ref_mean_yw
            wa_mean_neg_yw = wa_mean_neg_yw if wa_mean_neg_yw is not None else wa_mean_yw
            wr_mean_neg_yw = wr_mean_neg_yw if wr_mean_neg_yw is not None else wr_mean_yw

        # ---- Eq. 10: optionally re-do strong forward with masked images ----
        if not hp.disable_dpo and self._mask_image_in_dpo:
            masked_batch = self._zero_pixel_values(pos_batch)
            pol_dpo_sum, _ = self._logps_for_batch(model, masked_batch, is_no_grad=False)
            ref_dpo_sum, _ = self._strong_ref_logps(model, masked_batch)
            pol_dpo_sum_yw, pol_dpo_sum_yl = split_chosen_rejected(pol_dpo_sum)
            ref_dpo_sum_yw, ref_dpo_sum_yl = split_chosen_rejected(ref_dpo_sum)
        else:
            pol_dpo_sum_yw, pol_dpo_sum_yl = pol_pos_sum_yw, pol_pos_sum_yl
            ref_dpo_sum_yw, ref_dpo_sum_yl = ref_pos_sum_yw, ref_pos_sum_yl

        loss, components = m_wspo_total_loss(
            strong_pol_mean_yw=pol_mean_yw,
            strong_ref_mean_yw=ref_mean_yw,
            weak_aln_mean_yw=wa_mean_yw,
            weak_ref_mean_yw=wr_mean_yw,
            strong_pol_mean_neg_yw=pol_mean_neg_yw,
            strong_ref_mean_neg_yw=ref_mean_neg_yw,
            weak_aln_mean_neg_yw=wa_mean_neg_yw,
            weak_ref_mean_neg_yw=wr_mean_neg_yw,
            strong_pol_mean_yl=pol_mean_yl,
            strong_ref_mean_yl=ref_mean_yl,
            weak_aln_mean_yl=wa_mean_yl,
            weak_ref_mean_yl=wr_mean_yl,
            strong_pol_sum_yw=pol_dpo_sum_yw,
            strong_pol_sum_yl=pol_dpo_sum_yl,
            strong_ref_sum_yw=ref_dpo_sum_yw,
            strong_ref_sum_yl=ref_dpo_sum_yl,
            hp=hp,
        )

        prefix = "eval_" if train_eval == "eval" else ""
        chosen_rewards = (hp.dpo_temperature * (pol_pos_sum_yw - ref_pos_sum_yw)).detach()
        rejected_rewards = (hp.dpo_temperature * (pol_pos_sum_yl - ref_pos_sum_yl)).detach()
        metrics: dict[str, "torch.Tensor"] = {
            f"{prefix}rewards/chosen": chosen_rewards.mean(),
            f"{prefix}rewards/rejected": rejected_rewards.mean(),
            f"{prefix}rewards/accuracies": (chosen_rewards > rejected_rewards).float().mean(),
            f"{prefix}rewards/margins": (chosen_rewards - rejected_rewards).mean(),
            f"{prefix}logps/chosen": pol_pos_sum_yw.detach().mean(),
            f"{prefix}logps/rejected": pol_pos_sum_yl.detach().mean(),
        }
        for name, value in components.items():
            metrics[f"{prefix}{name}"] = value

        # Store as floats for the parent's `log` aggregator.
        metrics = {k: v.float().mean().item() if torch.is_tensor(v) else float(v) for k, v in metrics.items()}
        return loss, metrics
