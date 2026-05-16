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

r"""m-WSPO training workflow.

Loads the four required policies (`pi_theta^s`, `pi_ref^s`, `pi_r^w`,
`pi_ref^w`), wires them into `MWSPOTrainer`, and kicks off training.
Only the strong policy carries gradients (and LoRA adapters when
`finetuning_type=lora`); everything else is frozen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ...data import MWSPOPairwiseDataCollator, get_dataset, get_template_and_fix_tokenizer
from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...hparams import FinetuningArguments, ModelArguments
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push, create_ref_model
from .trainer import MWSPOTrainer


if TYPE_CHECKING:
    from transformers import PreTrainedModel, Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments


logger = logging.get_logger(__name__)


def _load_frozen_mllm(
    model_name_or_path: str,
    adapter_name_or_path: Optional[str],
    template_model_args: "ModelArguments",
) -> "PreTrainedModel":
    r"""Load a frozen multimodal model (used for the weak pair).

    Reuses LLaMA-Factory's `load_model(..., is_trainable=False)` so the
    correct dtype, vision tower, and processor configuration are picked up
    from `template_model_args`. Adapters are optional.
    """
    weak_model_args = ModelArguments.copyfrom(
        template_model_args,
        model_name_or_path=model_name_or_path,
        adapter_name_or_path=adapter_name_or_path,
    )
    weak_finetuning_args = FinetuningArguments()
    weak_tokenizer = load_tokenizer(weak_model_args)["tokenizer"]
    weak_model = load_model(
        weak_tokenizer,
        weak_model_args,
        weak_finetuning_args,
        is_trainable=False,
        add_valuehead=False,
    )
    weak_model.eval()
    for p in weak_model.parameters():
        p.requires_grad_(False)

    return weak_model


def run_m_wspo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    r"""End-to-end m-WSPO training run."""
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Pairwise multimodal data uses the "rm" stage processor (matching DPO).
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)

    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    data_collator = MWSPOPairwiseDataCollator(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        neg_image_strategy=finetuning_args.mwspo_neg_image_strategy,
        **tokenizer_module,
    )

    # Strong reference model. With LoRA we follow CustomDPOTrainer's trick of
    # using the base model with adapters disabled (`ref_model = None`); for
    # full / freeze finetuning we materialise a frozen copy.
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None and (not training_args.do_train):
            ref_model: Optional[Any] = model
        else:
            ref_model = create_ref_model(model_args, finetuning_args)
    else:
        ref_model = None

    # Skip loading the weak pair entirely when the cache covers the dataset
    # AND the user hasn't specified them. Saves significant GPU memory.
    weak_ref_model = (
        _load_frozen_mllm(
            finetuning_args.mwspo_weak_ref_model, finetuning_args.mwspo_weak_ref_adapters, model_args
        )
        if finetuning_args.mwspo_weak_ref_model is not None
        else None
    )
    weak_aligned_model = (
        _load_frozen_mllm(
            finetuning_args.mwspo_weak_aligned_model, finetuning_args.mwspo_weak_aligned_adapters, model_args
        )
        if finetuning_args.mwspo_weak_aligned_model is not None
        else None
    )

    logger.info_rank0(
        f"Loaded m-WSPO policies: strong={model_args.model_name_or_path}, "
        f"weak_ref={finetuning_args.mwspo_weak_ref_model or '<cache-only>'}, "
        f"weak_aligned={finetuning_args.mwspo_weak_aligned_model or '<cache-only>'}, "
        f"cache={finetuning_args.mwspo_weak_logprob_cache or '<none>'}."
    )

    trainer = MWSPOTrainer(
        model=model,
        ref_model=ref_model,
        weak_ref_model=weak_ref_model,
        weak_aligned_model=weak_aligned_model,
        args=training_args,
        finetuning_args=finetuning_args,
        processor=tokenizer_module.get("processor"),
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="rm"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss", "rewards/accuracies", "L_wspo", "L_img_w", "L_img_l", "L_dpo"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += [f"eval_{k}_loss" for k in dataset_module["eval_dataset"].keys()]
            else:
                keys += ["eval_loss"]

            plot_loss(training_args.output_dir, keys=keys)

    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        if id(model) == id(ref_model):
            remove_keys = [key for key in metrics if "rewards" in key]
            for key in remove_keys:
                metrics.pop(key)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
