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

r"""Offline pre-computation of weak-pair log-probabilities for m-WSPO.

The unified m-WSPO loss (Eq. 11) requires *two* extra forward passes per step
through the weak reference / weak DPO-aligned MLLMs. Since both are frozen,
their per-(sample, image, response) mean log-probabilities can be cached
ahead of time and read at training time. This script does the caching pass.

Output schema (parquet):
    sample_id              : int64    # matches the `_sample_id` column added by `_get_preprocessed_dataset`
    weak_aligned_lp_yw_mw  : float32  # mean lp of y_w  given (x, m_w)
    weak_ref_lp_yw_mw      : float32
    weak_aligned_lp_yw_ml  : float32  # mean lp of y_w  given (x, m_l)
    weak_ref_lp_yw_ml      : float32
    weak_aligned_lp_yl_mw  : float32  # mean lp of y_l  given (x, m_w)
    weak_ref_lp_yl_mw      : float32

The trainer (`MWSPOTrainer`) reads this parquet at startup; when caching is
enabled, weak forward passes are skipped entirely if all sample_ids in a
training batch are present in the cache. Cache misses fall back to live
weak forwards if the weak models are still loaded, otherwise raise.

Usage:
    python -m scripts.m_wspo.cache_weak_logprobs \
        --dataset rlhf_v_mwspo \
        --weak_ref_model llava-hf/llava-1.5-7b-hf \
        --weak_aligned_model your-org/llava-1.5-7b-dpo-rlhfv \
        --template vicuna \
        --output cache/rlhf_v_mwspo_weak_lp.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING

import pandas as pd
import torch
from tqdm import tqdm


# Allow `python scripts/m_wspo/cache_weak_logprobs.py ...` from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "src"))

from llamafactory.data import MWSPOPairwiseDataCollator, get_dataset, get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.extras.constants import IGNORE_INDEX  # noqa: E402
from llamafactory.hparams import DataArguments, FinetuningArguments, ModelArguments  # noqa: E402
from llamafactory.model import load_model, load_tokenizer  # noqa: E402
from llamafactory.train.m_wspo.trainer import _build_neg_batch  # noqa: E402
from llamafactory.train.trainer_utils import get_batch_logps  # noqa: E402


if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Registered dataset name (e.g. rlhf_v_mwspo).")
    parser.add_argument("--weak_ref_model", required=True, help="Path or HF id of pi_ref^w.")
    parser.add_argument("--weak_aligned_model", required=True, help="Path or HF id of pi_r^w.")
    parser.add_argument(
        "--template_model_path",
        default=None,
        help="HF id whose tokenizer/processor to use for input prep "
        "(defaults to weak_ref_model; ideally matches the strong policy's tokenizer).",
    )
    parser.add_argument("--template", required=True, help="Chat template name (e.g. vicuna, llava_next).")
    parser.add_argument("--output", required=True, help="Destination parquet path.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--cutoff_len", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--neg_image_strategy", choices=["in_batch", "pre_mined"], default="in_batch")
    return parser.parse_args()


def _build_inference_pipeline(args: argparse.Namespace):
    template_model_path = args.template_model_path or args.weak_ref_model
    model_args = ModelArguments(
        model_name_or_path=template_model_path,
        trust_remote_code=True,
        infer_dtype="bfloat16" if args.bf16 else "auto",
    )
    data_args = DataArguments(
        dataset=args.dataset,
        template=args.template,
        cutoff_len=args.cutoff_len,
        max_samples=args.max_samples,
        preprocessing_num_workers=4,
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Reuse a no-op `Seq2SeqTrainingArguments` only for `seed` / `output_dir`.
    from transformers import Seq2SeqTrainingArguments

    training_args = Seq2SeqTrainingArguments(output_dir="/tmp/m_wspo_cache", do_train=False)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    train_dataset = dataset_module["train_dataset"]

    collator = MWSPOPairwiseDataCollator(
        template=template,
        model=None,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        neg_image_strategy=args.neg_image_strategy,
        **tokenizer_module,
    )

    return tokenizer_module, train_dataset, collator, model_args


def _make_loader(dataset, collator, batch_size: int) -> "DataLoader":
    from torch.utils.data import DataLoader

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collator)


@torch.no_grad()
def _model_logps(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    labels = batch.pop("labels")
    logits = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
    sum_logps, valid_length = get_batch_logps(logits, labels, label_pad_token_id=IGNORE_INDEX)
    return sum_logps / valid_length.clamp(min=1)


def main() -> None:
    args = parse_args()
    tokenizer_module, train_dataset, collator, ref_model_args = _build_inference_pipeline(args)

    weak_ref_args = ModelArguments.copyfrom(ref_model_args, model_name_or_path=args.weak_ref_model)
    weak_aln_args = ModelArguments.copyfrom(ref_model_args, model_name_or_path=args.weak_aligned_model)
    fa = FinetuningArguments()  # default; not used for training
    weak_ref_model = load_model(
        load_tokenizer(weak_ref_args)["tokenizer"], weak_ref_args, fa, is_trainable=False, add_valuehead=False
    )
    weak_aligned_model = load_model(
        load_tokenizer(weak_aln_args)["tokenizer"], weak_aln_args, fa, is_trainable=False, add_valuehead=False
    )
    weak_ref_model.eval()
    weak_aligned_model.eval()

    loader = _make_loader(train_dataset, collator, args.batch_size)

    rows: list[dict[str, float]] = []
    for batch in tqdm(loader, desc="caching weak logps"):
        sample_id_tensor = batch.get("sample_id")
        if sample_id_tensor is None:
            raise RuntimeError(
                "Pairwise batch is missing `sample_id`; rebuild your dataset cache so "
                "`_get_preprocessed_dataset` re-injects per-row IDs."
            )

        sample_ids = sample_id_tensor.tolist()
        pos = {k: v for k, v in batch.items() if not k.startswith("neg_") and k != "sample_id"}
        neg = _build_neg_batch({k: v for k, v in batch.items() if k != "sample_id"})

        bsz = pos["input_ids"].size(0) // 2

        wa_pos = _model_logps(weak_aligned_model, dict(pos))
        wr_pos = _model_logps(weak_ref_model, dict(pos))
        wa_neg = _model_logps(weak_aligned_model, dict(neg))
        wr_neg = _model_logps(weak_ref_model, dict(neg))

        wa_yw_mw, wa_yl_mw = wa_pos[:bsz], wa_pos[bsz:]
        wr_yw_mw, wr_yl_mw = wr_pos[:bsz], wr_pos[bsz:]
        wa_yw_ml = wa_neg[:bsz]
        wr_yw_ml = wr_neg[:bsz]

        for i in range(bsz):
            rows.append(
                {
                    "sample_id": int(sample_ids[i]),
                    "weak_aligned_lp_yw_mw": float(wa_yw_mw[i]),
                    "weak_ref_lp_yw_mw": float(wr_yw_mw[i]),
                    "weak_aligned_lp_yw_ml": float(wa_yw_ml[i]),
                    "weak_ref_lp_yw_ml": float(wr_yw_ml[i]),
                    "weak_aligned_lp_yl_mw": float(wa_yl_mw[i]),
                    "weak_ref_lp_yl_mw": float(wr_yl_mw[i]),
                }
            )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}.")


if __name__ == "__main__":
    main()
