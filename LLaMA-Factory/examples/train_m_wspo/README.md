# m-WSPO: Multimodal Weak-to-Strong Preference Optimization

This directory contains training configs for the **m-WSPO** stage described
in *m-WSPO: Multimodal Weak-to-Strong Preference Optimization* (Ishmam,
Hossain, Fahim). The stage is implemented inside
`src/llamafactory/train/m_wspo/`; see `agent.md` at the repository root for
the design and equation references.

## Unified objective (Eq. 11)

```
L_m-WSPO = L_WSPO + alpha * L_img_WSPO + beta_dpo * L_DPO^m
L_img_WSPO = L_img_WSPO^winning + L_img_WSPO^losing
```

| Hyperparam              | YAML key                | Default | Description                                                  |
| ----------------------- | ----------------------- | ------- | ------------------------------------------------------------ |
| gamma                   | `mwspo_gamma`           | 0.1     | weak-to-strong regularization in Eqs. 6, 8, 9.               |
| alpha                   | `mwspo_alpha`           | 0.5     | weight on L_img_WSPO in Eq. 11.                              |
| beta_dpo                | `mwspo_beta_dpo`        | 1.0     | weight on L_DPO^m in Eq. 11.                                 |
| dpo_temperature         | `mwspo_dpo_temperature` | 0.1     | temperature inside `sigma(.)` in Eq. 10 (NOT the Eq. 11 weight). |

## Configs in this directory

| File                                          | Purpose                                                  |
| --------------------------------------------- | -------------------------------------------------------- |
| `prep_llava_1_5_7b_dpo.yaml`                  | **Run first.** Vanilla DPO on LLaVA-1.5 7B → produces `pi_r^w` for the LLaVA recipe. |
| `prep_qwen2_vl_2b_dpo.yaml`                   | **Run first.** Vanilla DPO on Qwen2-VL 2B → produces `pi_r^w` for the Qwen recipe.  |
| `llava_next_13b_m_wspo.yaml`                  | LLaVA-Next 13B (strong) + LLaVA-1.5 7B pair (weak).      |
| `qwen2_vl_7b_m_wspo.yaml`                     | Qwen2-VL 7B (strong) + Qwen2-VL 2B pair (weak).          |
| `ablations/ablation_1_wspo_only.yaml`         | L_WSPO only (alpha=0, beta_dpo=0).                       |
| `ablations/ablation_2_wspo_plus_img.yaml`     | L_WSPO + L_img_WSPO (beta_dpo=0).                        |
| `ablations/ablation_3_full.yaml`              | Full m-WSPO (mirrors the LLaVA-Next default).            |
| `ablations/ablation_4_text_only_dpo.yaml`     | Replace L_DPO^m with text-only DPO (mask images).        |

### End-to-end recipe

```
1. (one-time) Train pi_r^w via vanilla DPO:
       llamafactory-cli train examples/train_m_wspo/prep_llava_1_5_7b_dpo.yaml
       # (or merge the LoRA back into the base, or pass mwspo_weak_aligned_adapters: <path>)

2. (optional) Pre-compute weak log-probabilities to skip weak forwards at training time:
       python scripts/m_wspo/cache_weak_logprobs.py \
           --dataset rlhf_v_mwspo \
           --weak_ref_model     llava-hf/llava-1.5-7b-hf \
           --weak_aligned_model saves/llava_1_5_7b/dpo_rlhfv \
           --template vicuna \
           --output cache/rlhf_v_mwspo_weak_lp.parquet

3. Train m-WSPO:
       llamafactory-cli train examples/train_m_wspo/llava_next_13b_m_wspo.yaml
```

## Placeholders to fill in before training

Each YAML contains `<YOUR-...>` strings you must replace with concrete
values. The required ones are:

1. **`mwspo_weak_aligned_model`** -- HuggingFace path/id of a *DPO-aligned*
   weak MLLM. Produced by `prep_llava_1_5_7b_dpo.yaml` /
   `prep_qwen2_vl_2b_dpo.yaml`. If you keep the LoRA separate, point this
   at the base model and pass the adapters via
   `mwspo_weak_aligned_adapters: saves/llava_1_5_7b/dpo_rlhfv`.
2. **`dataset: rlhf_v_mwspo`** -- registered in `data/dataset_info.json`.
   Drop a sharegpt-formatted preference file at `data/rlhf_v_mwspo.json`
   with the schema:
   ```json
   {
     "conversations": [
       {"from": "human", "value": "<image>\nDescribe the image."},
       {"from": "gpt", "value": "..."}
     ],
     "chosen":   {"from": "gpt", "value": "preferred response"},
     "rejected": {"from": "gpt", "value": "dispreferred response"},
     "images":   ["images/0001.jpg"],
     "images_neg": ["images/0001_negative.jpg"]
   }
   ```
   `images_neg` is **optional** and only consumed when
   `mwspo_neg_image_strategy: pre_mined`. To enable it in
   `data/dataset_info.json`, add the column mapping:
   ```json
   "rlhf_v_mwspo": {
     ...
     "columns": {
       ...,
       "images_neg": "images_neg"
     }
   }
   ```
   The full RLHF-V dataset (28k pairs) is available at
   `llamafactory/RLHF-V` on HF Hub.

## Launching training

```bash
# Strong base config:
llamafactory-cli train examples/train_m_wspo/llava_next_13b_m_wspo.yaml

# Ablation:
llamafactory-cli train examples/train_m_wspo/ablations/ablation_1_wspo_only.yaml
```

For multi-GPU runs, the configs already point at
`examples/deepspeed/ds_z3_config.json`; launch with `accelerate launch` or
the project's distributed runner.

## How m_l (mismatched image) is selected

Two strategies (`mwspo_neg_image_strategy`):

1. **`in_batch`** (default) -- in the collator, for each sample `i` set
   `m_l := images[(i + 1) % B]`. Cheapest option, and the in-batch images
   already come from the *same* dataset distribution.
2. **`pre_mined`** -- if your dataset rows include an extra `images_neg`
   column (for example, a hard-negative image of the same scene with a
   different object), the collator passes those through instead. Falls
   back to in-batch shifting when the column is missing.

> For models with **variable-resolution image grids** (Qwen2-VL,
> LLaVA-Next), pre-resize all images to the same canonical resolution. The
> in-batch swap re-collates the m_l batch independently, but extreme grid
> mismatches across the batch still hurt training-time efficiency. Set
> `image_max_pixels` in the YAML to enforce a cap.

## Smoke tests (`agent.md` §12)

Run the unit tests for the loss math:

```bash
pytest tests/m_wspo -q
```

The full §12 checklist still includes three integration-level tests that
require a real dataset & GPU:

1. Reproduce vanilla multimodal DPO on the same dataset to establish the
   upper-bound baseline.
2. Reproduce text-only WSPO using the reference repo on a small text
   preference set to validate the loss math you ported.
3. One-step overfit: train on 8 samples for 200 steps; verify that
   `L_wspo`, `L_img_w`, `L_img_l`, and `L_dpo` all decrease.

## Optional: caching weak log-probs

The two frozen weak models contribute ~50% of the per-step compute. Their
mean log-probabilities are deterministic, so you can precompute them once
and point the trainer at the parquet:

```bash
python scripts/m_wspo/cache_weak_logprobs.py \
    --dataset rlhf_v_mwspo \
    --weak_ref_model     llava-hf/llava-1.5-7b-hf \
    --weak_aligned_model saves/llava_1_5_7b/dpo_rlhfv \
    --template vicuna \
    --output cache/rlhf_v_mwspo_weak_lp.parquet
```

Then in the m-WSPO YAML, switch on caching:

```yaml
mwspo_cache_weak_logprobs: true
mwspo_weak_logprob_cache:  cache/rlhf_v_mwspo_weak_lp.parquet
# (optional) drop the weak models entirely once the cache covers the dataset:
# mwspo_weak_ref_model:     null
# mwspo_weak_aligned_model: null
```

The trainer will skip both weak forward passes whenever every `sample_id`
in the batch is present in the cache. Cache misses fall back to live weak
forwards if the weak models are still loaded; otherwise they raise a
clear error.

> **Stable IDs.** `_get_preprocessed_dataset` injects a `_sample_id`
> column on every row before tokenization. The collator surfaces it as
> `sample_id` in the training batch, and the producer above writes it as
> the parquet key. As long as you don't reorder the dataset between cache
> creation and training, the lookup is exact.

## Implementation map

| Spec section (`agent.md`) | File                                                             |
| ------------------------- | ---------------------------------------------------------------- |
| §4.1 STAGES               | `src/llamafactory/extras/constants.py`                           |
| §4.2 Args                 | `src/llamafactory/hparams/finetuning_args.py` (`MWSPOArguments`) |
| §4.3 tuner routing        | `src/llamafactory/train/tuner.py`                                |
| §5    data schema         | `data/dataset_info.json` (`rlhf_v_mwspo`)                        |
| §5.2  m_l collator        | `src/llamafactory/data/collator.py` (`MWSPOPairwiseDataCollator`)|
| §6    workflow            | `src/llamafactory/train/m_wspo/workflow.py`                      |
| §7    log-prob helpers    | `src/llamafactory/train/m_wspo/losses.py` (`mean_lp_from_sum`)   |
| §8    losses              | `src/llamafactory/train/m_wspo/losses.py`                        |
| §9    trainer             | `src/llamafactory/train/m_wspo/trainer.py`                       |
| §10   example YAML        | `examples/train_m_wspo/llava_next_13b_m_wspo.yaml`               |
| §11   ablations           | `examples/train_m_wspo/ablations/`                               |
| §12   smoke tests         | `tests/m_wspo/test_losses.py`                                    |
