<div align="center">
<h1>m-WSPO: Multimodal Weak-to-Strong Preference Optimization</h1>

Farhan Ishmam, &lt;Hossain&gt;, &lt;Fahim&gt;

[<a href="./agent.md">Spec</a>] | [<a href="./LLaMA-Factory/examples/train_m_wspo/README.md">m-WSPO Recipes</a>] | [<a href="./wspo-ref/README.md">Upstream WSPO (text-only)</a>]

</div>

---

## Introduction

**m-WSPO** extends Weak-to-Strong Preference Optimization (WSPO; Zhu et al.,
ICLR 2025) to the multimodal setting. Given a *strong* multimodal LLM
`pi^M` (e.g. LLaVA-Next 13B, Qwen2-VL 7B) and a frozen *weak pair*
`(pi_ref^w, pi_r^w)` (e.g. LLaVA-1.5 7B before / after vanilla DPO),
m-WSPO trains the strong model with a single objective that combines
three signals:

```
L_m-WSPO = L_WSPO  +  alpha * L_img_WSPO  +  beta_dpo * L_DPO^m

L_WSPO       : match the strong logp-shift to the weak pair's shift (Eq. 6).
L_img_WSPO   : cross-modal contrastive penalty on mismatched images (Eqs. 8 + 9).
L_DPO^m      : multimodal DPO conditioned on (image, prompt) (Eq. 10).
```

See [`agent.md`](./agent.md) for the full derivation and design notes.

## What's in this repo

| Path                                                     | Role                                                                                |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| [`agent.md`](./agent.md)                                 | Algorithmic spec — equations, data schema, model loading, ablations.                |
| [`LLaMA-Factory/`](./LLaMA-Factory)                      | **Implementation.** A fork of LLaMA-Factory with the new `m_wspo` training stage.   |
| [`LLaMA-Factory/src/llamafactory/train/m_wspo/`](./LLaMA-Factory/src/llamafactory/train/m_wspo) | Loss math, trainer, workflow.                                                       |
| [`LLaMA-Factory/examples/train_m_wspo/`](./LLaMA-Factory/examples/train_m_wspo)                | Training YAMLs (main configs, ablations, prep DPO, README).                         |
| [`LLaMA-Factory/scripts/m_wspo/`](./LLaMA-Factory/scripts/m_wspo)                              | Offline weak-logprob caching script.                                                |
| [`LLaMA-Factory/tests/m_wspo/`](./LLaMA-Factory/tests/m_wspo)                                  | Unit tests for the loss math and cache plumbing.                                    |
| [`wspo-ref/`](./wspo-ref)                                | Upstream text-only WSPO reference. Used to validate the ported math.                |

## Installation

m-WSPO is built on LLaMA-Factory, so the install steps mirror upstream.

```bash
conda create -n mwspo python=3.10 -y
conda activate mwspo

cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# multimodal extras (required for the MLLMs we target)
pip install qwen-vl-utils accelerate>=0.34
```

> Tested on 4×H100 (80 GB) for LLaVA-Next 13B + LLaVA-1.5 7B.
> Smaller pairs (e.g. Qwen2-VL 7B + Qwen2-VL 2B) fit on 2×A100 (40 GB).

## End-to-end recipe

All commands below assume `cd LLaMA-Factory`.

### 1. Prepare your dataset

m-WSPO consumes a sharegpt-formatted preference set with images. The
default registration is `rlhf_v_mwspo` in `data/dataset_info.json`.
Drop a JSON file at `data/rlhf_v_mwspo.json` with rows of the shape:

```json
{
  "conversations": [
    {"from": "human", "value": "<image>\nDescribe what is in the photo."}
  ],
  "chosen":   {"from": "gpt", "value": "preferred response"},
  "rejected": {"from": "gpt", "value": "dispreferred response"},
  "images":   ["images/0001.jpg"],
  "images_neg": ["images/0001_negative.jpg"]
}
```

`images_neg` is **optional** — only consumed when
`mwspo_neg_image_strategy: pre_mined`. Without it, the collator does
in-batch image rotation to construct `m_l`.

The full RLHF-V dataset (28 k pairs) is available at
`llamafactory/RLHF-V` on the HF Hub.

### 2. Train the DPO-aligned weak model `pi_r^w`

m-WSPO needs a *DPO-aligned* weak checkpoint. We ship two one-shot prep
configs:

```bash
# LLaVA recipe
llamafactory-cli train examples/train_m_wspo/prep_llava_1_5_7b_dpo.yaml

# Qwen recipe
llamafactory-cli train examples/train_m_wspo/prep_qwen2_vl_2b_dpo.yaml
```

Either merge the resulting LoRA back into the base, or pass it through
`mwspo_weak_aligned_adapters:` in step 4.

### 3. (optional) Cache the weak log-probabilities

Both weak models are frozen, so their per-sample mean log-probabilities
are constant across training. Pre-computing them once eliminates ~50 %
of the per-step compute:

```bash
python scripts/m_wspo/cache_weak_logprobs.py \
    --dataset rlhf_v_mwspo \
    --weak_ref_model     llava-hf/llava-1.5-7b-hf \
    --weak_aligned_model saves/llava_1_5_7b/dpo_rlhfv \
    --template vicuna \
    --output cache/rlhf_v_mwspo_weak_lp.parquet
```

Then in your m-WSPO YAML:

```yaml
mwspo_cache_weak_logprobs: true
mwspo_weak_logprob_cache:  cache/rlhf_v_mwspo_weak_lp.parquet
# (optional) drop the weak models entirely once the cache is full:
# mwspo_weak_ref_model:     null
# mwspo_weak_aligned_model: null
```

### 4. Train m-WSPO

```bash
# LLaVA-Next 13B (strong) + LLaVA-1.5 7B pair (weak)
llamafactory-cli train examples/train_m_wspo/llava_next_13b_m_wspo.yaml

# Qwen2-VL 7B (strong) + Qwen2-VL 2B pair (weak)
llamafactory-cli train examples/train_m_wspo/qwen2_vl_7b_m_wspo.yaml
```

For ablations, see
[`examples/train_m_wspo/`](./LLaMA-Factory/examples/train_m_wspo) — there
are four pre-built configs covering `L_WSPO` only, `L_WSPO + L_img_WSPO`,
full m-WSPO, and text-only DPO (image-masked).

## Hyperparameters at a glance

| YAML key                  | Default | Description                                                  |
| ------------------------- | ------- | ------------------------------------------------------------ |
| `mwspo_gamma`             | `0.1`   | Weak-to-strong regularization in Eqs. 6 / 8 / 9.              |
| `mwspo_alpha`             | `0.5`   | Weight on `L_img_WSPO` in Eq. 11.                            |
| `mwspo_beta_dpo`          | `1.0`   | Weight on `L_DPO^m` in Eq. 11.                               |
| `mwspo_dpo_temperature`   | `0.1`   | DPO temperature inside `sigma(.)` (Eq. 10).                  |
| `mwspo_neg_image_strategy`| `in_batch` | `in_batch` rotates images within the batch; `pre_mined` uses `images_neg`. |
| `mwspo_disable_*`         | `false` | Per-component disable flags for ablations.                   |

The full set is documented in
[`examples/train_m_wspo/README.md`](./LLaMA-Factory/examples/train_m_wspo/README.md).

## Tests

CPU-only smoke tests cover the loss math and the cache lookup:

```bash
cd LLaMA-Factory
pytest tests/m_wspo -q
```

The integration-level checklist (real dataset + GPU) lives in
[`agent.md`](./agent.md) §12: reproduce vanilla multimodal DPO, reproduce
text-only WSPO against [`wspo-ref/`](./wspo-ref), and the 8-sample
overfit test.

## Acknowledgments

- m-WSPO is implemented on top of
  [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
- The text-only WSPO reference (under [`wspo-ref/`](./wspo-ref)) is from
  [Zhu et al., ICLR 2025](https://openreview.net/forum?id=f7KxfUrRSb).

## Citation

If you use this work, please cite both m-WSPO and upstream WSPO:

```
@misc{ishmam2026mwspo,
  title  = {m-WSPO: Multimodal Weak-to-Strong Preference Optimization},
  author = {Ishmam, Farhan and Hossain and Fahim},
  year   = {2026},
  note   = {In preparation}
}

@inproceedings{zhu2025weaktostrong,
  title     = {Weak-to-Strong Preference Optimization: Stealing Reward from Weak Aligned Model},
  author    = {Zhu, Wenhong and He, Zhiwei and Wang, Xiaofeng and Liu, Pengfei and Wang, Rui},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025},
  url       = {https://openreview.net/forum?id=f7KxfUrRSb}
}
```
