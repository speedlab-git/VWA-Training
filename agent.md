> **Target paper:** *m-WSPO: Multimodal Weak-to-Strong Preference Optimization* (Ishmam, Hossain, Fahim).
> **Base repository:** **[hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — used for multimodal infra (vision towers, processors, DPO stage, dataset registry, DeepSpeed/LoRA).
> **Reference implementation:** **[zwhong714/weak-to-strong-preference-optimization](https://github.com/zwhong714/weak-to-strong-preference-optimization)** — port the WSPO loss math from here; do **not** use it as the base.
>
> Goal: Add a new `m_wspo` training stage to LLaMA-Factory that combines (1) Weak-to-Strong Transfer, (2) Vision-oriented WSPO, and (3) multimodal DPO into the unified Eq. 11.

---

## 1. High-Level Objective

```
L_m-WSPO  =  L_WSPO  +  alpha * L_img_WSPO  +  beta_dpo * L_DPO^m
L_img_WSPO = L_img_WSPO^winning + L_img_WSPO^losing
```

Three mechanisms:

1. **Weak-to-Strong Transfer** — match the strong MLLM\'s log-prob shift to the weak pair\'s shift on `(x, m_w, y_w)`.
2. **Vision-oriented WSPO** — cross-modal contrastive penalty using mismatched image `m_l` (winning term) and losing response `y_l` (losing term).
3. **Self-Preference Optimization** — multimodal DPO conditioned on `(m, x)` over `(y_w, y_l)`.

> **Symbol disambiguation.** The paper overloads `beta`: it is both the DPO temperature inside `sigma(.)` (Eq. 10) **and** the weight on `L_DPO^m` in unified Eq. 11. We split them into `dpo_temperature` (Eq. 10) and `beta_dpo` (Eq. 11 weight).

---

## 2. Why LLaMA-Factory as the Base

m-WSPO is ~75% multimodal plumbing + ~25% novel loss. LLaMA-Factory already provides:

- First-class MLLMs: LLaVA-1.5/1.6, Qwen2-VL, InternVL2, MiniCPM-V, Pixtral, Llama-3.2-Vision.
- A multimodal DPO stage (`stage: dpo`) with image conditioning, chat templates, vision-token masking, and a frozen reference model.
- Dataset registry for `(prompt, chosen, rejected, images)`.
- LoRA / QLoRA, DeepSpeed Zero-2/3, FSDP, BF16 — one-flag toggles.

The original WSPO repo is text-only; we port only its **loss math** (~150 LOC).

---

## 3. Repository Setup

### 3.1 Clone & branch
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout -b m-wspo

# Pull the WSPO repo elsewhere as a reference
git clone https://github.com/zwhong714/weak-to-strong-preference-optimization.git ../wspo-ref
```

### 3.2 Files added / modified inside LLaMA-Factory

```
LLaMA-Factory/
├── src/llamafactory/
│   ├── hparams/
│   │   └── finetuning_args.py          # MODIFY: add MWSPOArguments dataclass
│   ├── data/
│   │   ├── parser.py                   # MODIFY: register "m_wspo" stage uses pairwise+image schema
│   │   ├── collator.py                 # MODIFY: also yield m_l (mismatched image) per sample
│   │   └── template.py                 # (unchanged; reuse multimodal templates)
│   ├── train/
│   │   ├── m_wspo/                     # NEW package (mirrors structure of train/dpo/)
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py              # MWSPOTrainer  (subclasses CustomDPOTrainer)
│   │   │   ├── workflow.py             # run_m_wspo(...)
│   │   │   └── losses.py               # ported WSPO + vision-oriented losses
│   │   └── tuner.py                    # MODIFY: route stage=="m_wspo" -> run_m_wspo
│   └── extras/constants.py             # MODIFY: add "m_wspo" to STAGES
├── examples/train_m_wspo/
│   ├── llava_next_13b_m_wspo.yaml      # NEW: end-to-end training config
│   └── qwen2_vl_7b_m_wspo.yaml         # NEW
├── data/
│   └── dataset_info.json               # MODIFY: register multimodal preference dataset
└── AGENT.md                            # this file
```

### 3.3 Extra dependencies
LLaMA-Factory already pulls everything needed; nothing else to install.

---

## 4. Register the New Stage

### 4.1 `src/llamafactory/extras/constants.py`
```python
STAGES = ("pt", "sft", "rm", "ppo", "dpo", "kto", "m_wspo")
```

### 4.2 `src/llamafactory/hparams/finetuning_args.py`
Add a dataclass and merge it into `FinetuningArguments`:
```python
@dataclass
class MWSPOArguments:
    gamma:           float = 0.1   # WSPO regularization (Eqs. 6, 8, 9)
    alpha:           float = 0.5   # weight on L_img_WSPO  (Eq. 11)
    beta_dpo:        float = 1.0   # weight on L_DPO^m     (Eq. 11)
    dpo_temperature: float = 0.1   # beta inside sigma(.)  (Eq. 10)

    weak_ref_model:     Optional[str] = None  # path/HF id of pi_ref^w
    weak_aligned_model: Optional[str] = None  # path/HF id of pi_r^w   (DPO-aligned weak)

    # Engineering
    cache_weak_logprobs: bool = True
    weak_logprob_cache:  Optional[str] = None  # parquet path
```

### 4.3 `src/llamafactory/train/tuner.py`
```python
elif finetuning_args.stage == "m_wspo":
    from .m_wspo import run_m_wspo
    run_m_wspo(model_args, data_args, training_args, finetuning_args, callbacks)
```

---

## 5. Data Schema

m-WSPO needs **five** fields per example: `(x, m_w, m_l, y_w, y_l)`. LLaMA-Factory\'s pairwise multimodal schema already provides `(x, m_w, y_w, y_l)`. We extend it to also yield `m_l`.

### 5.1 `data/dataset_info.json` entry (example: RLHF-V style)
```json
"rlhf_v_mwspo": {
  "file_name": "rlhf_v_mwspo.json",
  "ranking": true,
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "chosen":   "chosen",
    "rejected": "rejected",
    "images":   "images"
  },
  "tags": {
    "role_tag": "from", "content_tag": "value",
    "user_tag": "human", "assistant_tag": "gpt"
  }
}
```

### 5.2 Mismatched image `m_l`
Two strategies inside `data/collator.py` (extend `PairwiseDataCollatorWithPadding`):

1. **In-batch sampling (default).** For each sample `i`, set `m_l := images[(i + 1) % B]`. Cheap, no preprocessing.
2. **Pre-mined hard negatives.** Add an optional `"images_neg"` column to the dataset and load it directly when present. Use this for harder visual contrast (e.g., same scene, different object).

The collator must return a dict containing both `pixel_values` (for `m_w`) and `pixel_values_neg` (for `m_l`), both already passed through the model\'s image processor.

---

## 6. Model Loading: Four Policies

In `train/m_wspo/workflow.py`, load all four policies. LLaMA-Factory\'s `load_model` is reused; only `pi_theta^s` carries gradients and LoRA adapters.

```python
def run_m_wspo(model_args, data_args, training_args, finetuning_args, callbacks):
    tokenizer_module = load_tokenizer(model_args)
    processor = tokenizer_module["processor"]

    # Strong policy (trainable, LoRA on top)
    strong_pol = load_model(tokenizer_module["tokenizer"], model_args,
                            finetuning_args, is_trainable=True,
                            add_valuehead=False)

    # Strong reference (frozen copy)
    strong_ref = load_reference_model(model_args, finetuning_args)  # eval(), no grad

    # Weak pair (frozen) -- override model_name_or_path
    weak_ref = load_frozen_mllm(finetuning_args.weak_ref_model,     processor)
    weak_aln = load_frozen_mllm(finetuning_args.weak_aligned_model, processor)

    dataset_module = get_dataset(model_args, data_args, training_args,
                                 stage="m_wspo", **tokenizer_module)

    trainer = MWSPOTrainer(
        model=strong_pol,
        ref_model=strong_ref,
        weak_ref_model=weak_ref,
        weak_aligned_model=weak_aln,
        args=training_args,
        finetuning_args=finetuning_args,
        processor=processor,
        callbacks=callbacks,
        **dataset_module,
    )
    trainer.train()
    trainer.save_model()
```

`load_frozen_mllm` is a thin helper that reuses `load_model` with `is_trainable=False`, sets `requires_grad=False` on every parameter, and calls `.eval()`. Wrap with DeepSpeed Zero-3 partitioning if memory is tight.

---

## 7. Log-Prob Helpers

In `train/m_wspo/losses.py`, define **two** helpers — the distinction matters because Eqs. 6, 8, 9 carry an explicit `1/|y|` normalizer but Eq. 10 does not:

```python
def _per_token_logp(model, processor, x, m, y_ids):
    # Returns Tensor[B, T_y] -- per-token log-probs of y given (x, m).
    # Image tokens and prompt tokens are masked to 0 (then dropped via mask).
    ...

def mean_lp(model, processor, x, m, y_ids):           # Eqs. 6, 8, 9
    lp = _per_token_logp(model, processor, x, m, y_ids)
    mask = (y_ids != -100).float()
    return (lp * mask).sum(-1) / mask.sum(-1).clamp(min=1)

def sum_lp(model, processor, x, m, y_ids):            # Eq. 10 (standard DPO form)
    lp = _per_token_logp(model, processor, x, m, y_ids)
    mask = (y_ids != -100).float()
    return (lp * mask).sum(-1)
```

Reuse LLaMA-Factory\'s existing `get_batch_logps` from `train/dpo/trainer.py` as the starting point — it already handles image-token masking for the supported MLLMs.

---

## 8. Loss Implementations (`train/m_wspo/losses.py`)

> **Convention.** Eqs. 6, 8, 9 use `mean_lp`. Eq. 10 uses `sum_lp`. Mixing these silently rescales `dpo_temperature` and breaks reproducibility.

### 8.1 Weak-to-Strong Transfer — Eq. 6
```python
def loss_wspo_mm(strong_pol, strong_ref, weak_aln, weak_ref,
                 batch, gamma, processor):
    x, m_w, y_w = batch["prompt"], batch["pixel_values"], batch["chosen_ids"]
    s = gamma * (mean_lp(strong_pol, processor, x, m_w, y_w)
               - mean_lp(strong_ref, processor, x, m_w, y_w))
    w =          mean_lp(weak_aln,   processor, x, m_w, y_w) \\
               - mean_lp(weak_ref,   processor, x, m_w, y_w)
    return ((s - w) ** 2).mean()
```

### 8.2 Vision-oriented WSPO — Eqs. 7-9
```python
def loss_img_wspo_winning(strong_pol, strong_ref, weak_aln, weak_ref,
                          batch, gamma, processor):                   # Eq. 8
    x, m_l, y_w = batch["prompt"], batch["pixel_values_neg"], batch["chosen_ids"]
    w =          mean_lp(weak_aln,   processor, x, m_l, y_w) \\
               - mean_lp(weak_ref,   processor, x, m_l, y_w)
    s = gamma * (mean_lp(strong_pol, processor, x, m_l, y_w)
               - mean_lp(strong_ref, processor, x, m_l, y_w))
    return ((w - s) ** 2).mean()

def loss_img_wspo_losing(strong_pol, strong_ref, weak_aln, weak_ref,
                         batch, gamma, processor):                    # Eq. 9
    x, m_w, y_l = batch["prompt"], batch["pixel_values"], batch["rejected_ids"]
    w =          mean_lp(weak_aln,   processor, x, m_w, y_l) \\
               - mean_lp(weak_ref,   processor, x, m_w, y_l)
    s = gamma * (mean_lp(strong_pol, processor, x, m_w, y_l)
               - mean_lp(strong_ref, processor, x, m_w, y_l))
    return ((w - s) ** 2).mean()
```

### 8.3 Multimodal DPO — Eq. 10 (uses `sum_lp`, NOT length-normalized)
```python
def loss_dpo_mm(strong_pol, strong_ref, batch, dpo_temperature, processor):
    x, m, y_w, y_l = (batch["prompt"], batch["pixel_values"],
                      batch["chosen_ids"], batch["rejected_ids"])
    pi_w = sum_lp(strong_pol, processor, x, m, y_w) - sum_lp(strong_ref, processor, x, m, y_w)
    pi_l = sum_lp(strong_pol, processor, x, m, y_l) - sum_lp(strong_ref, processor, x, m, y_l)
    return -F.logsigmoid(dpo_temperature * (pi_w - pi_l)).mean()
```

### 8.4 Unified objective — Eq. 11
```python
def m_wspo_total_loss(models, batch, hp, processor):
    L_wspo  = loss_wspo_mm(*models, batch, hp.gamma, processor)
    L_imgW  = loss_img_wspo_winning(*models, batch, hp.gamma, processor)
    L_imgL  = loss_img_wspo_losing (*models, batch, hp.gamma, processor)
    L_dpo   = loss_dpo_mm(models[0], models[1], batch, hp.dpo_temperature, processor)
    total   = L_wspo + hp.alpha * (L_imgW + L_imgL) + hp.beta_dpo * L_dpo
    return total, {"L_wspo": L_wspo, "L_img_w": L_imgW,
                   "L_img_l": L_imgL, "L_dpo": L_dpo}
```

---

## 9. Trainer (`train/m_wspo/trainer.py`)

Subclass LLaMA-Factory\'s `CustomDPOTrainer` so we inherit its multimodal batch handling, ref-model wrapping, and DeepSpeed integration:

```python
from ..dpo.trainer import CustomDPOTrainer

class MWSPOTrainer(CustomDPOTrainer):
    def __init__(self, *, weak_ref_model, weak_aligned_model, processor,
                 finetuning_args, **kw):
        super().__init__(finetuning_args=finetuning_args, **kw)
        self.weak_ref = self._prepare_frozen(weak_ref_model)
        self.weak_aln = self._prepare_frozen(weak_aligned_model)
        self.processor = processor
        self.hp = finetuning_args   # carries gamma, alpha, beta_dpo, dpo_temperature

    def _prepare_frozen(self, model):
        if model is None: return None
        for p in model.parameters(): p.requires_grad_(False)
        model.eval()
        if self.is_deepspeed_enabled:
            model = self._prepare_deepspeed(model)   # reuse parent helper
        else:
            model = self.accelerator.prepare_model(model, evaluation_mode=True)
        return model

    def compute_loss(self, model, inputs, return_outputs=False):
        models = (model, self.ref_model, self.weak_aln, self.weak_ref)
        loss, metrics = m_wspo_total_loss(models, inputs, self.hp, self.processor)
        self.log(metrics)
        return (loss, None) if return_outputs else loss
```

Engineering notes:
- All three frozen models live on the same accelerator; with DeepSpeed Zero-3 they are sharded across ranks.
- For LLaVA-Next-13B + LLaVA-1.5-7B weak pair, expect ~80 GB peak with bf16 + LoRA on 2x A100-80G via Zero-3 + activation checkpointing.
- **Cache weak log-probs offline** (`--cache_weak_logprobs true`): a one-time pass writes per-example `mean_lp(weak_aln/weak_ref)` for the three `(image, response)` combos to parquet keyed by sample id. Trainer reads from cache and skips the two weak forward passes — typically a 2-3x speedup.

---

## 10. Example Config — `examples/train_m_wspo/llava_next_13b_m_wspo.yaml`

```yaml
# ---- model ----
model_name_or_path: llava-hf/llava-v1.6-vicuna-13b-hf
trust_remote_code: true

# ---- method ----
stage: m_wspo
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 64
lora_alpha: 128

# ---- m-wspo specific ----
gamma:           0.1
alpha:           0.5
beta_dpo:        1.0
dpo_temperature: 0.1
weak_ref_model:     llava-hf/llava-1.5-7b-hf
weak_aligned_model: <your-org>/llava-1.5-7b-dpo-rlhfv
cache_weak_logprobs: true
weak_logprob_cache:  cache/rlhf_v_mwspo_weak_lp.parquet

# ---- dataset ----
dataset: rlhf_v_mwspo
template: llava_next
cutoff_len: 2048
max_samples: 100000
preprocessing_num_workers: 16

# ---- output ----
output_dir: saves/llava_next_13b/m_wspo
overwrite_output_dir: true

# ---- optimization ----
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 5.0e-7
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
ddp_timeout: 180000000
deepspeed: examples/deepspeed/ds_z3_config.json
gradient_checkpointing: true

# ---- logging / eval ----
logging_steps: 10
save_steps: 500
report_to: wandb
```

Launch:
```bash
llamafactory-cli train examples/train_m_wspo/llava_next_13b_m_wspo.yaml
```

---

## 11. Evaluation Protocol

After each epoch, merge LoRA into `pi_theta^s` and run:

- **Hallucination:** POPE, MMHal-Bench, Object HalBench
- **General VQA:** MME, MM-Vet, LLaVA-Bench (in-the-wild)
- **Preference win-rate** vs. `pi_ref^s` judged by GPT-4o on a 200-prompt held-out set

LLaMA-Factory ships an evaluation CLI; pair with VLMEvalKit for the standard MLLM benchmarks.

### Required ablations (mirror paper\'s three mechanisms)
1. `L_WSPO` only (`alpha=0`, `beta_dpo=0`).
2. `L_WSPO + L_img_WSPO` (`beta_dpo=0`).
3. Full `L_m-WSPO` (default).
4. Replace `L_DPO^m` with vanilla DPO on text-only (mask image) — sanity check that visual conditioning matters.

---

## 12. Smoke Tests Before the Real Run

1. **Reproduce vanilla multimodal DPO** in LLaMA-Factory on the same dataset → establishes the upper-bound baseline.
2. **Reproduce text-only WSPO** using the reference repo on a small text preference set → validates the loss math you ported.
3. **Unit-test `loss_wspo_mm`** with `gamma=0` and `weak_aln == weak_ref` → loss must equal `(gamma*(s_pol - s_ref))^2 = 0`. Catches sign/order bugs.
4. **Unit-test `loss_dpo_mm`** with `strong_pol == strong_ref` → loss must equal `-log sigma(0) = log 2`. Catches `sum_lp` vs `mean_lp` confusion.
5. **One-step overfit test**: train on 8 samples for 200 steps; verify `L_wspo`, `L_img_w`, `L_img_l`, `L_dpo` all decrease.

---

## 13. Implementation Checklist

- [ ] Fork LLaMA-Factory; create `m-wspo` branch.
- [ ] Add `m_wspo` to `STAGES` and route in `tuner.py`.
- [ ] Add `MWSPOArguments` dataclass.
- [ ] Extend pairwise multimodal collator to emit `pixel_values_neg` (`m_l`).
- [ ] Implement `mean_lp` / `sum_lp` reusing `get_batch_logps`.
- [ ] Port WSPO losses (Eqs. 6, 8, 9) with `mean_lp`; multimodal DPO (Eq. 10) with `sum_lp`.
- [ ] Subclass `CustomDPOTrainer` -> `MWSPOTrainer`; load weak pair as frozen.
- [ ] Add weak-log-prob offline caching pipeline.
- [ ] Author `llava_next_13b_m_wspo.yaml` and `qwen2_vl_7b_m_wspo.yaml`.
- [ ] Pass all five smoke tests in §12.
- [ ] Train m-WSPO; evaluate on POPE / MMHal / MM-Vet / LLaVA-Bench.
- [ ] Run the four ablations in §11.

---

## 14. References

- Rafailov et al., 2023. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS.
- Zhu et al., 2024. *Weak-to-Strong Preference Optimization: Stealing Reward from Weak Aligned Model.* arXiv:2410.18640.
- LLaMA-Factory: <https://github.com/hiyouga/LLaMA-Factory>
- WSPO reference impl: <https://github.com/zwhong714/weak-to-strong-preference-optimization>