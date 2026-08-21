# VEC-DPO

This repository implements **VEC-DPO**, a visual evidence-calibrated preference optimization framework for mitigating hallucination in multimodal large language models.

VEC-DPO is built on top of a LLaVA-style multimodal training pipeline. Instead of relying only on response-level preference labels, VEC-DPO decomposes model responses into fine-grained visual claims, estimates their visual evidence support, constructs evidence-calibrated preference pairs, and reweights the DPO objective according to the evidence gap between chosen and rejected responses.

<p align="center">
  <img src="assets/fig1.png" width="850">
</p>

---

## Overview

Multimodal large language models often generate fluent responses that contain visual hallucinations. Standard preference optimization methods usually compare entire responses, but a single response may contain multiple visual claims with different evidence support.

VEC-DPO introduces a claim-level evidence calibration pipeline:

```text
Candidate Responses
    ↓
Visual Claim Extraction
    ↓
Visual Evidence Verification
    ↓
Evidence-Calibrated Pair Construction
    ↓
VEC-DPO Training
```

## Project Structure

```text
.
├── llava/                  # Multimodal backbone
├── muffin/
│   ├── vec_data/           # Claim extraction, evidence verification, pair construction
│   ├── datasets/           # VEC-DPO dataset and collator
│   ├── losses/             # VEC-DPO loss
│   ├── trainers/           # VEC-DPO trainer
│   └── train/              # Training entry
├── script/train/           # Training launch scripts
├── eval/                   # Evaluation scripts
├── data/                   # Data files
└── run.sh                  # Main launcher
```

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-name>

conda create -n vecdpo python=3.10 -y
conda activate vecdpo

pip install -r requirements.txt

pip install flash-attn --no-build-isolation
pip install peft deepspeed accelerate bitsandbytes
```
## Usage

### Data Preparation

VEC-DPO supports existing multimodal DPO-style preference data. Each training sample should contain an image, a question, a chosen response, and a rejected response.

The final training data should follow this format:

```json
{
  "image": "xxx.jpg",
  "question": "What is shown in the image?",
  "chosen": "A visually grounded response.",
  "rejected": "A hallucinated response.",
  "evidence_gap": 1.2,
  "evidence_weight": 1.6
}
```

Run preprocessing modules with `python -m` from the repository root. For
example, after candidate generation:

```bash
python -m muffin.vec_data.claim_extractor \
  --input data/candidates.json \
  --output data/candidates_with_claims.json \
  --mode heuristic

python -m muffin.vec_data.build_vec_pairs \
  --input data/candidates_with_verified_claims.json \
  --output data/vec_dpo_train.json \
  --alpha 0.5
```

Evidence must be verified before pair construction. The `llm_prompt` modes
only export prompts for an external LLM/VLM; they do not make API calls. Parse
external outputs with `VisualClaimExtractor.parse_llm_claims` and
`VisualEvidenceVerifier.parse_llm_verification_output`, or provide cached
`supported` / `uncertain` / `unsupported` annotations. Pair construction
rejects responses that contain no verified visual claims.

When `evidence_gap` is present, training recomputes
`evidence_weight = 1 + evidence_alpha * evidence_gap` and clamps it to the
configured range. This keeps the command-line `evidence_alpha` authoritative.

### Training

After replacing the placeholder paths in `run.sh`, run VEC-DPO training with:

```bash
bash run.sh
```

The default configuration in `run.sh` is:

```bash
bash script/train/llava15_train_vec_dpo.sh \
    VEC_DPO_llava15_7b \
    "[Path of your LLaVA model]" \
    "[Path of your vision tower model]" \
    data/vec_dpo_train.json \
    0,1,2,3 \
    5e-6 \
    0.1 \
    0.5
```

Arguments:

```text
1. experiment name
2. base LLaVA model path
3. vision tower path
4. VEC-DPO training data path
5. GPU ids
6. learning rate
7. DPO beta
8. evidence alpha
```

### Evaluation

After training, evaluate the checkpoint using the scripts under `script/eval/`.

Some benchmarks require LLM-based assessment, such as DeepSeek-V3, GPT-3.5, or GPT-4.

#### HallusionBench

Download the HallusionBench questions, annotations, and figures, then run:

```bash
bash script/eval/eval_hallusion.sh [ckpt_path] [base_path or "No"] [YOUR_DEEPSEEK_API_KEY] [GPU_ID]
```

#### Object-HalBench

Download COCO annotations and install supplementary packages:

```python
import nltk
nltk.download('wordnet')
nltk.download('punkt')
```

```bash
python -m spacy download en_core_web_trf
```

Then run:

```bash
bash script/eval/eval_objhal.sh [ckpt_path] [base_path or "No"] [YOUR_OPENAI_API_KEY] [GPU_ID]
```

#### MMHal-Bench

Download MMHal-Bench data, then run:

```bash
bash script/eval/eval_mmhal.sh [ckpt_path] [base_path or "No"] [YOUR_OPENAI_API_KEY] [GPU_ID]
```

#### AMBER

Download AMBER data and images, and install the supplementary model:

```bash
python -m spacy download en_core_web_lg
```

Then run:

```bash
bash script/eval/eval_amber.sh [ckpt_path] [base_path or "No"] [GPU_ID] [data_dir]
```

#### MMStar

Download MMStar data, then run:

```bash
bash script/eval/eval_mmstar.sh [ckpt_path] [base_path or "No"] [GPU_ID] [data_dir]
```
