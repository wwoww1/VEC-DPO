#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

EXP_NAME=$1
MODEL_PATH=$2
VISION_TOWER_PATH=$3
DATA_PATH=$4
GPU_IDS=$5
LEARNING_RATE=$6
DPO_BETA=$7
EVIDENCE_ALPHA=$8

if [ -z "$EXP_NAME" ]; then
  EXP_NAME="vec_dpo_llava15_7b"
fi

if [ -z "$MODEL_PATH" ]; then
  echo "MODEL_PATH is required."
  exit 1
fi

if [ -z "$VISION_TOWER_PATH" ]; then
  echo "VISION_TOWER_PATH is required."
  exit 1
fi

if [ -z "$DATA_PATH" ]; then
  echo "DATA_PATH is required."
  exit 1
fi

if [ -z "$GPU_IDS" ]; then
  GPU_IDS="0"
fi

if [ -z "$LEARNING_RATE" ]; then
  LEARNING_RATE="5e-6"
fi

if [ -z "$DPO_BETA" ]; then
  DPO_BETA="0.1"
fi

if [ -z "$EVIDENCE_ALPHA" ]; then
  EVIDENCE_ALPHA="0.5"
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=false

NUM_GPUS=$(echo "${GPU_IDS}" | awk -F',' '{print NF}')

OUTPUT_DIR="checkpoints/${EXP_NAME}"

mkdir -p "${OUTPUT_DIR}"

echo "========== VEC-DPO Training =========="
echo "EXP_NAME          : ${EXP_NAME}"
echo "MODEL_PATH        : ${MODEL_PATH}"
echo "VISION_TOWER_PATH : ${VISION_TOWER_PATH}"
echo "DATA_PATH         : ${DATA_PATH}"
echo "GPU_IDS           : ${GPU_IDS}"
echo "NUM_GPUS          : ${NUM_GPUS}"
echo "LEARNING_RATE     : ${LEARNING_RATE}"
echo "DPO_BETA          : ${DPO_BETA}"
echo "EVIDENCE_ALPHA    : ${EVIDENCE_ALPHA}"
echo "OUTPUT_DIR        : ${OUTPUT_DIR}"
echo "======================================"

torchrun \
  --nnodes=1 \
  --nproc_per_node=${NUM_GPUS} \
  --master_port=29600 \
  --module muffin.train.train_vec_dpo \
  --deepspeed script/zero3_offload.json \
  --model_name_or_path "${MODEL_PATH}" \
  --version llava_v1 \
  --data_path "${DATA_PATH}" \
  --image_folder "" \
  --vision_tower "${VISION_TOWER_PATH}" \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_vision_select_feature patch \
  --image_aspect_ratio pad \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 2 \
  --learning_rate ${LEARNING_RATE} \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to tensorboard \
  --dpo_beta ${DPO_BETA} \
  --evidence_alpha ${EVIDENCE_ALPHA} \
  --use_evidence_weight True \
  --normalize_weighted_loss False \
  --min_evidence_weight 0.5 \
  --max_evidence_weight 2.0 \
  --loss_type sigmoid
