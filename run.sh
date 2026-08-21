#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

bash script/train/llava15_train_vec_dpo.sh \
  VEC_DPO_llava15_7b \
  /path/to/llava-v1.5-7b \
  /path/to/clip-vit-large-patch14-336 \
  data/vec_dpo_train.json \
  0,1,2,3 \
  5e-6 \
  0.1 \
  0.5
