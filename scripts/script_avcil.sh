#!/bin/bash
set -e

GPU=${1:-0}
DATASET=${2:-VGGSound_balance}
SCENARIO=${3:-full}      # smoke | full | reverse
SEED=${4:-0}
RESULTS_DIR=${5:-"../results"}

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && cd .. && pwd )"
SRC_DIR="$PROJECT_DIR/src"

echo "GPU: $GPU"
echo "DATASET: $DATASET"
echo "SCENARIO: $SCENARIO"
echo "SEED: $SEED"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "SRC_DIR: $SRC_DIR"
echo "RESULTS_DIR: $RESULTS_DIR"

# default
NUM_TASKS=10
NEPOCHS=200
BATCH_SIZE=128
NUM_WORKERS=0
LR=1e-3
WD=1e-4
SCHEDULE_STEP="100"
MEMORY=1500
EXP_NAME="avcil_${DATASET}_${SCENARIO}_s${SEED}"

# scenario switch
if [ "$SCENARIO" = "smoke" ]; then
  NUM_TASKS=3
  NEPOCHS=2
  BATCH_SIZE=32
  NUM_WORKERS=0
  MEMORY=200
  EXP_NAME="smoke_${DATASET}_s${SEED}"
elif [ "$SCENARIO" = "full" ]; then
  # keep defaults
  :
elif [ "$SCENARIO" = "reverse" ]; then
  # 你可绑定到特定 reverse 数据集配置
  # DATASET=VGGSound_reverse (前提是你在 dataset_config 里注册过)
  NUM_TASKS=10
  NEPOCHS=200
  MEMORY=500
  EXP_NAME="reverse_${DATASET}_s${SEED}"
else
  echo "Unknown scenario: $SCENARIO"
  exit 1
fi

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$GPU \
python3 -u $SRC_DIR/main_incremental.py \
  --exp-name $EXP_NAME \
  --results-path $RESULTS_DIR \
  --datasets $DATASET \
  --approach avcil \
  --network avcil_net \
  --seed $SEED \
  --gpu 0 \
  --num-tasks $NUM_TASKS \
  --nepochs $NEPOCHS \
  --batch-size $BATCH_SIZE \
  --num-workers $NUM_WORKERS \
  --lr $LR \
  --weight-decay $WD \
  --schedule_step $SCHEDULE_STEP \
  --num-exemplars $MEMORY \
  --instance-contrastive \
  --class-contrastive \
  --attn-score-distil \
  --instance-contrastive-temperature 0.05 \
  --class-contrastive-temperature 0.05 \
  --lam 0.5 \
  --lam-I 0.1 \
  --lam-C 1.0