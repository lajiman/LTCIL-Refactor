#!/bin/bash
set -e

GPU=${1:-0}
DATASET=${2:-VGGSound_balance}
SCENARIO=${3:-balance}   # smoke | balance | reverse | custom
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

# -----------------------------
# defaults (close to LTCIL)
# -----------------------------
NUM_TASKS=10
NEPOCHS=200
BATCH_SIZE=128
NUM_WORKERS=0
PIN_MEMORY=false
LR=1e-3
WD=1e-4
SCHEDULE_STEP=(50000)
MEMORY=500
EXP_NAME="avcil_${DATASET}_${SCENARIO}_s${SEED}"

# avcil losses (align train_incremental_ours.py)
USE_INSTANCE=true
USE_CLASS=true
USE_ATTN=true
TEMP_I=0.05
TEMP_C=0.05
LAM=0.5
LAM_I=0.1
LAM_C=1.0

# -----------------------------
# scenario switch
# -----------------------------
case "$SCENARIO" in
  smoke)
    NUM_TASKS=3
    NEPOCHS=2
    BATCH_SIZE=32
    NUM_WORKERS=0
    MEMORY=200
    EXP_NAME="smoke_${DATASET}_s${SEED}"
    ;;
  balance)
    # match @train_incremental.sh core training params
    # dataset should be VGGSound_balance
    NUM_TASKS=10
    NEPOCHS=200
    BATCH_SIZE=256
    NUM_WORKERS=4
    MEMORY=500
    LR=1e-3
    WD=1e-4
    SCHEDULE_STEP=(50000)
    EXP_NAME="balance_${DATASET}_s${SEED}"
    ;;
  reverse)
    NUM_TASKS=10
    NEPOCHS=200
    BATCH_SIZE=256
    NUM_WORKERS=0
    MEMORY=500
    LR=1e-3
    WD=1e-4
    SCHEDULE_STEP=(50000)
    EXP_NAME="reverse_${DATASET}_s${SEED}"
    ;;
  custom)
    # keep defaults, user can override by editing script or env vars
    ;;
  *)
    echo "Unknown scenario: $SCENARIO"
    echo "Supported: smoke | balance | reverse | custom"
    exit 1
    ;;
esac

# loss flags
LOSS_FLAGS=()
if [ "$USE_INSTANCE" = true ]; then LOSS_FLAGS+=(--instance-contrastive); fi
if [ "$USE_CLASS" = true ]; then LOSS_FLAGS+=(--class-contrastive); fi
if [ "$USE_ATTN" = true ]; then LOSS_FLAGS+=(--attn-score-distil); fi

PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$GPU \
python3 -u $SRC_DIR/main_incremental.py \
  --exp-name "$EXP_NAME" \
  --results-path "$RESULTS_DIR" \
  --datasets "$DATASET" \
  --approach avcil \
  --network avcil_net \
  --seed "$SEED" \
  --gpu 0 \
  --num-tasks "$NUM_TASKS" \
  --nepochs "$NEPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --pin-memory "$PIN_MEMORY" \
  --lr "$LR" \
  --weight-decay "$WD" \
  --schedule_step "${SCHEDULE_STEP[@]}" \
  --num-exemplars "$MEMORY" \
  "${LOSS_FLAGS[@]}" \
  --instance-contrastive-temperature "$TEMP_I" \
  --class-contrastive-temperature "$TEMP_C" \
  --lam "$LAM" \
  --lam-I "$LAM_I" \
  --lam-C "$LAM_C"