#!/usr/bin/env bash
set -uo pipefail
export PYTHONUNBUFFERED=1

if [ $# -lt 2 ]; then
  echo "usage: $0 {fcos|fasterrcnn|dino} {kitti|bdd100k|cityscapes|nuimages} [baseline] [r0025] [r0050] [r0075]" >&2
  exit 1
fi

MODEL=$1
DATA=$2
shift 2
RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(baseline r0025 r0050 r0075)

DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"

case "$MODEL:$DATA" in
  fcos:cityscapes)       TRAIN_CFG=configs/train/fcos_cityscapes.yaml ;;
  fcos:*)                TRAIN_CFG=configs/train/fcos.yaml ;;
  fasterrcnn:cityscapes) TRAIN_CFG=configs/train/fasterrcnn_cityscapes.yaml ;;
  fasterrcnn:*)          TRAIN_CFG=configs/train/fasterrcnn.yaml ;;
  dino:kitti)            TRAIN_CFG=configs/train/dino_12ep.yaml ;;
  dino:*)                TRAIN_CFG=configs/train/dino_24ep.yaml ;;
  *) echo "unknown detector: $MODEL" >&2; exit 1 ;;
esac

for RUN in "${RUNS[@]}"; do
  case "$RUN" in
    baseline) NAME=baseline; HR_CFG=configs/remiss/hard_replay_off.yaml ;;
    r0025)    NAME=HR-R0025; HR_CFG=configs/remiss/hard_replay_r0025.yaml ;;
    r0050)    NAME=HR-R0050; HR_CFG=configs/remiss/hard_replay_r0050.yaml ;;
    r0075)    NAME=HR-R0075; HR_CFG=configs/remiss/hard_replay_r0075.yaml ;;
    *) echo "unknown run: $RUN" >&2; exit 1 ;;
  esac

  OUT="runs/train/$DATA/$MODEL/$NAME"
  mkdir -p "$OUT"
  python tools/train.py \
    --config "$TRAIN_CFG" \
    --model "configs/models/$MODEL.yaml" \
    --data "$DATA" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --output-dir "$OUT" \
    --missbank-config configs/remiss/missbank.yaml \
    --hard-replay-config "$HR_CFG" \
    2>&1 | tee -a "$OUT/train.log"
done
