#!/usr/bin/env bash
# Launch L-NSDL agentic VL SFT.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_SIZE="${MODEL_SIZE:-3b}"
DATA_DIR="${DATA_DIR:-$(pwd)/data/lnsdl/sft/v2/gpt56_terra_image_p58_full}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs}"
DEVICE="${DEVICE:-}"
AUTO_TORCHRUN="${AUTO_TORCHRUN:-1}"
DRY_RUN="${DRY_RUN:-0}"

case "$MODEL_SIZE" in
  2b)
    # P10.S2 capacity grid, previous-generation 2B cell (Qwen2-VL-2B). Like 3b it
    # fits on one card, so seeds run in parallel; CONFIG/OUTPUT_DIR always come
    # from the caller (there is no non-seeded 2b baseline).
    CONFIG="${CONFIG:?CONFIG is required for MODEL_SIZE=2b}"
    DEFAULT_OUTPUT_DIR="$(pwd)/checkpoints/lnsdl/sft/v3_3/qwen2-vl-2b-gpt56-terra-image-p513-obsbranch"
    DEFAULT_DEVICE="0"
    ;;
  3b)
    CONFIG="${CONFIG:-configs/lnsdl-agentic-vl-sft-qwen25-3b.yaml}"
    DEFAULT_OUTPUT_DIR="$(pwd)/checkpoints/lnsdl/sft/v2/qwen25-vl-3b-gpt56-terra-image-p58"
    DEFAULT_DEVICE="0"
    ;;
  7b)
    CONFIG="${CONFIG:-configs/lnsdl-agentic-vl-sft-qwen25-7b.yaml}"
    DEFAULT_OUTPUT_DIR="$(pwd)/checkpoints/lnsdl/sft/v2/qwen25-vl-7b-gpt56-terra-image-p58"
    DEFAULT_DEVICE="auto"
    ;;
  *)
    echo "Unsupported MODEL_SIZE=$MODEL_SIZE; expected 2b, 3b or 7b." >&2
    exit 2
    ;;
esac

DEVICE="${DEVICE:-$DEFAULT_DEVICE}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
LOG="${LOG:-$LOG_DIR/lnsdl_agentic_sft_${MODEL_SIZE}_${TS}.log}"
EXIT_FILE="${EXIT_FILE:-$LOG_DIR/lnsdl_agentic_sft_${MODEL_SIZE}_${TS}.exit}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
rm -f "$EXIT_FILE"
ln -sfn "$(basename "$LOG")" "$LOG_DIR/lnsdl_agentic_sft_${MODEL_SIZE}.log.latest"
ln -sfn "$(basename "$EXIT_FILE")" "$LOG_DIR/lnsdl_agentic_sft_${MODEL_SIZE}.exit.latest"

auto_torchrun_arg=("--auto-torchrun")
if [[ "$AUTO_TORCHRUN" == "0" ]]; then
  auto_torchrun_arg=("--no-auto-torchrun")
fi

extra=()
if [[ "$DRY_RUN" == "1" ]]; then
  extra+=("--dry_run")
fi
if [[ -n "${MODEL_PATH:-}" ]]; then
  extra+=("--model-path" "$MODEL_PATH")
fi
if [[ -n "${GOAL_LOSS_WEIGHT:-}" ]]; then
  extra+=("--goal-loss-weight" "$GOAL_LOSS_WEIGHT")
fi
if [[ -n "${PLAN_LOSS_WEIGHT:-}" ]]; then
  extra+=("--plan-loss-weight" "$PLAN_LOSS_WEIGHT")
fi

echo "[$(date -Is)] Launching L-NSDL agentic SFT" | tee "$LOG"
echo "model_size=$MODEL_SIZE" | tee -a "$LOG"
echo "config=$CONFIG" | tee -a "$LOG"
echo "data_dir=$DATA_DIR" | tee -a "$LOG"
echo "output_dir=$OUTPUT_DIR" | tee -a "$LOG"
echo "device=$DEVICE" | tee -a "$LOG"
echo "goal_loss_weight=${GOAL_LOSS_WEIGHT:-config}" | tee -a "$LOG"
echo "plan_loss_weight=${PLAN_LOSS_WEIGHT:-config}" | tee -a "$LOG"

set +e
"$PYTHON_BIN" scripts/run_vl_sft.py \
  --config "$CONFIG" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  "${auto_torchrun_arg[@]}" \
  "${extra[@]}" \
  "$@" \
  2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e

echo "$status" > "$EXIT_FILE"
echo "[$(date -Is)] Finished with status=$status" | tee -a "$LOG"
exit "$status"
