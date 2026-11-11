#!/usr/bin/env bash
# Parallel launcher for L-NSDL agentic VIKI-L2 evaluation.

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_SIZE="${MODEL_SIZE:-3b}"
# Benchmark to evaluate: VIKI-L2 (default, splits test/val) or LOCUS (splits
# test_id/ood_s/ood_m/ood_x). Same eval recipe either way -- only the split
# path and the reported score column differ, so a cell stays comparable across
# both benchmarks.
LEVEL="${LEVEL:-VIKI-L2}"
SPLITS="${SPLITS:-val,test}"
SHARDS="${SHARDS:-2}"
NUM_GPUS="${NUM_GPUS:-2}"
# First physical GPU to use. Default 0 keeps existing invocations byte-identical;
# set e.g. GPU_OFFSET=1 NUM_GPUS=1 to run an eval on card 1 while card 0 is busy
# training. Only affects placement, never the eval recipe.
GPU_OFFSET="${GPU_OFFSET:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_TAG="${CHECKPOINT_TAG:-gpt56-terra-image-p58}"
SLM_WEIGHTS="${SLM_WEIGHTS:-$(pwd)/checkpoints/lnsdl/sft/v2/qwen25-vl-${MODEL_SIZE}-${CHECKPOINT_TAG}/final}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(pwd)/results/lnsdl/v2/viki_l2}"
OUTPUT_BUCKET="${OUTPUT_BUCKET:-agentic_sft_qwen25_vl_${MODEL_SIZE}_${CHECKPOINT_TAG//-/_}}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.0}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$LOG_DIR"

extra=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  extra+=("--max-samples" "$MAX_SAMPLES")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  extra+=("--overwrite")
fi
# TASK_NAME accepts a comma-separated list; --task-name is `action="append"`
# on the python side, so each name becomes its own flag (A0b G5 needs five
# LOCUS cells in one run).
if [[ -n "${TASK_NAME:-}" ]]; then
  IFS=',' read -r -a _task_names <<< "$TASK_NAME"
  for _tn in "${_task_names[@]}"; do
    [[ -n "$_tn" ]] && extra+=("--task-name" "$_tn")
  done
fi
if [[ -n "${DETECTOR_WEIGHTS:-}" ]]; then
  extra+=("--detector-weights" "$DETECTOR_WEIGHTS")
fi
# P5.10.2 crop-and-look probe: auto-attach ≤N crops to model detect calls.
if [[ -n "${DETECT_ATTACH_CROPS:-}" ]]; then
  extra+=("--detect-attach-crops" "$DETECT_ATTACH_CROPS")
fi
# P5.10.4 perception bracket: DETECTOR_BACKEND in
# {none, ultralytics_yolo26, yoloworld_openvocab, off, oracle}.
if [[ -n "${DETECTOR_BACKEND:-}" ]]; then
  extra+=("--detector-backend" "$DETECTOR_BACKEND")
fi
# D2 generality bracket: the open-vocabulary backend has no args.yaml next to its
# weights, so its class vocabulary must be supplied explicitly. Without it the arm
# would load with an empty vocabulary and silently equal DETECTOR_BACKEND=off.
if [[ -n "${DETECTOR_DATA_YAML:-}" ]]; then
  extra+=("--detector-data-yaml" "$DETECTOR_DATA_YAML")
fi
# P10.S1 A-axis inversion: detect additionally returns already_at_destination.
# Composes with DETECTOR_BACKEND to give the matched-perception 2x2.
if [[ "${DETECT_VERDICT:-0}" == "1" ]]; then
  extra+=("--detect-verdict")
fi
# Table 3 block 1 crop-off arm: keep symbolic evidence, drop crop pixels.
if [[ "${DETECT_STRIP_CROPS:-0}" == "1" ]]; then
  extra+=("--detect-strip-crops")
fi
# Cluster 10 guard: truncate each generation to the first [TOOL_CALL].
if [[ "${TRUNCATE_FIRST_TOOL_CALL:-0}" == "1" ]]; then
  extra+=("--truncate-first-tool-call")
fi
# P10.S2 y-axis: Phi-off arm (schedule/verify/expand unavailable, detect kept).
if [[ "${DISABLE_PHI:-0}" == "1" ]]; then
  extra+=("--disable-phi")
fi
# P10.S2 trained Phi-off arm: also swap in the system prompt the --phi-off SFT
# checkpoint was trained under. Set alongside DISABLE_PHI=1 (the gate then only
# ever fires if the model hallucinates a tool the prompt never mentioned, which
# is exactly the diagnostic we want to read).
if [[ "${PHI_OFF_PROMPT:-0}" == "1" ]]; then
  extra+=("--phi-off-prompt")
fi
# P6.1 point 4-AR (eval-only): force a Phi/schedule pass on direct-emit plans.
# Mutually exclusive with DISABLE_PHI / PHI_OFF_PROMPT (the python layer errors out).
if [[ "${FORCE_PHI_RESCHEDULE:-0}" == "1" ]]; then
  extra+=("--force-phi-reschedule")
fi
# A0f-W1: which samples the forcing covers. "always" = the A0f decision itself
# (final plan is always Phi-expanded); unset/"direct_emit" keeps the frozen
# point 4-AR lower-bound semantics. Python layer rejects scope without the flag.
if [[ -n "${FORCE_PHI_SCOPE:-}" ]]; then
  extra+=("--force-phi-scope" "$FORCE_PHI_SCOPE")
fi
# A0k (facts:E24c, 2026-09-04): forced arm's final plan = Phi's own schedule
# (python default "phi_schedule"); "reexpand_ops" only reproduces pre-fix buckets.
if [[ -n "${FORCE_PHI_FINAL_PLAN:-}" ]]; then
  extra+=("--force-phi-final-plan" "$FORCE_PHI_FINAL_PLAN")
fi
# A0g② (v3.4 eval): rejected forced schedule -> ToolReason feedback turn +
# extra repair rounds instead of silent fallback. Requires FORCE_PHI_RESCHEDULE=1.
if [[ "${FORCE_PHI_REJECT_FEEDBACK:-0}" == "1" ]]; then
  extra+=("--force-phi-reject-feedback")
fi
# A1 decoding hardening (opt-in; default off preserves greedy decode).
if [[ -n "${REPETITION_PENALTY:-}" ]]; then
  extra+=("--repetition-penalty" "$REPETITION_PENALTY")
fi
if [[ -n "${STOP_STRING:-}" ]]; then
  # STOP_STRING may be a comma-separated list, e.g. "</pre>,<pre>"
  IFS=',' read -ra _stops <<< "$STOP_STRING"
  for s in "${_stops[@]}"; do extra+=("--stop-string" "$s"); done
fi

pids=()
slot=0
IFS=',' read -ra SPLIT_ARR <<< "$SPLITS"
for split in "${SPLIT_ARR[@]}"; do
  for ((i=0; i<SHARDS; i++)); do
    gpu=$(( GPU_OFFSET + slot % NUM_GPUS ))
    log="$LOG_DIR/eval_${OUTPUT_BUCKET}_${split}_shard${i}of${SHARDS}.log"
    echo "[launch] split=$split shard=$i/$SHARDS gpu=$gpu log=$log"
    CUDA_VISIBLE_DEVICES="$gpu" \
      "$PYTHON_BIN" scripts/eval_lnsdl_agentic_viki_l2.py \
        --model-size "$MODEL_SIZE" \
        --slm-weights "$SLM_WEIGHTS" \
        --level "$LEVEL" \
        --split "$split" \
        --num-shards "$SHARDS" --shard-index "$i" \
        --device cuda:0 \
        --output-dir "$OUTPUT_ROOT" \
        --result-bucket "$OUTPUT_BUCKET" \
        --max-tool-rounds "$MAX_TOOL_ROUNDS" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        "${extra[@]}" \
        > "$log" 2>&1 &
    pids+=($!)
    slot=$(( slot + 1 ))
    sleep 4
  done
done

echo "[launch] all ${#pids[@]} processes started; waiting..."
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "[launch] pid=$pid FAILED"
    fail=$(( fail + 1 ))
  fi
done
echo "[launch] done; failures=$fail / ${#pids[@]}"
exit "$fail"
