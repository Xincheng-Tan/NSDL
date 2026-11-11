#!/usr/bin/env bash
# A0b G1: distil the LOCUS-Bench TRAIN split into an SFT dataset.
#
# The recipe is scripts/run_v3_3_rebuild.sh argument for argument. Only three
# things differ, and they are all "which corpus", never "which method":
#   --source locus   read data/lnsdl_bench/ramp63k/<split>/data.parquet, index
#                    renders by ROW INDEX out of render_manifest.json (LOCUS
#                    task_id is the grid CELL, shared by 210 rows, so it cannot
#                    key an image), and gate the status curriculum on
#                    locus_meta.cell[0] >= LOCUS_STATUS_MIN_S instead of the
#                    three VIKI task names.
#   --output-dir     data/lnsdl/sft/v3_3_locus/...
#   --yolo-device    CPU by default; this build must never contend with a
#                    training job for a card.
#
# CURRICULUM GATE (a configuration decision, not a constant of nature -- it is
# written into meta.json and must be disclosed): VIKI triggers the "look before
# you act" curriculum on three task names. LOCUS has no VIKI task names; its
# equivalent axis is S, where s>=1 is exactly the state-judgement load
# (docs/benchmark.md S axis). s>=1 fires on 4620/6300 = 73% of train rows.
#
# NO INHERITABLE TEACHER CACHE. The v3.2/v3.3 rebuilds could seed from the
# previous cache because the prompts were unchanged. LOCUS prompts are new, so
# all ~6300 think calls are fresh. The cache file is still written incrementally
# and is NOT deleted between attempts, so a retry resumes rather than re-pays.
#
# Usage:
#   bash scripts/run_locus_sft_build.sh                  # full 6300
#   MAX_SAMPLES=200 OUT=/tmp/probe bash scripts/run_locus_sft_build.sh
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY=python

# TEACHER ENDPOINT (changed 2026-08-26; the v3.3 default is dead).
# v3.3 distilled through relay (endpoint 2, relay.example.com, `responses`).
# On 2026-08-26 that endpoint began answering EVERY request with 403 "please
# use a standard Codex client" -- both hostnames (relay.example.com and
# relay.example.com), text and image alike, in 1.2-1.7s. It is a client-fingerprint
# block, not a rate limit and not a black window, so retrying does not help.
# Measured replacement: relay (endpoint 3) serving the same gpt-5.6-terra,
# 24/24 successful image calls at both 6 and 12 workers, median 14-16s on a
# full-size 2.96MB LOCUS render. Note relay is also the endpoint the paper's
# reported numbers go through (memory: relay-api-silent-failures-2026-08-02),
# and the reason relay was only ever acceptable for distillation is that it
# randomly injects Kiro/Codex system prompts -- so this switch moves the
# supervision onto the cleaner of the two, not onto a fallback.
API_BASE="${API_BASE:-https://relay.example.com/v1}"
ENDPOINT_INDEX="${ENDPOINT_INDEX:-3}"
API_PROTOCOL="${API_PROTOCOL:-auto}"
# 6 workers is the v3.3 value. LOCUS pays ~1 image call (~16s) plus up to two
# text calls per row over 6300 rows, which is ~7.6h at 6 workers and ~3.8h at
# 12 -- and 12 measured identically clean (24/24, median 14.4s). The knob is
# concurrency only; it cannot change what the teacher is asked or answers.
TEACHER_WORKERS="${TEACHER_WORKERS:-12}"

# A0f-W2 (2026-08-31): "balanced" is the frozen opschain default (67.2% of rows
# carry a schedule demo). SCHEDULE_POLICY=always injects the demo on every row
# whose Phi reconstruction matches GT -- with the ops_chain contract that is
# 6300/6300, so density goes to ~100%. Cache note: teacher_cache_key hashes the
# tool-call list, so rows that ALREADY had a demo keep their exact keys (cache
# hit), while previously demo-less rows re-distil (~1/3 of train, ~1-2 h at 12
# workers). Seed the new OUT's teacher_think_cache.jsonl from the opschain dir
# before launching to get that reuse.
SCHEDULE_POLICY="${SCHEDULE_POLICY:-balanced}"
LOCUS_SPLIT="${LOCUS_SPLIT:-train}"
LOCUS_STATUS_MIN_S="${LOCUS_STATUS_MIN_S:-1}"
OUT="${OUT:-$ROOT/data/lnsdl/sft/v3_3_locus/gpt56_terra_locus_train_full}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
YOLO_DEVICE="${YOLO_DEVICE:-cpu}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOG="${LOG:-$ROOT/logs/locus_sft_build_${TS}.log}"
EXIT_FILE="${EXIT_FILE:-$ROOT/logs/locus_sft_build_${TS}.exit}"

mkdir -p "$OUT" "$ROOT/logs"
rm -f "$EXIT_FILE"
ln -sfn "$(basename "$LOG")" "$ROOT/logs/locus_sft_build.log.latest"
ln -sfn "$(basename "$EXIT_FILE")" "$ROOT/logs/locus_sft_build.exit.latest"

{
echo "[locus] start $(date -Is)"
echo "[locus] split=$LOCUS_SPLIT out=$OUT yolo=$YOLO_DEVICE max_samples=$MAX_SAMPLES min_s=$LOCUS_STATUS_MIN_S"

# PREFLIGHT. The teacher phase writes nothing to this log until a think comes
# back clean, so a dead endpoint and a slow one look identical for hours -- that
# is exactly how the first LOCUS attempt burned 2h for 0 cached thinks. Pay one
# real call, in the real request shape, before committing.
echo "[locus] preflight $(date -Is)"
if ! "$PY" scripts/preflight_teacher_api.py \
    --endpoint-index "$ENDPOINT_INDEX" \
    --api-base-url "$API_BASE" \
    --api-model gpt-5.6-terra \
    --api-protocol "$API_PROTOCOL" \
    --image "$ROOT/data/lnsdl_bench/ramp63k/${LOCUS_SPLIT}/images/00000.png"; then
  echo "[locus] preflight FAILED -- refusing to start a multi-hour build against a dead endpoint"
  echo "3" > "$EXIT_FILE"
  exit 3
fi

# PROGRESS. The builder itself is silent during the teacher phase. Sample the
# incrementally-written cache so the log carries a rate, and a stall is visible
# within five minutes instead of at the end.
(
  while sleep 300; do
    n=$(wc -l < "$OUT/teacher_think_cache.jsonl" 2>/dev/null || echo 0)
    echo "[locus] progress $(date -Is) cached_thinks=$n"
  done
) &
PROGRESS_PID=$!
trap 'kill "$PROGRESS_PID" 2>/dev/null' EXIT

status=1
for attempt in 1 2 3; do
  echo "[locus] build attempt $attempt start $(date -Is)"
  if "$PY" scripts/build_lnsdl_sft_data.py \
      --source locus \
      --sketch-variant ops_chain \
      --locus-split "$LOCUS_SPLIT" \
      --locus-status-min-s "$LOCUS_STATUS_MIN_S" \
      --max-samples "$MAX_SAMPLES" \
      --output-dir "$OUT" \
      --think-source api \
      --think-style ad_checklist \
      --query-source instruction \
      --teacher-model gpt-5.6-terra \
      --endpoint-index "$ENDPOINT_INDEX" \
      --api-base-url "$API_BASE" \
      --api-protocol "$API_PROTOCOL" \
      --teacher-include-image \
      --teacher-think-mode blind \
      --teacher-workers "$TEACHER_WORKERS" \
      --teacher-api-retries 10 \
      --teacher-api-retry-sleep 3.0 \
      --teacher-quality-retries "${TEACHER_QUALITY_RETRIES:-8}" \
      --schedule-policy "$SCHEDULE_POLICY" \
      --goal-coverage-policy auto \
      --status-curriculum \
      --pre-goal-tools \
      --repair-rate 0.03 \
      --goal-coverage-direct-rate 0.85 \
      --crop-demo-policy curriculum \
      --crop-demo-other-rate 0.05 \
      --crop-demo-max 2 \
      --yolo-device "$YOLO_DEVICE"; then
    echo "[locus] build SUCCEEDED $(date -Is)"
    echo "[locus] running dataset audit"
    if "$PY" scripts/audit_lnsdl_sft_data.py --data-dir "$OUT" > "$OUT/audit_report.json" 2>&1; then
      echo "[locus] audit PASSED -> $OUT/audit_report.json"
      status=0
    else
      echo "[locus] audit FAILED -> $OUT/audit_report.json"
      status=2
    fi
    break
  fi
  echo "[locus] build attempt $attempt FAILED $(date -Is); cache preserved, will retry"
done

if [[ "$status" != "0" ]]; then
  echo "[locus] giving up with status=$status $(date -Is)"
fi
kill "$PROGRESS_PID" 2>/dev/null
echo "[locus] finished status=$status $(date -Is)"
echo "$status" > "$EXIT_FILE"
exit "$status"
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
