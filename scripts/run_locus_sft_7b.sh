#!/usr/bin/env bash
# A0b G1: the LOCUS-Bench MAIN ROW -- 7B SFT trained directly on LOCUS train.
#
# WHY THIS EXISTS. Every LOCUS number in the paper today comes from a checkpoint
# that was SFT'd on VIKI-L2 and then evaluated on LOCUS, i.e. every row is a
# TRANSFER row. docs/plan.md A0b G1 requires one row where the model was trained
# on the benchmark's own train split, so the transfer rows can be labelled as
# transfer rather than read as the benchmark's headline result.
#
# Contract, mirroring scripts/run_v3_3_7b_sft_seed.sh:
#   - model.path = foundation Qwen2.5-VL-7B-Instruct, NOT a VIKI SFT checkpoint
#     (initialising from VIKI would make this a second transfer row).
#   - recipe (lr / epochs / goal_loss_weight / freeze / fsdp / internal_val_seed)
#     byte-identical to v3.3; the only variable is the data.
#   - the four frozen LOCUS eval splits are never read here.
#
# 7B uses FSDP full_shard and ONE run occupies ALL FOUR cards -- unlike v3.3,
# which fit on two. LOCUS renders are 1920x1080 vs VIKI-L2's 600x337 and the
# processor downscales nothing (max_pixels 12,845,056), so a sample carries
# ~2645 vision tokens instead of ~286 and the sequence runs 6246 mean / 6653
# max instead of 4461 / 5346. On two cards both ranks OOM at step 1 on the fp32
# logits upcast (seq x 152064 x 4 B); see the config comment. Sharding over
# four ranks with gradient_accumulation_steps halved keeps the effective batch
# at 8, so the recipe is still v3.3's -- but it means the two seeds run
# SEQUENTIALLY, not side by side.
#
# n=1 seed is a direction, not a result -- see the W4-b seed band (7B OOD
# +/-.051 across 3 seeds); further seeds are backfilled later.
#
# Usage: SEED=42 bash scripts/run_locus_sft_7b.sh
set -uo pipefail
cd "$(dirname "$0")/.."

SEED="${SEED:-42}"

# LOCUS renders are 1920x1080 (9.2x the vision tokens of VIKI-L2's 600x337) and
# the yaml therefore turns gradient checkpointing ON. Fragmentation is still the
# margin here -- the OOM that forced this change showed 3.65 GiB reserved-but-
# unallocated against a 3.90 GiB request on rank 0 -- so give the caching
# allocator expandable segments too. Harmless if the run has headroom.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export MODEL_SIZE=7b
export CONFIG="configs/lnsdl-agentic-vl-sft-qwen25-7b-v3_3-locus-seed${SEED}.yaml"
export DATA_DIR="${DATA_DIR:-$(pwd)/data/lnsdl/sft/v3_3_locus/gpt56_terra_locus_train_full}"
export OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/checkpoints/lnsdl/sft/v3_3_locus/qwen25-vl-7b-locus-seed${SEED}}"
export DEVICE="${DEVICE:-auto}"
export TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
export LOG="logs/lnsdl_locus_sft_7b_seed${SEED}_${TS}.log"
export EXIT_FILE="logs/lnsdl_locus_sft_7b_seed${SEED}_${TS}.exit"

# HARD GUARD. run_vl_sft.py derives torchrun --nproc_per_node from the visible
# device count, and the yaml's accum is set for a world size of 4. Running this
# on 2 cards would silently train at effective batch 4, on 8 at batch 16 --
# either way a different recipe from v3.3 wearing the same run_name.
nvis="$(python -c 'import torch;print(torch.cuda.device_count())')"
if [[ "$nvis" != "4" ]]; then
  echo "Refusing to start: $nvis CUDA device(s) visible, need exactly 4." >&2
  echo "The config sets gradient_accumulation_steps=2 for a world size of 4" >&2
  echo "(effective batch 8, matching v3.3). Set CUDA_VISIBLE_DEVICES=0,1,2,3." >&2
  exit 2
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config $CONFIG" >&2
  exit 2
fi
if [[ ! -f "$DATA_DIR/samples.jsonl" ]]; then
  echo "Missing LOCUS SFT dataset: $DATA_DIR/samples.jsonl" >&2
  echo "Build it first with scripts/run_locus_sft_build.sh (CPU + teacher API, no GPU)." >&2
  exit 2
fi

exec bash scripts/train_lnsdl_agentic_sft.sh
