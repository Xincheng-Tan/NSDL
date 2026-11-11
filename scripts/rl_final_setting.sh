#!/usr/bin/env bash
# The reinforcement-learning setting behind the "NSDL 7B (SFT + RL)" rows of
# Tables 1 and 2. Wraps scripts/run_dapo_lnsdl_7b.sh with the exact knobs used.
#
#   reward  = 0.40 R_task + 0.30 R_goal + 0.15 R_goal_plan
#             - 0.05 P_format - 0.05 P_sched_repeat      (all read against GT)
#   advantage = scalar (GRPO group normalisation) on the DAPO stack
#   KL 0.01, 30 steps, checkpoints every 5 steps, final (step 30) reported
#
# Usage:
#   SFT_CKPT=checkpoints/nsdl-7b-sft  TRAIN_FILE=data/rl/viki_l2_train/train.parquet \
#   VAL_FILE=data/rl/viki_l2_train/probe512.parquet  EXP_NAME=nsdl-7b-rl-viki \
#   bash scripts/rl_final_setting.sh
set -euo pipefail
: "${SFT_CKPT:?}" "${TRAIN_FILE:?}" "${VAL_FILE:?}" "${EXP_NAME:?}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}" ROLLOUT_TP=1 \
MODEL_SIZE="${MODEL_SIZE:-7b}" \
LNSDL_ADV_MODE=scalar \
LNSDL_W_TASK=0.40 LNSDL_W_GOAL=0.30 LNSDL_W_GOAL_PLAN=0.15 \
LNSDL_W_FAITHFUL=0 LNSDL_W_STRUCT=0 LNSDL_W_STRUCT_PHI=0 LNSDL_W_STRUCT_SKETCH=0 \
LNSDL_STRUCT_SKETCH_EXEC=1 LNSDL_P_FORMAT_V2=0.05 LNSDL_P_SCHED_REPEAT=0.05 \
LNSDL_R_TASK_SOURCE=plan LNSDL_R_PLAN_SOURCE=plan \
LNSDL_TOOLSET_CONTRACT=v341 LNSDL_PHI_OPEN_PLACEMENT=early \
TRAIN_PROMPT_BSZ=64 TRAIN_PROMPT_MINI_BSZ=16 \
ROLLOUT_MAX_MODEL_LEN=9216 MAX_RESPONSE_LENGTH=3072 LIMIT_IMAGE_PER_PROMPT=11 \
LNSDL_SKIP_UPWEIGHT=2.0 KL_LOSS_COEF=0.01 \
ACTOR_OPTIMIZER_OFFLOAD=False \
TOTAL_TRAINING_STEPS=30 TEST_FREQ=5 SAVE_FREQ=5 VAL_BEFORE_TRAIN=True \
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}" \
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}" \
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}" \
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}" \
bash scripts/run_dapo_lnsdl_7b.sh
