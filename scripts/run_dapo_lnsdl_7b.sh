#!/usr/bin/env bash
# P5.3 · VIKI-L2-train scoped RL for L-NSDL (DAPO + small KL anchor to P5.7).
#
# Design: docs/dataset_and_train.md §E.1 + docs/plan.md §P5.3.
# Reward:  verl/verl/utils/reward_score/lnsdl.py  (data_source = "lnsdl")
#   R_total = 0.40 R_task + 0.30 R_goal + 0.15 R_faithful + 0.15 R_struct
#             + LNSDL_R_TOOL_USE R_tool_use - P_format - P_nonterm - P_cost
#
# DAPO knobs (each maps to a recorded failure, see §E.1.1):
#   - Dynamic sampling  (algorithm.filter_groups, metric=acc_first_score by default):
#     T3 composite filter: keep prompt groups with R_task/acc variance first,
#     then use dense score variance as a fallback when exact-success variance is
#     too sparse. Set FILTER_GROUPS_METRIC=score to reproduce the old dense
#     filter, or =acc for exact-signal-only diagnostics.
#     This is the root-cause fix
#     for the recorded GRPO collapse (7B SFT .970 -> all-correct groups ->
#     advantage~=0), while still preserving dense shaping for all-wrong groups.
#   - Clip-higher       (clip_ratio_low=0.2 / high=0.28): anti bias-collapse.
#   - Token-level loss  (loss_agg_mode=token-mean): long multi-turn trajectories.
#   - Overlong shaping  (reward_model.overlong_buffer): never-close soft penalty,
#     complements the in-reward P_nonterm against P5.2.4 tool explosion.
#   - Small KL anchor   (use_kl_loss=True, kl_loss_coef=1e-3): DELIBERATE
#     deviation from vanilla DAPO (which drops KL) -- the project's dominant
#     risk is collapse, and we must preserve the P5.7 tool-protocol prior. Raise
#     this coef if a red-line (tool-call explosion / stacking / opval) blows.
#
# Init: P5.8 v2 SFT ckpt. Historical P5.7 runs require explicit overrides.
# Data: official VIKI-L2 train ONLY. val.parquet (OOD 11-*) is NEVER read here;
#       VAL_FILE is the ID test split (test.parquet), matching VIKI-R hygiene.
#
# MULTI-TURN TOOL ROLLOUT (wired): rollout.name=vllm_multiturn drives the
#     tool_runtime loop ([TOOL_CALL] -> tool_result -> continue) inside rollout
#     (verl/.../vllm_rollout/vllm_multiturn_rollout.py + nsdl/rl/tool_loop_adapter.py).
#     It returns a per-token loss_mask (tool-observation tokens masked) and a
#     per-sample trace in extra_info['trace'] so the lnsdl reward reads real
#     emitted/executed/stopped_reason. STATUS: GPU path has run; full RL is gated
#     on a clean revised 20-30 step short run.
# DATA: frozen RL parquet v3 (2026-07-22; official train + ID test, never OOD
#     val; prompt byte-identical to the v3.3 SFT training side):
#     python scripts/prepare_rl_data.py --mode lnsdl \
#         --lnsdl-output-dir data/lnsdl/rl/v3/viki_l2_train
#     -> rows carry prompt / images / data_source='lnsdl' / lnsdl_sample /
#        reward_model.ground_truth{gt_dag_json,gt_goal_json,time_steps,robots,
#        init_pos,goal_constraints,temporal_constraints,idle_robots}.

set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# --------------------------------------------------------------------------
# Paths / experiment
# --------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MODEL_SIZE="${MODEL_SIZE:-7b}"
# Frozen init (plan.md P5.3'.0-4, 2026-07-22): v3.3 final + rollout guard ON.
SFT_CKPT="${SFT_CKPT:-${REPO_ROOT}/checkpoints/lnsdl/sft/v3_3/qwen25-vl-${MODEL_SIZE}-gpt56-terra-image-p513-obsbranch/final}"
EXP_NAME="${EXP_NAME:-lnsdl-dapo-qwen25-vl-${MODEL_SIZE}-v33init-${TS}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/lnsdl/rl/v3/${EXP_NAME}}"
# RL parquet must carry data_source="lnsdl" + full ground_truth fields
# {gt_dag_json, gt_goal_json, time_steps, robots, init_pos, goal_constraints,
#  temporal_constraints, idle_robots} (see scripts/prepare_rl_data.py).
TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/data/lnsdl/rl/v3/viki_l2_train/train.parquet}"
VAL_FILE="${VAL_FILE:-${REPO_ROOT}/data/lnsdl/rl/v3/viki_l2_train/test.parquet}"  # ID test only
LOG="${LOG:-logs/dapo_lnsdl_${MODEL_SIZE}_${TS}.log}"
EXIT_FILE="${EXIT_FILE:-logs/dapo_lnsdl_${MODEL_SIZE}_${TS}.exit}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"
mkdir -p logs "$RAY_TMPDIR"
: > "$LOG"; rm -f "$EXIT_FILE"

export PYTHONPATH="${REPO_ROOT}/verl:${REPO_ROOT}:${PYTHONPATH:-}"
# vLLM's CuMemAllocator memory pool is incompatible with PyTorch
# expandable_segments. If an outer shell left it enabled, clear it before
# constructing the vLLM rollout engine.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    echo "[$(date -Is)] unset PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF} because vLLM CuMemAllocator is incompatible with expandable_segments" >&2
    unset PYTORCH_CUDA_ALLOC_CONF
fi
if [[ -n "${VLLM_ATTENTION_BACKEND:-}" ]]; then
    export VLLM_ATTENTION_BACKEND
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export RAY_TMPDIR
# The host exposes ~1TiB host memory, but Ray may see only a 200GB cgroup and
# kill checkpoint workers at the default 0.95 threshold. This does not allocate
# more memory; it only lets the save path use the available headroom.
ray_memory_usage_threshold="${RAY_MEMORY_USAGE_THRESHOLD:-${RAY_memory_usage_threshold:-0.99}}"
export RAY_memory_usage_threshold="${ray_memory_usage_threshold}"
# Reward weights are env-overridable (docs/dataset_and_train §E.1.2 defaults).
export LNSDL_W_TASK="${LNSDL_W_TASK:-0.40}"
export LNSDL_W_GOAL="${LNSDL_W_GOAL:-0.30}"
export LNSDL_W_FAITHFUL="${LNSDL_W_FAITHFUL:-0.15}"
export LNSDL_W_STRUCT="${LNSDL_W_STRUCT:-0.15}"
# Guard against the observed dense-reward shortcut: wrong trajectories should
# still get shaped, but must not score like near-successes.
export LNSDL_TASK_FAIL_DENSE_SCALE="${LNSDL_TASK_FAIL_DENSE_SCALE:-0.70}"
export LNSDL_TASK_FAIL_SCORE_CAP="${LNSDL_TASK_FAIL_SCORE_CAP:-0.45}"
export LNSDL_R_TOOL_USE="${LNSDL_R_TOOL_USE:-0.0}"
# E6 (docs/RL_debug.md §3(3)): stacking protocol shaping. The stack gate drops
# W_TASK from a protocol-dirty (stacked / emitted!=executed) trajectory so a
# "stacked-but-suppress-recovered" winner can no longer out-score a clean one
# (E5 root fix). With the gate carrying the deterrent, P_format demotes to a
# TIE-BREAKER (0.05/cap 0.10) — bigger P_format was already shown to eat score
# (hard fmt40) and could never out-weigh W_TASK anyway.
export LNSDL_STACK_GATE_TASK="${LNSDL_STACK_GATE_TASK:-1}"
export LNSDL_P_FORMAT="${LNSDL_P_FORMAT:-0.05}"
export LNSDL_P_FORMAT_CAP="${LNSDL_P_FORMAT_CAP:-0.10}"
# T1 (docs/RL_debug.md §8): charge tool cost ONLY on trivial+solved samples,
# so P_cost stops being the dominant "use fewer tools" gradient on the majority
# of solved, non-trivial prompts. Needs the trivial_structure ground_truth flag.
export LNSDL_COND_PCOST="${LNSDL_COND_PCOST:-1}"
# T2 (docs/RL_debug.md §8): OPTIONAL, task-blind up-weight of skip-decision
# train samples (where R_goal has variance). UNSET/1.0 keeps the exact official
# train distribution (docs/agent.md). Set >1.0 for a DIAGNOSTIC signal-boost run;
# such a checkpoint is diagnostic, not a headline reported model.
if [[ -n "${LNSDL_SKIP_UPWEIGHT:-}" ]]; then export LNSDL_SKIP_UPWEIGHT; fi
if [[ -n "${LNSDL_SKIP_CURRICULUM:-}" ]]; then export LNSDL_SKIP_CURRICULUM; fi
if [[ -n "${LNSDL_SKIP_CURRICULUM_NONSKIP_WEIGHT:-}" ]]; then export LNSDL_SKIP_CURRICULUM_NONSKIP_WEIGHT; fi

# --------------------------------------------------------------------------
# DAPO algorithm hyper-parameters
# --------------------------------------------------------------------------
# Advantage composition (docs/plan.md §P5.3'.2/.2b). LNSDL_ADV_MODE selects the
# variant inside ONE code path (verl/verl/trainer/ppo/lnsdl_advantage.py):
#   scalar        = weighted-sum-then-normalise (CFA-off control, numerically
#                   identical to the stock grpo estimator)
#   gdpo          = per-component group norm + sum + batch norm, no zero-variance
#                   mask (arXiv 2601.05242) -- the key control
#   cfa           = per-component group norm + std_g < tau => A_k := 0
#   cfa_batchnorm = cfa + batch norm, isolates the batch-norm factor
#   signadv       = 2*r-1 on the task axis (arXiv 2605.07689), collapse baseline
# Set LNSDL_ADV_MODE=off to fall back to the stock estimator path entirely.
export LNSDL_ADV_MODE="${LNSDL_ADV_MODE:-cfa}"
export LNSDL_CFA_TAU="${LNSDL_CFA_TAU:-0.001}"
export LNSDL_CFA_W_A="${LNSDL_CFA_W_A:-1.0}"
export LNSDL_CFA_W_B="${LNSDL_CFA_W_B:-1.0}"
export LNSDL_CFA_W_T="${LNSDL_CFA_W_T:-1.0}"
export LNSDL_CFA_AXIS_A_MODE="${LNSDL_CFA_AXIS_A_MODE:-merge}"
export LNSDL_CFA_NONTERM_GUARD="${LNSDL_CFA_NONTERM_GUARD:-1}"
export LNSDL_COLLAPSE_LOG="${LNSDL_COLLAPSE_LOG:-${REPO_ROOT}/results/lnsdl/rl/${EXP_NAME}/collapse.jsonl}"
# Raw per-trajectory reward breakdown for the offline estimator contrast
# (docs/plan.md §P5.3'.2b protocol E). Mode independent, so ANY run leaves behind
# a batch every mode in ADV_MODES can be replayed on. Set LNSDL_ADV_DUMP=off to
# skip it.
export LNSDL_ADV_DUMP="${LNSDL_ADV_DUMP:-${REPO_ROOT}/results/lnsdl/rl/${EXP_NAME}/adv_batch.jsonl}"
if [[ "${LNSDL_ADV_DUMP}" == "off" ]]; then
    unset LNSDL_ADV_DUMP
fi
mkdir -p "$(dirname "$LNSDL_COLLAPSE_LOG")"
if [[ "${LNSDL_ADV_MODE}" == "off" ]]; then
    adv_estimator=grpo             # DAPO is built on the GRPO group-relative base
else
    adv_estimator=lnsdl_axis       # per-axis composition; mode from LNSDL_ADV_MODE
fi
# small KL anchor (deliberate deviation; see header)
use_kl_in_reward=False
use_kl_loss=True
kl_loss_coef="${KL_LOSS_COEF:-0.001}"
kl_loss_type=low_var_kl
# clip-higher
clip_ratio_low="${CLIP_LOW:-0.2}"
clip_ratio_high="${CLIP_HIGH:-0.28}"
# Damping knobs (docs/debug_rl.md 5.5.4). Both defaults are byte-identical to
# the probe512 step-0 anchor, so an unset env leaves every prior run
# reproducible. NEGATIVE entropy_coeff = entropy PENALTY: the actor computes
# `policy_loss = pg_loss - entropy_loss * entropy_coeff`, so a negative coeff
# adds +|coeff|*entropy to the minimised loss and pushes entropy DOWN. A
# positive value is the usual entropy BONUS and is the wrong sign for damping.
entropy_coeff="${ENTROPY_COEFF:-0}"
actor_lr="${ACTOR_LR:-1e-6}"
loss_agg_mode=token-mean
# dynamic sampling
enable_filter_groups="${ENABLE_FILTER_GROUPS:-True}"
filter_groups_metric="${FILTER_GROUPS_METRIC:-acc_first_score}"  # T3: acc variance first, score variance fallback
max_num_gen_batches="${MAX_NUM_GEN_BATCHES:-12}"
filter_groups_on_max_gen_batches="${FILTER_GROUPS_ON_MAX_GEN_BATCHES:-pad}"
# never-close soft shaping (token-level, complements in-reward P_nonterm)
enable_overlong_buffer=True
overlong_buffer_len="${OVERLONG_BUFFER_LEN:-512}"
overlong_penalty_factor=1.0
# batch / group sizes (7B on 2 GPUs; raise n_resp to 16 if budget allows)
train_prompt_bsz="${TRAIN_PROMPT_BSZ:-64}"
gen_prompt_bsz="${GEN_PROMPT_BSZ:-$((train_prompt_bsz * 3))}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-8}"
train_prompt_mini_bsz="${TRAIN_PROMPT_MINI_BSZ:-16}"
max_prompt_length="${MAX_PROMPT_LENGTH:-4096}"
max_response_length="${MAX_RESPONSE_LENGTH:-3072}"
temperature="${ROLLOUT_TEMPERATURE:-1.0}"   # exploration (SFT eval used 0.0)
max_tool_rounds="${MAX_TOOL_ROUNDS:-3}"
checkpoint_contents="${CKPT_CONTENTS:-model,hf_model,optimizer,extra}"
checkpoint_contents="${checkpoint_contents// /}"
checkpoint_contents_hydra="[${checkpoint_contents}]"
# Multi-turn prompts grow after each tool observation. vLLM's default
# max_model_len=prompt+response is too tight for agentic rollout, so reserve an
# explicit context margin and keep chunked-prefill's token budget in sync.
mt_context_margin="${MT_CONTEXT_MARGIN:-2048}"
rollout_max_model_len="${ROLLOUT_MAX_MODEL_LEN:-$((max_prompt_length + max_response_length + mt_context_margin))}"
rollout_max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-${rollout_max_model_len}}"
mt_context_guard_tokens="${MT_CONTEXT_GUARD_TOKENS:-128}"
# Rollout tensor parallelism. Default 1, and that is a correctness requirement,
# not a performance preference (docs/debug_rl.md §Q5): under
# `distributed_executor_backend=external_launcher` EVERY rank runs the whole
# tool loop -- detector forward included -- and each decides on its own whether
# a trajectory is done. Any per-rank difference in a tool return (non-
# deterministic detector, image-decode drift, a timeout) makes the ranks' done
# sets diverge, so they call the COLLECTIVE `generate()` a different number of
# times and the first one to finish waits forever on the next collective. It
# looks like a hang with `utilization.gpu 100%` at 76 W -- NCCL busy-wait, not
# compute. Runs that survived TP=2 did so because their tool returns happened
# to agree rank-for-rank; it is data-dependent, and crop usage raises the odds
# of divergence. TP=1 removes the collective from the loop entirely (each rank
# owns a whole engine and rollout parallelises over prompts instead), which is
# the path the §4.6 variance audit already drove for 7200 rollouts.
rollout_tp="${ROLLOUT_TP:-1}"
if [[ "${rollout_tp}" -gt 1 && "${ALLOW_MULTITURN_ROLLOUT_TP_GT1:-0}" != "1" ]]; then
    echo "[FATAL] ROLLOUT_TP=${rollout_tp} with rollout.name=vllm_multiturn is the deadlock configuration of docs/debug_rl.md §Q5." >&2
    echo "        The multi-turn tool loop runs per-rank while generate() is collective; diverging done-sets hang the run." >&2
    echo "        Fix the rollout loop (rank-0 decides + broadcasts) before raising TP, or set ALLOW_MULTITURN_ROLLOUT_TP_GT1=1 to override knowingly." >&2
    exit 1
fi

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
log "DAPO L-NSDL scoped RL: size=${MODEL_SIZE} init=${SFT_CKPT}"
log "train=${TRAIN_FILE} val(ID-test)=${VAL_FILE} exp=${EXP_NAME}"
log "context: prompt=${max_prompt_length} response=${max_response_length} max_model_len=${rollout_max_model_len} max_num_batched_tokens=${rollout_max_num_batched_tokens} guard=${mt_context_guard_tokens}"
log "filter_groups: enable=${enable_filter_groups} metric=${filter_groups_metric} max_num_gen_batches=${max_num_gen_batches} on_max=${filter_groups_on_max_gen_batches}"
log "reward_recipe: patch_stack=${LNSDL_REWARD_PATCH_STACK:-<unset=0>} (0 = P5.3'.1 four positive axes, no negative term; set 1 to restore the pre-2026-08-03 penalty stack for ablation. ALL reward_guard/reward_E6/reward_T1/reward_T4 knobs below are INERT unless patch_stack=1)"
log "reward_guard: task_fail_dense_scale=${LNSDL_TASK_FAIL_DENSE_SCALE} task_fail_score_cap=${LNSDL_TASK_FAIL_SCORE_CAP} struct_only=${LNSDL_TASK_FAIL_SCALE_STRUCT_ONLY:-1}"
log "reward_E6_stacking: stack_gate_task=${LNSDL_STACK_GATE_TASK} p_format=${LNSDL_P_FORMAT} p_format_cap=${LNSDL_P_FORMAT_CAP}"
log "protocol_P9+cluster10: stop_after_first_tool_call=${STOP_AFTER_FIRST_TOOL_CALL:-True} (default ON per plan.md P5.3'.0-2': v3.3 init stacks ~50% ID turns; guard = behavioral pre-parse truncation + trained-token tail masking, matches eval --truncate-first-tool-call)"
log "reward_T1: cond_pcost=${LNSDL_COND_PCOST}  sampler_T2: skip_upweight=${LNSDL_SKIP_UPWEIGHT:-<unset=uniform>} skip_curriculum=${LNSDL_SKIP_CURRICULUM:-0} nonskip_weight=${LNSDL_SKIP_CURRICULUM_NONSKIP_WEIGHT:-0.0}"
log "reward_T4: r_tool_use_weight=${LNSDL_R_TOOL_USE}"
log "advantage: adv_estimator=${adv_estimator} mode=${LNSDL_ADV_MODE} tau=${LNSDL_CFA_TAU} w_A=${LNSDL_CFA_W_A} w_B=${LNSDL_CFA_W_B} w_T=${LNSDL_CFA_W_T} axis_a_mode=${LNSDL_CFA_AXIS_A_MODE} nonterm_guard=${LNSDL_CFA_NONTERM_GUARD} collapse_log=${LNSDL_COLLAPSE_LOG}"
log "overlong_filter: LNSDL_OVERLONG_FILTER=${LNSDL_OVERLONG_FILTER:-<unset=0>} (M1, docs/debug_rl.md §3.5: 1 = zero the advantage of trajectories that hit MAX_RESPONSE_LENGTH; s24/s30 frozen recipes ran with it OFF)"
log "damping: entropy_coeff=${entropy_coeff} (0 = anchor / no damping; NEGATIVE = entropy penalty) actor_lr=${actor_lr} clip=${clip_ratio_low}/${clip_ratio_high}"
log "checkpoint: contents=${checkpoint_contents_hydra} drop_optimizer_state_on_light_save=${LNSDL_DROP_OPTIMIZER_STATE_ON_LIGHT_SAVE:-last}"
log "trainer: total_training_steps=${TOTAL_TRAINING_STEPS:--1} total_epochs=${TOTAL_EPOCHS:-4} save_freq=${SAVE_FREQ:-50} test_freq=${TEST_FREQ:-50} val_before_train=${VAL_BEFORE_TRAIN:-True}"
log "memory: pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF:-<unset>} ray_memory_usage_threshold=${RAY_memory_usage_threshold} actor_optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-False} actor_param_offload=${ACTOR_PARAM_OFFLOAD:-False} actor_adamw_foreach=${ACTOR_ADAMW_FOREACH:-False} vllm_gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
log "rollout_parallel: tensor_model_parallel_size=${rollout_tp} n_gpus_per_node=${N_GPUS_PER_NODE:-2} (TP=1 is §Q5-mandated for vllm_multiturn; TP>1 needs ALLOW_MULTITURN_ROLLOUT_TP_GT1=1)"

set +e
"$PYTHON_BIN" -m recipe.dapo.src.main_dapo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.on_max_gen_batches=${filter_groups_on_max_gen_batches} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=${SFT_CKPT} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.foreach="${ACTOR_ADAMW_FOREACH:-False}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}" \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.checkpoint.contents="${checkpoint_contents_hydra}" \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD:-False}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-False}" \
    actor_rollout_ref.rollout.name=vllm_multiturn \
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens} \
    actor_rollout_ref.rollout.multi_turn.max_tool_rounds="${max_tool_rounds}" \
    actor_rollout_ref.rollout.multi_turn.max_tool_calls_per_turn=1 \
    actor_rollout_ref.rollout.multi_turn.stop_after_first_tool_call="${STOP_AFTER_FIRST_TOOL_CALL:-True}" \
    actor_rollout_ref.rollout.multi_turn.max_context_tokens=${rollout_max_model_len} \
    actor_rollout_ref.rollout.multi_turn.context_guard_tokens=${mt_context_guard_tokens} \
    actor_rollout_ref.rollout.multi_turn.batch_rounds="${BATCH_ROUNDS:-True}" \
    actor_rollout_ref.rollout.multi_turn.scene_parser_backend=rule \
    actor_rollout_ref.rollout.multi_turn.scene_facts_refiner_backend=none \
    actor_rollout_ref.rollout.multi_turn.detector_backend="${DETECTOR_BACKEND:-none}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${rollout_tp}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}" \
    actor_rollout_ref.rollout.limit_mm_per_prompt.image="${LIMIT_IMAGE_PER_PROMPT:-1}" \
    actor_rollout_ref.rollout.limit_mm_per_prompt.video="${LIMIT_VIDEO_PER_PROMPT:-0}" \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}" \
    actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD:-True}" \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='LNSDL-DAPO' \
    trainer.experiment_name=${EXP_NAME} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-2}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ:-50}" \
    trainer.test_freq="${TEST_FREQ:-50}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS:--1}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-4}" \
    "$@" \
    2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e
echo "$status" > "$EXIT_FILE"
log "finished with status=${status}"
exit "$status"
