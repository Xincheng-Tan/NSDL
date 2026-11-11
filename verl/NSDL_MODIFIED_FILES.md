# verl: what is upstream and what is ours

`verl/` is a vendored copy of [verl](https://github.com/volcengine/verl)
(Apache-2.0, `verl/version/version` = 0.2.0.dev at import time) with the DAPO
recipe. Its `tests/`, `docs/`, `examples/` and `docker/` trees are omitted here
to keep the review repository small; nothing in them is used by NSDL.

Files added or modified for NSDL (everything else is upstream):

| Path | Change |
|---|---|
| `verl/utils/reward_score/lnsdl.py` | NSDL reward: `R_task` / `R_goal` / `R_goal_plan` and the two penalties, all read against benchmark ground truth; env knobs `LNSDL_W_*`, `LNSDL_P_*`, `LNSDL_R_TASK_SOURCE`, `LNSDL_R_PLAN_SOURCE` |
| `verl/utils/reward_score/__init__.py` | registers the `lnsdl` and `locus_bench` data sources |
| `verl/utils/reward_score/enactgraph.py` | earlier reward variants kept for reference; not used by the reported runs |
| `verl/utils/reward_score/utils/**`, `eval_gt.py`, `eval_viki_2.py` | copy of the official VIKI-L2 plan checker (from the VIKI-R repository, Apache-2.0) used by the reward |
| `verl/trainer/ppo/lnsdl_advantage.py` | advantage modes (`LNSDL_ADV_MODE`); the reported runs use `scalar`, i.e. plain GRPO group normalisation |
| `verl/trainer/ppo/core_algos.py`, `metric_utils.py`, `ray_trainer.py` | hook the advantage module and per-axis logging into the trainer |
| `verl/workers/rollout/vllm_rollout/vllm_multiturn_rollout.py` | multi-turn tool rollout: `detect` / `schedule` are answered by `nsdl.rl.tool_loop_adapter` inside the rollout |
| `verl/workers/actor/dp_actor.py`, `verl/workers/fsdp_workers.py`, `verl/models/transformers/qwen2_vl.py`, `verl/utils/tokenizer.py` | Qwen2.5-VL multi-image position ids, image-token accounting, offload settings |
| `recipe/dapo/src/dapo_ray_trainer.py`, `recipe/dapo/src/config/dapo_trainer.yaml` | DAPO recipe wiring for the multi-turn rollout |

Internal identifiers such as `lnsdl` / `LNSDL_*` / `enactgraph` are historical
code names for what the paper calls NSDL.
