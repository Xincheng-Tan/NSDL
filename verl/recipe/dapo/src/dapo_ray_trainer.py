# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from pprint import pprint
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage, AdvantageEstimator
from verl.trainer.ppo import lnsdl_advantage
from verl.trainer.ppo.metric_utils import (compute_data_metrics, compute_throughout_metrics, compute_timing_metrics,
                                           reduce_metrics)


def _reward_extra_metrics(reward_extra_infos_dict: dict[str, list]) -> dict[str, float]:
    """Summarize scalar reward breakdowns for console/wandb red-line monitoring."""
    metrics: dict[str, float] = {}
    for key, values in reward_extra_infos_dict.items():
        try:
            arr = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if arr.size == 0:
            continue
        metrics[f"reward/{key}/mean"] = float(np.mean(arr))
        if key in {"score", "acc", "r_goal", "r_faithful", "r_struct", "n_tool_calls", "stacked", "nonterm"}:
            metrics[f"reward/{key}/min"] = float(np.min(arr))
            metrics[f"reward/{key}/max"] = float(np.max(arr))
    return metrics


def _lnsdl_skip_reward_metrics(non_tensor_batch: dict) -> dict[str, float]:
    """Task-blind P2 diagnostics split by structural skip-decision samples."""
    if "skip_sample" not in non_tensor_batch:
        return {}

    try:
        skip_arr = np.asarray(non_tensor_batch["skip_sample"], dtype=np.float32)
    except (TypeError, ValueError):
        return {}
    if skip_arr.size == 0:
        return {}

    valid = skip_arr >= 0
    if not np.any(valid):
        return {}

    skip_mask = valid & (skip_arr > 0.5)
    nonskip_mask = valid & (skip_arr <= 0.5)
    metrics = {
        "train/skip_sample/valid_responses": float(np.sum(valid)),
        "train/skip_sample/skip_responses": float(np.sum(skip_mask)),
        "train/skip_sample/nonskip_responses": float(np.sum(nonskip_mask)),
        "train/skip_sample/skip_response_frac": float(np.mean(skip_mask[valid])),
    }

    split_keys = ("score", "acc", "r_task", "r_goal", "r_faithful", "r_struct", "n_tool_calls")
    std_keys = {"score", "acc", "r_goal", "n_tool_calls"}
    for key in split_keys:
        if key not in non_tensor_batch:
            continue
        try:
            arr = np.asarray(non_tensor_batch[key], dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if arr.shape[0] != skip_arr.shape[0]:
            continue
        for name, mask in (("skip", skip_mask), ("nonskip", nonskip_mask)):
            if not np.any(mask):
                continue
            vals = arr[mask]
            metrics[f"reward/{name}/{key}/mean"] = float(np.mean(vals))
            if key in std_keys:
                metrics[f"reward/{name}/{key}/std"] = float(np.std(vals))

    return metrics


def _lnsdl_skip_advantage_metrics(batch: DataProto) -> dict[str, float]:
    """Measure how much token-level advantage budget lands on skip samples."""
    if "skip_sample" not in batch.non_tensor_batch or "advantages" not in batch.batch:
        return {}

    try:
        skip_arr = np.asarray(batch.non_tensor_batch["skip_sample"], dtype=np.float32)
    except (TypeError, ValueError):
        return {}

    advantages = batch.batch["advantages"]
    if skip_arr.shape[0] != advantages.shape[0]:
        return {}

    response_length = batch.batch["responses"].shape[-1]
    response_mask = batch.batch["attention_mask"][:, -response_length:].to(advantages.device)
    denom = response_mask.sum(dim=-1).clamp_min(1)
    seq_abs_adv = (advantages.abs() * response_mask).sum(dim=-1) / denom
    seq_adv = (advantages * response_mask).sum(dim=-1) / denom

    skip_tensor = torch.as_tensor(skip_arr, device=advantages.device)
    valid_mask = skip_tensor >= 0
    if not bool(valid_mask.any().item()):
        return {}

    skip_mask = valid_mask & (skip_tensor > 0.5)
    nonskip_mask = valid_mask & (skip_tensor <= 0.5)
    metrics: dict[str, float] = {}

    for name, mask in (("skip", skip_mask), ("nonskip", nonskip_mask)):
        if not bool(mask.any().item()):
            continue
        metrics[f"critic/advantages/{name}_abs_mean"] = float(seq_abs_adv[mask].mean().detach().cpu().item())
        metrics[f"critic/advantages/{name}_mean"] = float(seq_adv[mask].mean().detach().cpu().item())

    total_abs = seq_abs_adv[valid_mask].sum()
    if float(total_abs.detach().cpu().item()) > 0:
        skip_abs = seq_abs_adv[skip_mask].sum() if bool(skip_mask.any().item()) else total_abs.new_tensor(0.0)
        metrics["critic/advantages/skip_abs_share"] = float((skip_abs / total_abs).detach().cpu().item())

    return metrics


def _ordered_unique(values) -> list:
    """Return unique prompt ids in first-seen order."""
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _select_prompt_groups(batch: DataProto,
                          prompt_uids: list,
                          *,
                          max_prompts: int | None = None,
                          exclude_uids: set | None = None) -> tuple[DataProto | None, list]:
    """Select complete response groups for the requested prompt ids."""
    if batch is None or len(batch) == 0:
        return None, []

    exclude_uids = exclude_uids or set()
    uid2idxs = defaultdict(list)
    for idx, uid in enumerate(batch.non_tensor_batch["uid"]):
        uid2idxs[uid].append(idx)

    selected_idxs = []
    selected_uids = []
    seen = set()
    for uid in prompt_uids:
        if uid in seen or uid in exclude_uids or uid not in uid2idxs:
            continue
        selected_idxs.extend(uid2idxs[uid])
        selected_uids.append(uid)
        seen.add(uid)
        if max_prompts is not None and len(selected_uids) >= max_prompts:
            break

    if not selected_idxs:
        return None, []
    return batch[selected_idxs], selected_uids


def _concat_nonempty_batches(batches: list[DataProto | None]) -> DataProto | None:
    batches = [batch for batch in batches if batch is not None and len(batch) > 0]
    if not batches:
        return None
    if len(batches) == 1:
        return batches[0]
    return DataProto.concat(batches)


_ACC_FIRST_SCORE_METRICS = {"acc_first_score", "acc_then_score", "acc_score_fallback", "composite"}


def _prompt_group_variance_uids(non_tensor_batch: dict, metric_name: str) -> tuple[list, dict]:
    """Return prompt uids whose rollout group has non-zero metric variance."""
    if metric_name not in non_tensor_batch:
        raise KeyError(f"filter_groups metric {metric_name!r} is not available in non_tensor_batch")

    prompt_uid2metric_vals = defaultdict(list)
    for uid, metric_val in zip(non_tensor_batch["uid"], non_tensor_batch[metric_name]):
        prompt_uid2metric_vals[uid].append(metric_val)

    prompt_uid2metric_std = {
        prompt_uid: np.std(metric_vals)
        for prompt_uid, metric_vals in prompt_uid2metric_vals.items()
    }
    kept_prompt_uids = [
        uid for uid, std in prompt_uid2metric_std.items()
        if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
    ]
    return kept_prompt_uids, prompt_uid2metric_std


def _acc_first_score_filter_uids(non_tensor_batch: dict) -> tuple[list, list, dict[str, float]]:
    """T3 filter: acc-variance prompts first, score-variance prompts as fallback."""
    acc_uids, acc_std = _prompt_group_variance_uids(non_tensor_batch, "acc")
    score_uids, score_std = _prompt_group_variance_uids(non_tensor_batch, "score")

    acc_uid_set = set(acc_uids)
    score_fallback_uids = [uid for uid in score_uids if uid not in acc_uid_set]
    stats = {
        "candidate_prompts": float(len(acc_std)),
        "acc_var_candidate_prompts": float(len(acc_uids)),
        "score_var_candidate_prompts": float(len(score_uids)),
        "score_fallback_candidate_prompts": float(len(score_fallback_uids)),
    }
    return acc_uids, score_fallback_uids, stats


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        fallback_pool_batch = None
        score_fallback_pool_batch = None
        num_candidate_prompt_groups = 0
        num_acc_var_candidate_prompt_groups = 0
        num_score_var_candidate_prompt_groups = 0
        num_score_fallback_candidate_prompt_groups = 0
        num_score_fallback_prompt_groups = 0
        selected_acc_filter_uids = set()
        selected_score_filter_uids = set()
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation. L-NSDL multi-turn rollout also
                # needs the opaque sample payload to rebuild scene_facts/tools.
                gen_non_tensor_keys = ['raw_prompt_ids']
                if 'multi_modal_inputs' in new_batch.non_tensor_batch.keys():
                    gen_non_tensor_keys.extend(['multi_modal_data', 'multi_modal_inputs'])
                if 'lnsdl_sample' in new_batch.non_tensor_batch.keys():
                    gen_non_tensor_keys.append('lnsdl_sample')
                gen_batch = new_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=gen_non_tensor_keys,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch['uid'] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    with _timer('reward', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result['reward_tensor']
                            reward_extra_infos_dict = reward_result['reward_extra_info']
                        except Exception as e:
                            print(f'Error in reward_fn: {e}')
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch['token_level_scores'] = reward_tensor

                        print(f'{list(reward_extra_infos_dict.keys())=}')
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({
                                k: np.array(v) for k, v in reward_extra_infos_dict.items()
                            })
                            metrics.update(_reward_extra_metrics(reward_extra_infos_dict))

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch,
                                                                     kl_ctrl=self.kl_ctrl_in_reward,
                                                                     kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(
                                kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch['token_level_rewards'] = new_batch.batch['token_level_scores']

                    actual_prompt_in_batch = None
                    # Protocol E (docs/plan.md §P5.3'.2b): the PRE-filter dump.
                    # Dynamic sampling below keeps only prompt groups that have
                    # accuracy variance, so an advantage-collapse rate measured on
                    # what survives is near zero on the task axis by construction
                    # -- i.e. exactly blind to the collapse CFA is supposed to
                    # handle. This site sees the whole generation batch, before
                    # any selection. No-op unless LNSDL_ADV_DUMP is set.
                    lnsdl_advantage.log_axis_batch(
                        new_batch.non_tensor_batch,
                        new_batch.non_tensor_batch['uid'],
                        extra={'global_step': int(self.global_steps),
                               'gen_batch': int(num_gen_batches)},
                        stage='pre_filter',
                    )
                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size, we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch['token_level_scores'].sum(
                                dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch['token_level_scores'].sum(
                                dim=-1).numpy()

                        unfiltered_batch = new_batch
                        fallback_pool_batch = _concat_nonempty_batches([fallback_pool_batch, unfiltered_batch])
                        prompt_bsz = self.config.data.train_batch_size

                        if metric_name in _ACC_FIRST_SCORE_METRICS:
                            acc_uids, score_fallback_uids, filter_stats = _acc_first_score_filter_uids(
                                unfiltered_batch.non_tensor_batch)
                            num_candidate_prompt_groups += int(filter_stats["candidate_prompts"])
                            num_acc_var_candidate_prompt_groups += int(filter_stats["acc_var_candidate_prompts"])
                            num_score_var_candidate_prompt_groups += int(filter_stats["score_var_candidate_prompts"])
                            num_score_fallback_candidate_prompt_groups += int(
                                filter_stats["score_fallback_candidate_prompts"])

                            acc_batch, acc_selected_uids = _select_prompt_groups(unfiltered_batch, acc_uids)
                            batch = _concat_nonempty_batches([batch, acc_batch])
                            num_prompt_in_batch += len(acc_selected_uids)
                            selected_acc_filter_uids.update(acc_selected_uids)

                            score_fallback_batch, _ = _select_prompt_groups(unfiltered_batch, score_fallback_uids)
                            score_fallback_pool_batch = _concat_nonempty_batches(
                                [score_fallback_pool_batch, score_fallback_batch])
                        else:
                            kept_prompt_uids, prompt_uid2metric_std = _prompt_group_variance_uids(
                                unfiltered_batch.non_tensor_batch, metric_name)
                            num_candidate_prompt_groups += len(prompt_uid2metric_std)

                            num_prompt_in_batch += len(kept_prompt_uids)

                            new_batch, _ = _select_prompt_groups(unfiltered_batch, kept_prompt_uids)
                            batch = _concat_nonempty_batches([batch, new_batch])

                        if num_prompt_in_batch < prompt_bsz:
                            print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f'{num_gen_batches=}. Keep generating...')
                                continue

                            if metric_name in _ACC_FIRST_SCORE_METRICS:
                                existing_uids = set()
                                if batch is not None and len(batch) > 0:
                                    existing_uids = set(_ordered_unique(batch.non_tensor_batch["uid"]))
                                needed_prompts = prompt_bsz - num_prompt_in_batch
                                score_batch, score_uids = _select_prompt_groups(
                                    score_fallback_pool_batch,
                                    _ordered_unique(score_fallback_pool_batch.non_tensor_batch["uid"])
                                    if score_fallback_pool_batch is not None else [],
                                    max_prompts=needed_prompts,
                                    exclude_uids=existing_uids,
                                )
                                batch = _concat_nonempty_batches([batch, score_batch])
                                num_prompt_in_batch += len(score_uids)
                                num_score_fallback_prompt_groups += len(score_uids)
                                selected_score_filter_uids.update(score_uids)
                                if score_uids:
                                    metrics["train/filter_groups/score_fallback"] = 1.0
                                    metrics["train/filter_groups/score_fallback_prompts"] = float(len(score_uids))

                            if num_prompt_in_batch >= prompt_bsz:
                                # score fallback was sufficient; no zero-variance padding needed.
                                pass
                            else:
                                on_max_gen_batches = self.config.algorithm.filter_groups.get(
                                    "on_max_gen_batches", "raise")
                                if on_max_gen_batches == "raise":
                                    raise ValueError(
                                        f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
                                    )

                                if on_max_gen_batches != "pad":
                                    raise ValueError(
                                        f"Unsupported algorithm.filter_groups.on_max_gen_batches={on_max_gen_batches!r}. "
                                        "Expected 'pad' or 'raise'."
                                    )

                                existing_uids = set()
                                if batch is not None and len(batch) > 0:
                                    existing_uids = set(_ordered_unique(batch.non_tensor_batch["uid"]))
                                needed_prompts = prompt_bsz - num_prompt_in_batch
                                fallback_batch, fallback_uids = _select_prompt_groups(
                                    fallback_pool_batch,
                                    _ordered_unique(fallback_pool_batch.non_tensor_batch["uid"]),
                                    max_prompts=needed_prompts,
                                    exclude_uids=existing_uids,
                                )
                                kept_before_fallback = num_prompt_in_batch
                                batch = _concat_nonempty_batches([batch, fallback_batch])
                                num_prompt_in_batch += len(fallback_uids)
                                metrics["train/filter_groups/fallback"] = 1.0
                                metrics["train/filter_groups/fallback_prompts"] = float(len(fallback_uids))
                                metrics["train/filter_groups/kept_prompts_before_fallback"] = float(
                                    kept_before_fallback)
                                metrics["train/filter_groups/accepted_prompts"] = float(num_prompt_in_batch)
                                print(
                                    f"filter_groups reached {max_num_gen_batches=} with "
                                    f"{kept_before_fallback} kept prompts out of "
                                    f"{num_candidate_prompt_groups} candidates; padding "
                                    f"{len(fallback_uids)} zero-variance prompt groups."
                                )

                                if batch is None or len(batch) == 0 or num_prompt_in_batch < prompt_bsz:
                                    raise ValueError(
                                        f'{num_gen_batches=} >= {max_num_gen_batches=} but only '
                                        f'{num_prompt_in_batch}/{prompt_bsz} prompt groups were available after fallback.'
                                    )

                        # Align the batch
                        traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                        batch = batch[:traj_bsz]
                        actual_prompt_in_batch = len(_ordered_unique(batch.non_tensor_batch["uid"]))
                        final_prompt_uids = set(_ordered_unique(batch.non_tensor_batch["uid"]))
                        if "acc" in batch.non_tensor_batch:
                            final_acc_var_uids, _ = _prompt_group_variance_uids(batch.non_tensor_batch, "acc")
                            metrics["train/filter_groups/final_acc_var_prompts"] = float(len(final_acc_var_uids))
                            metrics["train/filter_groups/final_acc_var_frac"] = (
                                float(len(final_acc_var_uids)) / float(actual_prompt_in_batch)
                                if actual_prompt_in_batch > 0 else 0.0
                            )
                        if "score" in batch.non_tensor_batch:
                            final_score_var_uids, _ = _prompt_group_variance_uids(batch.non_tensor_batch, "score")
                            metrics["train/filter_groups/final_score_var_prompts"] = float(len(final_score_var_uids))
                            metrics["train/filter_groups/final_score_var_frac"] = (
                                float(len(final_score_var_uids)) / float(actual_prompt_in_batch)
                                if actual_prompt_in_batch > 0 else 0.0
                            )
                        if metric_name in _ACC_FIRST_SCORE_METRICS:
                            final_acc_selected = len(final_prompt_uids & selected_acc_filter_uids)
                            final_score_selected = len(final_prompt_uids & selected_score_filter_uids)
                            metrics["train/filter_groups/acc_var_candidate_prompts"] = float(
                                num_acc_var_candidate_prompt_groups)
                            metrics["train/filter_groups/score_var_candidate_prompts"] = float(
                                num_score_var_candidate_prompt_groups)
                            metrics["train/filter_groups/score_fallback_candidate_prompts"] = float(
                                num_score_fallback_candidate_prompt_groups)
                            metrics["train/filter_groups/acc_first_prompts"] = float(final_acc_selected)
                            metrics["train/filter_groups/score_fallback_prompts"] = float(final_score_selected)
                            metrics["train/filter_groups/score_fallback_used_prompts"] = float(
                                num_score_fallback_prompt_groups)
                            metrics["train/filter_groups/acc_first_frac"] = (
                                float(final_acc_selected) / float(actual_prompt_in_batch)
                                if actual_prompt_in_batch > 0 else 0.0
                            )

                    reward_metric_keys = (
                        "score", "acc", "r_task", "r_goal", "r_faithful", "r_struct",
                        "r_dense_raw", "task_fail_dense_scale", "p_format", "p_nonterm", "p_cost",
                        "n_tool_calls", "stacked", "nonterm", "skip_sample",
                    )
                    batch_reward_infos = {
                        key: batch.non_tensor_batch[key]
                        for key in reward_metric_keys
                        if key in batch.non_tensor_batch
                    }
                    if batch_reward_infos:
                        # Multi-gen DAPO overwrites metrics while sampling; re-log the
                        # final update batch after filtering/padding/alignment.
                        metrics.update(_reward_extra_metrics(batch_reward_infos))
                        metrics.update(_lnsdl_skip_reward_metrics(batch_reward_infos))
                    metrics.setdefault("train/filter_groups/fallback", 0.0)
                    metrics.setdefault("train/filter_groups/fallback_prompts", 0.0)
                    metrics.setdefault("train/filter_groups/score_fallback", 0.0)
                    metrics.setdefault("train/filter_groups/score_fallback_prompts", 0.0)
                    metrics.setdefault("train/filter_groups/score_fallback_used_prompts", 0.0)
                    metrics.setdefault(
                        "train/filter_groups/kept_prompts_before_fallback",
                        float(actual_prompt_in_batch if actual_prompt_in_batch is not None else num_prompt_in_batch),
                    )
                    if actual_prompt_in_batch is not None:
                        metrics["train/filter_groups/accepted_prompts"] = float(actual_prompt_in_batch)
                        metrics["train/filter_groups/candidate_prompts"] = float(num_candidate_prompt_groups)
                    else:
                        metrics.setdefault("train/filter_groups/accepted_prompts", float(num_prompt_in_batch))
                        metrics.setdefault("train/filter_groups/candidate_prompts", float(num_candidate_prompt_groups))

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)
                    metrics.update(_lnsdl_skip_advantage_metrics(batch))
                    # Collapse diagnostics for the per-axis advantage modes
                    # (docs/plan.md §P5.3'.2b): ACR / per-axis mask rate /
                    # within-group std / axis grad share. Absent for the stock
                    # estimators, so this is a no-op there.
                    if 'lnsdl_adv_diagnostics' in batch.meta_info:
                        metrics.update(lnsdl_advantage.flat_metrics(
                            batch.meta_info.pop('lnsdl_adv_diagnostics')))

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                            (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or
                                                              self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                fallback_pool_batch = None
                score_fallback_pool_batch = None
                num_candidate_prompt_groups = 0
                num_acc_var_candidate_prompt_groups = 0
                num_score_var_candidate_prompt_groups = 0
                num_score_fallback_candidate_prompt_groups = 0
                num_score_fallback_prompt_groups = 0
                selected_acc_filter_uids = set()
                selected_score_filter_uids = set()

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1

        progress_bar.close()
        completed_steps = max(0, self.global_steps - 1)
        raise RuntimeError(
            f"Training dataloader exhausted after completing {completed_steps} optimizer steps, "
            f"before trainer.total_training_steps={self.total_training_steps}. "
            "Increase trainer.total_epochs or disable dataloader-state resume for this continuation. "
            "DAPO dynamic sampling may consume multiple dataloader batches per optimizer step."
        )
