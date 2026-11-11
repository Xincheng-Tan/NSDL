import numpy as np

from recipe.dapo.src.dapo_ray_trainer import (
    _acc_first_score_filter_uids,
    _prompt_group_variance_uids,
)


def test_acc_first_score_filter_prioritizes_acc_variance():
    non_tensor_batch = {
        "uid": np.array(["a", "a", "b", "b", "c", "c", "d", "d"], dtype=object),
        "acc": np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "score": np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.3, 0.7, 0.2], dtype=np.float32),
    }

    acc_uids, score_fallback_uids, stats = _acc_first_score_filter_uids(non_tensor_batch)

    assert acc_uids == ["a"]
    assert score_fallback_uids == ["b", "d"]
    assert stats["candidate_prompts"] == 4.0
    assert stats["acc_var_candidate_prompts"] == 1.0
    assert stats["score_var_candidate_prompts"] == 3.0
    assert stats["score_fallback_candidate_prompts"] == 2.0


def test_prompt_group_variance_keeps_singletons_for_backward_compatibility():
    non_tensor_batch = {
        "uid": np.array(["a", "a", "b"], dtype=object),
        "acc": np.array([1.0, 1.0, 0.0], dtype=np.float32),
    }

    kept_uids, std_by_uid = _prompt_group_variance_uids(non_tensor_batch, "acc")

    assert kept_uids == ["b"]
    assert std_by_uid["a"] == 0.0
    assert std_by_uid["b"] == 0.0
