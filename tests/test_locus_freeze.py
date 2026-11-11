"""v1.0 冻结章的判据（docs/benchmark.md §9「共同冻结门」）。

盖章脚本唯一的价值是它**会拒绝**。写 `bench_version=v1.0` 本身是一行赋值；难的
是保证这一行只在四道门都从**产物**上过了的时候才发生，而且冻结的那份数据/权重
就是被审计过的那一份。下面按"每种能让冻结变成谎言的方式"逐条钉：

1. 构造门里任意一键翻假（含 Φ 一致率跌破 0.995、S4 成对残缺）⇒ 拒绝；
2. 规模门未过、或 `lookup` 又出现在 train 侧（工具集 2026-07-22 已冻结为 5 工具）⇒ 拒绝；
3. §4.6 用的是 pilot 阶段读数（`stage != freeze`），或审计跑的 train parquet 与将被
   冻结的那份哈希不同 ⇒ 拒绝——否则"方差门过了"说的是另一份数据；
4. §7-7 的 `undetermined`/`saturated` 被当成通过 ⇒ 拒绝（§7.9 不变式 1：覆盖不足
   不是通过）；全零格没有构造证据裁定 ⇒ 拒绝（裁定只能是"真难度"，删样本不是修法）；
5. §7.9 的近零观察名单必须从产物**算出来**而不是抄进去，且与全零格并列披露；
6. 盖章要同时落在 `config` 与 `audit_report` 两处（下游两条读法都存在），并且
   冻结后数据被改一个字节，`--verify-frozen` 必须报出来——P9.0a 起跑前靠它。
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from freeze_locus_bench import (  # noqa: E402
    FROZEN_VERSION,
    NEAR_ZERO_BEST_THRESHOLD,
    PENDING_SENTINEL,
    GateFailure,
    build_disclosure,
    build_freeze_manifest,
    check_construction_gates,
    check_rl_variance_gate,
    check_scale_gate,
    check_strong_baseline_gate,
    near_zero_watchlist,
    stamp_pilot_report,
    verify_freeze_preconditions,
    verify_frozen,
)


# --------------------------------------------------------------------------- #
# fixtures: ramp63k 实际报的形状，全部干净
# --------------------------------------------------------------------------- #
CLEAN_AUDIT_REPORT = {
    "bench_version": "v0-pilot",
    "n_instances": 6860,
    "realization_balance": {"ok": True},
    "axis_orthogonality": {"ok": True},
    "family_train_coverage": {"ok": True},
    "composition_holdout": {"ok": True},
    "goal_text_recoverability": {"goals_checked": 11372, "n_violations": 0, "ok": True},
    "dual_validator": {"simenv_pass_rate": 1.0, "phi_checked": 6860, "phi_ok": 6860,
                       "phi_agreement": 1.0, "gate": 0.995, "ok": True},
    "s4_counterfactual_pairing": {"pairs_checked": 30, "incomplete_pairs": 0, "failures": 0,
                                  "post_open_observation_claims": 0, "ok": True},
    "operator_train_support": {"structural_ok": True, "scale_gate_ok": True, "lookup_count": 0},
    "strong_baseline_prerun": {"gate": "benchmark.md §7-7"},
    "eval_fairness": {"ok": True},
}

TRAIN_SHA = "e6a18908b4778091a4a3766e51f33b38ed8f4ac721d83e5c5bffdfe1ff6f8943"

CLEAN_RL_AUDIT = {
    "schema": "rl_variance_audit/1",
    "stage": "freeze",
    "gates": {"ok": True, "failed": []},
    "provenance": {
        "dataset": {"path": "data/lnsdl_bench/ramp63k/train/data.parquet",
                    "sha256": TRAIN_SHA, "content_hashed": True},
        "checkpoint": {"sha256": "314b26a7", "content_hashed": True},
        "reward": {"sha256": "07ba55b7"},
    },
    "diagnostics_all": {"acr": 0.2111},
    "totals": {"n_prompts": 900, "n_rollouts": 7200},
}

CLEAN_STRONG_BASELINE = {
    "strong_baseline_prerun": {
        "gate": "benchmark.md §7-7",
        "zero_coverage_splits": {},
        "coverage_incomplete_rows": {},
        "axis_monotonicity": {"S": {"monotone_non_increasing": True},
                              "M": {"monotone_non_increasing": True}},
        "family_cell_verdicts": {
            "cells": {
                "F10_s0m0": {"verdict": "discriminative", "best": 1.0, "n_rows_qualifying": 8},
                "F5_s2m3": {"verdict": "discriminative", "best": 0.05, "n_rows_qualifying": 8},
                "F5_s3m3": {"verdict": "discriminative", "best": 0.05, "n_rows_qualifying": 8},
                "F8_s1m3": {"verdict": "discriminative", "best": 0.05, "n_rows_qualifying": 8},
                "F9_s0m4": {"verdict": "discriminative", "best": 0.05, "n_rows_qualifying": 8},
                "F8_s0m3": {"verdict": "all_zero", "best": 0.0, "n_rows_qualifying": 8},
            }
        },
        "summary": {
            "n_rows": 8,
            "grid": {"saturated": [], "all_zero": [], "undetermined": [], "n_cells": 21},
            "family_cell": {"saturated": [], "all_zero": ["F8_s0m3"], "undetermined": [],
                            "n_cells": 49},
            "all_zero_adjudication": {"verdict": "real_difficulty", "cells": ["F8_s0m3"]},
            "ok": True,
            "blocking_reason": None,
        },
    },
    "eval_fairness": {
        "ok": True,
        "all_models_both_channels": True,
        "api_sources_present": ["relay"],
        "official_api_rerun_present": False,
    },
}


def _report(audit_report=None):
    return {"config": {}, "audit_report": copy.deepcopy(audit_report or CLEAN_AUDIT_REPORT)}


def _verify(audit_report=None, rl=None, sb=None, train_sha=TRAIN_SHA):
    return verify_freeze_preconditions(
        pilot_report=_report(audit_report),
        rl_audit=copy.deepcopy(rl or CLEAN_RL_AUDIT),
        strong_baseline_audit=copy.deepcopy(sb or CLEAN_STRONG_BASELINE),
        train_parquet_sha256=train_sha,
    )


# --------------------------------------------------------------------------- #
# 基线：干净产物四门全过
# --------------------------------------------------------------------------- #
def test_clean_artifacts_pass_all_four_gates():
    verdict = _verify()
    assert verdict["ok"] is True
    assert verdict["failed"] == []
    assert set(verdict["gates"]) == {
        "gate1_construction", "gate2_scale", "gate3_rl_variance", "gate4_strong_baseline",
    }


# --------------------------------------------------------------------------- #
# 门1 构造门
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["realization_balance", "axis_orthogonality",
                                 "family_train_coverage", "composition_holdout",
                                 "goal_text_recoverability"])
def test_any_construction_key_false_blocks_the_stamp(key):
    audit = copy.deepcopy(CLEAN_AUDIT_REPORT)
    audit[key] = {"ok": False}
    assert _verify(audit)["failed"] == ["gate1_construction"]


def test_phi_agreement_below_gate_blocks():
    """§7-6 的门是 0.995：Φ 重建一致率是"意图 sketch 可机械重建"的唯一证据。"""
    audit = copy.deepcopy(CLEAN_AUDIT_REPORT)
    audit["dual_validator"] = {"simenv_pass_rate": 1.0, "phi_agreement": 0.99,
                               "gate": 0.995, "ok": True}
    assert not check_construction_gates(audit)["ok"]


def test_simenv_pass_rate_must_be_exactly_one():
    audit = copy.deepcopy(CLEAN_AUDIT_REPORT)
    audit["dual_validator"] = dict(audit["dual_validator"], simenv_pass_rate=0.999)
    assert not check_construction_gates(audit)["ok"]


@pytest.mark.parametrize("field", ["incomplete_pairs", "failures", "post_open_observation_claims"])
def test_s4_pairing_residue_blocks_even_when_ok_flag_is_true(field):
    """`ok: True` 与残留计数并存时，以计数为准——S4 成对样式是轴 S 的反捷径根据。"""
    audit = copy.deepcopy(CLEAN_AUDIT_REPORT)
    audit["s4_counterfactual_pairing"][field] = 1
    assert not check_construction_gates(audit)["ok"]


# --------------------------------------------------------------------------- #
# 门2 规模门 + 工具集冻结
# --------------------------------------------------------------------------- #
def test_scale_gate_not_reached_blocks():
    audit = copy.deepcopy(CLEAN_AUDIT_REPORT)
    audit["operator_train_support"]["scale_gate_ok"] = False
    assert _verify(audit)["failed"] == ["gate2_scale"]


def test_lookup_reappearing_in_train_blocks():
    """工具集 2026-07-22 冻结为 5 工具，lookup 永久移除；train 侧出现即冻结面不一致。"""
    audit = copy.deepcopy(CLEAN_AUDIT_REPORT)
    audit["operator_train_support"]["lookup_count"] = 3
    assert not check_scale_gate(audit)["ok"]


# --------------------------------------------------------------------------- #
# 门3 §4.6 RL 组内方差
# --------------------------------------------------------------------------- #
def test_pilot_stage_variance_audit_cannot_close_the_freeze_gate():
    """§4.6 明说 pilot 每 cell 30 prompts 的读数不能替代冻结审计。"""
    rl = copy.deepcopy(CLEAN_RL_AUDIT)
    rl["stage"] = "pilot"
    assert _verify(rl=rl)["failed"] == ["gate3_rl_variance"]


def test_variance_gate_failure_blocks():
    rl = copy.deepcopy(CLEAN_RL_AUDIT)
    rl["gates"] = {"ok": False, "failed": ["axis_A_per_cell_S1S2"]}
    assert not check_rl_variance_gate(rl, train_parquet_sha256=TRAIN_SHA)["ok"]


def test_variance_audit_must_have_run_on_the_data_being_frozen():
    """审计跑在别的 parquet 上时，"方差门过了"说的是另一份 benchmark。"""
    result = check_rl_variance_gate(copy.deepcopy(CLEAN_RL_AUDIT),
                                    train_parquet_sha256="deadbeef")
    assert not result["ok"]
    failed = [item["check"] for item in result["checks"] if not item["ok"]]
    assert failed == ["audited_train_parquet_is_the_frozen_one"]


def test_unhashed_provenance_blocks():
    """`content_hashed=false` 意味着 provenance 只是路径字符串，冻结不接受。"""
    rl = copy.deepcopy(CLEAN_RL_AUDIT)
    rl["provenance"]["checkpoint"]["content_hashed"] = False
    assert not check_rl_variance_gate(rl, train_parquet_sha256=TRAIN_SHA)["ok"]


# --------------------------------------------------------------------------- #
# 门4 §7-7 / §7-8
# --------------------------------------------------------------------------- #
def test_undetermined_cells_are_not_a_pass():
    """§7.9 不变式 1：覆盖不足 = 不知道，不是通过。"""
    sb = copy.deepcopy(CLEAN_STRONG_BASELINE)
    sb["strong_baseline_prerun"]["summary"]["grid"]["undetermined"] = ["s2m3"]
    assert _verify(sb=sb)["failed"] == ["gate4_strong_baseline"]


def test_saturated_cells_block():
    sb = copy.deepcopy(CLEAN_STRONG_BASELINE)
    sb["strong_baseline_prerun"]["summary"]["family_cell"]["saturated"] = ["F1_s0m0"]
    assert not check_strong_baseline_gate(CLEAN_AUDIT_REPORT, sb)["ok"]


def test_all_zero_cell_without_construction_adjudication_blocks():
    """全零格允许留在 benchmark，但必须先有构造证据裁定；无裁定不能盖章。"""
    sb = copy.deepcopy(CLEAN_STRONG_BASELINE)
    sb["strong_baseline_prerun"]["summary"]["all_zero_adjudication"] = {}
    assert not check_strong_baseline_gate(CLEAN_AUDIT_REPORT, sb)["ok"]


def test_all_zero_adjudication_must_cover_exactly_the_all_zero_cells():
    """裁了别的格 = 没裁这个格。"""
    sb = copy.deepcopy(CLEAN_STRONG_BASELINE)
    sb["strong_baseline_prerun"]["summary"]["all_zero_adjudication"]["cells"] = ["F1_s0m0"]
    assert not check_strong_baseline_gate(CLEAN_AUDIT_REPORT, sb)["ok"]


def test_pending_sentinel_in_pilot_report_blocks():
    """`pending:P7.5` 还在，就说明门4 的回填从未发生。"""
    audit = copy.deepcopy(CLEAN_AUDIT_REPORT)
    audit["strong_baseline_prerun"] = PENDING_SENTINEL
    assert _verify(audit)["failed"] == ["gate4_strong_baseline"]


def test_zero_coverage_split_blocks():
    """零覆盖 run 会让"全零交集"判据静默失效（§7.7 实测：49 格命中 0 而非 2）。"""
    sb = copy.deepcopy(CLEAN_STRONG_BASELINE)
    sb["strong_baseline_prerun"]["zero_coverage_splits"] = {"claude-sonnet-5__image": ["ood_x"]}
    assert not check_strong_baseline_gate(CLEAN_AUDIT_REPORT, sb)["ok"]


# --------------------------------------------------------------------------- #
# §7.9 披露口径
# --------------------------------------------------------------------------- #
def test_near_zero_watchlist_is_computed_not_hardcoded():
    watchlist = near_zero_watchlist(CLEAN_STRONG_BASELINE["strong_baseline_prerun"])
    assert [item["cell"] for item in watchlist] == ["F5_s2m3", "F5_s3m3", "F8_s1m3", "F9_s0m4"]
    assert all(item["best"] <= NEAR_ZERO_BEST_THRESHOLD for item in watchlist)


def test_all_zero_cell_is_not_double_counted_in_the_watchlist():
    """全零格单独披露；观察名单是"离全零一个样本"的那一档，两者并列不重叠。"""
    cells = [item["cell"] for item in
             near_zero_watchlist(CLEAN_STRONG_BASELINE["strong_baseline_prerun"])]
    assert "F8_s0m3" not in cells


def test_disclosure_carries_all_zero_watchlist_and_official_rerun_todo():
    disclosure = build_disclosure(CLEAN_STRONG_BASELINE)
    assert disclosure["all_zero_family_cells"] == ["F8_s0m3"]
    assert len(disclosure["near_zero_watchlist"]) == 4
    assert disclosure["official_api_rerun_present"] is False
    # 官方 API 复跑未做 ⇒ 必须以待办形式留在冻结件里，不能被"两行都披露"的说法吞掉。
    assert disclosure["post_freeze_todo"]


def test_disclosure_todo_clears_once_official_rerun_lands():
    sb = copy.deepcopy(CLEAN_STRONG_BASELINE)
    sb["eval_fairness"]["official_api_rerun_present"] = True
    assert build_disclosure(sb)["post_freeze_todo"] == []


# --------------------------------------------------------------------------- #
# 盖章本身
# --------------------------------------------------------------------------- #
def _manifest():
    return build_freeze_manifest(
        verification=_verify(),
        data_block={"root": "data/lnsdl_bench/ramp63k",
                    "splits": {"train": {"path": "data/lnsdl_bench/ramp63k/train/data.parquet",
                                         "bytes": 10, "sha256": TRAIN_SHA}}},
        detector_block={"versioned_checkpoint": {"path": "checkpoints/yolo26l_locus_full34_v1.pt",
                                                 "bytes": 53102430, "sha256": "7259d6aa"},
                        "class_order": {"n_classes": 34},
                        "inference_caliber": {"confidence_threshold": 0.25, "iou_threshold": 0.70,
                                              "max_det": 64}},
        disclosure=build_disclosure(CLEAN_STRONG_BASELINE),
    )


def test_manifest_refuses_to_build_when_a_gate_failed():
    failing = _verify(rl=dict(copy.deepcopy(CLEAN_RL_AUDIT), stage="pilot"))
    with pytest.raises(GateFailure):
        build_freeze_manifest(verification=failing, data_block={}, detector_block={},
                              disclosure={})


def test_stamp_writes_both_read_paths():
    """下游两条读法都存在（top-level config 与 audit_report），只写一处会各说各话。"""
    stamped = stamp_pilot_report(_report(), _manifest())
    assert stamped["config"]["bench_version"] == FROZEN_VERSION
    assert stamped["audit_report"]["bench_version"] == FROZEN_VERSION


def test_stamp_embeds_gate_artifact_pointers():
    freeze = stamp_pilot_report(_report(), _manifest())["audit_report"]["freeze"]
    pointers = freeze["gate_artifacts"]
    assert pointers["rl_variance_audit"].endswith("rl_variance_audit.json")
    assert pointers["strong_baseline_audit"].endswith("audit_strong_baseline.json")
    assert freeze["gate_verdicts"] == {"gate1_construction": True, "gate2_scale": True,
                                       "gate3_rl_variance": True, "gate4_strong_baseline": True}
    assert freeze["detector"]["inference_caliber"]["max_det"] == 64
    assert freeze["disclosure"]["all_zero_family_cells"] == ["F8_s0m3"]


# --------------------------------------------------------------------------- #
# P9.0a 起跑前检查
# --------------------------------------------------------------------------- #
def test_verify_frozen_rejects_an_unstamped_report():
    result = verify_frozen(pilot_report=_report(), manifest=_manifest(),
                           recheck_data_hashes=False)
    assert result["ok"] is False
    assert any("bench_version" in problem for problem in result["problems"])


def test_verify_frozen_accepts_a_stamped_report():
    manifest = _manifest()
    stamped = stamp_pilot_report(_report(), manifest)
    assert verify_frozen(pilot_report=stamped, manifest=manifest,
                         recheck_data_hashes=False)["ok"] is True


def test_verify_frozen_detects_post_freeze_data_drift(tmp_path, monkeypatch):
    """冻结后数据被改一个字节 ⇒ P9.0a 必须拒绝起跑，而不是安静地跑在新数据上。"""
    import freeze_locus_bench as freeze_module

    parquet = tmp_path / "train" / "data.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"frozen")
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    monkeypatch.setattr(freeze_module, "REPO_ROOT", tmp_path)

    manifest = {
        "bench_version": FROZEN_VERSION,
        "gates": {"gate1_construction": {"ok": True}},
        "data": {"splits": {"train": {"path": "train/data.parquet", "bytes": 6,
                                      "sha256": freeze_module.sha256_file(parquet)}}},
        "detector": {"versioned_checkpoint": {"path": "w.pt",
                                              "sha256": freeze_module.sha256_file(weights)},
                     "class_order": {"n_classes": 34},
                     "inference_caliber": {"max_det": 64}},
    }
    stamped = stamp_pilot_report(_report(), dict(manifest, schema="locus_bench_freeze/1",
                                                 frozen_utc="x", artifact_pointers={},
                                                 disclosure={}))
    assert verify_frozen(pilot_report=stamped, manifest=manifest)["ok"] is True

    parquet.write_bytes(b"frozeN")  # 同长度、不同内容：字节数判据抓不到，哈希抓得到
    drifted = verify_frozen(pilot_report=stamped, manifest=manifest)
    assert drifted["ok"] is False
    assert any("sha256" in problem for problem in drifted["problems"])


def test_verify_frozen_rejects_a_manifest_carrying_a_failed_gate():
    manifest = dict(_manifest())
    manifest["gates"] = dict(manifest["gates"])
    manifest["gates"]["gate3_rl_variance"] = {"ok": False}
    stamped = stamp_pilot_report(_report(), _manifest())
    result = verify_frozen(pilot_report=stamped, manifest=manifest, recheck_data_hashes=False)
    assert result["ok"] is False


def test_freeze_manifest_is_json_serializable():
    json.dumps(_manifest(), ensure_ascii=False)
