#!/usr/bin/env python
"""LOCUS-Bench v1.0 冻结章（benchmark.md §9「共同冻结门」）。

写入 ``bench_version=v1.0`` 不是改一个字符串，而是一次**联合冻结**：benchmark 数据
与 detector 必须在同一时刻被钉住，否则 headline 结果里"感知口径"这一维是浮动的
（§9 共同冻结门、YOLO.md §7.3）。因此本脚本做三件事，缺一不可：

1. **从产物复验四道门**（不是从散文，也不是从人的记忆）：
   门1 构造门 / 门2 规模门（`pilot_report.audit_report`）、门3 RL 组内方差
   （`audits/rl_variance_audit.json`，§4.6）、门4 强基线判别性 + eval 公平性
   （`ramp63k/audit_strong_baseline.json`，§7-7/§7-8）。任何一门未过或未判定
   （`undetermined`）⇒ 拒绝盖章，benchmark 保持 `v0-pilot`（§9 末条）。
2. **哈希并记录冻结面**：五个 split parquet、detector versioned checkpoint、34 类
   顺序、global/per-label conf、IoU、max_det、detector train/val manifest、人工复核
   记录指针（§9/YOLO.md §7.3 逐条要求的那张清单）。
3. **落披露口径**：§7.9 的近零观察名单（`best <= 1/20` 的族-格）与全零
   `F8_s0m3` 并列写进冻结件，`official_api_rerun_present=false` 记为**冻结后待办**
   而不是被静默吞掉。

判据全部是产物级的，所以 `verify_freeze_preconditions()` 也被 P9.0a 复用为
"headline run 起跑前检查"：冻结件在、四门在、盘上数据与冻结哈希一致，才允许起跑。

用法::

    python scripts/freeze_locus_bench.py                 # 干跑，只报判决
    python scripts/freeze_locus_bench.py --apply         # 真盖章
    python scripts/freeze_locus_bench.py --verify-frozen # P9.0a 起跑前检查
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BENCH_ROOT = REPO_ROOT / "data" / "lnsdl_bench"
PILOT_REPORT_PATH = BENCH_ROOT / "ramp63k" / "pilot_report.json"
RL_VARIANCE_AUDIT_PATH = BENCH_ROOT / "audits" / "rl_variance_audit.json"
STRONG_BASELINE_AUDIT_PATH = BENCH_ROOT / "ramp63k" / "audit_strong_baseline.json"
FREEZE_MANIFEST_PATH = BENCH_ROOT / "ramp63k" / "freeze_manifest.json"

FROZEN_VERSION = "v1.0"
PILOT_VERSION = "v0-pilot"
PENDING_SENTINEL = "pending:P7.5"

SPLITS = ("train", "test_id", "ood_s", "ood_m", "ood_x")

# §7.9：判据用 max，而 max 落在 1/20 上时"判别性"离"全零"只有一个样本的运气。
# 这些格不阻塞冻结（§7-7 的触发条件是严格全零），但必须与全零格并列披露。
NEAR_ZERO_BEST_THRESHOLD = 0.05

# §9 / YOLO.md §7.3 要求与 benchmark 同时冻结的 detector 面。
DETECTOR_WEIGHTS_PATH = REPO_ROOT / "checkpoints" / "yolo26l_locus_full34_v1.pt"
DETECTOR_RUN_DIR = REPO_ROOT / "outputs" / "yolo_runs" / "yolo26l_locus_full34_v1"
DETECTOR_CLASS_LIST_PATH = REPO_ROOT / "configs" / "yolo" / "locus_full34_class_list.txt"
DETECTOR_DATASET_DIR = REPO_ROOT / "outputs" / "yolo_locus" / "locus_full34_det_ramp63k"
DETECTOR_STAGING_MANIFEST = (
    REPO_ROOT / "outputs" / "yolo_locus" / "shared_detector_train_ramp63k"
    / "detector_data_manifest.json"
)
# 人工复核记录（YOLO.md §5/§7.5：玻璃柜补框 198 图 + 透明队列判定）。
HUMAN_REVIEW_POINTERS = (
    REPO_ROOT / "outputs" / "yolo_locus" / "glass_review_ramp63k_train" / "merge_report.json",
    REPO_ROOT / "outputs" / "yolo_locus" / "glass_review_ramp63k_train" / "DECISIONS.tsv",
    REPO_ROOT / "outputs" / "yolo_locus" / "review_transparent_train",
    REPO_ROOT / "outputs" / "yolo_locus" / "review_transparent_eval",
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class GateFailure(Exception):
    """冻结前置未过。调用方应保持 v0-pilot 并停在原地。"""


def _check(results: list[dict[str, Any]], name: str, ok: bool, detail: Any) -> bool:
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


# --------------------------------------------------------------------------- #
# 门1 / 门2：构造门与规模门（pilot_report.audit_report）
# --------------------------------------------------------------------------- #
def check_construction_gates(audit_report: dict[str, Any]) -> dict[str, Any]:
    """§7-1..§7-6 + §6.3 的构造门，逐键从产物读。"""
    checks: list[dict[str, Any]] = []

    for key in ("realization_balance", "axis_orthogonality", "family_train_coverage",
                "composition_holdout", "goal_text_recoverability"):
        block = audit_report.get(key)
        ok = isinstance(block, dict) and block.get("ok") is True
        _check(checks, f"{key}.ok", ok, None if ok else block)

    dual = audit_report.get("dual_validator") or {}
    gate = float(dual.get("gate", 0.995))
    phi_agreement = dual.get("phi_agreement")
    _check(
        checks,
        "dual_validator.phi_agreement",
        isinstance(phi_agreement, (int, float)) and phi_agreement >= gate,
        {"phi_agreement": phi_agreement, "gate": gate,
         "phi_checked": dual.get("phi_checked"), "phi_ok": dual.get("phi_ok")},
    )
    _check(
        checks,
        "dual_validator.simenv_pass_rate",
        dual.get("simenv_pass_rate") == 1.0,
        {"simenv_pass_rate": dual.get("simenv_pass_rate")},
    )

    pairing = audit_report.get("s4_counterfactual_pairing") or {}
    # S4 成对样式：不完整对 / 重放失败 / "开柜后才能观察"的越权声明都必须是 0。
    pairing_ok = (
        pairing.get("ok") is True
        and pairing.get("incomplete_pairs") == 0
        and pairing.get("failures") == 0
        and pairing.get("post_open_observation_claims") == 0
    )
    _check(checks, "s4_counterfactual_pairing", pairing_ok, pairing or None)

    return {
        "gate": "benchmark.md §7-1..§7-6 / §6.3（构造门）",
        "checks": checks,
        "ok": all(item["ok"] for item in checks),
    }


def check_scale_gate(audit_report: dict[str, Any]) -> dict[str, Any]:
    """§4.4 规模门 + 工具集冻结（lookup 已永久移除，train 侧必须为 0）。"""
    checks: list[dict[str, Any]] = []
    support = audit_report.get("operator_train_support") or {}

    _check(checks, "operator_train_support.structural_ok",
           support.get("structural_ok") is True, support.get("structural_ok"))
    _check(checks, "operator_train_support.scale_gate_ok",
           support.get("scale_gate_ok") is True, support.get("scale_gate_ok"))
    # 工具集 2026-07-22 冻结为 5 工具，lookup 永久移除；train 侧出现即冻结面不一致。
    _check(checks, "operator_train_support.lookup_count==0",
           support.get("lookup_count") == 0, support.get("lookup_count"))

    n_instances = audit_report.get("n_instances")
    _check(checks, "n_instances>0", isinstance(n_instances, int) and n_instances > 0, n_instances)

    return {
        "gate": "benchmark.md §4.4（规模门）+ 工具集冻结",
        "checks": checks,
        "n_instances": n_instances,
        "ok": all(item["ok"] for item in checks),
    }


# --------------------------------------------------------------------------- #
# 门3：RL 组内方差（§4.6）
# --------------------------------------------------------------------------- #
def check_rl_variance_gate(
    audit: dict[str, Any],
    *,
    train_parquet_sha256: str | None = None,
) -> dict[str, Any]:
    """§4.6。除了 `gates.ok`，还要求审计跑的**就是**将被冻结的那份 train 数据。"""
    checks: list[dict[str, Any]] = []

    _check(checks, "schema", str(audit.get("schema", "")).startswith("rl_variance_audit/"),
           audit.get("schema"))
    # pilot 阶段每 cell 30 prompts 的读数不能替代冻结审计（§4.6）。
    _check(checks, "stage==freeze", audit.get("stage") == "freeze", audit.get("stage"))

    gates = audit.get("gates") or {}
    _check(checks, "gates.ok", gates.get("ok") is True, gates.get("ok"))
    _check(checks, "gates.failed==[]", gates.get("failed") in ([], None), gates.get("failed"))

    provenance = audit.get("provenance") or {}
    dataset = provenance.get("dataset") or {}
    _check(checks, "provenance.dataset.content_hashed",
           dataset.get("content_hashed") is True, dataset.get("content_hashed"))

    checkpoint = provenance.get("checkpoint") or {}
    _check(checks, "provenance.checkpoint.content_hashed",
           checkpoint.get("content_hashed") is True, checkpoint.get("content_hashed"))

    if train_parquet_sha256 is not None:
        audited = dataset.get("sha256")
        _check(
            checks,
            "audited_train_parquet_is_the_frozen_one",
            audited == train_parquet_sha256,
            {"audit_sha256": audited, "on_disk_sha256": train_parquet_sha256},
        )

    return {
        "gate": "benchmark.md §4.6（RL 组内方差冻结门）",
        "artifact": _rel(RL_VARIANCE_AUDIT_PATH),
        "checks": checks,
        "diagnostics": audit.get("diagnostics_all"),
        "totals": audit.get("totals"),
        "sft_init_sha256": checkpoint.get("sha256"),
        "reward_module_sha256": (provenance.get("reward") or {}).get("sha256"),
        "ok": all(item["ok"] for item in checks),
    }


# --------------------------------------------------------------------------- #
# 门4：强基线判别性 + eval 公平性（§7-7 / §7-8）
# --------------------------------------------------------------------------- #
def near_zero_watchlist(strong_baseline: dict[str, Any]) -> list[dict[str, Any]]:
    """§7.9：`best <= 1/20` 的族-格。判别性与全零之间只隔一个样本的运气。"""
    cells = ((strong_baseline.get("family_cell_verdicts") or {}).get("cells") or {})
    watchlist = [
        {"cell": name,
         "best": entry.get("best"),
         "verdict": entry.get("verdict"),
         "n_rows_qualifying": entry.get("n_rows_qualifying")}
        for name, entry in cells.items()
        if isinstance(entry, dict)
        and isinstance(entry.get("best"), (int, float))
        and 0.0 < entry["best"] <= NEAR_ZERO_BEST_THRESHOLD
    ]
    return sorted(watchlist, key=lambda item: item["cell"])


def check_strong_baseline_gate(
    audit_report: dict[str, Any],
    strong_baseline_audit: dict[str, Any],
) -> dict[str, Any]:
    """§7-7 判别性 + §7-8 eval 公平性。"""
    checks: list[dict[str, Any]] = []

    # 回填必须真的发生过：`pilot_report` 里还留着 sentinel 就等于门4 没关。
    for key in ("strong_baseline_prerun", "eval_fairness"):
        value = audit_report.get(key)
        _check(checks, f"audit_report.{key} backfilled",
               isinstance(value, dict) and value != PENDING_SENTINEL,
               value if not isinstance(value, dict) else "dict")

    prerun = strong_baseline_audit.get("strong_baseline_prerun") or {}
    fairness = strong_baseline_audit.get("eval_fairness") or {}
    summary = prerun.get("summary") or {}

    _check(checks, "strong_baseline.summary.ok", summary.get("ok") is True,
           {"ok": summary.get("ok"), "blocking_reason": summary.get("blocking_reason")})

    for scope in ("grid", "family_cell"):
        block = summary.get(scope) or {}
        # 饱和 = 不可判别；undetermined = 覆盖率不足，**不是**通过（§7.9 不变式 1）。
        _check(checks, f"summary.{scope}.saturated==[]", not block.get("saturated"),
               block.get("saturated"))
        _check(checks, f"summary.{scope}.undetermined==[]", not block.get("undetermined"),
               block.get("undetermined"))

    # 全零格允许存在，但必须已按构造证据裁定（且裁定只能是"真难度"，不许删样本）。
    all_zero = (summary.get("family_cell") or {}).get("all_zero") or []
    adjudication = summary.get("all_zero_adjudication") or {}
    if all_zero:
        _check(checks, "all_zero_adjudicated_by_construction_evidence",
               adjudication.get("verdict") == "real_difficulty"
               and sorted(adjudication.get("cells") or []) == sorted(all_zero),
               {"all_zero": all_zero, "adjudication": adjudication.get("verdict"),
                "cells": adjudication.get("cells")})

    _check(checks, "coverage: zero_coverage_splits=={}", not prerun.get("zero_coverage_splits"),
           prerun.get("zero_coverage_splits"))
    _check(checks, "coverage: coverage_incomplete_rows=={}",
           not prerun.get("coverage_incomplete_rows"), prerun.get("coverage_incomplete_rows"))

    _check(checks, "eval_fairness.ok", fairness.get("ok") is True, fairness.get("ok"))
    _check(checks, "eval_fairness.all_models_both_channels",
           fairness.get("all_models_both_channels") is True,
           fairness.get("all_models_both_channels"))

    return {
        "gate": "benchmark.md §7-7（判别性）/ §7-8（eval 公平性）",
        "artifact": _rel(STRONG_BASELINE_AUDIT_PATH),
        "checks": checks,
        "n_rows": summary.get("n_rows"),
        "grid": summary.get("grid"),
        "family_cell": summary.get("family_cell"),
        "axis_monotonicity": prerun.get("axis_monotonicity"),
        "ok": all(item["ok"] for item in checks),
    }


def build_disclosure(strong_baseline_audit: dict[str, Any]) -> dict[str, Any]:
    """§7.9 披露口径：全零格 + 近零观察名单 + 官方 API 复跑仍是冻结后待办。"""
    prerun = strong_baseline_audit.get("strong_baseline_prerun") or {}
    fairness = strong_baseline_audit.get("eval_fairness") or {}
    summary = prerun.get("summary") or {}
    all_zero = (summary.get("family_cell") or {}).get("all_zero") or []
    watchlist = near_zero_watchlist(prerun)
    return {
        "spec": "benchmark.md §7.9",
        "all_zero_family_cells": all_zero,
        "all_zero_verdict": (summary.get("all_zero_adjudication") or {}).get("verdict"),
        "near_zero_watchlist": watchlist,
        "near_zero_threshold_best": NEAR_ZERO_BEST_THRESHOLD,
        "statement": (
            "族-格判别性判据用 max：一个方法做对就算该格有判别性。但 "
            f"{len(watchlist)} 个格的 best 恰好落在 1/20 上，与全零只差一个样本的运气。"
            f"报告时必须把它们与全零格 {all_zero} 并列披露，"
            "不得只说“49 格里只有 1 格全零”。"
        ),
        "api_sources_present": fairness.get("api_sources_present"),
        "official_api_rerun_present": fairness.get("official_api_rerun_present"),
        "post_freeze_todo": (
            []
            if fairness.get("official_api_rerun_present")
            else ["官方 API（gpt-4o / Gemini 等）复跑单列一行，与 relay 行两行都披露（§7-7）"]
        ),
    }


# --------------------------------------------------------------------------- #
# 冻结面：数据 + detector
# --------------------------------------------------------------------------- #
def build_data_freeze_block(bench_dir: Path = BENCH_ROOT / "ramp63k") -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for split in SPLITS:
        parquet = bench_dir / split / "data.parquet"
        if not parquet.exists():
            raise GateFailure(f"split parquet 缺失，无法冻结：{_rel(parquet)}")
        splits[split] = {
            "path": _rel(parquet),
            "bytes": parquet.stat().st_size,
            "sha256": sha256_file(parquet),
        }
    return {"root": _rel(bench_dir), "splits": splits}


def _load_detector_names(dataset_dir: Path) -> list[str]:
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        return []
    import yaml  # 局部导入：本脚本的门1–门4 复验路径不该依赖 yaml 可用。

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = payload.get("names")
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda k: int(k))]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def build_detector_freeze_block() -> dict[str, Any]:
    """§9 / YOLO.md §7.3 逐条：路径+SHA256、34 类顺序、conf/IoU/max_det、manifest、人工复核。"""
    from nsdl.yolo_detector import (  # 就地读取真实默认口径，不复制一份常量
        DEFAULT_YOLO26_WEIGHTS,
        UltralyticsYoloDetector,
        _PER_LABEL_MAX_CONFIDENCE_THRESHOLD,
    )

    weights = Path(DEFAULT_YOLO26_WEIGHTS)
    if weights.resolve() != DETECTOR_WEIGHTS_PATH.resolve():
        raise GateFailure(
            "检测器默认权重与冻结目标不一致："
            f"default={_rel(weights)} frozen_target={_rel(DETECTOR_WEIGHTS_PATH)}。"
            "所有 headline 方法必须共用同一冻结 detector（§9）。"
        )
    if not weights.exists():
        raise GateFailure(f"detector 权重缺失：{_rel(weights)}")

    # 推理侧默认阈值 = 构造器默认参数，直接从签名读，避免文档与代码漂移。
    import inspect

    signature = inspect.signature(UltralyticsYoloDetector.__init__)
    defaults = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }

    class_list = [
        line.strip()
        for line in DETECTOR_CLASS_LIST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    data_yaml_names = _load_detector_names(DETECTOR_DATASET_DIR)
    order_matches = bool(data_yaml_names) and data_yaml_names == class_list
    if not order_matches:
        raise GateFailure(
            "34 类顺序不一致：`configs/yolo/locus_full34_class_list.txt` 与训练 "
            "`data.yaml` 的 names 必须逐位相同，否则冻结的类序不是权重的类序。"
        )

    manifests: dict[str, Any] = {}
    for label, path in (("detector_dataset_manifest", DETECTOR_DATASET_DIR / "dataset_manifest.json"),
                        ("detector_staging_manifest", DETECTOR_STAGING_MANIFEST),
                        ("detector_train_args", DETECTOR_RUN_DIR / "args.yaml"),
                        ("detector_val_gate_report", DETECTOR_RUN_DIR / "gate_report.json"),
                        ("detector_eval_holdout_report", DETECTOR_RUN_DIR / "eval_holdout_report.json")):
        manifests[label] = (
            {"path": _rel(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            if path.is_file()
            else {"path": _rel(path), "present": False}
        )

    human_review = []
    for path in HUMAN_REVIEW_POINTERS:
        entry: dict[str, Any] = {"path": _rel(path), "present": path.exists()}
        if path.is_file():
            entry["sha256"] = sha256_file(path)
        human_review.append(entry)

    return {
        "spec": "benchmark.md §9 共同冻结门 / YOLO.md §7.3",
        "versioned_checkpoint": {
            "path": _rel(weights),
            "bytes": weights.stat().st_size,
            "sha256": sha256_file(weights),
        },
        "run_dir": _rel(DETECTOR_RUN_DIR),
        "class_order": {
            "source": _rel(DETECTOR_CLASS_LIST_PATH),
            "n_classes": len(class_list),
            "names": class_list,
            "data_yaml_order_matches": order_matches,
        },
        "inference_caliber": {
            "confidence_threshold": defaults.get("confidence_threshold"),
            "per_label_confidence_threshold": dict(_PER_LABEL_MAX_CONFIDENCE_THRESHOLD),
            "iou_threshold": defaults.get("iou_threshold"),
            "max_det": defaults.get("max_det"),
        },
        "manifests": manifests,
        "human_review_records": human_review,
        "bracket_note": (
            "所有 headline 方法共用这一个 detector；`off`/`oracle` 只是 eval-only "
            "perception bracket，不得替代主结果（§9 / YOLO.md §7.3）。"
        ),
    }


# --------------------------------------------------------------------------- #
# 冻结前置总检
# --------------------------------------------------------------------------- #
def verify_freeze_preconditions(
    *,
    pilot_report: dict[str, Any] | None = None,
    rl_audit: dict[str, Any] | None = None,
    strong_baseline_audit: dict[str, Any] | None = None,
    train_parquet_sha256: str | None = None,
) -> dict[str, Any]:
    """四门复验，返回结构化判决。P9.0a 起跑前检查复用同一函数。"""
    pilot_report = pilot_report if pilot_report is not None else _load_json(PILOT_REPORT_PATH)
    rl_audit = rl_audit if rl_audit is not None else _load_json(RL_VARIANCE_AUDIT_PATH)
    strong_baseline_audit = (
        strong_baseline_audit
        if strong_baseline_audit is not None
        else _load_json(STRONG_BASELINE_AUDIT_PATH)
    )

    audit_report = pilot_report.get("audit_report") or {}
    gates = {
        "gate1_construction": check_construction_gates(audit_report),
        "gate2_scale": check_scale_gate(audit_report),
        "gate3_rl_variance": check_rl_variance_gate(
            rl_audit, train_parquet_sha256=train_parquet_sha256
        ),
        "gate4_strong_baseline": check_strong_baseline_gate(audit_report, strong_baseline_audit),
    }
    failed = sorted(name for name, block in gates.items() if not block["ok"])
    return {"gates": gates, "failed": failed, "ok": not failed}


def build_freeze_manifest(
    *,
    verification: dict[str, Any],
    data_block: dict[str, Any],
    detector_block: dict[str, Any],
    disclosure: dict[str, Any],
) -> dict[str, Any]:
    if not verification["ok"]:
        raise GateFailure(
            "冻结前置未过：" + ", ".join(verification["failed"])
            + "。benchmark 保持 v0-pilot，不得启动 P9 headline run（§9 末条）。"
        )
    return {
        "schema": "locus_bench_freeze/1",
        "spec": "docs/benchmark.md §9（共同冻结门）",
        "bench_version": FROZEN_VERSION,
        "frozen_utc": _utc_now(),
        "gates": verification["gates"],
        "data": data_block,
        "detector": detector_block,
        "disclosure": disclosure,
        "artifact_pointers": {
            "pilot_report": _rel(PILOT_REPORT_PATH),
            "rl_variance_audit": _rel(RL_VARIANCE_AUDIT_PATH),
            "strong_baseline_audit": _rel(STRONG_BASELINE_AUDIT_PATH),
            "freeze_manifest": _rel(FREEZE_MANIFEST_PATH),
        },
    }


def stamp_pilot_report(pilot_report: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """把 v1.0 写进 `config` 与 `audit_report` 两处，并挂上冻结指针。

    两处都写是因为下游读法不一：`scripts/audit_rl_variance.py` 读 top-level 的
    `bench_version`，报告样例读 `audit_report.bench_version`。只写一处会让另一条
    路径继续报 `v0-pilot`。
    """
    pilot_report.setdefault("config", {})["bench_version"] = FROZEN_VERSION
    audit_report = pilot_report.setdefault("audit_report", {})
    audit_report["bench_version"] = FROZEN_VERSION
    audit_report["freeze"] = {
        "schema": manifest["schema"],
        "bench_version": manifest["bench_version"],
        "frozen_utc": manifest["frozen_utc"],
        "manifest": _rel(FREEZE_MANIFEST_PATH),
        "gate_artifacts": manifest["artifact_pointers"],
        "gate_verdicts": {name: block["ok"] for name, block in manifest["gates"].items()},
        "detector": {
            "versioned_checkpoint": manifest["detector"]["versioned_checkpoint"],
            "n_classes": manifest["detector"]["class_order"]["n_classes"],
            "inference_caliber": manifest["detector"]["inference_caliber"],
        },
        "data_sha256": {
            split: entry["sha256"] for split, entry in manifest["data"]["splits"].items()
        },
        "disclosure": manifest["disclosure"],
    }
    return pilot_report


def verify_frozen(
    *,
    pilot_report: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    recheck_data_hashes: bool = True,
) -> dict[str, Any]:
    """P9.0a 起跑前检查：冻结件在、四门在、盘上数据仍与冻结哈希一致。"""
    pilot_report = pilot_report if pilot_report is not None else _load_json(PILOT_REPORT_PATH)
    problems: list[str] = []

    if manifest is None:
        if not FREEZE_MANIFEST_PATH.exists():
            return {"ok": False, "bench_version": (pilot_report.get("config") or {}).get(
                "bench_version"), "problems": [f"冻结件缺失：{_rel(FREEZE_MANIFEST_PATH)}"]}
        manifest = _load_json(FREEZE_MANIFEST_PATH)

    for scope, block in (("config", pilot_report.get("config") or {}),
                         ("audit_report", pilot_report.get("audit_report") or {})):
        if block.get("bench_version") != FROZEN_VERSION:
            problems.append(f"{scope}.bench_version={block.get('bench_version')!r} != {FROZEN_VERSION!r}")

    if manifest.get("bench_version") != FROZEN_VERSION:
        problems.append(f"manifest.bench_version={manifest.get('bench_version')!r}")

    failed = [name for name, block in (manifest.get("gates") or {}).items() if not block.get("ok")]
    if failed:
        problems.append("冻结件里存在未过的门：" + ", ".join(sorted(failed)))

    if recheck_data_hashes:
        for split, entry in ((manifest.get("data") or {}).get("splits") or {}).items():
            path = REPO_ROOT / entry["path"]
            if not path.exists():
                problems.append(f"{split}: 数据缺失 {entry['path']}")
                continue
            if path.stat().st_size != entry.get("bytes"):
                problems.append(f"{split}: 字节数与冻结件不符")
                continue
            if sha256_file(path) != entry.get("sha256"):
                problems.append(f"{split}: sha256 与冻结件不符（数据在冻结后被改动）")
        weights_entry = (manifest.get("detector") or {}).get("versioned_checkpoint") or {}
        weights_path = REPO_ROOT / weights_entry.get("path", "")
        if not weights_path.exists():
            problems.append(f"detector 权重缺失：{weights_entry.get('path')}")
        elif sha256_file(weights_path) != weights_entry.get("sha256"):
            problems.append("detector 权重 sha256 与冻结件不符")

    return {"ok": not problems, "bench_version": FROZEN_VERSION if not problems else None,
            "problems": problems}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_verdict(verification: dict[str, Any]) -> None:
    for name, block in verification["gates"].items():
        mark = "PASS" if block["ok"] else "FAIL"
        print(f"[{mark}] {name} — {block['gate']}")
        for item in block["checks"]:
            if not item["ok"]:
                print(f"        ✗ {item['check']}: {json.dumps(item['detail'], ensure_ascii=False)[:200]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true",
                        help="真写入 bench_version=v1.0（默认只干跑报判决）")
    parser.add_argument("--verify-frozen", action="store_true",
                        help="P9.0a 起跑前检查：冻结件是否在且盘上数据未漂移")
    parser.add_argument("--skip-data-hash", action="store_true",
                        help="跳过 split parquet 重哈希（只在反复调试判决时用；--apply 忽略本开关）")
    args = parser.parse_args(argv)

    if args.verify_frozen:
        result = verify_frozen(recheck_data_hashes=not args.skip_data_hash)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 1

    train_sha = None
    data_block = None
    if args.apply or not args.skip_data_hash:
        data_block = build_data_freeze_block()
        train_sha = data_block["splits"]["train"]["sha256"]

    verification = verify_freeze_preconditions(train_parquet_sha256=train_sha)
    _print_verdict(verification)
    if not verification["ok"]:
        print("\n拒绝盖章：benchmark 保持 v0-pilot，不得启动 P9 headline run（§9 末条）。")
        return 1

    detector_block = build_detector_freeze_block()
    disclosure = build_disclosure(_load_json(STRONG_BASELINE_AUDIT_PATH))
    manifest = build_freeze_manifest(
        verification=verification,
        data_block=data_block,
        detector_block=detector_block,
        disclosure=disclosure,
    )

    print(f"\n四门全过。detector = {manifest['detector']['versioned_checkpoint']['path']}"
          f" sha256={manifest['detector']['versioned_checkpoint']['sha256'][:16]}…")
    print(f"披露：全零格 {disclosure['all_zero_family_cells']}，"
          f"近零观察名单 {[item['cell'] for item in disclosure['near_zero_watchlist']]}，"
          f"official_api_rerun_present={disclosure['official_api_rerun_present']}")

    if not args.apply:
        print("\n干跑结束（未写盘）。确认后加 --apply。")
        return 0

    backup = PILOT_REPORT_PATH.with_suffix(PILOT_REPORT_PATH.suffix + ".bak_freeze")
    shutil.copy2(PILOT_REPORT_PATH, backup)
    pilot_report = stamp_pilot_report(_load_json(PILOT_REPORT_PATH), manifest)
    PILOT_REPORT_PATH.write_text(
        json.dumps(pilot_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    FREEZE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n已盖章 bench_version={FROZEN_VERSION}")
    print(f"  {_rel(PILOT_REPORT_PATH)}（备份 {_rel(backup)}）")
    print(f"  {_rel(FREEZE_MANIFEST_PATH)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
