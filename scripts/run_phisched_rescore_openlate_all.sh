#!/usr/bin/env bash
# plan.md 序 30（2026-09-11 17:00，用户裁决「显式 openlate 重派生现在起」）：对**每一个已有 `<bucket>__phisched_v2` 派生桶**
# 的源桶，用显式 `LNSDL_PHI_OPEN_PLACEMENT=late_freehand` 重派生 `<bucket>__phisched_v2_openlate`（零 GPU）。
#   · 不碰代码默认值（仍 early）⇒ 与在跑链（s63/s64）链内的 `__phisched_v2` 派生互不干扰；
#   · rescore 脚本按「显式模式 ≠ 默认」判 strip_auto（剥旧 auto_open 重插）并自动加 `_openlate` 后缀（rescore_forced_phi_bucket_offline.py:108/415）；
#   · 切默认之后再跑本脚本会因 strip_auto=False 而退化成不剥离 ⇒ 切默认前必须先把 rescore 脚本改成显式 `--open-placement`（序 30 第二半）。
# 桶集合 = 三个结果根 + constant_offload 下所有 `*__phisched_v2`（排除 smoke），即序 26 定义的全部进表桶 + 之后新出的桶。
# 用法：bash scripts/run_phisched_rescore_openlate_all.sh [--only-missing]
set -uo pipefail
cd "$(dirname "$0")/.."
PY=python
OUT=results/lnsdl/audits/rescore_phisched_v2_openlate
ONLY_MISSING="${1:-}"
STATUS="${STATUS:-logs/phisched_openlate_all_$(date +%Y%m%d_%H%M).status}"
export LNSDL_PHI_OPEN_PLACEMENT=late_freehand LNSDL_TOOLSET_CONTRACT=v341
mkdir -p "$OUT"
say() { echo "[$(date '+%F %T%z')] $*" | tee -a "$STATUS"; }
"$PY" -c "import nsdl.planning_tool as p; assert p.phi_open_placement_mode()=='late_freehand', p.phi_open_placement_mode()" \
  || { say "**停** —— late_freehand 未生效（显式 env 或默认均可，序 30 切默认后默认即 late）"; exit 1; }
run() {  # level src_dir
  local level="$1" d="$2" b; b=$(basename "$d")
  if [ "$ONLY_MISSING" = "--only-missing" ] && [ -d "${d}__phisched_v2_openlate" ] && [ "$(ls "${d}__phisched_v2_openlate"/*/*.json 2>/dev/null | wc -l)" = "$(ls "${d}__phisched_v2"/*/*.json 2>/dev/null | wc -l)" ]; then say "  跳过 $b（openlate 已满）"; return 0; fi
  "$PY" scripts/rescore_forced_phi_bucket_offline.py --buckets "$d" --level "$level" --rescheduler v2 --out-dir "$OUT" --write-derived-bucket 2>&1 | grep -v "^Not Found Entity" | tail -1 | cut -c1-160
  say "  $b → openlate $(ls "${d}__phisched_v2_openlate"/*/*.json 2>/dev/null | wc -l)/$(ls "${d}__phisched_v2"/*/*.json 2>/dev/null | wc -l)（v2 → openlate；模式 $(ls "${d}__phisched_v2_openlate"/*/*.json 2>/dev/null | head -1 | xargs -r "$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('agentic_eval',{}).get('phi_open_placement'))")）"
}
say "=== 序 30 openlate 全桶重派生启动（out=$OUT）==="
for root_level in "results/lnsdl/v3_3/locus:locus" "results/lnsdl/v3_4_1/viki_l2:viki" "results/lnsdl/rl_v3/viki_l2:viki"; do
  root="${root_level%%:*}"; level="${root_level##*:}"
  for v2 in $(ls -d "$root"/*__phisched_v2 2>/dev/null | grep -v smoke | sort); do run "$level" "${v2%__phisched_v2}"; done
done
for v2 in $(ls -d results/lnsdl/baselines/constant_offload/*__phisched_v2 2>/dev/null | sort); do
  case "$v2" in *locus*) lv=locus ;; *) lv=viki ;; esac; run "$lv" "${v2%__phisched_v2}"
done
"$PY" scripts/summarize_phisched_rescore.py "$OUT" > "$OUT/summary.md" 2>/dev/null && say "summary → $OUT/summary.md（$(grep -c '^|' "$OUT/summary.md") 行）"
say "=== 收工：openlate 桶 $(ls -d results/lnsdl/*/*/*__phisched_v2_openlate results/lnsdl/baselines/constant_offload/*__phisched_v2_openlate 2>/dev/null | wc -l) 个 ==="
