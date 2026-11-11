#!/usr/bin/env bash
# verl FSDP shards -> HF safetensors，用于把 RL checkpoint 送进标准 eval 管线。
#
# 为什么需要这一步：actor 目录里的 `huggingface/` 子目录只有 tokenizer/config
# （verl 的 `checkpoint.contents=[model,hf_model,optimizer,extra]` 在 FSDP 后端下
# 并不写出合并后的权重），权重只在 `model_world_size_4_rank_*.pt` 里分片存着。
#
# 两个容易漏的点：
#   1) merger 只 `save_pretrained` 模型权重 + config，**不写 tokenizer / processor**。
#      VL eval 需要 preprocessor_config.json + chat_template.json，否则
#      AutoProcessor 加载失败。这里从 **ckpt 自己的 `huggingface/`** 补齐
#      （而不是从 SFT init 拷），因为那份才是本 run 真正用过的 tokenizer。
#      `generation_config.json` ckpt 侧没有，从 SFT init 拷。
#   2) world_size 写在文件名里；4 卡 run 的分片只能按 4 合。
#
# 用法：RUN=<exp_name> STEP=10 bash scripts/merge_rl_ckpt.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

RUN="${RUN:?need RUN=<exp_name>}"
STEP="${STEP:?need STEP=<int>}"
SFT_DIR="${SFT_DIR:-$ROOT/checkpoints/lnsdl/sft/v3_3/qwen25-vl-7b-gpt56-terra-image-p513-obsbranch/final}"
SRC="$ROOT/checkpoints/lnsdl/rl/v3/$RUN/global_step_$STEP/actor"
DST="${DST:-$ROOT/checkpoints/lnsdl/rl/v3/_merged/${RUN}_step${STEP}}"

[[ -d "$SRC" ]] || { echo "[merge] 缺 $SRC" >&2; exit 1; }
n=$(ls -1 "$SRC" | wc -l)
[[ "$n" -ge 13 ]] || { echo "[merge] $SRC 只有 $n 个条目 (<13)，可能是残 checkpoint，拒绝合并" >&2; exit 1; }

echo "[merge] src=$SRC"
echo "[merge] dst=$DST"
mkdir -p "$DST"

python3 verl/scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path "$SFT_DIR" \
    --local_dir "$SRC" \
    --target_dir "$DST"

# tokenizer / processor：优先 ckpt 自带的那一份（这些 merger 不写，缺了就 cp -n 补）
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt added_tokens.json \
         special_tokens_map.json chat_template.json preprocessor_config.json; do
    if [[ -f "$SRC/huggingface/$f" ]]; then
        cp -n "$SRC/huggingface/$f" "$DST/$f" && echo "[merge] copied (ckpt) $f"
    elif [[ -f "$SFT_DIR/$f" ]]; then
        cp -n "$SFT_DIR/$f" "$DST/$f" && echo "[merge] copied (sft)  $f"
    fi
done

# generation_config.json：**必须强制覆盖**，不能 cp -n。
# `model.save_pretrained` 会自己写一份 `_from_model_config: true` 的，它把训练期
# `config.use_cache=false` 继承了进来 —— generate 于是每个 decode step 重算整个前缀
# （实测 457 s vs 18 s，慢 25×），而且 `eos_token_id` 只剩 151645（丢了 151643）、
# 没有 `repetition_penalty` / `attn_implementation`。那不只是慢，是**换了解码口径**。
# baseline 用的就是 SFT final 这一份，照抄才能可比。
cp -f "$SFT_DIR/generation_config.json" "$DST/generation_config.json"
echo "[merge] overwrote generation_config.json from SFT init (禁止 cp -n)"
python3 - "$DST" <<'GC'
import json, sys, os
g = json.load(open(os.path.join(sys.argv[1], "generation_config.json")))
assert g.get("use_cache") is not False, "generation_config.use_cache=false -> KV cache 关闭，评测会慢 25 倍"
assert g.get("_from_model_config") is not True, "generation_config 仍是 save_pretrained 自动生成的那份"
print("[merge] generation_config OK:", {k: g.get(k) for k in ("eos_token_id","repetition_penalty","use_cache")})
GC

echo "[merge] --- 产物 ---"
ls -1 "$DST"
python3 - "$DST" <<'PY'
import json, sys, os
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
assert os.path.exists(idx), "缺 model.safetensors.index.json"
m = json.load(open(idx))
print("[merge] 权重张量数 =", len(m["weight_map"]), "总字节 =", m["metadata"]["total_size"])
for f in ("config.json", "preprocessor_config.json", "chat_template.json", "tokenizer.json"):
    assert os.path.exists(os.path.join(d, f)), f"缺 {f}"
print("[merge] 必需文件齐备 OK")
PY
echo "[merge] done: $DST"
