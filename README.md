# NSDL: Neuro-Symbolic Division of Labor for Multi-Robot Task Planning

Xincheng Tan<sup>1</sup>, Ke Wang<sup>1</sup>, Yining Di<sup>2</sup>, Minfeng Zhu<sup>1</sup>

<sup>1</sup>Zhejiang University &nbsp; <sup>2</sup>The Hong Kong Polytechnic University

Code, model weights and LOCUS-Bench of the paper (arXiv preprint, 2026). Section, table and
figure numbers below refer to the arXiv version. Correspondence to Minfeng Zhu
(minfeng_zhu@zju.edu.cn); issues are welcome.

Internal identifiers such as `lnsdl` / `LNSDL_*` are historical code names for
what the paper calls NSDL. Code comments occasionally cite an internal design
log (`plan.md`, `benchmark.md`, `debug_*.md`, "memory:" notes, run ids such as
`s55`, sequence numbers); that log is not part of this release and nothing in
it is needed to run the code.

## What is here

NSDL divides planning labor by decision type. A trained vision-language model
(VLM) makes the decisions that must be judged from observed evidence and writes
them into a plan sketch (`[GOAL]` / `[ASSIGN]` / `[SKETCH]`); a deterministic
solver Φ expands the sketch into the executable joint plan. Two tools are
exposed to the model: `detect` (evidence only) and `schedule` (Φ).

| Path | Contents |
|---|---|
| `nsdl/` | The method. `tool_runtime.py` (tool loop), `planning_tool.py` + `micro_planner.py` + `macro_graph.py` (Φ: sketch expansion, scheduling, opening-prerequisite insertion), `validator.py` (feasibility), `perc_tool.py` + `yolo_detector.py` (`detect`), `benchmark_scoring.py` + `evaluation.py` (scoring, incl. the LOCUS empty-plan extension), `slm_backends/` (local VLM / API backends), `rl/tool_loop_adapter.py` (tool loop inside RL rollouts), `toolset_contract.py` (the `{detect, schedule}` tool contract used throughout the paper; it is the default) |
| `scripts/` | Data construction, training, evaluation and audit entry points (see the table below) |
| `configs/` | SFT configs of every trained row in the paper |
| `verl/` | verl + DAPO with the NSDL reward, the multi-turn tool rollout and the official VIKI-L2 plan checker used for scoring; `verl/NSDL_MODIFIED_FILES.md` lists exactly what differs from upstream |
| `locus_bench/` | LOCUS-Bench generator (task families, grid instantiation, rendering, validation) and a few rendered samples; the data itself is on Hugging Face (below) |
| `eval/VIKI-L2/` | Upstream VIKI-L2 inference scripts kept for the native-protocol baseline |
| `tests/` | Offline unit tests (no GPU, no data) for Φ, scoring, reward and the tool runtime |
| `results/PAPER_TABLE_MAP.md` | Which run identifier backs which table row |

## Artifacts (Hugging Face, `XinchengTan`)

| Artifact | Repository | Paper row |
|---|---|---|
| NSDL 7B (SFT), Qwen2.5-VL-7B | `XinchengTan/nsdl-7b-sft` (VIKI-L2), `XinchengTan/nsdl-7b-sft-locus` (LOCUS) | Table I / Table II main rows; Table III; Fig. 3 |
| NSDL 7B (SFT + RL) | `XinchengTan/nsdl-7b-rl-viki`, `XinchengTan/nsdl-7b-rl-locus` | Table I / Table II |
| NSDL 3B (SFT) | `XinchengTan/nsdl-3b-sft-viki`, `XinchengTan/nsdl-3b-sft-locus` | Table I / Table II |
| NSDL 8B (SFT), Qwen3-VL-8B | `XinchengTan/nsdl-8b-qwen3vl-sft-viki`, `XinchengTan/nsdl-8b-qwen3vl-sft-locus` | Table I / Table II |
| Detector, VIKI (20 classes) | `XinchengTan/nsdl-detector-viki20` | all VIKI-L2 tool-taking rows |
| Detector, LOCUS (34 classes) | `XinchengTan/nsdl-detector-locus34` | all LOCUS tool-taking rows |
| LOCUS-Bench | `XinchengTan/locus-bench` (dataset) | Sec. IV, Table II, Fig. 3 |

Each model card carries the sha256 of its files; the dataset's `MANIFEST.json`
carries the sha256 of every parquet file.

## Setup

```bash
conda env create -f NSDL.yml -n NSDL && conda activate NSDL
pip install -e verl                        # RL stack; the VIKI-L2 plan checker inside it is also used for scoring
# detectors (expected at these paths)
hf download XinchengTan/nsdl-detector-viki20  best.pt                      --local-dir checkpoints
hf download XinchengTan/nsdl-detector-locus34 yolo26l_locus_full34_v1.pt   --local-dir checkpoints
# VIKI-L2 (public): place VIKI-Bench under data/viki/VIKI-L2/{train,test,val}
# LOCUS-Bench: place the dataset under data/lnsdl_bench/ramp63k/{train,test_id,ood_s,ood_m,ood_x}
hf download XinchengTan/locus-bench --repo-type dataset --local-dir data/lnsdl_bench/ramp63k
```

Teacher-model calls (SFT trace construction and the closed-source baseline
only) use any OpenAI-compatible endpoint: copy `api_keys.example.json` to
`api_keys.json` (git-ignored) or set `NSDL_API_KEYS`, and adjust
`API_BASE_URL_LIST` / `API_MODEL_LIST` in `nsdl/llm_api.py`. Evaluation and
training of NSDL itself need no API.

## Quick start: evaluate the main row

```bash
hf download XinchengTan/nsdl-7b-sft       --local-dir checkpoints/nsdl-7b-sft
hf download XinchengTan/nsdl-7b-sft-locus --local-dir checkpoints/nsdl-7b-sft-locus
# VIKI-L2, ID (test) and OOD (val)
python scripts/eval_lnsdl_agentic_viki_l2.py --level VIKI-L2 --split test --model-size 7b \
    --slm-weights checkpoints/nsdl-7b-sft --truncate-first-tool-call \
    --output-dir results/viki_l2 --result-bucket nsdl_7b_sft
# LOCUS, one split
python scripts/eval_lnsdl_agentic_viki_l2.py --level LOCUS --split ood_m --model-size 7b \
    --slm-weights checkpoints/nsdl-7b-sft-locus --truncate-first-tool-call \
    --output-dir results/locus --result-bucket nsdl_7b_sft
# Reported numbers adopt Φ's expansion of the final sketch (Sec. V-A, Appendix I);
# derive that reading from a finished bucket with
python scripts/rescore_locus_bucket_offline.py --help
```

`scripts/eval_lnsdl_agentic_viki_l2_parallel.sh` shards the same run over GPUs.
`--truncate-first-tool-call` is the decoding stop of Sec. V-A (applied to every
locally decoded tool-taking arm). Ablations of Table III are flags of the same
script: `--detector-backend off` (detect off), `--detect-strip-crops` (crops
disabled), `--detector-backend oracle` with `LNSDL_ORACLE_SCOPE=query|full`
(ground-truth relations, query scope or all entities), `--disable-phi` /
`--phi-off-prompt` (Φ removed), and omitting `--truncate-first-tool-call`
(decoding stop off).

## Reproducing training

1. **SFT traces from benchmark ground truth** (Sec. III-E): `scripts/build_lnsdl_sft_data.py`
   (VIKI-L2) and `scripts/run_locus_sft_build.sh` (LOCUS). Tool calls in the
   traces are answered by the same Φ used at test time.
2. **SFT**: `scripts/run_vl_sft.py --config configs/<row>.yaml` (see `configs/README.md`).
3. **RL with Φ in the loop** (Sec. III-E): `scripts/prepare_rl_data.py` then
   `scripts/rl_final_setting.sh` (the exact reward and knobs of the reported
   runs); `scripts/merge_rl_ckpt.sh` exports the step-30 checkpoint.
4. **Detectors**: `scripts/train_yolo26_detector.py` on the official train
   splits only (Sec. III-D).
5. **LOCUS-Bench**: `locus_bench/README.md`.

## Tests

```bash
pytest tests -q
```

## License

Code: Apache-2.0 (`LICENSE`). Third-party components and their licenses are
listed in `THIRD_PARTY_NOTICES.md`. The 7B and 8B checkpoints derive from
Qwen2.5-VL-7B and Qwen3-VL-8B (Apache-2.0). The 3B checkpoints derive from
Qwen2.5-VL-3B and follow its Qwen Research License (non-commercial research
use). The YOLO26 detectors derive from Ultralytics (AGPL-3.0). LOCUS-Bench
licensing is stated on its dataset card.

## Citation

```bibtex
@misc{tan2026nsdl,
  title  = {{NSDL}: Neuro-Symbolic Division of Labor for Multi-Robot Task Planning},
  author = {Tan, Xincheng and Wang, Ke and Di, Yining and Zhu, Minfeng},
  year   = {2026},
  note   = {arXiv preprint}
}
```
