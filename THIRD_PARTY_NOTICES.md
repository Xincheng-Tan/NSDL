# Third-party components

| Component | Used for | License | Location here |
|---|---|---|---|
| VIKI-R / VIKI-Bench (MARS-EAI/VIKI-R) | VIKI-L2 benchmark, official plan checker, data-production pipeline used to render LOCUS-Bench | Apache-2.0 | `eval/VIKI-L2/`, `verl/verl/utils/reward_score/utils/`, `locus_bench/generator/` (builds on the `data-pipeline` branch) |
| verl (volcengine/verl) and the DAPO recipe | RL training stack | Apache-2.0 | `verl/` |
| ManiSkill3 / SAPIEN | Simulator behind the VIKI renderer and plan checker | Apache-2.0 | dependency |
| RoboCasa | Kitchen scenes rendered in VIKI and LOCUS | code MIT; assets CC BY 4.0 | dependency (not redistributed) |
| VIKI-Assets (FACEONG/VIKI-Assets) | Object models rendered in VIKI and LOCUS; some objects originate from PartNet-Mobility (non-commercial research terms) | as stated by the upstream | not redistributed; download from the upstream |
| Ultralytics YOLO26 | `detect` tool detectors | AGPL-3.0 | dependency; fine-tuned weights on Hugging Face |
| Qwen2.5-VL-7B, Qwen3-VL-8B (Alibaba) | Backbones of the 7B and 8B rows | Apache-2.0 | dependency; fine-tuned weights on Hugging Face |
| Qwen2.5-VL-3B (Alibaba) | Backbone of the 3B rows | Qwen Research License (non-commercial research use) | dependency; fine-tuned weights on Hugging Face |
| YOLO-World | zero-shot detector swap (Table IV) | GPL-3.0 | dependency |
