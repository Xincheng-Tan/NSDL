# LOCUS-Bench

A two-axis benchmark for embodied multi-robot task planning (Sec. IV of the paper): axis S
grades state judgment, axis M grades mechanical structure; 21 populated cells,
10 task families, 49 family-cells; train 6300, ID test 180, OOD-S 140, OOD-M
200, OOD-X 40. Scenes are rendered with the official VIKI renderer over six
heterogeneous robots and a 34-class vocabulary.

## Data

Dataset repository: `XinchengTan/locus-bench` on Hugging Face. Each split
is a parquet file (`prompt`, `images` (PNG bytes), `reward_model.ground_truth`
with `locus_meta` carrying the cell, family, split flags and instantiation
seeds) plus `configs/*.yaml` (scene instantiation), `gt_labels/` (detector
boxes exported from instance masks) and `render_manifest.json`.
`samples/` here holds a handful of rendered frames with their ground truth so
the format can be inspected without downloading the dataset.

Scoring reuses the official VIKI-L2 scorer plus the empty-plan extension
(`nsdl/benchmark_scoring.py: compute_locus_l2_acc`).

## Generator

`generator/` builds on the VIKI-R `data-pipeline` branch (RoboFactory,
Apache-2.0); install that first and download VIKI-Assets and the RoboCasa
scenes from their upstream sources (3D assets are not redistributed here).

| File | Role |
|---|---|
| `locus_task_pool.py` | the ten task families and their (S, M) cells |
| `locus_generate.py` | grid instantiation: layouts, distractors, paraphrase and state realizations, split assignment |
| `locus_validate.py` | replay of every reference plan through the official simulator; construction-rule checks |
| `locus_render_pack.py`, `locus_render_masked.py`, `locus_mask_export.py` | rendering, instance-mask export, detector labels |
| `locus_pilot.py` | end-to-end driver |
| `test_locus_*.py` | unit tests of the above |
