# Table rows and their run identifiers

Numbers live in the paper; this file only says which run each row comes from,
so that a re-run can be compared like for like. A "bucket" is the directory
name `--result-bucket` writes under `--output-dir`; the reported reading is the
`__phisched_v2_openlate` derivation of each bucket (Φ's expansion of the final
sketch adopted as the plan, opening ops placed `late_freehand`), produced by
`scripts/rescore_locus_bucket_offline.py` / `scripts/run_phisched_rescore_openlate_all.sh`.

## Table I (VIKI-L2; ID = test 1800, OOD = val 1218)

| Row | Checkpoint (HF) | Bucket |
|---|---|---|
| Prompt-only | none (rules + Φ) | `p61_constant_viki_l2_v341` |
| Full-translation (NSDL-SFT ckpt) | `nsdl-7b-sft` reused as translator | `p61_full_translation_viki_l2_v341` |
| Trained-translation | translator trained on the same rows (`configs/*viki-translator-frontend*`) | `p61_full_translation_viki_l2_v341_roletrained` |
| NSDL 3B (SFT) | `nsdl-3b-sft-viki` | `agentic_sft_qwen25_vl_3b_v341_obsbranch_seed42_truncfirst` |
| NSDL 7B (SFT) | `nsdl-7b-sft` | `agentic_sft_qwen25_vl_7b_v341_obsbranch_seed42_truncfirst` |
| NSDL 7B (SFT + RL) | `nsdl-7b-rl-viki` (step 30) | `rl_formal_scalar_v341f_step30` |
| NSDL 8B, Qwen3-VL (SFT) | `nsdl-8b-qwen3vl-sft-viki` | `agentic_sft_qwen3_vl_8b_v341_obsbranch_seed42_a1` |
| NSDL 7B w/o Φ | trained on the same corpus with Φ removed (`configs/*phioff*`) | `agentic_sft_qwen25_vl_7b_v341_phioff_seed42_truncfirst` |

## Table II (LOCUS; test_id 180 / ood_s 140 / ood_m 200 / ood_x 40)

| Row | Checkpoint (HF) | Bucket |
|---|---|---|
| Qwen2.5-VL-7B zero-shot (our tools) | base model | `locus_base_zeroshot_v341_cap524288_7b` |
| gpt-5.6-terra (our tools) | API | `g6` collection |
| VIKI-R 7B (op-level / native protocol) | official VIKI-R weights | `baselines/constant_offload/locus_transfer_vikir_official_7b`; `scripts/eval_vikir_native_protocol.py` |
| Formalize-then-plan, prompt-only | none | `p61_constant_locus_v341_cap524288` |
| Full-translation (NSDL-SFT ckpt) | `nsdl-7b-sft` reused | `p61_full_translation_locus_v341_cap524288` |
| Trained-translation | translator trained on LOCUS rows | `p61_full_translation_locus_cap524288_v341_roletrained` |
| NSDL-SFT trained on VIKI (transfer) | `nsdl-7b-sft` | `locus_transfer_v341_cap524288_7b_seed42` |
| NSDL 3B (SFT) | `nsdl-3b-sft-locus` (seed 1234) | `locus_g1_mainrow_v341_cap524288_3b_seed1234` |
| NSDL 7B (SFT) | `nsdl-7b-sft-locus` | `locus_g1_mainrow_v341_cap524288_7b_seed42` |
| NSDL 7B (SFT + RL) | `nsdl-7b-rl-locus` (step 30) | `locus_rlformal_locus_scalar_v341f_step30` |
| NSDL 8B, Qwen3-VL (SFT) | `nsdl-8b-qwen3vl-sft-locus` | `locus_g1_mainrow_v341_cap524288_8b_seed42` |
| NSDL 7B w/o Φ | Φ-off trained | `locus_phioff_v341_cap524288_7b_seed42` |
| GT sketch → Φ | none | `scripts/run_locus_phi_upper_bound.py` (`phi_sketch_opschain`) |

## Table III (ablations on the NSDL 7B (SFT) checkpoint)

| Row | Flags / bucket suffix |
|---|---|
| detect off | `--detector-backend off`; `_bracket_off` |
| YOLO, crops (NSDL-SFT) | main row |
| YOLO, crops disabled | `--detect-strip-crops` |
| GT rel., query scope | `--detector-backend oracle`, `LNSDL_ORACLE_SCOPE=query`; `_bracket_oracle` |
| GT rel., all entities | `--detector-backend oracle`, `LNSDL_ORACLE_SCOPE=full`; `_bracket_oracle_full` |
| GT rel., detector's forms | ground-truth relations restricted to the detector vocabulary (`LNSDL_ORACLE_VOCAB`); `_bracket_oracle_yolovocab` |
| Decoding stop off | without `--truncate-first-tool-call`; `_notrunc` |

Fig. 3 is produced by `scripts/gen_fig3_locus_heatmap.py` from the LOCUS
buckets above.
