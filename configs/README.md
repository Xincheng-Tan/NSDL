# SFT configurations of the trained rows

| Config | Row |
|---|---|
| `lnsdl-agentic-vl-sft-qwen25-7b-v3_4_1-seed42-4gpu.yaml` | NSDL 7B (SFT), VIKI-L2 main row |
| `lnsdl-agentic-vl-sft-qwen25-3b-v3_4_1-seed42-2gpu.yaml` | NSDL 3B (SFT), VIKI-L2 |
| `lnsdl-agentic-vl-sft-qwen3-8b-v3_4_1-seed42.yaml` | NSDL 8B Qwen3-VL (SFT), VIKI-L2 |
| `lnsdl-agentic-vl-sft-qwen25-7b-v3_4_1-locus-cap524288-seed42-4gpu.yaml` | NSDL 7B (SFT), LOCUS |
| `lnsdl-agentic-vl-sft-qwen25-3b-v3_4_1-locus-cap524288-seed1234-2gpu.yaml` | NSDL 3B (SFT), LOCUS |
| `lnsdl-agentic-vl-sft-qwen3-8b-v3_4_1-locus-cap524288-seed42.yaml` | NSDL 8B Qwen3-VL (SFT), LOCUS |
| `lnsdl-agentic-vl-sft-qwen25-7b-v3_4_1-locus-phioff-cap524288-seed42-4gpu.yaml` | NSDL 7B w/o Φ, LOCUS |
| `lnsdl-agentic-vl-sft-qwen25-7b-v3_4_1-viki-translator-frontend-seed42-4gpu.yaml` | Trained-translation pipeline frontend, VIKI-L2 |
| `lnsdl-agentic-vl-sft-qwen25-7b-v3_4_1-ctrl-terra-seed1234-4gpu.yaml` | teacher-ablation control (Appendix VI) |
| `lnsdl-agentic-vl-sft-qwen25-7b-v3_4_1-teacher-qwen25vl72b-seed1234-4gpu.yaml` | teacher ablation, Qwen2.5-VL-72B teacher |
| `lnsdl-agentic-vl-sft-qwen25-7b-v3_4_1-teacher-qwen3vl32b-seed1234-4gpu.yaml` | teacher ablation, Qwen3-VL-32B teacher |

`cap524288` = VLM input capped at 524288 pixels (LOCUS renders are 1920x1080;
the detector reads native resolution). `yolo/locus_full34_class_list.txt` is
the 34-class vocabulary of the LOCUS detector.
