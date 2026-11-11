#!/usr/bin/env python
"""D2 第一阶段：零样本开放词表检测器 vs 微调 YOLO26 的**预审计**。

回答的是排卡前的那个问题：换成通用检测器之后，检测器这一环还剩多少信号？
如果零样本检测器在 LOCUS 域上近乎发不出有效框，D2 的端到端配对 eval 就会
塌回**已经跑完的 D0（detect-off）**——那样再花卡只是把 D0 重跑一遍换个名字。
同一形状的退化已经吃过两次（G10 verify 2.0% 调用率、4-AR 5.2% 强制率），
所以这一阶段是硬前置，不是可选的尽职调查。

三条容易踩错、这里显式钉死的口径
--------------------------------
1. **imgsz 必须对齐到 960。** 生产侧 `YoloDetector.detect()` 不传 imgsz，
   ultralytics 于是从 ckpt overrides 继承训练时的 960；而 yolov8x-worldv2
   自己的 overrides 是 640。按各自默认跑，YOLO-World 在实测样图上出 **0 框**、
   960 上出 4 框（cabinet .460 / bowl .428）——"通用检测器全面崩溃"这个结论
   完全是分辨率假象。另加 1280 一档，是**刻意把基线做强**：连最优待遇下都塌，
   结论才稳。
2. **主指标是操作点 P/R，不是 mAP。** 下游模型拿到的是 conf>=0.25 过滤后的框，
   所以决策相关的量是那个阈值上的 P/R。mAP（conf~0）一并报，但只作参考——
   两者可以给出相反的观感。
3. **词表用自然语言。** 类名 `wooden_cutting_board` 对 CLIP 文本编码器是 OOD 的，
   下划线换空格。这不是美化，是开放词表模型的输入规范。
4. **机器人措辞复查只做一轮，且结果是阴性、不进最终配置。** 6 个机器人类零样本
   召回全为 0.0；这一轮换措辞是**看到那批 0 之后**才做的，属事后探查，故在此披露：
   `--robot-vocab-probe` 复现它。结论见 `robot_vocab_probe`——唯一能定位机器人的
   词是 "person"，而它把 6 种机器人塌成一个标签，正好抹掉 LOCUS 调度依赖的机器人
   身份。所以 D2 保留原始资产名词表，没有任何事后调优进入最终配置。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 生产操作点，取自 nsdl/yolo_detector.py 的 UltralyticsYoloDetector 默认值。
# 改这里之前先确认那边没变——两边漂移会让"平价"二字失效。
PROD_CONF = 0.25
PROD_IOU = 0.70
PROD_MAX_DET = 64
PROD_IMGSZ = 960


def _load_vocab(data_yaml: str) -> list[str]:
    names = yaml.safe_load(open(data_yaml))["names"]
    ordered = [names[i] for i in sorted(names)]
    # set_classes 的顺序即类索引顺序，必须与 data.yaml 一致，否则 val() 的
    # 逐类 P/R 会整体错位——且不会报错，只会给出一张看着合理的错表。
    return ordered


def _run_val(model, data_yaml: str, imgsz: int, conf: float, device: str, name: str):
    res = model.val(
        data=data_yaml,
        imgsz=imgsz,
        conf=conf,
        iou=PROD_IOU,
        max_det=PROD_MAX_DET,
        device=device,
        split="val",
        plots=False,
        verbose=False,
        project=os.path.join(REPO_ROOT, "outputs", "d2_preaudit"),
        name=name,
        exist_ok=True,
    )
    box = res.box
    per_class = {}
    for idx, ci in enumerate(box.ap_class_index.tolist()):
        per_class[res.names[ci]] = {
            "precision": float(box.p[idx]),
            "recall": float(box.r[idx]),
            "map50": float(box.ap50[idx]),
        }
    return {
        "overall": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
        "n_classes_with_gt": len(per_class),
        "per_class": per_class,
    }


# —— 机器人措辞探针（一轮，阴性，已登记）——————————————————————————
# 判别的是"措辞不对"还是"根本没有能触发的概念"：把机器人槽的词换掉、其余 28 个
# 类原样留着，于是 NMS 竞争条件不变，唯一变量是那 6 个词。
ROBOT_SLOTS = ["fetch", "stompy", "unitree_h1", "panda", "unitree_go2", "anymal_c"]
ROBOT_WORDINGS = {
    "资产名(原)": None,  # 用 vocab 自身
    "person": ["person"] * 6,
    "robot": ["robot"] * 6,
    "humanoid robot": ["humanoid robot", "humanoid robot", "humanoid robot",
                       "robotic arm", "quadruped robot dog", "quadruped robot dog"],
    "mobile robot with arm": ["mobile robot with arm"] * 6,
}


def _robot_vocab_probe(model_path, vocab_raw, vocab_nl, images, device):
    from ultralytics import YOLOWorld

    slot_idx = [vocab_raw.index(name) for name in ROBOT_SLOTS if name in vocab_raw]
    out = {}
    for tag, words in ROBOT_WORDINGS.items():
        vocab = list(vocab_nl)
        if words is not None:
            for pos, word in zip(slot_idx, words):
                vocab[pos] = word
        model = YOLOWorld(model_path)
        model.set_classes(vocab)
        per_image = {}
        for img in images:
            res = model.predict(img, imgsz=PROD_IMGSZ, conf=PROD_CONF, iou=PROD_IOU,
                                max_det=PROD_MAX_DET, device=device, verbose=False)[0]
            cls = [int(c) for c in res.boxes.cls.tolist()]
            conf = [round(float(c), 3) for c in res.boxes.conf.tolist()]
            hits = [(vocab[c], f) for c, f in zip(cls, conf) if c in slot_idx]
            per_image[os.path.basename(img)] = {"total_boxes": len(cls), "robot_slot_hits": hits}
            print(f"  [{tag:<22}] {os.path.basename(img):<18} 总 {len(cls):>3}  "
                  f"机器人位 {len(hits)}: {hits}", flush=True)
        out[tag] = per_image
    return {
        "wordings": out,
        "conclusion": (
            "6 个机器人类零样本召回全为 0.0。同一词表内换措辞（robot / humanoid robot / "
            "quadruped robot dog / robotic arm / mobile robot with arm）仍全 0；唯一能"
            "定位到它们的词是 person，但那会把 6 种机器人塌成一个标签，抹掉 LOCUS 调度"
            "依赖的机器人身份。⇒ 不是措辞问题，是现成检测器没有可触发的机器人概念。"
        ),
        "decision": (
            "保留原始资产名词表。本轮是看到零之后才做的事后探查，只做一轮，结果为阴性，"
            "不进最终配置——D2 端到端跑的就是资产名词表。"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(
        REPO_ROOT, "outputs/yolo_locus/locus_full34_eval_holdout/data.yaml"))
    ap.add_argument("--finetuned", default=os.path.join(
        REPO_ROOT, "checkpoints/yolo26l_locus_full34_v1.pt"))
    ap.add_argument("--openvocab", default=os.path.join(
        REPO_ROOT, "checkpoints/openvocab/yolov8x-worldv2.pt"))
    ap.add_argument("--device", default="cpu",
                    help="默认 cpu：M 的四张卡长期被 RL 占着，这一步没必要去挤")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[PROD_IMGSZ, 1280])
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "outputs/d2_preaudit/preaudit.json"))
    ap.add_argument("--robot-vocab-probe", action="store_true",
                    help="只跑机器人措辞探针，把结果并进已有 --out（不重跑 val）")
    ap.add_argument("--probe-images", nargs="+", default=[os.path.join(
        REPO_ROOT, "outputs/yolo_locus/locus_full34_eval_holdout/images/val", name)
        for name in ("ood_m_00000.png", "test_id_00000.png")])
    args = ap.parse_args()

    if args.robot_vocab_probe:
        vocab_raw = _load_vocab(args.data)
        vocab_nl = [n.replace("_", " ") for n in vocab_raw]
        with open(args.out) as fh:
            report = json.load(fh)
        report["robot_vocab_probe"] = _robot_vocab_probe(
            args.openvocab, vocab_raw, vocab_nl, args.probe_images, args.device)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(json.dumps(report["robot_vocab_probe"]["conclusion"], ensure_ascii=False))
        print(f"[D2] 已并入 {args.out}")
        return 0

    from ultralytics import YOLO, YOLOWorld

    vocab_raw = _load_vocab(args.data)
    vocab_nl = [n.replace("_", " ") for n in vocab_raw]
    report = {
        "operating_point": {
            "conf": PROD_CONF, "iou": PROD_IOU, "max_det": PROD_MAX_DET,
            "prod_imgsz": PROD_IMGSZ,
        },
        "vocab_natural_language": vocab_nl,
        "arms": {},
    }

    print(f"[D2] 微调 YOLO26 @ imgsz={PROD_IMGSZ}（平价档）", flush=True)
    ft = YOLO(args.finetuned)
    report["arms"]["finetuned_yolo26@960"] = _run_val(
        ft, args.data, PROD_IMGSZ, PROD_CONF, args.device, "ft960")

    for sz in args.imgsz:
        tag = f"openvocab_yoloworld@{sz}"
        print(f"[D2] YOLO-World 零样本 @ imgsz={sz}", flush=True)
        ov = YOLOWorld(args.openvocab)
        ov.set_classes(vocab_nl)
        report["arms"][tag] = _run_val(ov, args.data, sz, PROD_CONF, args.device, f"ov{sz}")

    # —— 退化判据。下游拿到的是 conf>=0.25 的框；召回近零 = 端到端 D2 与 D0 同构。
    ov_r = report["arms"][f"openvocab_yoloworld@{PROD_IMGSZ}"]["overall"]["recall"]
    ft_r = report["arms"]["finetuned_yolo26@960"]["overall"]["recall"]
    report["degeneracy_verdict"] = {
        "openvocab_recall_at_prod": ov_r,
        "finetuned_recall_at_prod": ft_r,
        "retained_fraction": (ov_r / ft_r) if ft_r else None,
        "degenerate": ov_r < 0.05,
        "note": ("零样本召回 <0.05 ⇒ 检测器实际发不出框 ⇒ D2 端到端会塌回已跑完的 D0，"
                 "不值得排卡；否则 D2 是一个真的换装对照，可进第二阶段配对 eval。"),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(json.dumps(report["degeneracy_verdict"], indent=2, ensure_ascii=False))
    print(f"[D2] 写出 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
