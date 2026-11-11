#!/usr/bin/env python3
"""Figure 3 (single-column): LOCUS S x M grid heatmaps -- one absolute panel plus difference panels.

python scripts/gen_fig3_locus_heatmap.py                         # 2x2: (a) SFT acc, (b) - base, (c) - detect off, (d) RL - SFT; early-Phi caliber
python scripts/gen_fig3_locus_heatmap.py --caliber openlate      # after plan #30 re-derives every bucket
python scripts/gen_fig3_locus_heatmap.py --panels main,base,detect_off   # three panels -> 1 x 3 layout (RL out of the table)
python scripts/gen_fig3_locus_heatmap.py --region-shapes first   # ID / OOD-X outlines only on panel (a)

Outputs docs/paper_writing/fig/fig3_locus_heatmap.{pdf,png,svg} and
fig3_materials_locus_grid.json (every number drawn, per cell, with n and the
family composition of the cell). Layout and rationale: fig3_design_philosophy.md.

Data source: results/lnsdl/audits/official_replay_taxonomy/<arm>.json (plan #22
replay of the official judger; each row carries `cell` = F<family>_s<S>m<M> and
`class` in {correct, err, infeasible, goal_unsat, achieved_but_long}). Per
S x M cell we pool every family-cell that lands there. Panel (a) is the SFT main
row's accuracy; every other panel is a WITHIN-cell difference against it,
which is the only reading the grid supports across rows (S rows differ in
family composition, see design doc section 3).

Tweaking text positions: everything is in LAYOUT below, in grid units
(one cell = 1 unit; x runs 0..5 left->right over M0..M4, y runs 0..5
top->bottom over S0..S4, so y < 0 is above the grid and x > 5 is right of it).
Edit, re-run, open the .png.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "results/lnsdl/audits/official_replay_taxonomy"
OUT = REPO / "docs/paper_writing/fig/fig3_locus_heatmap"

for _f in glob.glob("/usr/share/fonts/opentype/urw-base35/NimbusSans-*.otf"):
    fm.fontManager.addfont(_f)
plt.rcParams.update({
    "font.family": "Nimbus Sans",
    "font.size": 7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.5,
})

# ----------------------------------------------------------------------------- data
# arm key -> bucket stem (without the __phisched_v2[_openlate].json suffix)
ARMS = {
    "main": "locus_g1_mainrow_v341_cap524288_7b_seed42",           # NSDL-7B, SFT only (table 2 main row)
    "base": "locus_base_zeroshot_v341_cap524288_7b",               # same backbone, zero-shot
    "detect_off": "locus_g1_mainrow_v341_cap524288_7b_seed42_bracket_off",  # main ckpt, detect disabled
    "transfer": "locus_transfer_v341_cap524288_7b_seed42",         # VIKI-trained ckpt evaluated on LOCUS
    "rl": "locus_rlformal_locus_scalar_v341f_step30",              # s55 (final setting) RL step30, init = main; v341d = s46 (superseded)
}
# Difference panels: title and direction. "rl" is the one panel read as (arm - main).
DELTA_PANELS = {
    "base": ("Δ vs. base zero-shot", "main_minus_arm"),
    "detect_off": ("Δ vs. detect off", "main_minus_arm"),
    "transfer": ("Δ vs. VIKI-transfer", "main_minus_arm"),
    "rl": ("Δ: +RL vs. SFT", "arm_minus_main"),
}
SUFFIX = {"early": "__phisched_v2.json", "openlate": "__phisched_v2_openlate.json"}

S_LEVELS = ["S0", "S1", "S2", "S3", "S4"]
M_LEVELS = ["M0", "M1", "M2", "M3", "M4"]
# The grid is not Cartesian: these four cells cannot be instantiated (draft section 5, para 17).
ABSENT = {(2, 4), (3, 4), (4, 0), (4, 4)}
ID_BLOCK = (0, 2, 0, 2)  # S0-2 x M0-2 = train / test_id coverage

# ----------------------------------------------------------------------------- look
SEQ = LinearSegmentedColormap.from_list("seq", ["#f7f7f7", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])
DIV = LinearSegmentedColormap.from_list("div", ["#b2182b", "#ef8a62", "#fddbc7", "#ffffff", "#d1e5f0", "#67a9cf", "#2166ac"])
ABSENT_FACE = "#e6e6e6"
ID_EDGE = "#222222"
REGION_COLOR = "#333333"

LAYOUT = {
    "fig_width_in": 3.4,                 # ieeeconf column width
    "row_height_in": 1.75,               # figure height per panel row (colorbars included)
    "margins": dict(left=0.10, right=0.94, top=0.83, bottom=0.17, wspace=0.55),
    "row_height_in_2x2": 1.4,
    "margins_2x2": dict(left=0.13, right=0.855, top=0.935, bottom=0.095, wspace=0.72, hspace=0.38),
    "cell_value_fontsize": 5.6,
    "tick_fontsize": 6,
    "title_fontsize": 6.6, "title_pad": 9, "title_loc": "center",
    "axis_label_fontsize": 7,
    "region_fontsize": 5.3, "region_fontweight": "bold",
    "region_linewidth": 1.0,             # OOD-M / OOD-S brackets and the OOD-X dashed box
    "id_box_linewidth": 1.1,
    # Region annotations on the absolute panel only. (x, y) in grid units.
    "id_label":   dict(x=0.0,  y=-0.18, text="train / ID", ha="left",   va="bottom"),
    "oodm_line":  dict(x0=3.02, x1=4.98, y=-0.12),
    "oodm_label": dict(x=4.0,  y=-0.18, text="OOD-M",      ha="center", va="bottom"),
    "oods_line":  dict(x=5.12, y0=3.02, y1=4.98),
    "oods_label": dict(x=5.2,  y=4.0,   text="OOD-S",      ha="left",   va="center", rotation=270),
    "oodx_box":   dict(x=3, y=3, w=1, h=2),
    "oodx_label": dict(x=3.5,  y=5.55,  text="OOD-X",      ha="center", va="top"),
    # Colorbars: vertical strips to the right of each panel, as fractions of the panel's width.
    # The gap leaves room for the OOD-S bracket + label on panel (a); all panels share it for alignment.
    "cbar_width_frac": 0.05, "cbar_gap_frac": 0.19, "cbar_tick_fontsize": 5,
    "xlabel_y": 0.012,                   # figure fraction from the bottom (shared "depth M" label)
    "ylabel_x": 0.012,                   # figure fraction from the left (shared "depth S" label)
}


# ----------------------------------------------------------------------------- helpers
def load_rows(arm: str, caliber: str) -> list[dict]:
    fn = AUDIT / (ARMS[arm] + SUFFIX[caliber])
    if not fn.exists():
        raise SystemExit(f"missing bucket for arm={arm} caliber={caliber}: {fn}")
    d = json.loads(fn.read_text())
    rows = d["rows"]
    return list(rows.values()) if isinstance(rows, dict) else rows


def grid_of(rows: list[dict]):
    """(acc[5,5], n[5,5], families{(s,m): {family: [correct, n]}}) pooled over families per S x M cell."""
    n = np.zeros((5, 5), int)
    c = np.zeros((5, 5), int)
    fam: dict = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for r in rows:
        mo = re.match(r"(F\d+)_s(\d)m(\d)$", r["cell"])
        f, s, m = mo.group(1), int(mo.group(2)), int(mo.group(3))
        ok = r["class"] == "correct"
        n[s, m] += 1
        c[s, m] += ok
        fam[(s, m)][f][0] += ok
        fam[(s, m)][f][1] += 1
    with np.errstate(invalid="ignore"):
        acc = np.where(n > 0, c / np.maximum(n, 1), np.nan)
    for s, m in ABSENT:
        assert n[s, m] == 0, f"cell S{s}xM{m} is declared absent but has {n[s, m]} rows"
    return acc, n, fam


def fmt_acc(v):
    return "1.0" if v >= 0.995 else f"{v:.2f}".lstrip("0")


def fmt_delta(v):
    if abs(v) < 0.005:
        return "0"
    return ("+" if v > 0 else "−") + f"{abs(v):.2f}".lstrip("0")


def draw_panel(ax, values, *, title, cmap, norm, fmt, annotate_regions, region_shapes=True):
    L = LAYOUT
    ax.set_xlim(0, 5)
    ax.set_ylim(5, 0)
    ax.set_aspect("equal")
    for s in range(5):
        for m in range(5):
            if (s, m) in ABSENT:
                ax.add_patch(Rectangle((m, s), 1, 1, facecolor=ABSENT_FACE, edgecolor="white",
                                       hatch="////", linewidth=0.6))
                continue
            v = values[s, m]
            rgba = cmap(norm(v))
            ax.add_patch(Rectangle((m, s), 1, 1, facecolor=rgba, edgecolor="white", linewidth=0.6))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(m + 0.5, s + 0.5, fmt(v), ha="center", va="center", fontsize=L["cell_value_fontsize"],
                    color="white" if lum < 0.5 else "#111111")
    s0, s1, m0, m1 = ID_BLOCK
    # clip_on=False: the box's left/top edges lie on the axes boundary and would otherwise be
    # clipped to half their width, making the right/bottom edges look thicker.
    if region_shapes:
        ax.add_patch(Rectangle((m0, s0), m1 - m0 + 1, s1 - s0 + 1, fill=False, edgecolor=ID_EDGE,
                               linewidth=L["id_box_linewidth"], zorder=5, clip_on=False))
        b = L["oodx_box"]
        ax.add_patch(Rectangle((b["x"], b["y"]), b["w"], b["h"], fill=False, edgecolor=REGION_COLOR,
                               linewidth=L["region_linewidth"], linestyle=(0, (2, 1.5)), zorder=5, clip_on=False))
    ax.set_xticks(np.arange(5) + 0.5)
    ax.set_xticklabels(M_LEVELS, fontsize=L["tick_fontsize"])
    ax.set_yticks(np.arange(5) + 0.5)
    ax.set_yticklabels(S_LEVELS, fontsize=L["tick_fontsize"])
    ax.tick_params(length=0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=L["title_fontsize"], pad=L["title_pad"], loc=L["title_loc"])
    if not annotate_regions:
        return
    lab = dict(fontsize=L["region_fontsize"], fontweight=L["region_fontweight"], color=REGION_COLOR, clip_on=False)
    t = L["id_label"]
    ax.text(t["x"], t["y"], t["text"], ha=t["ha"], va=t["va"], fontweight="bold",
            fontsize=L["region_fontsize"], color=ID_EDGE, clip_on=False)
    ln = L["oodm_line"]
    ax.plot([ln["x0"], ln["x1"]], [ln["y"], ln["y"]], lw=L["region_linewidth"], color=REGION_COLOR,
            solid_capstyle="butt", clip_on=False)
    t = L["oodm_label"]
    ax.text(t["x"], t["y"], t["text"], ha=t["ha"], va=t["va"], **lab)
    ln = L["oods_line"]
    ax.plot([ln["x"], ln["x"]], [ln["y0"], ln["y1"]], lw=L["region_linewidth"], color=REGION_COLOR,
            solid_capstyle="butt", clip_on=False)
    t = L["oods_label"]
    ax.text(t["x"], t["y"], t["text"], ha=t["ha"], va=t["va"], rotation=t.get("rotation", 0), **lab)
    t = L["oodx_label"]
    ax.text(t["x"], t["y"], t["text"], ha=t["ha"], va=t["va"], **lab)


def add_vcolorbar(fig, ax, cmap, norm, ticks, ticklabels):
    """Vertical colorbar to the right of one axes."""
    L = LAYOUT
    bb = ax.get_position()
    w = L["cbar_width_frac"] * bb.width
    cax = fig.add_axes([bb.x1 + L["cbar_gap_frac"] * bb.width, bb.y0, w, bb.height])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="vertical", ticks=ticks)
    cb.ax.set_yticklabels(ticklabels)
    cb.ax.tick_params(labelsize=L["cbar_tick_fontsize"], length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    return cb


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caliber", choices=list(SUFFIX), default="early",
                    help="Phi open-placement caliber of the buckets; every panel must share it")
    ap.add_argument("--panels", default="main,base,detect_off,rl",
                    help="comma list; first must be 'main' (absolute acc), the rest are difference panels "
                         f"from {sorted(DELTA_PANELS)}; 3 panels -> 1x3, 4 panels -> 2x2")
    ap.add_argument("--rl-caliber", choices=list(SUFFIX), default=None,
                    help="preview only: compute the RL panel (RL and its SFT init) on a different caliber "
                         "than the other panels; the final figure must use one caliber throughout")
    ap.add_argument("--region-shapes", choices=["all", "first"], default="all",
                    help="draw the ID box and the OOD-X dashed box on every panel (all) or only on panel (a) (first)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    panels = args.panels.split(",")
    assert panels[0] == "main" and all(p in DELTA_PANELS for p in panels[1:]), panels
    assert len(panels) in (3, 4), "supported layouts: 3 panels (1x3) or 4 panels (2x2)"
    L = LAYOUT

    main_acc, main_n, main_fam = grid_of(load_rows("main", args.caliber))
    grids, calibers = {"main": main_acc}, {"main": args.caliber}
    for p in panels[1:]:
        cal = args.rl_caliber if (p == "rl" and args.rl_caliber) else args.caliber
        acc, n, _ = grid_of(load_rows(p, cal))
        assert (n == main_n).all(), f"arm {p} disagrees with main on per-cell n"
        grids[p], calibers[p] = acc, cal
    rl_cal = calibers.get("rl", args.caliber)
    main_for_rl = main_acc if rl_cal == args.caliber else grid_of(load_rows("main", rl_cal))[0]

    deltas = {}
    for p in panels[1:]:
        ref = main_for_rl if p == "rl" else main_acc
        deltas[p] = (grids[p] - ref) if DELTA_PANELS[p][1] == "arm_minus_main" else (ref - grids[p])
    lim = max(0.7, float(np.nanmax(np.abs(np.stack(list(deltas.values()))))))
    seq_norm, div_norm = plt.Normalize(0, 1), TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

    nrow, ncol = (1, 3) if len(panels) == 3 else (2, 2)
    row_h = L["row_height_in"] if nrow == 1 else L["row_height_in_2x2"]
    fig, axes = plt.subplots(nrow, ncol, figsize=(L["fig_width_in"], row_h * nrow))
    axes = np.atleast_1d(axes).ravel()
    fig.subplots_adjust(**(L["margins"] if nrow == 1 else L["margins_2x2"]))

    letters = "abcd"
    draw_panel(axes[0], main_acc, title=f"({letters[0]}) acc, NSDL 7B (SFT)", cmap=SEQ, norm=seq_norm,
               fmt=fmt_acc, annotate_regions=True)
    for i, p in enumerate(panels[1:], start=1):
        draw_panel(axes[i], deltas[p], title=f"({letters[i]}) {DELTA_PANELS[p][0]}", cmap=DIV, norm=div_norm,
                   fmt=fmt_delta, annotate_regions=False, region_shapes=(args.region_shapes == "all"))
    # One shared label per axis for the whole figure (row / column tick labels repeat per panel).
    grid_x = 0.5 * (axes[0].get_position().x0 + axes[ncol - 1].get_position().x1)
    grid_y = 0.5 * (axes[0].get_position().y1 + axes[-1].get_position().y0)
    fig.text(grid_x, L["xlabel_y"], "structural planning depth M", ha="center", va="bottom",
             fontsize=L["axis_label_fontsize"])
    fig.text(L["ylabel_x"], grid_y, "state judgment depth S", ha="left", va="center", rotation=90,
             fontsize=L["axis_label_fontsize"])

    # Colorbars: one per panel, to its right; every difference panel shares the same limits.
    lim_lab = f"{lim:.1f}".lstrip("0")
    add_vcolorbar(fig, axes[0], SEQ, seq_norm, [0, 0.5, 1], ["0", ".5", "1"])
    for i in range(1, len(panels)):
        add_vcolorbar(fig, axes[i], DIV, div_norm, [-lim, 0, lim], [f"−{lim_lab}", "0", f"+{lim_lab}"])

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(out.with_suffix("." + ext), dpi=600 if ext == "png" else None)

    cells = {}
    for s in range(5):
        for m in range(5):
            if (s, m) in ABSENT:
                continue
            entry = {
                "n": int(main_n[s, m]),
                "split": ("test_id" if s <= 2 and m <= 2 else "ood_s" if m <= 2 else "ood_m" if s <= 2 else "ood_x"),
                "families": {f: {"correct": v[0], "n": v[1]} for f, v in sorted(main_fam[(s, m)].items())},
                "acc_main_sft": round(float(main_acc[s, m]), 4),
            }
            for p in panels[1:]:
                entry[f"acc_{p}"] = round(float(grids[p][s, m]), 4)
                entry[f"delta_{p}"] = round(float(deltas[p][s, m]), 4)
            cells[f"S{s}xM{m}"] = entry
    materials = {
        "figure": str(out.with_suffix(".pdf").relative_to(REPO)) if REPO in out.parents else str(out.with_suffix(".pdf")),
        "panels": panels,
        "calibers": calibers,
        "mixed_caliber_preview_only": len(set(calibers.values())) > 1,
        "buckets": {p: ARMS[p] + SUFFIX[calibers[p]] for p in panels},
        "delta_definitions": {p: DELTA_PANELS[p] for p in panels[1:]},
        "absent_cells": [f"S{s}xM{m}" for s, m in sorted(ABSENT)],
        "id_block": "S0-S2 x M0-M2",
        "region_shapes": args.region_shapes,
        "diverging_limit": round(lim, 3),
        "cells": cells,
    }
    mat = out.parent / (out.name.replace("fig3_locus_heatmap", "fig3_materials_locus_grid") + ".json")
    mat.write_text(json.dumps(materials, indent=2, ensure_ascii=False))
    print(f"wrote {out}.{{pdf,png,svg}} + {mat.name}  panels={panels} calibers={calibers}")


if __name__ == "__main__":
    main()
