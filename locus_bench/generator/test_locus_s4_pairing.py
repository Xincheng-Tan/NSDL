"""P7.4b S4 counterfactual pairing gates (benchmark.md §4.2/§7.10)."""

import json
import os
import random
import sys

_ROBOFACTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROBOFACTORY)
sys.path.insert(0, os.path.join(_ROBOFACTORY, "script_general"))

from locus_generate import (  # noqa: E402
    instantiate_counterfactual_pair, instantiate_locus_task,
)
from locus_render_pack import (  # noqa: E402
    OPAQUE_STYLES, TRANSPARENT_STYLES, build_render_config,
)


def _target_pos(inst):
    target = inst["meta"]["mask_map"]["mask1"]
    return next(list(pos) for name, pos in inst["init_pos"].items()
                if name.rsplit("_", 1)[0] == target)


def _has_open(inst):
    return any(op[0] == "Open" for step in inst["ground_truth"]
               for op in step.values())


def _objects_without_target(cfg, target):
    return {
        obj["name"]: obj for obj in cfg["objects"]
        if obj["name"].rsplit("_", 1)[0] != target
    }


def test_s4_counterfactual_pair_invariants():
    for m in (1, 2, 3):
        hidden, counter = instantiate_counterfactual_pair(
            "F4", 4, m, random.Random(100 + m),
            inject_distractors=False)
        assert hidden["meta"]["state_realization"] == {"mask1": "cabinet"}
        assert counter["meta"]["state_realization"] == {"mask1": "counter"}
        assert hidden["description"] == counter["description"]
        assert hidden["layout_id"] == counter["layout_id"]
        assert hidden["meta"]["mask_map"] == counter["meta"]["mask_map"]
        assert hidden["meta"]["counterfactual_pair_id"] == \
            counter["meta"]["counterfactual_pair_id"]
        assert hidden["meta"]["style_seed"] == counter["meta"]["style_seed"]
        assert hidden["meta"]["base_mechanical_signature"] == \
            counter["meta"]["base_mechanical_signature"]
        assert hidden["meta"]["state_entailed_open"]
        assert not counter["meta"]["state_entailed_open"]
        assert _target_pos(hidden) == ["cabinet"]
        assert _target_pos(counter) != ["cabinet"]
        assert _has_open(hidden)
        if m < 3:
            assert not _has_open(counter)


def test_s4_pair_render_style_camera_and_non_target_assets_match():
    hidden, counter = instantiate_counterfactual_pair(
        "F4", 4, 2, random.Random(9), inject_distractors=False)
    hidden_cfg, hidden_style = build_render_config(hidden)
    counter_cfg, counter_style = build_render_config(counter)
    assert hidden_style == counter_style
    assert hidden_style in OPAQUE_STYLES
    assert hidden_cfg["cameras"] == counter_cfg["cameras"]
    assert hidden_cfg["agents"] == counter_cfg["agents"]
    target = hidden["meta"]["mask_map"]["mask1"]
    assert json.dumps(_objects_without_target(hidden_cfg, target),
                      sort_keys=True) == \
        json.dumps(_objects_without_target(counter_cfg, target),
                   sort_keys=True)


def test_non_s4_cabinet_control_remains_transparent():
    inst = instantiate_locus_task(
        "F4", 2, 1, random.Random(7),
        realization={"mask1": "cabinet"}, inject_distractors=False)
    _, style = build_render_config(inst)
    assert not inst["meta"]["opaque_container"]
    assert style in TRANSPARENT_STYLES
    assert not inst["meta"]["state_entailed_open"]

