"""P7.4b family/operator bridge acceptance gates (benchmark.md §4.5)."""

import json
import os
import random
import sys
from collections import Counter

_ROBOFACTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROBOFACTORY)
sys.path.insert(0, os.path.join(_ROBOFACTORY, "script_general"))

from locus_generate import generate_locus_cell, instantiate_locus_task  # noqa: E402
from locus_task_pool import iter_cells, make_template  # noqa: E402


def _actions(tpl):
    return [op[0] for step in tpl["ground_truth"] for op in step.values()]


def test_explicit_registry_and_train_family_coverage():
    cells = list(iter_cells())
    assert len(cells) == 49
    assert not any(f == "F4" and s == 3 for f, s, _ in cells)

    train = Counter(f for f, s, m in cells if s <= 2 and m <= 2)
    ood_families = {f for f, s, m in cells if s >= 3 or m >= 3}
    assert all(train[f] > 0 for f in ood_families)
    assert all(s <= 2 and m <= 2
               for f, s, m in cells if s <= 2 and m <= 2)


def test_state_operator_bridges_are_bounded():
    f3_s2 = make_template("F3", 2, 1)
    f3_s3 = make_template("F3", 3, 1)
    assert f3_s2["bridge_kind"] == "tag_quantifier_k2"
    assert len(f3_s2["state_realization_spec"]["candidates"]) == 2
    assert len(f3_s3["state_realization_spec"]["candidates"]) == 3

    f5_s2 = make_template("F5", 2, 2)
    f5_s3 = make_template("F5", 3, 2)
    assert f5_s2["bridge_kind"] == "bounded_clear_set_k2"
    assert len(f5_s2["state_realization_spec"]["candidates"]) == 2
    assert len(f5_s3["state_realization_spec"]["candidates"]) == 3

    f4_s2 = make_template("F4", 2, 1)
    f4_s4 = make_template("F4", 4, 1)
    assert f4_s2["bridge_kind"] == "transparent_fallback"
    assert not f4_s2["opaque_container"] and f4_s4["opaque_container"]
    assert f4_s2["description"] == f4_s4["description"]

    samples = generate_locus_cell("F3", 2, 1, 20, random.Random(4))
    for inst in samples:
        assert len(inst["meta"]["distractors"]) <= 1
        assert all(d["kind"] == "clutter"
                   for d in inst["meta"]["distractors"])
        assert inst["meta"]["observation_tool"] == {
            "name": "detect",
            "args": {"query": {"tag": "fruit"}},
        }


def test_mechanical_operator_bridges_hold_out_full_composition():
    f8_m2 = make_template("F8", 0, 2)
    f8_m3 = make_template("F8", 0, 3)
    assert f8_m2["bridge_kind"] == "independent_deposit_twin"
    assert f8_m2["mask3"] == ["rack"]
    assert "Open" not in _actions(f8_m2)
    assert f8_m3["mask3"] == ["cabinet"]
    assert "Open" in _actions(f8_m3)

    f9_m2 = make_template("F9", 0, 2)
    f9_m2_load = make_template("F9", 0, 2, bridge_variant="load")
    f9_m2_unload = make_template("F9", 0, 2, bridge_variant="unload")
    f9_m3 = make_template("F9", 0, 3)
    f9_m4 = make_template("F9", 0, 4)
    assert f9_m2["bridge_kind"] == "carrier_components_two_robot"
    assert f9_m3["bridge_kind"] == "load_push_subchain"
    assert len(f9_m2["robot_roles"]) == len(f9_m3["robot_roles"]) == 2
    assert len(f9_m4["robot_roles"]) == 3
    assert _actions(f9_m2) == ["Move", "Push"]
    assert _actions(f9_m2_load) == ["Reach", "Grasp", "Place"]
    assert _actions(f9_m2_unload) == ["Reach", "Grasp", "Place"]
    assert "Push" in _actions(f9_m3)
    assert sum(a == "Push" for a in _actions(f9_m4)) == 2
    assert any(goal[0]["status"].get("pos.name") == "<mask2>"
               for goal in f9_m3["goal_constraints"])
    assert f9_m4["goal_constraints"][0][0]["status"]["pos.name"] == "<mask3>"


def test_lookup_blacklist_and_detect_wire_shape():
    instances = [
        instantiate_locus_task(f, s, m, random.Random(i),
                               inject_distractors=False)
        for i, (f, s, m) in enumerate(iter_cells())
    ]
    serialized = json.dumps(instances, sort_keys=True).lower()
    assert '"name": "lookup"' not in serialized
    assert '"lookup"' not in serialized
    f3 = next(i for i in instances if i["meta"]["bench_family"] == "F3")
    assert f3["meta"]["observation_tool"]["name"] == "detect"
    assert f3["meta"]["observation_tool"]["args"] == {
        "query": {"tag": "fruit"}}
