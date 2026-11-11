from __future__ import annotations

from pathlib import Path


FULL_CLASS_LIST = [
    "fetch",
    "stompy",
    "unitree_h1",
    "panda",
    "unitree_go2",
    "anymal_c",
    "apple",
    "banana",
    "bottle",
    "bowl",
    "bread",
    "cabinet",
    "cardboardbox",
    "cooling_fan",
    "cup",
    "drawer",
    "fork",
    "fridge",
    "green_peas",
    "kettle",
    "knife",
    "meat",
    "oven",
    "peach",
    "pear",
    "pizza",
    "plate",
    "pumpkin",
    "rack",
    "scissors",
    "sink",
    "spoon",
    "table",
    "toaster",
    "tomato",
    "wine",
    "wooden_cutting_board",
    "kitchen_work_area",
    "kitchen_island_area",
]


def ordered_subset(labels: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    wanted = set(labels)
    return [label for label in FULL_CLASS_LIST if label in wanted]


# Pure semantic destination tokens in the current benchmark. They never get
# real init_pos slots in VIKI-L2 and only appear as Move/Place targets, so the
# planner can consume them directly from instruction semantics.
SEMANTIC_ONLY_TARGETS = [
    "rack",
    "table",
    "kitchen_work_area",
    "kitchen_island_area",
]

# These are still valid destination tokens when detector evidence is missing,
# but unlike semantic-only targets they can also be real movable objects.
SEMANTIC_FALLBACK_TARGETS = [
    "bowl",
    "plate",
]

# Default training labels: L1 robot visibility plus the small set of L2 labels
# whose visual presence/absence genuinely changes macro-graph construction.
# Current examples:
# - conditional dish verification: bowl / plate
# - ensure_all_fruits_on_table: apple / banana / peach / pear / tomato
# - serve_*_after_checking_cabinet / from_counter: apple / banana / bread
BASE_LABELS = [
    "fetch",
    "stompy",
    "unitree_h1",
    "panda",
    "unitree_go2",
    "anymal_c",
    "apple",
    "banana",
    "bowl",
    "bread",
    "peach",
    "pear",
    "plate",
    "tomato",
]

# Optional second-pass labels. These labels may still improve robustness or be
# useful for ablations, but current macro-graph generation does not require
# them as first-order visual anchors.
SUPPLEMENT_LABELS = [
    "cabinet",
    "bottle",
    "cardboardbox",
    "cooling_fan",
    "cup",
    "drawer",
    "fork",
    "fridge",
    "green_peas",
    "kettle",
    "knife",
    "meat",
    "oven",
    "pizza",
    "pumpkin",
    "scissors",
    "sink",
    "spoon",
    "toaster",
    "wine",
    "wooden_cutting_board",
]

# Historical pilot split is now aligned with the current two-folder review
# workflow:
# - xlabels: legacy 15-label group that was already human-checked
# - xlabels2: new 5-label group for cabinet/table/bowl/plate/sink
# The benchmark-driven base/supplement policy above remains separate.
PILOT_V1_LABELS = ordered_subset(
    [
        "fetch",
        "stompy",
        "unitree_h1",
        "panda",
        "unitree_go2",
        "anymal_c",
        "apple",
        "banana",
        "bread",
        "knife",
        "peach",
        "pear",
        "toaster",
        "tomato",
        "wooden_cutting_board",
    ]
)

# New second-pass labels that live in the separate xlabels2 folder.
PILOT_EXTENSION_LABELS = ordered_subset(
    [
        "cabinet",
        "table",
        "bowl",
        "plate",
        "sink",
    ]
)

PILOT_COMPLETE_LABELS = ordered_subset(PILOT_V1_LABELS + PILOT_EXTENSION_LABELS)

# LOCUS-Bench detector taxonomy. Preserve the deployed full20 class order so
# annotations/checkpoints remain easy to audit, then append the 14 classes that
# are visible operands or fixtures in the current LOCUS templates. Semantic
# regions (rack/work area/island area) stay out of the detector taxonomy.
LOCUS_FULL34_LABELS = [
    "fetch",
    "stompy",
    "unitree_h1",
    "panda",
    "unitree_go2",
    "anymal_c",
    "apple",
    "banana",
    "bowl",
    "bread",
    "cabinet",
    "knife",
    "peach",
    "pear",
    "plate",
    "sink",
    "table",
    "toaster",
    "tomato",
    "wooden_cutting_board",
    "bottle",
    "cardboardbox",
    "cup",
    "drawer",
    "fork",
    "fridge",
    "kettle",
    "meat",
    "oven",
    "pizza",
    "pumpkin",
    "scissors",
    "spoon",
    "wine",
]


FULL_BENCHMARK_LABELS = ordered_subset(BASE_LABELS + SUPPLEMENT_LABELS)

LABEL_SCOPES = {
    "full_ontology": FULL_CLASS_LIST,
    "base": BASE_LABELS,
    "supplement": SUPPLEMENT_LABELS,
    "full_benchmark": FULL_BENCHMARK_LABELS,
    "semantic_only": SEMANTIC_ONLY_TARGETS,
    "semantic_fallback": SEMANTIC_FALLBACK_TARGETS,
    "pilot_v1": PILOT_V1_LABELS,
    "pilot_extension": PILOT_EXTENSION_LABELS,
    "pilot_complete": PILOT_COMPLETE_LABELS,
    "locus_full34": LOCUS_FULL34_LABELS,
}

# Backward-compatible aliases for older local commands.
LEGACY_SCOPE_ALIASES = {
    "first_round": "base",
    "full_training": "full_benchmark",
}

CLASS_FILE_NAMES = {
    "full_class_list.txt": FULL_CLASS_LIST,
    "base_class_list.txt": BASE_LABELS,
    "supplement_class_list.txt": SUPPLEMENT_LABELS,
    "full_benchmark_class_list.txt": FULL_BENCHMARK_LABELS,
    "semantic_only_target_list.txt": SEMANTIC_ONLY_TARGETS,
    "semantic_fallback_target_list.txt": SEMANTIC_FALLBACK_TARGETS,
    "first_round_class_list.txt": BASE_LABELS,
    "full_training_class_list.txt": FULL_BENCHMARK_LABELS,
    "pilot_v1_class_list.txt": PILOT_V1_LABELS,
    "pilot_extension_class_list.txt": PILOT_EXTENSION_LABELS,
    "pilot_complete_class_list.txt": PILOT_COMPLETE_LABELS,
    "locus_full34_class_list.txt": LOCUS_FULL34_LABELS,
}


def get_label_scope(name: str) -> list[str]:
    scope_name = LEGACY_SCOPE_ALIASES.get(name, name)
    if scope_name not in LABEL_SCOPES:
        valid = ", ".join(sorted(set(LABEL_SCOPES) | set(LEGACY_SCOPE_ALIASES)))
        raise ValueError(f"Unknown label scope '{name}'. Expected one of: {valid}")
    return LABEL_SCOPES[scope_name]


def write_class_lists(output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for filename, labels in CLASS_FILE_NAMES.items():
        path = output_dir / filename
        path.write_text("\n".join(labels) + "\n", encoding="utf-8")
        written[filename] = str(path)
    return written


def _assert_partition() -> None:
    full_set = set(FULL_CLASS_LIST)
    semantic_only_set = set(SEMANTIC_ONLY_TARGETS)
    semantic_fallback_set = set(SEMANTIC_FALLBACK_TARGETS)
    base_set = set(BASE_LABELS)
    supplement_set = set(SUPPLEMENT_LABELS)
    benchmark_set = set(FULL_BENCHMARK_LABELS)
    pilot_v1_set = set(PILOT_V1_LABELS)
    pilot_extension_set = set(PILOT_EXTENSION_LABELS)
    pilot_complete_set = set(PILOT_COMPLETE_LABELS)
    locus_full34_set = set(LOCUS_FULL34_LABELS)

    if base_set & supplement_set:
        raise ValueError("Base and supplement label sets must not overlap.")
    if base_set | supplement_set != benchmark_set:
        raise ValueError("Base + supplement labels must cover the full benchmark label scope.")
    if benchmark_set & semantic_only_set:
        raise ValueError("Semantic-only targets must stay out of the default benchmark label scope.")
    if benchmark_set | semantic_only_set != full_set:
        raise ValueError("Benchmark labels plus semantic-only targets must cover the full ontology.")
    if semantic_fallback_set - benchmark_set:
        raise ValueError("Semantic fallback targets must remain in the benchmark label scope.")
    if pilot_v1_set & pilot_extension_set:
        raise ValueError("Pilot v1 and pilot extension label groups must not overlap.")
    if pilot_complete_set != pilot_v1_set | pilot_extension_set:
        raise ValueError("Pilot complete labels must equal pilot_v1 + pilot_extension.")
    if pilot_complete_set - full_set:
        raise ValueError("Pilot label groups must stay within the ontology.")
    if len(LOCUS_FULL34_LABELS) != 34 or len(locus_full34_set) != 34:
        raise ValueError("LOCUS full34 must contain exactly 34 unique labels.")
    if locus_full34_set - full_set:
        raise ValueError("LOCUS full34 labels must stay within the ontology.")
    if LOCUS_FULL34_LABELS[: len(PILOT_COMPLETE_LABELS)] != PILOT_COMPLETE_LABELS:
        raise ValueError("LOCUS full34 must preserve the deployed full20 class order.")


_assert_partition()
