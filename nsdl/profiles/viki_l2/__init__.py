from __future__ import annotations

from typing import Any

from nsdl.catalog import (
    CAP_ACTIVATION,
    CAP_CONTAINER_OPERATION,
    CAP_DUAL_EE,
    CAP_MANIPULATION,
    CAP_MOBILITY,
    CAP_PUSH_TRANSFER,
    ACTION_PRIMITIVES,
    EntityProfile,
    MOVE,
    NODE_KIND_SPECS,
    OPEN,
    PLACE,
    PUSH,
    REACH,
    GRASP,
    CLOSE,
    INTERACT,
    RobotProfile,
    normalize_entity_name,
)
from nsdl.profiles.base import (
    BenchmarkProfile,
    OpSpec,
    OpVocabProfile,
    StaticCatalogProvider,
    SystemPromptProvider,
)
from nsdl.profiles.viki_l2.emitter import VIKI_L2_DISPATCHER_EMITTER


VIKI_L2_OPERATION_SPECS = (
    OpSpec(
        name="acquire",
        description="Navigate to and pick up a portable item with a mobile manipulator.",
        required_capabilities=(CAP_MOBILITY, CAP_MANIPULATION),
        aliases=("pickup", "pick_up"),
        system_prompt_block=(
            "- acquire: Navigate to and pick up an item. Requires: mobility + manipulation."
        ),
    ),
    OpSpec(
        name="local_acquire",
        description="Pick up an item already inside a stationary/local manipulator workspace.",
        required_capabilities=(CAP_MANIPULATION,),
        aliases=("local_pickup", "local_pick_up"),
        system_prompt_block=(
            "- local_acquire: Pick up an item already in a local workspace, e.g. a Panda beside the item. "
            "Requires: manipulation."
        ),
    ),
    OpSpec(
        name="carry_to",
        description="Move while holding an item/tool to a destination without placing it.",
        required_capabilities=(CAP_MOBILITY,),
        aliases=("carry", "transport"),
        system_prompt_block=(
            "- carry_to: Move while holding an item/tool to a destination, without placing it. Requires: mobility."
        ),
    ),
    OpSpec(
        name="deposit",
        description="Place a held item into or onto a destination.",
        required_capabilities=(CAP_MANIPULATION,),
        aliases=("place", "put"),
        system_prompt_block=(
            "- deposit: Place a held item at destination. Requires: manipulation. "
            "If dest is openable+container, open must happen before final placement."
        ),
    ),
    OpSpec(
        name="open",
        description="Open an openable container before depositing into it or retrieving from it.",
        required_capabilities=(CAP_CONTAINER_OPERATION,),
        system_prompt_block="- open: Open an openable container. Requires: container_operation.",
    ),
    OpSpec(
        name="close",
        description="Close an openable container after interaction is finished.",
        required_capabilities=(CAP_CONTAINER_OPERATION,),
        system_prompt_block=(
            "- close: Close an openable container. Requires: container_operation. "
            "Emit only when the instruction explicitly asks to close/shut a container; "
            "VIKI-L2 traces never require this op."
        ),
    ),
    OpSpec(
        name="process_at",
        description="Process an item at a station (wash/cook/clean/freeze/...).",
        required_capabilities=(CAP_MANIPULATION, CAP_ACTIVATION),
        aliases=("wash", "cook", "clean", "freeze"),
        system_prompt_block=(
            "- process_at: Process item at a station. Requires: manipulation + activation. "
            "`params.process` \u2208 {\"wash\",\"cook\",\"clean\",\"freeze\"} (default \"wash\"); "
            "choose by station tag (wash_station\u2192wash; cookable_station/oven/microwave\u2192cook; "
            "cleaning_station\u2192clean; freezer\u2192freeze)."
        ),
    ),
    OpSpec(
        name="apply_tool",
        description="Use a held tool at a target surface or on an item (cut/slice/peel/...).",
        required_capabilities=(CAP_ACTIVATION,),
        aliases=("cut", "slice", "tool_use", "use_tool", "peel"),
        system_prompt_block=(
            "- apply_tool: Use a held tool at a target/surface (e.g. cut fruit on board with knife). "
            "Requires: activation. `params.tool` names the held tool; "
            "`params.effect` \u2208 {\"cut\",\"slice\",\"peel\"} (default \"cut\"); "
            "choose by tool/target tags (sharp+sliceable\u2192slice; sharp+cuttable\u2192cut)."
        ),
    ),
    OpSpec(
        name="activate",
        description="Activate or interact with a fixture after its load preconditions are satisfied.",
        required_capabilities=(CAP_ACTIVATION,),
        system_prompt_block=(
            "- activate: Activate fixture. Requires: activation. "
            "`params.polarity` ∈ {\"on\",\"off\"} (default \"on\"); "
            "choose \"off\" when the instruction says \"turn/switch off\", \"shut off\", "
            "\"power down\", \"deactivate\"; otherwise \"on\"."
        ),
    ),
    OpSpec(
        name="load_carrier",
        description="Place a held item into a pushable carrier.",
        required_capabilities=(CAP_MANIPULATION,),
        aliases=("load", "load_into_carrier"),
        system_prompt_block=(
            "- load_carrier: Place a held item into a pushable carrier. Requires: manipulation."
        ),
    ),
    OpSpec(
        name="push_carrier",
        description="Push a carrier toward a robot slot or handoff boundary.",
        required_capabilities=(CAP_PUSH_TRANSFER,),
        aliases=("push", "push_transfer"),
        system_prompt_block=(
            "- push_carrier: Push a carrier toward a robot slot or handoff boundary. Requires: push_transfer."
        ),
    ),
    OpSpec(
        name="unload_carrier",
        description="Retrieve an item from a carrier and place it at the final destination.",
        required_capabilities=(CAP_MANIPULATION,),
        aliases=("unload", "unload_from_carrier"),
        system_prompt_block=(
            "- unload_carrier: Retrieve an item from a carrier and place it at the final destination. "
            "Requires: manipulation."
        ),
    ),
    OpSpec(
        name="handoff",
        description="Transfer custody of an item across a capability boundary.",
        system_prompt_block=(
            "- handoff: Abstract custody transfer between robots at capability boundary."
        ),
    ),
    OpSpec(
        name="inspect",
        description="Observe or check an area, object, or fixture.",
        required_capabilities=(CAP_MOBILITY,),
        system_prompt_block="- inspect: Check area/object. Requires: mobility.",
    ),
)

VIKI_L2_OP_VOCAB = OpVocabProfile(ops=VIKI_L2_OPERATION_SPECS)

CAPABILITY_VOCAB = (
    CAP_MOBILITY,
    CAP_MANIPULATION,
    CAP_CONTAINER_OPERATION,
    CAP_ACTIVATION,
    CAP_PUSH_TRANSFER,
    CAP_DUAL_EE,
)

ENTITY_PROFILES = {
    "apple": EntityProfile("apple", "asset", ("portable", "food", "fruit", "graspable"), ("apple",)),
    "banana": EntityProfile("banana", "asset", ("portable", "food", "fruit", "graspable"), ("banana",)),
    "bread": EntityProfile("bread", "asset", ("portable", "food", "graspable"), ("bread", "loaf of bread")),
    "bottle": EntityProfile("bottle", "asset", ("portable", "graspable"), ("bottle",)),
    "bowl": EntityProfile(
        "bowl",
        "asset",
        ("portable", "container", "support", "place_target", "graspable"),
        ("bowl",),
    ),
    "cabinet": EntityProfile(
        "cabinet",
        "fixture",
        ("container", "openable", "place_target", "storage"),
        ("cabinet", "kitchen cabinet"),
    ),
    "cardboardbox": EntityProfile(
        "cardboardbox",
        "asset",
        # NOTE (F1, 2026-05-22): also carry `portable`/`graspable` so that
        # `single_move_asset_to_target / cardboardbox` plans (where the box
        # is the moved item, not a carrier) are not rejected by validator
        # V1b (`fixture_used_as_item` / `destination_used_as_item`).
        ("container", "place_target", "pushable_carrier", "portable", "graspable"),
        ("cardboardbox", "cardboard box", "box"),
    ),
    "counter": EntityProfile(
        "counter",
        "area",
        ("support", "area", "source_area"),
        ("counter", "countertop"),
    ),
    "cooling fan": EntityProfile(
        "cooling fan",
        "asset",
        ("portable", "graspable"),
        ("cooling fan", "fan"),
    ),
    "cup": EntityProfile("cup", "asset", ("portable", "container", "place_target", "graspable"), ("cup",)),
    "drawer": EntityProfile(
        "drawer",
        "fixture",
        ("container", "openable", "place_target", "inspectable"),
        ("drawer", "kitchen drawer"),
    ),
    "fork": EntityProfile("fork", "asset", ("portable", "graspable"), ("fork",)),
    "fridge": EntityProfile(
        "fridge",
        "fixture",
        ("container", "openable", "place_target", "inspectable"),
        ("fridge", "refrigerator"),
    ),
    "green peas": EntityProfile(
        "green peas",
        "asset",
        ("portable", "food", "graspable"),
        ("green peas", "peas"),
    ),
    "kettle": EntityProfile("kettle", "asset", ("portable", "graspable"), ("kettle",)),
    "kitchen island area": EntityProfile(
        "kitchen island area",
        "area",
        ("support", "area", "place_target"),
        ("kitchen island area", "kitchen island", "island area"),
    ),
    "kitchen work area": EntityProfile(
        "kitchen work area",
        "area",
        ("support", "area", "place_target"),
        ("kitchen work area", "work area"),
    ),
    "knife": EntityProfile(
        "knife",
        "asset",
        ("portable", "tool", "graspable", "interactable_tool"),
        ("knife",),
    ),
    "meat": EntityProfile("meat", "asset", ("portable", "food", "graspable"), ("meat",)),
    "oven": EntityProfile(
        "oven",
        "fixture",
        ("activatable", "inspectable"),
        ("oven",),
    ),
    "peach": EntityProfile("peach", "asset", ("portable", "food", "fruit", "graspable"), ("peach",)),
    "pear": EntityProfile("pear", "asset", ("portable", "food", "fruit", "graspable"), ("pear",)),
    "pizza": EntityProfile("pizza", "asset", ("portable", "food", "graspable"), ("pizza",)),
    "plate": EntityProfile(
        "plate",
        "asset",
        ("portable", "container", "support", "place_target", "graspable"),
        ("plate",),
    ),
    "pumpkin": EntityProfile("pumpkin", "asset", ("portable", "food", "graspable"), ("pumpkin",)),
    "rack": EntityProfile("rack", "fixture", ("support", "area", "place_target"), ("rack",)),
    "scissors": EntityProfile("scissors", "asset", ("portable", "graspable"), ("scissors",)),
    "sink": EntityProfile(
        "sink",
        "fixture",
        ("support", "place_target", "activatable", "wash_station"),
        ("sink", "tap"),
    ),
    "spoon": EntityProfile("spoon", "asset", ("portable", "graspable"), ("spoon",)),
    "table": EntityProfile("table", "fixture", ("support", "area", "place_target"), ("table",)),
    "toaster": EntityProfile(
        "toaster",
        "fixture",
        ("container", "place_target", "activatable"),
        ("toaster",),
    ),
    "tomato": EntityProfile("tomato", "asset", ("portable", "food", "fruit", "graspable"), ("tomato",)),
    "wine": EntityProfile("wine", "asset", ("portable", "graspable"), ("wine",)),
    "wooden cutting board": EntityProfile(
        "wooden cutting board",
        "fixture",
        ("support", "place_target", "cutting_surface"),
        ("wooden cutting board", "cutting board", "board"),
    ),
}

ROBOT_PROFILES = {
    "panda": RobotProfile(
        robot_type="panda",
        capabilities=(CAP_MANIPULATION, CAP_CONTAINER_OPERATION, CAP_ACTIVATION),
        benchmark_actions=(REACH, GRASP, PLACE, OPEN, CLOSE, INTERACT),
        action_primitives=("reach_object", "grasp_item", "place_item", "open_container", "close_container", "interact_with"),
        end_effector_num=1,
        embodiment_summary="Stationary single-arm manipulator for local grasping and container interaction.",
        hard_limits=(
            "No mobility base: cannot satisfy nodes that require navigation before execution.",
            "Single end effector only.",
        ),
        planning_notes=(
            "Use when the target is already within the manipulator workspace.",
            "Do not assign transport-heavy nodes that require moving across the scene.",
        ),
    ),
    "fetch": RobotProfile(
        robot_type="fetch",
        capabilities=(CAP_MOBILITY, CAP_MANIPULATION, CAP_CONTAINER_OPERATION, CAP_ACTIVATION),
        benchmark_actions=(MOVE, REACH, GRASP, PLACE, OPEN, CLOSE, INTERACT),
        action_primitives=(
            "move_to",
            "reach_object",
            "grasp_item",
            "place_item",
            "open_container",
            "close_container",
            "interact_with",
        ),
        end_effector_num=1,
        embodiment_summary="Mobile single-arm manipulator for navigation, pickup, placement, and opening fixtures.",
        hard_limits=(
            "Single end effector only.",
        ),
        planning_notes=(
            "Default choice for single-object relocate and retrieve tasks.",
            "Prefer dual-arm robots only when a node explicitly benefits from two end effectors.",
        ),
    ),
    "unitree_go2": RobotProfile(
        robot_type="unitree_go2",
        capabilities=(CAP_MOBILITY, CAP_PUSH_TRANSFER, CAP_ACTIVATION),
        benchmark_actions=(MOVE, PUSH, INTERACT),
        action_primitives=("move_to", "push_carrier", "interact_with"),
        end_effector_num=0,
        embodiment_summary="Legged mobile robot for traversal, inspection, activation, and pushing carriers.",
        hard_limits=(
            "No grasp or place primitives.",
            "Cannot open containers or perform pick-and-place nodes independently.",
        ),
        planning_notes=(
            "Assign only nodes that can be completed with mobility, push, or interact primitives alone.",
        ),
    ),
    "unitree_h1": RobotProfile(
        robot_type="unitree_h1",
        capabilities=(CAP_MOBILITY, CAP_MANIPULATION, CAP_CONTAINER_OPERATION, CAP_ACTIVATION, CAP_DUAL_EE),
        benchmark_actions=(MOVE, REACH, GRASP, PLACE, OPEN, CLOSE, INTERACT),
        action_primitives=(
            "move_to",
            "reach_object",
            "grasp_item",
            "place_item",
            "open_container",
            "close_container",
            "interact_with",
        ),
        end_effector_num=2,
        embodiment_summary="Mobile dual-arm humanoid for transport, manipulation, fixture interaction, and two-handed setups.",
        hard_limits=(
            "None within the current VIKI action vocabulary beyond normal scene reachability.",
        ),
        planning_notes=(
            "Good fit for nodes that need mobility plus manipulation and may benefit from dual-arm slack.",
        ),
    ),
    "stompy": RobotProfile(
        robot_type="stompy",
        capabilities=(CAP_MOBILITY, CAP_MANIPULATION, CAP_CONTAINER_OPERATION, CAP_ACTIVATION, CAP_DUAL_EE),
        benchmark_actions=(MOVE, REACH, GRASP, PLACE, OPEN, CLOSE, INTERACT),
        action_primitives=(
            "move_to",
            "reach_object",
            "grasp_item",
            "place_item",
            "open_container",
            "close_container",
            "interact_with",
        ),
        end_effector_num=2,
        embodiment_summary="Mobile dual-arm robot for manipulation-heavy tasks, transport, and fixture interaction.",
        hard_limits=(
            "None within the current VIKI action vocabulary beyond normal scene reachability.",
        ),
        planning_notes=(
            "Prefer when the node is manipulation-heavy and a dual-arm platform offers execution slack.",
        ),
    ),
    "anymal_c": RobotProfile(
        robot_type="anymal_c",
        capabilities=(CAP_MOBILITY, CAP_PUSH_TRANSFER, CAP_ACTIVATION),
        benchmark_actions=(MOVE, PUSH, INTERACT),
        action_primitives=("move_to", "push_carrier", "interact_with"),
        end_effector_num=0,
        embodiment_summary="Legged mobile robot focused on traversal, inspection, activation, and carrier pushing.",
        hard_limits=(
            "No grasp or place primitives.",
            "Cannot open containers or complete pick-and-place nodes independently.",
        ),
        planning_notes=(
            "Reserve for mobility-only, push-transfer, or interaction-only nodes.",
        ),
    ),
}

ROBOT_PRIORITY = {
    "fetch": 0,
    "unitree_h1": 1,
    "stompy": 2,
    "panda": 3,
    "unitree_go2": 4,
    "anymal_c": 5,
}

ROBOT_ALIAS_GROUPS = {
    "panda": ("panda",),
    "fetch": ("fetch",),
    "unitree_go2": ("unitree_go2", "unitree go2", "go2"),
    "unitree_h1": ("unitree_h1", "unitree h1", "h1"),
    "stompy": ("stompy",),
    "anymal_c": ("anymal_c", "anymal c"),
}

VIKI_L2_CATALOG = StaticCatalogProvider(
    capability_vocab=CAPABILITY_VOCAB,
    entity_profiles=ENTITY_PROFILES,
    robot_profiles=ROBOT_PROFILES,
    action_primitives=ACTION_PRIMITIVES,
    node_kind_specs=NODE_KIND_SPECS,
    robot_priority=ROBOT_PRIORITY,
    robot_alias_groups=ROBOT_ALIAS_GROUPS,
)


def _scene_instruction_lines(scene_facts: Any, affordance_context: Any) -> list[str]:
    return [f"Instruction: {scene_facts.instruction}"]


def _scene_robot_lines(scene_facts: Any, affordance_context: Any) -> list[str]:
    """Robots block (observation): slot=type plus optional anchor location.

    Robot anchor status is observation-grounded geometry (e.g. which Panda is
    on the sink support surface in push-assist tasks); folding it into the
    robot line keeps the [SCENE] block benchmark-agnostic and avoids exposing
    a separate `Robot anchor status:` marker.
    """
    anchors = scene_facts.robot_anchor_status or {}
    parts: list[str] = []
    for robot in affordance_context.robot_capabilities:
        anchor = anchors.get(robot.slot_id)
        if anchor:
            parts.append(f"{robot.slot_id}={robot.robot_type}@{anchor}")
        else:
            parts.append(f"{robot.slot_id}={robot.robot_type}")
    return [f"Robots: {', '.join(parts)}"]


VIKI_L2_SCENE_BLOCK_CONTRIBUTORS = (
    _scene_instruction_lines,
    _scene_robot_lines,
)


VIKI_L2_PROMPT_PROVIDER = SystemPromptProvider(
    intro_block=(
        "You are a vision-language multi-robot planning agent. Given the image, "
        "a natural-language instruction, and robot capabilities, produce one "
        "agentic trajectory and a final operation DAG. Read the image yourself; "
        "ask tools only when their structured observations or deterministic "
        "planning checks are useful."
    ),
    tag_semantics_block="""## Tag Semantics
- openable + container: Must open before deposit/retrieve. Requires container_operation capability. Use `close` to seal afterwards when the instruction asks for it.
- wash_station: Items can be washed here with `process_at` (params.process="wash"). Produces clean(item) state.
- cookable_station / oven / microwave: Items cooked here with `process_at` (params.process="cook").
- cleaning_station: Items cleaned here with `process_at` (params.process="clean").
- freezer: Items frozen here with `process_at` (params.process="freeze").
- activatable: Can be activated after items loaded. Requires activation capability.
- cutting_surface: Items placed here can be cut with a tool + `apply_tool` (params.effect="cut").
- pushable_carrier: Can be loaded with items, pushed by push_transfer robots, then unloaded by manipulators.
- portable + graspable: Can be picked up with acquire.
- sliceable: Item can be sliced with a tool; usually `apply_tool` with params.effect="slice".
- cookable: Item can be cooked at a cookable_station; usually `process_at` with params.process="cook".
- toggleable: Fixture can be toggled on/off; usually `activate` with appropriate params.polarity.
- breakable: Item can be destructively broken (rarely needed; not in core 14 op).
- dirty: Item needs cleaning; usually `process_at` at wash_station/cleaning_station.""",
    capability_reference_block="""## Capability Reference
- mobility: Can navigate (Move action)
- manipulation: Can pick/place (Reach, Grasp, Place actions)
- container_operation: Can open/close containers (Open, Close actions)
- activation: Can interact with fixtures (Interact action)
- push_transfer: Can push carriers (Push action)
- dual_ee: Has two end effectors, can hold two items simultaneously""",
    target_status_block="",
    robot_anchor_block="",
    inspection_role_block="",
    pattern_blocks=(
        """## Tool Specs
Tools are called by emitting exactly one JSON value after `[TOOL_CALL]`.
Use a single agentic trajectory; there is no route label and no separate coordination block.
Each assistant turn may emit at most one `[TOOL_CALL]` block. If another tool is needed, wait for the tool result, then continue in a new assistant turn.

- schedule: deterministic mechanical completion, resource scheduling, and feasibility.
  Args: {"ops":[intent op objects with robot fields], "assign":{"R1":{"items":[...]}, ...}}
  Returns: {"ok":bool, "ops":[complete DAG], "schedule":[micro steps], "inserted":[...], "reason":null|{...}}
  Call when the sketch has open-container, multi-robot transfer, carrier, handoff, end-effector, or dependency uncertainty.
- verify: feasibility check without asking the model to invent missing high-level targets.
  Args: {"ops":[op objects]}
  Returns: {"ok":bool, "code":null|string, "op_ids":[...], "message":string}
  Call when you want a structured diagnosis before finalizing or after a repair.
- verify_goal_coverage: read-only coverage diagnosis for your own `[GOAL]` against observed facts.
  Args: {"model_goal":{the GOAL object you emitted}, "observed_facts":{a detect result or {"relations":[...]}}}
  Returns: {"ok":bool, "code":null|string, "op_ids":[], "message":string}
  Call after `detect` when active/skip status depends on observed locations. It only reports contradictions between your declared goal and observations; it never rewrites the goal, never reads labels, and never returns destination-role status labels.
- expand: local lowering of one op/node into primitive actions.
  Args: {"op":{op object}}
  Returns: {"ok":bool, "primitives":[...], "end_state":[...], "reason":null|{...}}
  Call only for local lowering ambiguity; schedule already uses it internally.
- detect: query-scoped perception.
  Args: {"query":string or {"entities":[...]} or {"tag":"..."}}
  Returns: {"entities":[canonical names], "relations":["<entity> on|in|at|near <fixture>"], "confidences":{...}}
  Call when the image leaves entity presence or geometry uncertain. It returns observations only, never target/destination/status inference.
""",
        """## Intent Operation Vocabulary
`[PLAN_SKETCH]` should use only these intent ops:
- acquire: pick up an item; choose item and robot.
- deposit: place a held item at a destination; choose destination and robot.
- open: open an openable container; choose whether opening is needed and which robot opens it.
- process_at: process an item at a station, with params.process when needed.
- apply_tool: use a held tool on a target/surface, with params.tool/effect when needed.
- activate: activate/deactivate a fixture, with params.polarity when needed.
- inspect: embodied inspection when the task truly requires inspection.

Mechanical ops such as carry_to, handoff, load_carrier, push_carrier, and unload_carrier
are synthesized by `schedule` when needed. If you call `schedule` and it succeeds,
the final `[PLAN]` should adopt its returned complete DAG.
""",
        """## Polarity Cues (for activate op)
- "switch on / turn on / start / power on / activate / begin / start its operation" → params.polarity=on
- "switch off / turn off / shut off / power down / deactivate / stop / halt" → params.polarity=off
- If the instruction is silent on polarity, omit params.polarity (defaults to on).
- Polarity affects only the activate op; do not add params.polarity to other op_types.""",
    ),
    combination_rules_block="""## Planning Rules
- Complete coverage: Generate operations for every target item implied by the instruction. If N items need to reach a destination, produce N acquire operations or an equivalent carrier/transfer plan. Never stop after the first item.
- Single-robot multi-target ordering: When exactly one robot must deliver multiple target items to the same destination, every target item requires its own acquire op (one per item, not one shared). Choose between two layouts based on the robot's capabilities:
  - If the robot has dual_ee: emit N×acquire chained back-to-back, then a single deposit at the destination (carry all items in one trip).
  - If the robot lacks dual_ee: emit a per-item acquire+deposit chain in sequence (acquire item_1 → deposit item_1 → acquire item_2 → deposit item_2 → ...); each acquire is paired with its own deposit before the next acquire starts.
- Dual-hold merge: When multiple items share the same destination and the robot has dual_ee, the merged layout above is mandatory — do NOT split into separate deposit per item.
- Push direction: When [ROBOTS]/[SCENE] shows two same-type manipulators with anchor annotations (e.g. `R1=panda@on_sink_support_surface` and `R2=panda@off_sink_support_surface`), push the carrier FROM the off_sink slot TO the on_sink slot. The on_sink robot does the final unload_carrier.
- Robot slot universe is closed: Every `op.robot` value MUST be one of the slot ids listed in the [ROBOTS] block. Never invent slot ids beyond the provided set (e.g. do not use R4/R5 if only R1/R2/R3 are listed). When the same robot type appears multiple times, the slot id (R1, R2, …) is the only legal identifier.
- Capability-aware assignment: Before assigning `op.robot`, check that the robot's capability list in [ROBOTS] covers the op's requirement (acquire→mobility+manipulation, deposit→manipulation, open→container_operation, activate/process_at/apply_tool→activation). Stationary robots without mobility cannot run navigation-heavy acquire; pushing-only robots without manipulation cannot run deposit.
- Open is restricted to true openable containers: Emit `open` only for containers such as cabinets, fridges, microwaves, and drawers. Never open plain support surfaces such as tables, shelves, sofas, or counters.
- Cross-robot collaboration is expressed inside `[PLAN_SKETCH]`: if one robot acquires an item and another deposits it, that robot assignment is the collaboration intent. `schedule` derives handoff/carrier details.
- Use `<gap>` or `gaps` to mark deterministic details left for tools: `deps` for missing dependency edges and `precond` for missing mechanical preconditions.
""",
    output_format_block="""## Output Protocol
Produce one trajectory with these blocks. Do not use markdown fences. Do not emit a separate coordination block.
Use 0..N `[TOOL_CALL]` blocks as needed, then finish with `[PLAN]`.
Emit at most one `[TOOL_CALL]` block in any single assistant turn; multiple tools require multiple assistant/tool turns.

<think>brief grounded reasoning before high-level decisions or after tool feedback</think>

[GOAL]
{"targets":[{"item":"<name>","destination":"<fixture>","status":"active|skip","status_reason":"<reason>","status_line":"<item>: active|skip(<reason>)"}, ...],
 "active":["<item>", ...],
 "skip":["<item>", ...]}
- Decompose the instruction into target items. For EACH target, list its
  destination (or null if the instruction has none) and its current status.
- Allowed status values are exactly `active` and `skip`.
- `status_reason` is a concise discrete observation reason, e.g.
  `needs_relocation`, `observed_on_table`, `observed_in_cabinet`, or
  `observed_at_destination`. `status_line` mirrors the local decision
  (`bowl: active`, `plate: skip(observed_on_table)`) for easy checking.
- `active` = targets whose status is `active`; `skip` = targets whose status is
  `skip`.
- For tasks without a movement target (activate / process / inspect only),
  set `targets=[]`, `active=[]`, `skip=[]` and proceed.

[ASSIGN]
{"R1":{"items":["<item>", ...]},
 "R2":{"items":["<item>", ...]}, ...}
- One entry per slot id in [ROBOTS]. Use ALL the slot ids listed there
  (no fewer, no extras).
- `items` lists the active items handled by that slot (matches the
  entities of its acquire/load_carrier/deposit ops). Empty list if the
  slot is idle for this task.
- Capability check: each slot's assigned ops must be supported by its
  capability list in [ROBOTS] (see Planning Rules).

[PLAN_SKETCH]
[{"op_id":"o1","op_type":"acquire","entity":"...","dest":null,"deps":[],"robot":"R1","gaps":["deps"],"params":{}}, ...]
- Intent op list. Every op must include `robot`.
- Use only intent ops listed above. Leave deterministic details to tools via `gaps`.

[TOOL_CALL]
{"name":"schedule","args":{"ops":[...],"assign":{...}}}
- Tool call JSON must be a single object with `name` and `args`.
- After a tool result, continue with another `<think>` and either repair with another `[TOOL_CALL]` or output `[PLAN]`.

[PLAN]
[{"op_id":"op1","op_type":"...","entity":"...","dest":"...","deps":[...],"robot":"R1","params":{}}, ...]
- Topologically ordered list of operations.
- op_id: unique identifier (op1, op2, ...)
- op_type: an intent op, or a complete DAG op returned by `schedule`.
- entity: item/carrier/tool being operated on
- dest: destination fixture/area/robot slot when relevant
- deps: list of op_ids that must complete before this op
- robot: assigned robot slot (must appear in [ASSIGN])
- params: optional extra fields, e.g. {"carrier":"cardboardbox"},
  {"tool":"knife"}, {"polarity":"off"}.
- Coverage: every item in [GOAL].active must appear as the entity of at
  least one acquire / load_carrier op; items in [GOAL].skip must NOT
  appear. If [GOAL].active is non-empty, [PLAN] MUST be non-empty.""",
    contributors=VIKI_L2_SCENE_BLOCK_CONTRIBUTORS,
)

VIKI_L2_PROFILE = BenchmarkProfile(
    name="viki_l2",
    op_vocab=VIKI_L2_OP_VOCAB,
    catalog=VIKI_L2_CATALOG,
    prompt=VIKI_L2_PROMPT_PROVIDER,
    dispatcher_emitter=VIKI_L2_DISPATCHER_EMITTER,
)
