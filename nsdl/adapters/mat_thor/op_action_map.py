from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from nsdl.adapters.mat_thor.catalog_map import resolve_thor_object
from nsdl.schema import OperationStep


UNSUPPORTED_OP_TYPES = frozenset({
    "load_carrier",
    "push_carrier",
    "unload_carrier",
})

SUPPORTED_OP_TYPES = frozenset({
    "acquire",
    "local_acquire",
    "carry_to",
    "deposit",
    "open",
    "process_at",
    "apply_tool",
    "activate",
    "handoff",
    "inspect",
})


class UnsupportedOperationError(ValueError):
    pass


@dataclass(frozen=True)
class ThorAction:
    action: str
    object_ref: str | None = None
    op_type: str | None = None
    params: Mapping[str, Any] | None = None

    def to_plan_step(
        self,
        *,
        step_id: str,
        thor_objects: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "step_id": step_id,
            "action": self.action,
        }
        if self.op_type:
            payload["op_type"] = self.op_type
        resolved = resolve_thor_object(self.object_ref, thor_objects)
        if resolved is not None:
            payload["objectId"] = resolved.object_id
            payload["objectType"] = resolved.object_type
        elif self.object_ref:
            payload["objectId"] = self.object_ref
            payload["objectType"] = self.object_ref
        if self.params:
            payload["params"] = dict(self.params)
        return payload


def map_operation_to_thor_actions(op: OperationStep) -> list[ThorAction]:
    if op.op_type in UNSUPPORTED_OP_TYPES:
        raise UnsupportedOperationError(f"MAT-THOR does not support op_type={op.op_type!r}.")
    if op.op_type not in SUPPORTED_OP_TYPES:
        raise UnsupportedOperationError(f"Unknown op_type={op.op_type!r}.")

    if op.op_type in {"acquire", "local_acquire"}:
        return [
            ThorAction("GoToObject", op.entity, op.op_type),
            ThorAction("PickupObject", op.entity, op.op_type),
        ]
    if op.op_type == "carry_to":
        return [ThorAction("GoToObject", op.dest or op.entity, op.op_type)]
    if op.op_type == "deposit":
        return [
            ThorAction("GoToObject", op.dest, op.op_type),
            ThorAction("PutObject", op.dest, op.op_type),
        ]
    if op.op_type == "open":
        return [ThorAction("OpenObject", op.dest or op.entity, op.op_type)]
    if op.op_type == "process_at":
        station = (op.dest or "").lower()
        if any(token in station for token in ("sink", "faucet", "basin", "bath")):
            return [ThorAction("CleanObject", op.entity or op.dest, op.op_type)]
        if any(token in station for token in ("microwave", "stove", "toaster", "coffee")):
            return [ThorAction("CookObject", op.entity or op.dest, op.op_type)]
        return [ThorAction("OpenObject", op.dest or op.entity, op.op_type)]
    if op.op_type == "apply_tool":
        tool = str(op.params.get("tool", "")).lower()
        if "knife" in tool or "knife" in str(op.entity or "").lower():
            return [ThorAction("SliceObject", op.dest or op.entity, op.op_type)]
        return [ThorAction("Pass", op.dest or op.entity, op.op_type)]
    if op.op_type == "activate":
        polarity = str(op.params.get("polarity", "on")).strip().lower()
        action = "SwitchOff" if polarity == "off" else "SwitchOn"
        return [ThorAction(action, op.dest or op.entity, op.op_type)]
    if op.op_type == "handoff":
        return [
            ThorAction("DropHandObject", op.entity, op.op_type),
            ThorAction("PickupObject", op.entity, op.op_type),
        ]
    if op.op_type == "inspect":
        return [ThorAction("GoToObject", op.dest or op.entity, op.op_type)]
    raise UnsupportedOperationError(f"Unhandled op_type={op.op_type!r}.")


def op_mapping_coverage() -> dict[str, str]:
    coverage = {name: "supported" for name in sorted(SUPPORTED_OP_TYPES)}
    coverage.update({name: "unsupported" for name in sorted(UNSUPPORTED_OP_TYPES)})
    return dict(sorted(coverage.items()))


__all__ = [
    "SUPPORTED_OP_TYPES",
    "UNSUPPORTED_OP_TYPES",
    "ThorAction",
    "UnsupportedOperationError",
    "map_operation_to_thor_actions",
    "op_mapping_coverage",
]

