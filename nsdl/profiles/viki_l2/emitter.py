from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from nsdl.schema import DispatcherPayload, MicroStep


@dataclass(frozen=True)
class VikiL2DispatcherEmitter:
    """Emit the legacy VIKI-L2 dispatcher JSON shape."""

    def emit(self, payload: DispatcherPayload | Sequence[MicroStep]) -> list[dict[str, Any]]:
        steps = payload.micro_steps if isinstance(payload, DispatcherPayload) else payload
        return [
            {
                "step": step.step,
                "actions": {
                    robot: list(action)
                    for robot, action in step.actions.items()
                },
            }
            for step in steps
        ]


VIKI_L2_DISPATCHER_EMITTER = VikiL2DispatcherEmitter()
