"""Small, read-only bridge between an ElevenLabs tool call and the VLA."""

from __future__ import annotations

import os
import sys
import math
from pathlib import Path
from threading import Lock

JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
DEFAULT_INSTRUCTION = "pick up the object"


class ReadOnlyVLABridge:
    """Produce one six-joint prediction without connecting to the robot."""

    def __init__(self) -> None:
        self._model = None
        self._lock = Lock()

    @property
    def using_mock(self) -> bool:
        return not bool(os.environ.get("PI05_CHECKPOINT"))

    def predict(self) -> dict[str, float]:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A VLA prediction is already running")

        try:
            action = tuple(float(value) for value in self._predict_action())
            if len(action) != len(JOINT_NAMES) or not all(map(math.isfinite, action)):
                raise ValueError(f"Invalid VLA action: expected six finite values, got {action}")
            return {
                name: float(value)
                for name, value in zip(JOINT_NAMES, action, strict=True)
            }
        finally:
            self._lock.release()

    def _predict_action(self):
        checkpoint = os.environ.get("PI05_CHECKPOINT")
        if not checkpoint:
            # Lets the complete voice/tool path be tested while training runs.
            return [0.0] * len(JOINT_NAMES)

        if self._model is None:
            import numpy as np

            vla_dir = Path(__file__).resolve().parents[1] / "so101-pi05"
            sys.path.insert(0, str(vla_dir))
            from vla import Pi05VLA

            self._model = Pi05VLA(
                repo=checkpoint,
                device=os.environ.get("PI05_DEVICE", "cuda"),
            )

        import numpy as np

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        state = np.zeros(len(JOINT_NAMES), dtype=np.float32)
        return np.asarray(
            self._model.predict(
                image=image,
                image_side=image,
                state=state,
                instruction=os.environ.get("VLA_INSTRUCTION", DEFAULT_INSTRUCTION),
            ),
            dtype=np.float32,
        ).reshape(-1)


def format_positions(positions: dict[str, float]) -> str:
    return ", ".join(f"{name}={value:.3f}" for name, value in positions.items())
