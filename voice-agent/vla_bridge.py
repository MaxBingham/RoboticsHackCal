"""Small, read-only bridge between an ElevenLabs tool call and the VLA.

The backend is selected with the ``VLA_BACKEND`` environment variable:

* ``smolvla`` (default): loads the SmolVLA wrapper in ``so101-vla``. Runs on
  CPU/MPS/CUDA and uses a public checkpoint, so it works on a laptop.
* ``pi05``: loads the pi0.5 wrapper in ``so101-pi05``. Requires
  ``PI05_CHECKPOINT`` and, in practice, a CUDA GPU.
* ``mock``: returns six zeros without loading a model. Useful for exercising
  the full voice/tool path without a checkpoint.
"""

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

DEFAULT_BACKEND = "smolvla"
SMOLVLA_CHECKPOINT = "semi01/smolvla_official_so101_pickplace"
# The SmolVLA checkpoint was trained on a single task string; other wordings do
# not change its behaviour.
SMOLVLA_INSTRUCTION = "pink lego brick into the transparent box"
PI05_INSTRUCTION = "pick up the object"


def _backend() -> str:
    return os.environ.get("VLA_BACKEND", DEFAULT_BACKEND).strip().lower()


class ReadOnlyVLABridge:
    """Produce one six-joint prediction without connecting to the robot."""

    def __init__(self) -> None:
        self._model = None
        self._lock = Lock()

    @property
    def backend(self) -> str:
        return _backend()

    @property
    def using_mock(self) -> bool:
        return self.backend == "mock"

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
        backend = self.backend
        if backend == "mock":
            return [0.0] * len(JOINT_NAMES)
        if backend == "smolvla":
            return self._predict_smolvla()
        if backend == "pi05":
            return self._predict_pi05()
        raise ValueError(f"Unknown VLA_BACKEND {backend!r}; expected smolvla, pi05, or mock")

    def _predict_smolvla(self):
        import numpy as np

        if self._model is None:
            vla_dir = Path(__file__).resolve().parents[1] / "so101-vla"
            sys.path.insert(0, str(vla_dir))
            sys.modules.pop("vla", None)
            from vla import SmolVLA

            self._model = SmolVLA(
                repo=os.environ.get("SMOLVLA_CHECKPOINT", SMOLVLA_CHECKPOINT),
                device=os.environ.get("VLA_DEVICE") or None,
            )

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        state = np.zeros(len(JOINT_NAMES), dtype=np.float32)
        return np.asarray(
            self._model.predict(
                image=image,
                image_side=image,
                state=state,
                instruction=os.environ.get("VLA_INSTRUCTION", SMOLVLA_INSTRUCTION),
            ),
            dtype=np.float32,
        ).reshape(-1)

    def _predict_pi05(self):
        import numpy as np

        checkpoint = os.environ.get("PI05_CHECKPOINT")
        if not checkpoint:
            raise RuntimeError("VLA_BACKEND=pi05 requires PI05_CHECKPOINT to be set")

        if self._model is None:
            vla_dir = Path(__file__).resolve().parents[1] / "so101-pi05"
            sys.path.insert(0, str(vla_dir))
            sys.modules.pop("vla", None)
            from vla import Pi05VLA

            self._model = Pi05VLA(
                repo=checkpoint,
                device=os.environ.get("VLA_DEVICE", os.environ.get("PI05_DEVICE", "cuda")),
            )

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        state = np.zeros(len(JOINT_NAMES), dtype=np.float32)
        return np.asarray(
            self._model.predict(
                image=image,
                image_side=image,
                state=state,
                instruction=os.environ.get("VLA_INSTRUCTION", PI05_INSTRUCTION),
            ),
            dtype=np.float32,
        ).reshape(-1)


def format_positions(positions: dict[str, float]) -> str:
    return ", ".join(f"{name}={value:.3f}" for name, value in positions.items())
