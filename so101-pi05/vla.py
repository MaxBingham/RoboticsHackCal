"""π0.5 wrapper for SO-101 inference with a fine-tuned LeRobot checkpoint."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_chw_float(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Accept HWC uint8 RGB or CHW float [0, 1]; return unbatched CHW float32."""
    tensor = image.detach().cpu() if isinstance(image, torch.Tensor) else torch.from_numpy(np.asarray(image))
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"image must be HWC or CHW, got shape {tuple(tensor.shape)}")
    if tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1).contiguous()
    if tensor.shape[0] != 3:
        raise ValueError(f"expected 3 image channels, got shape {tuple(tensor.shape)}")
    tensor = tensor.float()
    return tensor / 255.0 if tensor.max() > 1.5 else tensor


def _to_state(state: np.ndarray | torch.Tensor | list[float]) -> torch.Tensor:
    tensor = torch.as_tensor(state, dtype=torch.float32).flatten()
    if tensor.numel() != len(JOINT_NAMES):
        raise ValueError(f"state must contain {len(JOINT_NAMES)} joints, got {tuple(tensor.shape)}")
    return tensor


def _to_numpy_action(action: Any) -> np.ndarray:
    if isinstance(action, dict):
        action = action.get("action", next(iter(action.values())))
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    return np.asarray(action, dtype=np.float32).reshape(-1)


def _feature_dim(config: PI05Config, collection: str, name: str) -> int:
    features = getattr(config, collection)
    if name not in features:
        raise ValueError(f"π0.5 checkpoint is missing required feature {name!r}")
    return int(features[name].shape[0])


def _validate_so101_config(config: PI05Config, repo: str) -> list[str]:
    state_dim = _feature_dim(config, "input_features", "observation.state")
    action_dim = _feature_dim(config, "output_features", "action")
    if state_dim != len(JOINT_NAMES) or action_dim != len(JOINT_NAMES):
        raise ValueError(
            f"{repo!r} has state/action dimensions {state_dim}/{action_dim}; "
            f"this runner requires an SO-101 checkpoint with 6/6. "
            "The generic lerobot/pi05_base checkpoint is padded to 32/32 and is not safe to run directly."
        )

    camera_keys = [name for name in config.input_features if name.startswith("observation.images.")]
    if not camera_keys:
        raise ValueError(f"{repo!r} does not define an image observation")

    feature_names = getattr(config, "action_feature_names", None)
    if feature_names and list(feature_names) != JOINT_NAMES:
        raise ValueError(
            f"{repo!r} action order is {list(feature_names)!r}; expected {JOINT_NAMES!r}"
        )
    return camera_keys


class Pi05VLA:
    def __init__(self, repo: str, device: str | None = None):
        self.repo = repo
        self.device = torch.device(device) if device else pick_device()
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

        config = PI05Config.from_pretrained(repo)
        self.camera_keys = _validate_so101_config(config, repo)
        config.device = str(self.device)

        self.policy = PI05Policy.from_pretrained(repo, config=config).to(self.device).eval()
        self.policy.reset()
        try:
            self.preprocess, self.postprocess = make_pre_post_processors(
                self.policy.config,
                repo,
                preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load π0.5 preprocessing statistics from {repo!r}. "
                "Use a complete fine-tuned LeRobot checkpoint containing "
                "policy_preprocessor.json, policy_postprocessor.json, and their state files."
            ) from exc

    def reset(self) -> None:
        """Clear the queued action chunk at an episode boundary."""
        self.policy.reset()

    def predict(
        self,
        image,
        state,
        instruction: str,
        image_side=None,
    ) -> np.ndarray:
        """Return one six-joint SO-101 action from one or two camera images."""
        primary = _to_chw_float(image)
        secondary = _to_chw_float(image if image_side is None else image_side)
        batch = {
            "observation.state": _to_state(state),
            "task": instruction,
        }
        for index, camera_key in enumerate(self.camera_keys):
            batch[camera_key] = primary if index == 0 else secondary

        observation = self.preprocess(batch)
        with torch.inference_mode():
            action = self.postprocess(self.policy.select_action(observation))
        action = _to_numpy_action(action)
        if action.shape != (len(JOINT_NAMES),) or not np.isfinite(action).all():
            raise ValueError(f"π0.5 returned an invalid SO-101 action: shape={action.shape}, value={action}")
        return action

    def predict_from_robot(
        self,
        observation: dict,
        instruction: str,
        camera_names: tuple[str, str | None] = ("front", None),
    ) -> dict[str, float]:
        """Convert a raw LeRobot observation into a ``robot.send_action`` payload."""
        primary_name, secondary_name = camera_names
        if primary_name not in observation:
            raise KeyError(f"observation is missing primary camera {primary_name!r}")
        secondary = observation[secondary_name] if secondary_name else None
        action = self.predict(
            image=observation[primary_name],
            image_side=secondary,
            state=_state_from_robot_obs(observation),
            instruction=instruction,
        )
        return action_to_robot_dict(action)


def action_to_robot_dict(action: np.ndarray) -> dict[str, float]:
    flat = np.asarray(action, dtype=np.float32).reshape(-1)
    if flat.size != len(JOINT_NAMES):
        raise ValueError(f"action must be 6-D, got {flat.shape}")
    return {name: float(value) for name, value in zip(JOINT_NAMES, flat, strict=True)}


def _state_from_robot_obs(observation: dict) -> np.ndarray:
    missing = [name for name in JOINT_NAMES if name not in observation]
    if missing:
        raise KeyError(f"observation is missing joints {missing}")
    return np.asarray([observation[name] for name in JOINT_NAMES], dtype=np.float32)
