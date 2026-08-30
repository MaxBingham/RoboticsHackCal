"""SmolVLA wrapper for offline SO-101 inference.

Expected teammate call (raw LeRobot observation):

    vla = SmolVLA()
    vla.reset()
    action = vla.predict_from_robot(robot.get_observation(), instruction)
    robot.send_action(action)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy, make_smolvla_pre_post_processors

REPO = "semi01/smolvla_official_so101_pickplace"
JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_chw_float(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Accept HWC uint8 RGB or CHW float32 [0, 1]; return unbatched CHW float32."""
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
    else:
        tensor = torch.from_numpy(np.asarray(image))

    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"image must be HWC or CHW, got shape {tuple(tensor.shape)}")

    # HWC -> CHW
    if tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1).contiguous()
    if tensor.shape[0] != 3:
        raise ValueError(f"expected 3 channels, got shape {tuple(tensor.shape)}")

    tensor = tensor.float()
    if tensor.max() > 1.5:
        tensor = tensor / 255.0
    return tensor


def _to_state(state: np.ndarray | torch.Tensor | list[float]) -> torch.Tensor:
    tensor = torch.as_tensor(state, dtype=torch.float32).flatten()
    if tensor.numel() != 6:
        raise ValueError(f"state must be 6 joints, got {tuple(tensor.shape)}")
    return tensor


def _to_numpy_action(action: Any) -> np.ndarray:
    if isinstance(action, dict):
        action = action.get("action", next(iter(action.values())))
    return np.asarray(action.detach().cpu()).reshape(-1).astype(np.float32)


def load_checkpoint_stats(repo: str = REPO) -> dict:
    """Mean/std stored inside the old checkpoint (no policy_preprocessor.json on the Hub)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = hf_hub_download(repo, "model.safetensors")
    sd = load_file(path)
    return {
        "observation.state": {
            "mean": sd["normalize_inputs.buffer_observation_state.mean"],
            "std": sd["normalize_inputs.buffer_observation_state.std"],
        },
        "action": {
            "mean": sd["unnormalize_outputs.buffer_action.mean"],
            "std": sd["unnormalize_outputs.buffer_action.std"],
        },
    }


def make_processors(policy, repo: str, device: torch.device):
    try:
        return make_pre_post_processors(
            policy.config,
            repo,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
    except Exception:
        policy.config.device = str(device)
        return make_smolvla_pre_post_processors(policy.config, dataset_stats=load_checkpoint_stats(repo))


class SmolVLA:
    def __init__(self, repo: str = REPO, device: str | None = None):
        self.repo = repo
        self.device = torch.device(device) if device else pick_device()
        self.policy = SmolVLAPolicy.from_pretrained(repo)
        self.policy.to(self.device)
        self.policy.eval()
        self.policy.reset()
        self.preprocess, self.postprocess = make_processors(self.policy, repo, self.device)

    def reset(self) -> None:
        """Call at the start of each episode so the 50-step action queue clears."""
        self.policy.reset()

    def predict(
        self,
        image,
        state,
        instruction: str,
        image_side=None,
    ) -> np.ndarray:
        """Return one 6-D SO-101 joint-position action.

        `image` is the overhead (`up`) camera. Pass `image_side` for the side camera.
        If `image_side` is omitted, the same frame is used for both views.
        """
        image_up = _to_chw_float(image)
        image_side = _to_chw_float(image if image_side is None else image_side)
        batch = {
            "observation.images.up": image_up,
            "observation.images.side": image_side,
            "observation.state": _to_state(state),
            "task": instruction,
        }
        obs = self.preprocess(batch)
        with torch.inference_mode():
            action = self.policy.select_action(obs)
            action = self.postprocess(action)
        return _to_numpy_action(action)

    def predict_from_robot(
        self,
        observation: dict,
        instruction: str,
        camera_map: dict[str, str] | None = None,
    ) -> dict[str, float]:
        """Drop-in for a LeRobot SO-101 loop.

        Pass `robot.get_observation()` and send the result to `robot.send_action()`.
        Camera keys default to `up` / `side`. If theirs are `front` / `wrist`:

            camera_map={"up": "front", "side": "wrist"}
        """
        image_up, image_side = _images_from_robot_obs(observation, camera_map)
        state = _state_from_robot_obs(observation)
        action = self.predict(
            image=image_up,
            image_side=image_side,
            state=state,
            instruction=instruction,
        )
        return {name: float(action[i]) for i, name in enumerate(JOINT_NAMES)}


def action_to_robot_dict(action: np.ndarray) -> dict[str, float]:
    """Convert a (6,) action vector to `robot.send_action()` keys."""
    flat = np.asarray(action, dtype=np.float32).reshape(-1)
    if flat.size != 6:
        raise ValueError(f"action must be 6-D, got {flat.shape}")
    return {name: float(flat[i]) for i, name in enumerate(JOINT_NAMES)}


def _images_from_robot_obs(
    observation: dict,
    camera_map: dict[str, str] | None,
) -> tuple[Any, Any]:
    aliases = {
        "up": ("up", "front", "top", "overhead"),
        "side": ("side", "wrist", "arm"),
    }
    if camera_map:
        aliases = {
            "up": (camera_map.get("up", "up"),),
            "side": (camera_map.get("side", "side"),),
        }

    def pick(role: str):
        for key in aliases[role]:
            if key in observation:
                return observation[key]
        available = [k for k in observation if not str(k).endswith(".pos")]
        raise KeyError(
            f"No {role} camera in observation. Tried {aliases[role]}. "
            f"Non-joint keys: {available}. Pass camera_map={{'up': '...', 'side': '...'}}."
        )

    return pick("up"), pick("side")


def _state_from_robot_obs(observation: dict) -> np.ndarray:
    missing = [name for name in JOINT_NAMES if name not in observation]
    if missing:
        raise KeyError(f"observation is missing joints {missing}")
    return np.array([observation[name] for name in JOINT_NAMES], dtype=np.float32)
