"""Run SmolVLA on a dummy observation. No robot required.

    conda activate lerobot
    cd ~/so101-vla
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    python run_inference.py
"""

from __future__ import annotations

import time

import numpy as np

from vla import JOINT_NAMES, SmolVLA

# The only task string in the checkpoint's training set. It saw no others,
# so wording changes do not change behaviour.
INSTRUCTION = "pink lego brick into the transparent box"


def dummy_image(color=(200, 30, 30)) -> np.ndarray:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[180:300, 250:390] = color
    return image


def dummy_state() -> np.ndarray:
    # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    return np.array([0.0, -90.0, 90.0, 90.0, 0.0, 0.0], dtype=np.float32)


def main() -> None:
    print("loading SmolVLA ...")
    t0 = time.perf_counter()
    vla = SmolVLA()
    print(f"loaded on {vla.device} in {time.perf_counter() - t0:.1f}s")

    image_up = dummy_image()
    image_side = dummy_image((180, 40, 40))
    state = dummy_state()

    latencies = []
    actions = []
    for i in range(2):
        t1 = time.perf_counter()
        action = vla.predict(
            image=image_up,
            state=state,
            instruction=INSTRUCTION,
            image_side=image_side,
        )
        latencies.append(time.perf_counter() - t1)
        actions.append(action)
        print(f"\ncall {i + 1}")
        print("  action:", np.round(action, 4).tolist())
        print("  dim:", action.shape)
        print(f"  latency_s: {latencies[-1]:.3f}")
        print("  finite:", bool(np.isfinite(action).all()))

    robot_obs = {
        "up": image_up,
        "side": image_side,
        **{name: float(value) for name, value in zip(JOINT_NAMES, state)},
    }
    robot_action = vla.predict_from_robot(robot_obs, INSTRUCTION)
    print("\nrobot.send_action payload:")
    for name, value in robot_action.items():
        print(f"  {name}: {value:.4f}")

    print("\n=== RESULT ===")
    print("instruction:", INSTRUCTION)
    print("joints:", JOINT_NAMES)
    print("action:", actions[0].tolist())
    print(f"first_chunk_latency_s: {latencies[0]:.3f}")
    print(f"queued_step_latency_s: {latencies[1]:.3f}")


if __name__ == "__main__":
    main()
