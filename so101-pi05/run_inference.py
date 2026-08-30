"""Run a fine-tuned π0.5 SO-101 checkpoint on a dummy observation."""

from __future__ import annotations

import argparse
import time

import numpy as np

from vla import JOINT_NAMES, Pi05VLA

DEFAULT_INSTRUCTION = "pick up the object"


def dummy_image(color=(200, 30, 30)) -> np.ndarray:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[180:300, 250:390] = color
    return image


def dummy_state() -> np.ndarray:
    # Unit-neutral synthetic values; this only verifies the inference plumbing.
    return np.zeros(len(JOINT_NAMES), dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test a fine-tuned π0.5 SO-101 checkpoint")
    parser.add_argument("--repo", required=True, help="Hugging Face repo or local checkpoint path")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(f"loading π0.5 checkpoint {args.repo!r} ...")
    t0 = time.perf_counter()
    vla = Pi05VLA(repo=args.repo, device=args.device)
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
            instruction=args.instruction,
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
    robot_action = vla.predict_from_robot(
        robot_obs,
        args.instruction,
        camera_names=("up", "side"),
    )
    print("\nrobot.send_action payload:")
    for name, value in robot_action.items():
        print(f"  {name}: {value:.4f}")

    print("\n=== RESULT ===")
    print("instruction:", args.instruction)
    print("joints:", JOINT_NAMES)
    print("action:", actions[0].tolist())
    print(f"first_chunk_latency_s: {latencies[0]:.3f}")
    print(f"queued_step_latency_s: {latencies[1]:.3f}")


if __name__ == "__main__":
    main()
