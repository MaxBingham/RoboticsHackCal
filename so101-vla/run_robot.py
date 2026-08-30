"""Drive a real SO-101 with SmolVLA.

Run on the Linux laptop with the arm and both cameras connected:

    python run_robot.py --port /dev/ttyACM0 --id my_follower \
        --up-camera 0 --side-camera 2

Find the port with `lerobot-find-port`. The arm must already be calibrated
(`lerobot-calibrate`). Keep a hand near the power switch on the first run.
"""

from __future__ import annotations

import argparse
import logging
import time

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.robot_utils import precise_sleep

from vla import JOINT_NAMES, REPO, SmolVLA

# The single task string this checkpoint was fine-tuned on. Changing the wording
# does not change its behaviour; only a fine-tune on new data does.
DEFAULT_INSTRUCTION = "pink lego brick into the transparent box"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True, help="Follower serial port, e.g. /dev/ttyACM0")
    p.add_argument("--id", default="so101_follower", help="Robot id used to find the calibration file")
    p.add_argument("--up-camera", type=int, default=0, help="OpenCV index for the overhead camera")
    p.add_argument("--side-camera", type=int, default=1, help="OpenCV index for the side camera")
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    p.add_argument("--policy", default=REPO, help="HF repo or local path of the SmolVLA checkpoint")
    p.add_argument("--fps", type=int, default=30, help="Control loop rate")
    p.add_argument("--duration", type=float, default=20.0, help="Seconds per episode")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument(
        "--max-relative-target",
        type=float,
        default=5.0,
        help="Safety clamp: max degrees any joint may move per step. Raise once the motion looks sane.",
    )
    p.add_argument(
        "--unsafe",
        action="store_true",
        help="Disable the relative-target clamp. The arm can then jump at full speed.",
    )
    return p.parse_args()


def build_robot(args: argparse.Namespace) -> SO101Follower:
    cameras = {
        "up": OpenCVCameraConfig(index_or_path=args.up_camera, width=640, height=480, fps=args.fps),
        "side": OpenCVCameraConfig(index_or_path=args.side_camera, width=640, height=480, fps=args.fps),
    }
    config = SO101FollowerConfig(
        port=args.port,
        id=args.id,
        cameras=cameras,
        use_degrees=True,
        max_relative_target=None if args.unsafe else args.max_relative_target,
    )
    return SO101Follower(config)


def warmup(vla: SmolVLA, instruction: str) -> None:
    """Compile kernels before the arm is live, so tick 0 is not a multi-second freeze."""
    import numpy as np

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    observation = {"up": blank, "side": blank, **{name: 0.0 for name in JOINT_NAMES}}
    t0 = time.perf_counter()
    vla.predict_from_robot(observation, instruction)
    vla.reset()
    print(f"warmup inference: {time.perf_counter() - t0:.2f}s")


def run_episode(robot: SO101Follower, vla: SmolVLA, args: argparse.Namespace, index: int) -> None:
    vla.reset()
    control_interval = 1.0 / args.fps
    start = time.perf_counter()
    ticks = 0
    stalls: list[float] = []

    print(f"\nepisode {index + 1}/{args.episodes} — {args.duration:.0f}s")
    while time.perf_counter() - start < args.duration:
        tick_start = time.perf_counter()

        observation = robot.get_observation()
        action = vla.predict_from_robot(observation, args.instruction)
        robot.send_action(action)

        ticks += 1
        elapsed = time.perf_counter() - tick_start
        if elapsed > control_interval:
            stalls.append(elapsed)
        precise_sleep(max(control_interval - elapsed, 0.0))

    wall = time.perf_counter() - start
    print(f"  {ticks} ticks in {wall:.1f}s ({ticks / wall:.1f} Hz average)")
    if stalls:
        # The policy plans n_action_steps ahead, then blocks to replan. The arm
        # holds position during each of these.
        print(f"  {len(stalls)} replans, worst {max(stalls):.2f}s, total {sum(stalls):.1f}s stalled")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    print(f"loading {args.policy} ...")
    t0 = time.perf_counter()
    vla = SmolVLA(repo=args.policy)
    print(f"loaded on {vla.device} in {time.perf_counter() - t0:.1f}s")
    print(f"instruction: {args.instruction!r}")

    if args.unsafe:
        print("WARNING: relative-target clamp disabled")
    else:
        print(f"safety clamp: {args.max_relative_target} deg/step")

    warmup(vla, args.instruction)

    robot = build_robot(args)
    robot.connect()
    print("robot connected")

    try:
        for index in range(args.episodes):
            run_episode(robot, vla, args, index)
            if index < args.episodes - 1:
                input("reset the scene, then press ENTER for the next episode...")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        robot.disconnect()
        print("robot disconnected")


if __name__ == "__main__":
    main()
