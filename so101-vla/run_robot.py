"""Run the bundled SmolVLA checkpoint on a physical SO-101 follower.

The checkpoint expects two cameras named ``up`` and ``side``. This runner is
intended as an integration test for the current one-camera hardware: it sends
the same ``front`` frame to both model inputs. Fine-tune a policy on the real
camera layout before expecting reliable task performance.

Motion is disabled unless ``--enable-motion`` is passed. Even with that flag,
the operator must type ``MOVE`` after the robot and camera connect.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from vla import JOINT_NAMES, REPO, SmolVLA

DEFAULT_INSTRUCTION = "pink lego brick into the transparent box"


def camera_source(value: str) -> int | Path:
    """Interpret a plain integer as a camera index and anything else as a path."""
    try:
        return int(value)
    except ValueError:
        return Path(value)


def state_from_observation(observation: dict) -> np.ndarray:
    missing = [name for name in JOINT_NAMES if name not in observation]
    if missing:
        raise KeyError(f"Robot observation is missing joint keys: {missing}")
    return np.asarray([observation[name] for name in JOINT_NAMES], dtype=np.float32)


def action_for_robot(action: np.ndarray) -> dict[str, float]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (len(JOINT_NAMES),):
        raise ValueError(f"Expected {len(JOINT_NAMES)} actions, got shape {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError(f"Model produced a non-finite action: {action.tolist()}")
    return {name: float(value) for name, value in zip(JOINT_NAMES, action, strict=True)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SmolVLA with one camera on an SO-101 follower.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--robot-port", "--port", dest="robot_port", default="/dev/tty.usbmodem58CD1770011")
    parser.add_argument("--robot-id", "--id", dest="robot_id", default="hack_follower")
    parser.add_argument("--camera", default="0", help="OpenCV camera index or device path")
    parser.add_argument("--camera-name", default="front")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--control-fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--repo", default=REPO, help="Local path or Hugging Face policy repo")
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu; auto-detected when omitted")
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=2.0,
        help="Maximum commanded change per joint and control step, in normalized units (full travel is 200)",
    )
    parser.add_argument(
        "--enable-motion",
        action="store_true",
        help="Actually send model actions. Without this, inference runs read-only.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    if args.control_fps <= 0:
        raise ValueError("--control-fps must be positive")
    if args.max_relative_target <= 0:
        raise ValueError("--max-relative-target must be positive")


def disconnect_safely(robot: SO101Follower) -> None:
    """Disable torque even when connection failed partway through setup."""
    if robot.is_connected:
        robot.disconnect()
        return

    # SOFollower.connect() opens the motor bus before its cameras. If a camera
    # fails to open, the aggregate is_connected property is false even though
    # the serial bus is still live, so clean up the components individually.
    if robot.bus.is_connected:
        robot.bus.disconnect(disable_torque=True)
    for camera in robot.cameras.values():
        if camera.is_connected:
            camera.disconnect()


def run(args: argparse.Namespace) -> None:
    validate_args(args)

    # Load the model before connecting the arm. A first-time download can take
    # several minutes, and there is no reason to leave the motors energized.
    print(f"Loading policy {args.repo!r} ...")
    vla = SmolVLA(repo=args.repo, device=args.device)
    print(f"Policy loaded on {vla.device}.")

    camera_config = OpenCVCameraConfig(
        index_or_path=camera_source(args.camera),
        width=args.width,
        height=args.height,
        fps=args.camera_fps,
    )
    robot_config = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras={args.camera_name: camera_config},
        max_relative_target=args.max_relative_target,
        disable_torque_on_disconnect=True,
        # lerobot/svla_so101_pickplace was recorded in RANGE_M100_100: its action
        # stats saturate at exactly +/-100, which only the normalized path clamps to.
        # Reading or writing degrees here would rescale every joint but the gripper.
        use_degrees=False,
    )
    robot = SO101Follower(robot_config)

    try:
        print(f"Connecting follower on {args.robot_port} and camera {args.camera!r} ...")
        robot.connect()
        observation = robot.get_observation()
        state = state_from_observation(observation)
        image = observation[args.camera_name]
        print(f"Connected. Camera frame: {image.shape}; joint state: {np.round(state, 2).tolist()}")
        print(f"Instruction: {args.instruction}")
        print(
            "Compatibility warning: this checkpoint was trained with separate up/side cameras; "
            "this integration test duplicates the front frame."
        )

        if args.enable_motion:
            confirmation = input(
                "Clear the workspace, keep a hand on the power switch, then type MOVE to begin: "
            )
            if confirmation.strip() != "MOVE":
                print("Confirmation not received; exiting without motion.")
                return
        else:
            print("READ-ONLY mode: predictions will be printed but not sent to the arm.")

        vla.reset()
        period_s = 1.0 / args.control_fps
        start = time.monotonic()
        deadline = start
        step = 0
        next_status = start

        while time.monotonic() - start < args.duration:
            observation = robot.get_observation()
            state = state_from_observation(observation)
            image = observation[args.camera_name]

            prediction_start = time.monotonic()
            action = vla.predict(
                image=image,
                image_side=image,
                state=state,
                instruction=args.instruction,
            )
            inference_s = time.monotonic() - prediction_start
            robot_action = action_for_robot(action)

            # Do not issue an action after a very slow inference has already
            # consumed the requested run duration.
            if time.monotonic() - start >= args.duration:
                break
            if args.enable_motion:
                robot.send_action(robot_action)

            step += 1
            now = time.monotonic()
            if now >= next_status:
                mode = "sent" if args.enable_motion else "predicted"
                rounded = np.round(action, 2).tolist()
                print(f"step={step:04d} {mode}={rounded} inference={inference_s:.3f}s")
                next_status = now + 1.0

            deadline += period_s
            sleep_s = deadline - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # Avoid accumulating an ever-growing timing debt when model
                # inference or camera capture misses the target cadence.
                deadline = time.monotonic()

        elapsed = time.monotonic() - start
        achieved_hz = step / elapsed if elapsed else math.nan
        print(f"Finished {step} control steps in {elapsed:.2f}s ({achieved_hz:.2f} Hz).")
    except KeyboardInterrupt:
        print("\nInterrupted by operator.")
    finally:
        if robot.is_connected or robot.bus.is_connected or any(
            camera.is_connected for camera in robot.cameras.values()
        ):
            print("Disconnecting follower and disabling torque ...")
            disconnect_safely(robot)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
