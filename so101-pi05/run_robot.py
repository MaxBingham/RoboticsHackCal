"""Run a fine-tuned π0.5 checkpoint on a physical SO-101 follower.

The checkpoint determines its own camera feature names. With one physical
camera, the runner duplicates that frame for every model camera input. Fine-tune
on the actual camera layout before expecting reliable task performance.

Motion is disabled unless ``--enable-motion`` is passed. Even with that flag,
the operator must type ``MOVE`` after the robot and camera connect.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from vla import JOINT_NAMES, Pi05VLA

DEFAULT_INSTRUCTION = "pick up the object"
CLAMP_WARNING_PREFIX = "Relative goal position magnitude had to be clamped to be safe."


class _ClampWarningFilter(logging.Filter):
    """Hide LeRobot's per-tick clamp dump; the runner prints a compact summary."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(CLAMP_WARNING_PREFIX)


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
        description="Run π0.5 with one or two cameras on an SO-101 follower.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--robot-port", "--port", dest="robot_port", default="/dev/tty.usbmodem58CD1770011")
    parser.add_argument("--robot-id", "--id", dest="robot_id", default="hack_follower")
    parser.add_argument("--camera", default="0", help="Primary/up OpenCV camera index or device path")
    parser.add_argument("--camera-name", default="front", help="LeRobot key for the primary/up camera")
    parser.add_argument(
        "--side-camera",
        default=None,
        help="Optional second OpenCV camera. The primary frame is duplicated when omitted.",
    )
    parser.add_argument("--side-camera-name", default="side", help="LeRobot key for the optional side camera")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--control-fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--repo",
        required=True,
        help="Fine-tuned six-action SO-101 Hugging Face repo or local checkpoint path",
    )
    parser.add_argument("--device", default="cuda", help="Torch inference device")
    parser.add_argument(
        "--joint-units",
        required=True,
        choices=("normalized", "degrees"),
        help="Must match the units used to record the checkpoint's training dataset",
    )
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
    if args.side_camera is not None and args.side_camera_name == args.camera_name:
        raise ValueError("--side-camera-name must differ from --camera-name")


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
    vla = Pi05VLA(repo=args.repo, device=args.device)
    print(f"Policy loaded on {vla.device}.")
    print(f"Checkpoint camera inputs: {vla.camera_keys}")

    cameras = {
        args.camera_name: OpenCVCameraConfig(
            index_or_path=camera_source(args.camera),
            width=args.width,
            height=args.height,
            fps=args.camera_fps,
        )
    }
    if args.side_camera is not None:
        cameras[args.side_camera_name] = OpenCVCameraConfig(
            index_or_path=camera_source(args.side_camera),
            width=args.width,
            height=args.height,
            fps=args.camera_fps,
        )

    robot_config = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=cameras,
        max_relative_target=args.max_relative_target,
        disable_torque_on_disconnect=True,
        use_degrees=args.joint_units == "degrees",
    )
    robot = SO101Follower(robot_config)

    clamp_filter = _ClampWarningFilter()
    root_logger = logging.getLogger()

    try:
        camera_description = repr(args.camera)
        if args.side_camera is not None:
            camera_description += f" plus side camera {args.side_camera!r}"
        print(f"Connecting follower on {args.robot_port} and camera {camera_description} ...")
        robot.connect()
        observation = robot.get_observation()
        state = state_from_observation(observation)
        image = observation[args.camera_name]
        image_side = observation[args.side_camera_name] if args.side_camera is not None else image
        print(
            f"Connected. Primary frame: {image.shape}; side frame: {image_side.shape}; "
            f"joint state: {np.round(state, 2).tolist()}"
        )
        print(f"Instruction: {args.instruction}")
        if args.side_camera is None:
            print(
                f"Using the primary frame for all {len(vla.camera_keys)} checkpoint camera inputs."
            )
        else:
            print("Using the primary frame for the first checkpoint camera and the side frame for the rest.")

        if args.enable_motion:
            confirmation = input(
                "Clear the workspace, keep a hand on the power switch, then type MOVE to begin: "
            )
            if confirmation.strip() != "MOVE":
                print("Confirmation not received; exiting without motion.")
                return
        else:
            print("READ-ONLY mode: predictions will be printed but not sent to the arm.")

        # LeRobot normally logs a multi-line warning for every clipped action.
        # The applied action and a clipped flag are reported below instead.
        root_logger.addFilter(clamp_filter)
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
            image_side = observation[args.side_camera_name] if args.side_camera is not None else image

            prediction_start = time.monotonic()
            action = vla.predict(
                image=image,
                image_side=image_side,
                state=state,
                instruction=args.instruction,
            )
            inference_s = time.monotonic() - prediction_start
            robot_action = action_for_robot(action)

            # Do not issue an action after a very slow inference has already
            # consumed the requested run duration.
            if time.monotonic() - start >= args.duration:
                break
            applied_action = None
            if args.enable_motion:
                applied = robot.send_action(robot_action)
                applied_action = np.asarray([applied[name] for name in JOINT_NAMES], dtype=np.float32)

            step += 1
            now = time.monotonic()
            if now >= next_status:
                requested = np.round(action, 2).tolist()
                if applied_action is None:
                    print(f"step={step:04d} predicted={requested} inference={inference_s:.3f}s")
                else:
                    applied = np.round(applied_action, 2).tolist()
                    clipped = not np.allclose(action, applied_action, atol=1e-3)
                    print(
                        f"step={step:04d} requested={requested} applied={applied} "
                        f"clipped={clipped} inference={inference_s:.3f}s"
                    )
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
        root_logger.removeFilter(clamp_filter)
        if robot.is_connected or robot.bus.is_connected or any(
            camera.is_connected for camera in robot.cameras.values()
        ):
            print("Disconnecting follower and disabling torque ...")
            disconnect_safely(robot)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
