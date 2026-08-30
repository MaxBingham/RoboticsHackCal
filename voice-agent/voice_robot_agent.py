"""voice-agent/voice_robot_agent.py
ElevenLabs Voice Interface for SO-101 Robotic Arm.

Architecture:
    Voice -> ElevenLabs Agent -> run_robot_task(task_id="peanut_handoff") -> Local Allowlist -> Policy -> SO-101

Usage:
    export ELEVENLABS_API_KEY="your-key"
    export ELEVENLABS_AGENT_ID="your-agent-id"

    # 1. Voice testing mode (dry-run without robot hardware):
    python voice_robot_agent.py --dry-run

    # 2. Hardware deployment mode:
    python voice_robot_agent.py --camera=/dev/video4 --enable-motion
"""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Ensure PortAudio is loaded from user local lib if present
local_portaudio = Path.home() / ".local" / "lib" / "libportaudio.so"
if local_portaudio.exists():
    try:
        ctypes.CDLL(str(local_portaudio))
    except Exception:
        pass

from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, ClientTools
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

# Configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID", "")

try:
    from config import ELEVENLABS_API_KEY as KEY_FROM_CONFIG, AGENT_ID as AGENT_FROM_CONFIG
    if KEY_FROM_CONFIG:
        ELEVENLABS_API_KEY = KEY_FROM_CONFIG
    if AGENT_FROM_CONFIG:
        AGENT_ID = AGENT_FROM_CONFIG
except ImportError:
    pass

# Immutable Local Allowlist of Approved Tasks
TASKS = {
    "peanut_handoff": {
        "instruction": (
            "Pick up a peanut from the table and present it in front "
            "of the person's mouth without touching the person."
        ),
        "checkpoint": (
            "/home/gardlae/RoboticsHackCal/lerobot/outputs/train/"
            "act_so101_nut_handoff_v3/checkpoints/last/pretrained_model"
        ),
    }
}

# Execution state
EXECUTION_LOCK = threading.Lock()
ACTIVE_PROCESS: subprocess.Popen | None = None
CLI_ARGS: argparse.Namespace | None = None


def run_robot_task(params: dict) -> dict[str, str]:
    """Client Tool: Execute an approved SO-101 robot task.
    
    Expected parameter from ElevenLabs agent:
        {"task_id": "peanut_handoff"}
    """
    global ACTIVE_PROCESS
    task_id = str(params.get("task_id", "")).strip()

    print("\n" + "=" * 65)
    print(" [ELEVENLABS CLIENT TOOL: run_robot_task]")
    print(f" Received task_id: '{task_id}'")

    # 1. Reject unknown or unapproved task IDs
    if task_id not in TASKS:
        msg = f"Rejected: Unknown task_id '{task_id}'. Only 'peanut_handoff' is approved."
        print(f" ERROR: {msg}")
        print("=" * 65 + "\n")
        return {"status": "error", "message": msg}

    # 2. Reject concurrent execution if robot is already busy
    if not EXECUTION_LOCK.acquire(blocking=False):
        msg = "Rejected: Robot is already executing a task. Please wait or say 'stop'."
        print(f" BUSY: {msg}")
        print("=" * 65 + "\n")
        return {"status": "busy", "message": msg}

    task_info = TASKS[task_id]
    instruction = task_info["instruction"]
    checkpoint = task_info["checkpoint"]

    print(f" Canonical Instruction: '{instruction}'")
    print(f" Checkpoint Target:     '{checkpoint}'")
    print("=" * 65)

    try:
        # Check dry-run mode
        if CLI_ARGS and CLI_ARGS.dry_run:
            print("\n[DRY RUN MODE] Simulating 10-second robot rollout...")
            for i in range(1, 11):
                if ACTIVE_PROCESS == "ABORTED":
                    raise KeyboardInterrupt("Dry run aborted by user")
                time.sleep(1)
                print(f"  Step {i}/10: executing trajectory...")
            print("[DRY RUN COMPLETE] Handoff finished.")
            return {"status": "success", "message": "Peanut handoff simulation completed."}

        # lerobot-rollout has no read-only preview mode: it always sends actions
        # to the motors. Without --enable-motion there is nothing safe to launch.
        if not (CLI_ARGS and CLI_ARGS.enable_motion):
            msg = "Motion disabled: rerun voice_robot_agent.py with --enable-motion to execute on hardware."
            print(f" REFUSED: {msg}")
            print("=" * 65 + "\n")
            return {"status": "error", "message": msg}

        # Hardware execution via lerobot-rollout, loading the ACT checkpoint directly
        # (this task's checkpoint is ACT, not SmolVLA, so it cannot go through
        # so101-vla/run_robot.py, which hardcodes the SmolVLA policy class).
        rollout_bin = Path(__file__).parent.parent / "lerobot" / ".venv" / "bin" / "lerobot-rollout"
        if not rollout_bin.exists():
            rollout_bin = Path("lerobot-rollout")
        camera = CLI_ARGS.camera if CLI_ARGS else "/dev/video4"
        cmd = [
            str(rollout_bin),
            "--strategy.type=base",
            f"--policy.path={checkpoint}",
            "--policy.n_action_steps=25",
            "--robot.type=so101_follower",
            "--robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_58CD177001-if00",
            "--robot.id=hack_follower",
            f"--robot.cameras={{front: {{type: opencv, index_or_path: {camera}, width: 640, height: 480, fps: 30}}}}",
            "--robot.max_relative_target=5.0",
            f"--task={instruction}",
            "--device=cuda",
            f"--duration={CLI_ARGS.duration if CLI_ARGS else 12.0}",
            "--display_data=false",
        ]

        print(f"\nLaunching robot subprocess: {' '.join(cmd)}")
        ACTIVE_PROCESS = subprocess.Popen(cmd)
        ret_code = ACTIVE_PROCESS.wait()
        ACTIVE_PROCESS = None

        if ret_code == 0:
            return {"status": "success", "message": "Peanut handoff completed successfully."}
        else:
            return {"status": "error", "message": f"Robot runner exited with code {ret_code}"}

    except Exception as e:
        print(f"Execution failed: {e}")
        return {"status": "error", "message": f"Execution error: {e}"}
    finally:
        EXECUTION_LOCK.release()


def stop_robot_task(params: dict | None = None) -> dict[str, str]:
    """Client Tool: Immediately halt and abort active robot execution."""
    global ACTIVE_PROCESS
    print("\n" + "!" * 65)
    print(" [ELEVENLABS CLIENT TOOL: stop_robot_task]")
    print(" EMERGENCY STOP TRIGGERED: Halting robot execution immediately.")
    print("!" * 65 + "\n")

    if ACTIVE_PROCESS and isinstance(ACTIVE_PROCESS, subprocess.Popen):
        try:
            ACTIVE_PROCESS.send_signal(signal.SIGINT)
            time.sleep(0.5)
            if ACTIVE_PROCESS.poll() is None:
                ACTIVE_PROCESS.terminate()
        except Exception as e:
            print(f"Error terminating process: {e}")
        ACTIVE_PROCESS = None
    elif CLI_ARGS and CLI_ARGS.dry_run:
        ACTIVE_PROCESS = "ABORTED"

    return {"status": "aborted", "message": "Robot movement halted immediately."}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ElevenLabs Voice Interface for SO-101 Robot.")
    parser.add_argument("--dry-run", action="store_true", help="Run in voice-only testing mode without hardware")
    parser.add_argument("--camera", default="/dev/video4", help="Camera device path (e.g. /dev/video4)")
    parser.add_argument("--duration", type=float, default=12.0, help="Episode rollout duration in seconds")
    parser.add_argument("--enable-motion", action="store_true", help="Enable physical robot motor execution")
    return parser


def main():
    global CLI_ARGS
    parser = build_parser()
    CLI_ARGS = parser.parse_args()

    if not ELEVENLABS_API_KEY or not AGENT_ID:
        print("=" * 65)
        print(" ERROR: Missing ElevenLabs Credentials")
        print("=" * 65)
        print("Please export your API key and Agent ID:")
        print("    export ELEVENLABS_API_KEY='your-elevenlabs-api-key'")
        print("    export ELEVENLABS_AGENT_ID='your-agent-id'")
        print("=" * 65)
        sys.exit(1)

    # Register client tools with ElevenLabs SDK
    client_tools = ClientTools()
    client_tools.register("run_robot_task", run_robot_task, is_async=False)
    client_tools.register("stop_robot_task", stop_robot_task, is_async=False)

    print("Connecting to ElevenLabs Conversational AI...")
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    conversation = Conversation(
        client=client,
        agent_id=AGENT_ID,
        requires_auth=True,
        audio_interface=DefaultAudioInterface(),
        client_tools=client_tools,
    )

    print("\n" + "=" * 65)
    print(" SO-101 Voice Interface LIVE")
    print(f" Mode:            {'DRY-RUN (Voice Only)' if CLI_ARGS.dry_run else 'HARDWARE CONNECTED'}")
    print(f" Approved Task:   'peanut_handoff'")
    print(f" Physical Motion: {'ENABLED' if CLI_ARGS.enable_motion else 'DISABLED (Read-Only)'}")
    print(" Speak to the microphone (e.g. 'Can you feed me a peanut?')")
    print(" Press Ctrl+C to stop.")
    print("=" * 65 + "\n")

    conversation.start_session()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping voice session...")
        conversation.end_session()
        conversation.wait_for_session_end()
        print("Session ended.")


if __name__ == "__main__":
    main()
