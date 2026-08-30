#!/usr/bin/env bash
# ==============================================================================
# record_so101.sh
# Record teleoperated demonstration episodes on physical SO-101 hardware using LeRobot.
#
# Default ports:
#   Leader Arm:    /dev/ttyACM0
#   Follower Arm:  /dev/serial/by-id/usb-1a86_USB_Single_Serial_58CD177001-if00 (or /dev/ttyACM1)
#   Camera:        /dev/video0
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_DIR="${SCRIPT_DIR}/../lerobot"

# Activate virtual environment
if [ -f "${LEROBOT_DIR}/.venv/bin/activate" ]; then
    source "${LEROBOT_DIR}/.venv/bin/activate"
elif [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "Using active conda environment: ${CONDA_DEFAULT_ENV}"
else
    echo "Warning: No active venv or conda detected. Trying system python."
fi

# Configurable parameters
REPO_ID="${1:-local/so101_pick_place}"
TASK_DESCRIPTION="${2:-pick pink lego brick and place in transparent box}"
NUM_EPISODES="${3:-50}"
CAMERA_PORT="${CAMERA_PORT:-/dev/video0}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_58CD177001-if00}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM0}"
FPS="${FPS:-30}"

echo "=========================================================="
echo " SO-101 Demonstration Recording"
echo "=========================================================="
echo " Dataset Repo ID:    ${REPO_ID}"
echo " Task Description:   ${TASK_DESCRIPTION}"
echo " Target Episodes:    ${NUM_EPISODES}"
echo " Follower Port:      ${FOLLOWER_PORT}"
echo " Leader Port:        ${LEADER_PORT}"
echo " Camera Source:      ${CAMERA_PORT}"
echo " FPS:                ${FPS}"
echo "=========================================================="

python -m lerobot.scripts.record \
    --robot.type=so101_follower \
    --robot.port="${FOLLOWER_PORT}" \
    --robot.teleop_port="${LEADER_PORT}" \
    --robot.cameras="{\"front\": \"${CAMERA_PORT}\"}" \
    --fps="${FPS}" \
    --task="${TASK_DESCRIPTION}" \
    --repo-id="${REPO_ID}" \
    --num-episodes="${NUM_EPISODES}"

