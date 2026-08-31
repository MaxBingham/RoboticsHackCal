# RoboticsHackCal

**Voice-controlled robotic peanut handoff** — built at the Robotics Hackathon at Cal.

Say *"can you feed me a peanut?"* and an SO-101 robotic arm picks up a peanut
and presents it in front of your mouth, driven by an ACT policy trained on our
own teleoperated demonstrations.

```
voice (ElevenLabs Conversational AI) ──> approved-task allowlist ──> lerobot-rollout ──> ACT policy ──> SO-101 arm
```

## How it works

1. **Demonstrations** — we teleoperated the SO-101 with a leader arm and
   recorded peanut-handoff episodes with LeRobot (single robot-mounted
   camera, 640×480 @ 30 fps).
2. **Training** — an ACT (Action Chunking Transformer) policy was trained
   for 20k steps on those episodes: image + 6-DOF joint state in, 6-DOF
   joint actions out.
3. **Deployment** — `lerobot-rollout` runs the checkpoint on the physical
   arm in a closed loop.
4. **Voice** — an ElevenLabs conversational agent exposes exactly one tool,
   `run_robot_task(task_id="peanut_handoff")`. A local allowlist maps that ID
   to the checkpoint and launches the rollout; saying "stop" halts the arm
   immediately.

## Repository layout

| Path | What it is |
|---|---|
| `voice-agent/` | ElevenLabs voice interface wired to the trained checkpoint |
| `so101-vla/` | SmolVLA experiments and generic record/train helper scripts |
| `lerobot/` *(local only, gitignored)* | LeRobot checkout holding the recorded datasets and trained checkpoints (14 GB — not pushed) |

## Trained checkpoints

Both live under `lerobot/outputs/train/` on the robot laptop:

| Checkpoint | Training data | Episodes |
|---|---|---|
| `act_so101_nut_handoff_v3` | `so101_nut_handoff_v3` | 29 |
| `act_so101_nut_handoff_all_v1` | all handoff recordings combined | 57 |

## Running the arm

Verified working on hardware:

```bash
cd ~/RoboticsHackCal/lerobot && source .venv/bin/activate

lerobot-rollout \
  --strategy.type=base \
  --policy.path=outputs/train/act_so101_nut_handoff_v3/checkpoints/last/pretrained_model \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_58CD177001-if00 \
  --robot.id=hack_follower \
  --robot.cameras='{front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}' \
  --robot.max_relative_target=5.0 \
  --task="Pick up the peanut and present it in front of the target." \
  --device=cuda \
  --duration=30 \
  --display_data=false
```

Notes:
- The ACT policy takes no language input — `--task` is logging metadata only.
- `Relative goal position ... clamped` log lines are the safety limiter
  working as intended.
- The arm starts moving as soon as the rollout begins; there is no preview
  mode. Disconnect the leader arm first.

## Voice control

```bash
cd ~/RoboticsHackCal/lerobot && source .venv/bin/activate
pip install -r ../voice-agent/requirements.txt

cd ~/RoboticsHackCal/voice-agent
python setup_elevenlabs_agent.py --api-key sk_your_key   # one-time agent creation
python voice_robot_agent.py --dry-run                    # voice only, no hardware
python voice_robot_agent.py --camera=/dev/video4 --enable-motion   # full stack
```

`setup_elevenlabs_agent.py` creates the ElevenLabs agent (system prompt and
tool schemas are defined in the file) and writes a gitignored `config.py`
with your credentials. `--enable-motion` is required for any hardware run;
without it the tool call is refused. Saying "stop", "cancel", or "abort"
interrupts the running rollout.

## `so101-vla/`

Earlier experiments with the pretrained
`semi01/smolvla_official_so101_pickplace` SmolVLA checkpoint (a pick-and-place
task, two-camera setup), plus reusable helpers:

- `run_inference.py` — offline SmolVLA smoke test, no robot needed
- `run_robot.py` — SmolVLA hardware runner (read-only by default; requires
  `--enable-motion` and typing `MOVE` before the arm moves)
- `record_so101.sh` / `train_policy.sh` — record teleop demonstrations and
  fine-tune a policy on them
- `transfer_weights.py` — experiment injecting a visual encoder pretrained on
  industrial manipulation video (World Context dataset) into SmolVLA

The final demo does not use SmolVLA — the ACT checkpoint above outperformed
it on our single-camera rig.

## Safety

- Never connect the leader arm during autonomous rollout.
- Start with a low `--robot.max_relative_target` and raise it gradually.
- Keep a hand near the power switch; Ctrl-C or a spoken "stop" halts the arm.
