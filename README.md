# RoboticsHackCal

An SO-101 robotic arm that picks up a peanut and presents it in front of a
person's mouth, controllable by voice:

```
voice command (ElevenLabs) -> lerobot-rollout -> ACT policy -> SO-101 arm
```

## Hardware

| | |
|---|---|
| Follower | SO-101, id `hack_follower`, port `/dev/serial/by-id/usb-1a86_USB_Single_Serial_58CD177001-if00` |
| Leader | id `hack_leader`, port `/dev/ttyACM0` — teleop only, not used for autonomous rollout |
| Camera | `/dev/video4`, 640×480 @ 30fps, LeRobot key `front` |

## Layout

```
RoboticsHackCal/
├── lerobot/          # LeRobot fork + this project's datasets, checkpoints, CLI tools (.venv here)
├── so101-vla/         # Off-the-shelf SmolVLA integration test — NOT the peanut model
├── voice-agent/       # ElevenLabs voice control, wired to the ACT peanut checkpoint
└── so101_nut_handoff_v3.tar.gz  # Packaged copy of the v3 dataset
```

## The trained model: ACT peanut handoff

ACT policy, single `front` camera, 6-DOF state/action, trained 20k steps.
Two checkpoints in `lerobot/outputs/train/`:

| Checkpoint | Episodes |
|---|---|
| `act_so101_nut_handoff_v3` | 29 |
| `act_so101_nut_handoff_all_v1` | 57 |

`so101_nut_handoff_v3_clean` (28 episodes, one bad episode removed) exists
but no checkpoint is trained on it yet — the `v3` checkpoint still trains on
the unfiltered set.

### Run it

Confirmed working:

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

- `--task` is metadata only — this policy has no language input, so the text doesn't affect behavior.
- `Relative goal position ... clamped` in the logs is the safety limiter working, not an error.
- No read-only preview mode — it starts sending actions as soon as it steps. Don't connect the leader arm.

If motion is still unreliable, untested ideas worth trying: `--policy.n_action_steps=25` (replans every ~0.8s instead of running a blind 3.3s chunk), or switch to `act_so101_nut_handoff_all_v1` (2x the data). Neither has been verified on hardware yet.

## `so101-vla/` — not the peanut model

Runs `semi01/smolvla_official_so101_pickplace`, someone else's checkpoint
fine-tuned on *"pink lego brick into the transparent box"* with two cameras
(`up`/`side`) — this rig duplicates its one `front` camera into both. Never
trained on a peanut; treat it as an integration test only.

- `run_inference.py` — sanity-check SmolVLA loads, no robot needed.
- `run_robot.py` — read-only by default; needs `--enable-motion` + typing `MOVE` to move. Loads `SmolVLAPolicy` only — don't point it at the ACT checkpoint.
- `record_so101.sh` / `train_policy.sh` — generic helpers for recording/training a new policy.
- `transfer_weights.py` — experimental, injects `WORLD_CONTEXT_EXPLORER_V3` encoder weights into SmolVLA. Unrelated to the ACT path.

## `voice-agent/` — ElevenLabs voice control

```
"can you feed me a peanut?" -> ElevenLabs agent -> run_robot_task -> lerobot-rollout -> act_so101_nut_handoff_v3
```

- `setup_elevenlabs_agent.py` — one-time: creates the agent, writes gitignored `config.py`.
- `voice_robot_agent.py` — runs the voice session; `run_robot_task` launches the rollout, `stop_robot_task` sends SIGINT to halt it.
- `test-colors.py` — dead prototype, not part of the working path.

```bash
cd ~/RoboticsHackCal/lerobot && source .venv/bin/activate
pip install elevenlabs   # voice-agent/requirements.txt is broken (UTF-16, wrong packages) — ignore it

cd ~/RoboticsHackCal/voice-agent
python setup_elevenlabs_agent.py --api-key sk_your_key_here
python voice_robot_agent.py --dry-run                              # voice only
python voice_robot_agent.py --camera=/dev/video4 --enable-motion   # real robot
```

`--enable-motion` is required for hardware runs — `run_robot_task` refuses otherwise, since `lerobot-rollout` has no read-only mode.

## Safety

- Never connect the leader arm during autonomous rollout.
- Start with a low `max_relative_target` and raise it gradually.
- Be ready with Ctrl-C (or say "stop" over voice) on any first run.
- `so101-vla/run_robot.py` requires typing `MOVE` before moving; `lerobot-rollout` does not — it moves as soon as it starts.
