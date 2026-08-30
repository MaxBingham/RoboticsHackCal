# so101-vla

Offline SmolVLA inference for SO-101. No robot required to test.

## Linux setup

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
pip install "lerobot[smolvla]"
```

Needs NVIDIA + CUDA for a usable control rate. CPU works but is slow.

## Run the dummy test

```bash
conda activate lerobot
cd so101-vla
python run_inference.py
```

First run downloads `semi01/smolvla_official_so101_pickplace` (~900 MB). You should see a 6-D `action`.

## Scope of the current checkpoint

`semi01/smolvla_official_so101_pickplace` was fine-tuned on 50 episodes of a **single** task:

```
pink lego brick into the transparent box
```

It is not language-general. Other instructions return actions, but the same learned motion. Picking a different object (nuts, bolts) needs new recorded episodes and a fine-tune.

## Drive the real arm

```bash
python run_robot.py --port /dev/ttyACM0 --id my_follower \
    --up-camera 0 --side-camera 2 --duration 20
```

Find the port with `lerobot-find-port`, and calibrate first with `lerobot-calibrate`. The script clamps each joint to 5 deg per step (`--max-relative-target`); raise it once the motion looks sane. `--unsafe` removes the clamp entirely.

### Expected timing

The policy plans 50 steps ahead, then blocks to replan. Measured on an M1 (CPU/MPS):

| | |
|---|---|
| Normal tick | ~6 ms |
| Replan, every 50 ticks | ~1.0 s |
| Result at 30 Hz target | ~16 Hz average, arm freezes ~1 s between chunks |

So the arm moves in bursts. A CUDA laptop should be substantially faster. If the stutter is still bad there, the real fix is LeRobot's async chunking, `lerobot-rollout --inference.type=rtc`, which is built for this.

## Use from the robot loop

`predict_from_robot` takes `robot.get_observation()` and returns a dict for `robot.send_action()`:

```python
from vla import SmolVLA

vla = SmolVLA()
vla.reset()  # once per episode

obs = robot.get_observation()
action = vla.predict_from_robot(
    obs,
    instruction="pink lego brick into the transparent box",
    # camera_map={"up": "front", "side": "wrist"},  # only if their camera names differ
)
robot.send_action(action)
```

Camera names default to `up` / `side` (also accepts `front`/`wrist`). Images should be RGB 480×640. Joint keys must be `shoulder_pan.pos`, `shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, `gripper.pos` — the same names LeRobot already uses.

Ask them to name cameras `up` and `side` at 640×480 if they can. That removes the `camera_map`.
