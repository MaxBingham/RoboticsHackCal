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

First run downloads the base VLM and
`semi01/smolvla_official_so101_pickplace` checkpoint (about 3 GB total). You
should see a 6-D `action`.

## Scope of the current checkpoint

`semi01/smolvla_official_so101_pickplace` was fine-tuned on 50 episodes of a **single** task:

```
pink lego brick into the transparent box
```

It is not language-general. Other instructions may return actions, but they do
not give the checkpoint a skill it was not trained on. Picking a different
object or using a substantially different scene needs recorded episodes and a
fine-tune.

## Run on one-camera SO-101 hardware

`run_robot.py` connects a physical follower and uses the same `front` image for
the checkpoint's `up` and `side` inputs. This is only an integration test: the
checkpoint was trained with two different camera views, so fine-tune on the
actual camera layout before expecting reliable task performance.

First run in read-only mode. This connects the arm and camera and prints model
predictions, but does not send actions:

```bash
python run_robot.py \
  --robot-port=/dev/tty.usbmodem58CD1770011 \
  --robot-id=hack_follower \
  --camera=0 \
  --duration=10
```

Then clear the workspace, keep a hand on the power switch, and explicitly
enable motion. The runner also requires typing `MOVE` after everything connects:

```bash
python run_robot.py \
  --robot-port=/dev/tty.usbmodem58CD1770011 \
  --robot-id=hack_follower \
  --camera=0 \
  --instruction="pink lego brick into the transparent box" \
  --max-relative-target=2 \
  --duration=10 \
  --enable-motion
```

Use `--device=mps` on Apple Silicon or `--device=cuda` on an NVIDIA machine if
auto-detection does not select the desired accelerator. The leader arm is not
used during autonomous inference. Press Ctrl-C or cut follower power to stop.

### Expected timing

The policy plans 50 steps ahead, then blocks to replan. Measured on an M1 (CPU/MPS):

| | |
|---|---|
| Normal tick | ~6 ms |
| Replan, every 50 ticks | ~1.0 s |
| Result at 30 Hz target | ~16 Hz average, arm freezes ~1 s between chunks |

So the arm moves in bursts. A CUDA laptop should be substantially faster. If the stutter is still bad there, the real fix is LeRobot's async chunking, `lerobot-rollout --inference.type=rtc`, which is built for this.

## Use from the robot loop

With one camera, pass the same frame to both visual inputs and convert the
returned vector to the six named motor targets:

```python
from vla import JOINT_NAMES, SmolVLA

vla = SmolVLA()
vla.reset()  # once per episode

obs = robot.get_observation()
state = [obs[name] for name in JOINT_NAMES]
action_vector = vla.predict(
    image=obs["front"],
    image_side=obs["front"],
    state=state,
    instruction="pink lego brick into the transparent box",
)
action = dict(zip(JOINT_NAMES, map(float, action_vector)))
robot.send_action(action)
```

Images should be RGB 480×640. Joint keys are `shoulder_pan.pos`,
`shoulder_lift.pos`, `elbow_flex.pos`, `wrist_flex.pos`, `wrist_roll.pos`, and
`gripper.pos`—the same names LeRobot uses.
