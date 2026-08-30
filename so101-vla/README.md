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

## Use from the robot loop

```python
from vla import SmolVLA

vla = SmolVLA()          # uses CUDA if available
vla.reset()              # once per episode

action = vla.predict(
    image=image_up,      # RGB, 480x640, HWC uint8 or CHW float [0, 1]
    image_side=image_side,
    state=robot_state,   # 6 joints: pan, lift, elbow, wrist_flex, wrist_roll, gripper
    instruction="Pick up the red cube and put it in the bowl.",
)
# action: np.ndarray (6,) — target joint positions, same units as the arm
```

Cameras must be named **up** (overhead) and **side**. If the hardware stack uses `front` / `wrist`, rename them before calling `predict`.
