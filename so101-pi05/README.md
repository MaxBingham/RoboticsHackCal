# so101-pi05

CUDA inference integration for a LeRobot π0.5 policy on the SO-101 follower.

## Checkpoint requirement

Use a π0.5 checkpoint fine-tuned for your SO-101 dataset. The runner validates
that the checkpoint has six state values, six actions, at least one camera, and
the standard SO-101 joint order before it downloads the full model.

Do **not** run `lerobot/pi05_base` directly on the robot. Its state and action
spaces are padded to 32 dimensions and it has not been adapted to this hardware.
The runner rejects it. A task-specific fine-tune is still required even if a
checkpoint has already been domain-adapted to SO-101.

The checkpoint must include LeRobot's saved `policy_preprocessor.json`,
`policy_postprocessor.json`, and normalization state files. State/action
normalization and camera feature names are loaded from those files.

## CUDA setup

```bash
conda create -y -n lerobot-pi05 python=3.12
conda activate lerobot-pi05
pip install "lerobot[pi]"
hf auth login
```

π0.5 uses the gated PaliGemma tokenizer/model assets. Accept their Hugging Face
license before the first run. A CUDA GPU is expected; the checkpoint is roughly
8 GB before runtime allocations.

## Smoke test without a robot

```bash
cd so101-pi05
python run_inference.py \
  --repo=YOUR_HF_ORG/YOUR_SO101_PI05_CHECKPOINT \
  --instruction="the exact task wording used during training" \
  --device=cuda
```

This checks checkpoint compatibility and prints six-dimensional actions.

## Run on SO-101 hardware

First run read-only. `--joint-units` must match the dataset used to fine-tune the
checkpoint. Current LeRobot datasets may use either calibrated degrees or the
older normalized ranges, and mixing them will produce incorrect commands.

```bash
python run_robot.py \
  --repo=YOUR_HF_ORG/YOUR_SO101_PI05_CHECKPOINT \
  --robot-port=/dev/ttyACM1 \
  --robot-id=hack_follower \
  --camera=/dev/video0 \
  --joint-units=degrees \
  --instruction="the exact task wording used during training" \
  --duration=10
```

With two useful workspace cameras:

```bash
python run_robot.py \
  --repo=YOUR_HF_ORG/YOUR_SO101_PI05_CHECKPOINT \
  --robot-port=/dev/ttyACM1 \
  --robot-id=hack_follower \
  --camera=/dev/video0 \
  --side-camera=/dev/video4 \
  --joint-units=degrees \
  --device=cuda \
  --duration=10
```

The primary frame is assigned to the checkpoint's first camera feature. The
side frame is assigned to its remaining camera features. With one physical
camera, the primary frame is duplicated for all model camera inputs; that is
only an integration test unless training used the same layout.

After validating predictions in read-only mode, motion must be enabled
explicitly:

```bash
python run_robot.py \
  --repo=YOUR_HF_ORG/YOUR_SO101_PI05_CHECKPOINT \
  --robot-port=/dev/ttyACM1 \
  --robot-id=hack_follower \
  --camera=/dev/video0 \
  --joint-units=degrees \
  --instruction="the exact task wording used during training" \
  --max-relative-target=2 \
  --duration=10 \
  --enable-motion
```

The runner still requires typing `MOVE` after the model, robot, and camera are
ready. Keep the workspace clear and a hand on the power switch. The relative
target limit, finite-action checks, read-only default, and torque-disabling
cleanup remain enabled.

## Fine-tuning

Start from `lerobot/pi05_base` during training, while deriving the six-joint
feature shapes and camera names from your SO-101 dataset:

```bash
lerobot-train \
  --dataset.repo_id=YOUR_HF_ORG/YOUR_SO101_DATASET \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --output_dir=outputs/so101-pi05
```

π0.5 normally uses quantile normalization. Ensure the dataset has `q01` and
`q99` statistics and test the resulting checkpoint read-only before motion.
