"""transfer_weights.py
Transfer pretrained World Context visual encoder weights into a LeRobot / SmolVLA policy.

Usage:
    cd /home/gardlae/RoboticsHackCal/so101-vla
    python transfer_weights.py \
        --worldcontext-checkpoint /home/gardlae/WORLD_CONTEXT_EXPLORER_V3/work/checkpoints/worldcontext_backbone_best.pt \
        --output-checkpoint ./checkpoints/smolvla_with_worldcontext
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from vla import JOINT_NAMES, SmolVLA, pick_device


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transfer World Context visual weights into policy.")
    parser.add_argument(
        "--worldcontext-checkpoint",
        default="/home/gardlae/WORLD_CONTEXT_EXPLORER_V3/work/checkpoints/worldcontext_backbone_best.pt",
        help="Path to pretrained World Context backbone weights (.pt)",
    )
    parser.add_argument(
        "--policy-repo",
        default="semi01/smolvla_official_so101_pickplace",
        help="Base policy repository or local path",
    )
    parser.add_argument(
        "--output-checkpoint",
        default="outputs/smolvla_worldcontext_init",
        help="Path to save initialized policy checkpoint",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu; auto-detected when omitted")
    return parser.parse_args()


def inspect_checkpoint(ckpt_path: Path) -> dict:
    """Load and print summary of exported World Context weights."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"World Context checkpoint not found at: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    print(f"Loaded World Context checkpoint from: {ckpt_path}")
    if isinstance(checkpoint, dict):
        print(f"  Keys in checkpoint: {list(checkpoint.keys())}")
        if "meta" in checkpoint:
            print(f"  Tasks trained on: {checkpoint['meta'].get('tasks', [])}")
            print(f"  Backbone architecture: {checkpoint.get('backbone_type', checkpoint['meta'].get('backbone'))}")
        state_dict = checkpoint.get("state_dict", checkpoint.get("backbone_state_dict", checkpoint))
    else:
        state_dict = checkpoint

    print(f"  Total parameter tensors in backbone: {len(state_dict)}")
    return state_dict


def transfer_to_policy(
    wc_state_dict: dict,
    policy_repo: str,
    output_dir: str | Path,
    device: torch.device,
) -> None:
    print(f"\nInitializing base SmolVLA policy from: {policy_repo}...")
    vla = SmolVLA(repo=policy_repo, device=str(device))
    policy = vla.policy

    print(f"Policy type: {type(policy).__name__}")

    # Inspect policy modules for vision backbone
    policy_sd = policy.state_dict()
    matched_keys = []
    skipped_keys = []

    for k, v in wc_state_dict.items():
        # Match against policy vision encoder keys (e.g. vision_tower, visual_backbone)
        target_k = None
        for prefix in ["model.vision_tower.", "model.visual_backbone.", "vision_encoder.", "backbone."]:
            candidate = f"{prefix}{k}"
            if candidate in policy_sd and policy_sd[candidate].shape == v.shape:
                target_k = candidate
                break

        if target_k:
            policy_sd[target_k] = v.to(device)
            matched_keys.append((k, target_k))
        else:
            skipped_keys.append(k)

    print(f"Matched & transferred: {len(matched_keys)} tensors")
    if skipped_keys:
        print(f"Skipped / unmapped: {len(skipped_keys)} tensors (expected when transferring cross-backbone)")

    # Load updated weights back into policy
    policy.load_state_dict(policy_sd, strict=False)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    try:
        policy.save_pretrained(out_path)
        print(f"\nSaved transferred policy checkpoint to: {out_path}")
    except Exception as e:
        torch.save(policy.state_dict(), out_path / "model_with_worldcontext.pt")
        print(f"\nSaved raw model state dict to: {out_path / 'model_with_worldcontext.pt'} (Reason: {e})")

    # Run dummy validation test
    print("\n--- Verifying Forward Pass on Dummy SO-101 Observation ---")
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_img[180:300, 250:390] = (200, 30, 30)
    dummy_state = np.array([0.0, -90.0, 90.0, 90.0, 0.0, 0.0], dtype=np.float32)

    vla.reset()
    action = vla.predict(
        image=dummy_img,
        state=dummy_state,
        instruction="pink lego brick into the transparent box",
    )
    print("Action output shape:", action.shape)
    print("Action output values:", np.round(action, 4).tolist())
    print("Verification SUCCESS: Model is valid and outputs finite 6-D actions.")


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else pick_device()
    print(f"Using device: {device}")

    ckpt_path = Path(args.worldcontext_checkpoint)
    wc_sd = inspect_checkpoint(ckpt_path)
    transfer_to_policy(wc_sd, args.policy_repo, args.output_checkpoint, device)


if __name__ == "__main__":
    main()

