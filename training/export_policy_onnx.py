#!/usr/bin/env python3
"""
Export a trained SB3 PPO policy to ONNX for use inside Isaac Sim.

Isaac Sim and PyTorch/SB3 cannot coexist in the same process (CUDA context
conflict causes a segfault on shutdown). This script runs outside Isaac Sim,
exports the policy network to ONNX, and saves VecNormalize statistics as .npz.
The demo script (nova_carter_rl_door_demo.py) then uses onnxruntime — no torch.

Usage:
  # On login node or any node with torch+SB3 installed:
  python training/export_policy_onnx.py \
    --checkpoint training/logs/sb3/NovaCarter-Patrol/2026-03-27_02-43-38/best_model.zip

  # Or via Isaac Lab's python (which has SB3):
  ~/.local/share/ov/pkg/isaac_lab*/isaaclab.sh -p training/export_policy_onnx.py \
    --checkpoint training/logs/sb3/NovaCarter-Patrol/2026-03-27_02-43-38/best_model.zip

Outputs (same directory as checkpoint):
  best_model.onnx          — policy network (obs → action, float32)
  best_model_vecnorm.npz   — VecNormalize running mean/var for obs normalization
"""

import argparse
import os
import pickle

import numpy as np
import torch
from stable_baselines3 import PPO

# Observation dimension must match training config exactly:
# LiDAR(20) + goal_pos_body(2) + goal_heading_cos_sin(2) + fwd_vel(1) + yaw_rate(1) = 26
OBS_DIM = 26


class OnnxablePolicy(torch.nn.Module):
    """Wrap the actor MLP so it can be traced by torch.onnx.export."""

    def __init__(self, mlp_extractor, action_net):
        super().__init__()
        self.mlp_extractor = mlp_extractor
        self.action_net = action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        latent_pi, _ = self.mlp_extractor(obs)
        return self.action_net(latent_pi)


def main():
    parser = argparse.ArgumentParser(description="Export SB3 PPO policy to ONNX.")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to best_model.zip (SB3 PPO checkpoint)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output .onnx path. Default: <checkpoint_dir>/best_model.onnx",
    )
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    run_dir = os.path.dirname(checkpoint_path)
    onnx_path = args.out or os.path.join(run_dir, "best_model.onnx")
    vecnorm_npz_path = onnx_path.replace(".onnx", "_vecnorm.npz")

    # ------------------------------------------------------------------
    # Load SB3 model
    # ------------------------------------------------------------------
    print(f"Loading checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, device="cpu")
    model.policy.set_training_mode(False)
    print(f"Policy architecture:\n{model.policy}")

    # ------------------------------------------------------------------
    # Export policy network to ONNX
    # ------------------------------------------------------------------
    onnxable = OnnxablePolicy(model.policy.mlp_extractor, model.policy.action_net)
    onnxable.eval()

    dummy_obs = torch.zeros(1, OBS_DIM, dtype=torch.float32)

    torch.onnx.export(
        onnxable,
        dummy_obs,
        onnx_path,
        opset_version=17,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    )
    print(f"Policy exported to: {onnx_path}")

    # ------------------------------------------------------------------
    # Export VecNormalize running statistics
    # ------------------------------------------------------------------
    vecnorm_pkl = os.path.join(run_dir, "best_model_vecnormalize.pkl")
    if not os.path.isfile(vecnorm_pkl):
        # Try alternate name
        vecnorm_pkl = os.path.join(run_dir, "model_vecnormalize.pkl")

    if os.path.isfile(vecnorm_pkl):
        with open(vecnorm_pkl, "rb") as f:
            vn = pickle.load(f)
        mean = np.array(vn.obs_rms.mean, dtype=np.float32)
        var  = np.array(vn.obs_rms.var,  dtype=np.float32)
        np.savez(vecnorm_npz_path, mean=mean, var=var)
        print(f"VecNormalize stats exported to: {vecnorm_npz_path}")
        print(f"  obs_mean[:5] = {mean[:5]}")
        print(f"  obs_var[:5]  = {var[:5]}")
    else:
        print(f"WARN: VecNormalize pkl not found in {run_dir} — skipping stats export.")
        print("  The demo will use identity normalization (obs unchanged).")

    # ------------------------------------------------------------------
    # Quick sanity check
    # ------------------------------------------------------------------
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        test_input = np.zeros((1, OBS_DIM), dtype=np.float32)
        result = sess.run(["action"], {"obs": test_input})[0]
        print(f"\nSanity check — zero obs → action = {result}")
        print("ONNX export verified.")
    except ImportError:
        print("\nonnxruntime not installed — skipping sanity check. Install with:")
        print("  pip install onnxruntime")


if __name__ == "__main__":
    main()
