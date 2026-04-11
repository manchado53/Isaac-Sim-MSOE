"""Train PPO agent for Nova Carter Patrol using Stable-Baselines3.

Usage (on a GPU node):
    cd ~/IsaacLab
    ./isaaclab.sh -p ~/isaac-sim-msoe/training/train_nova_carter.py --headless

    # With video recording:
    ./isaaclab.sh -p ~/isaac-sim-msoe/training/train_nova_carter.py --headless --video

    # Custom settings:
    ./isaaclab.sh -p ~/isaac-sim-msoe/training/train_nova_carter.py --headless --num_envs 512 --max_iterations 200
"""

import argparse
import contextlib
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Nova Carter Patrol with PPO (SB3).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Max policy update iterations (default: from yaml).")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume training from checkpoint.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- Now safe to import --
import os
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize


class DebugOverlayWrapper(gym.Wrapper):
    """Draws a debug overlay on rendered frames for training video diagnostics.

    Displays per-step info (step count, curriculum stage, distance to goal,
    last reward) and flashes the termination reason for 30 frames whenever
    an episode resets (TIMEOUT, BAD_ORIENTATION, GOAL_COMPLETE).

    Insert between the Isaac Lab env and RecordVideo so the overlay is baked
    into the recorded MP4.
    """

    _TERM_DISPLAY_FRAMES = 30  # how long to flash the termination reason

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._step_count = 0
        self._last_reward = 0.0
        self._ep_reward_sum = 0.0
        self._last_dist = 0.0
        self._term_reason: str | None = None
        self._term_frames_left = 0
        self._info_debug_str: str = ""
        # Try to load a monospace font; fall back to PIL default
        try:
            self._font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 18)
            self._font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 28)
        except Exception:
            self._font = ImageFont.load_default()
            self._font_large = self._font

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_curriculum_stage(self) -> int:
        try:
            isaac_env = self.env.unwrapped
            for term_cfg in isaac_env.curriculum_manager._class_term_cfgs:
                if hasattr(term_cfg.func, "current_stage"):
                    return int(term_cfg.func.current_stage)
        except Exception:
            pass
        return 0

    def _get_contact_force(self) -> float:
        try:
            isaac_env = self.env.unwrapped
            sensor = isaac_env.scene.sensors["contact_sensor"]
            if not hasattr(self, "_contact_debug_logged"):
                self._contact_debug_logged = True
                print(
                    f"[DEBUG] contact_sensor num_bodies={sensor._num_bodies}, "
                    f"max_force_history={sensor.data.net_forces_w_history.max().item():.4f}N",
                    flush=True,
                )
            # Use history to match undesired_contacts reward — net_forces_w alone misses
            # contacts that resolved before the last physics sub-step of this RL step.
            forces = sensor.data.net_forces_w_history  # (num_envs, history_length, num_bodies, 3)
            return float(torch.norm(forces[0], dim=-1).max().item())
        except Exception:
            return -1.0

    def _get_dist_to_goal(self) -> float:
        try:
            isaac_env = self.env.unwrapped
            cmd = isaac_env.command_manager.get_command("pose_command")
            return float(torch.norm(cmd[0, :2]).item())
        except Exception:
            return -1.0

    def _parse_term_reason(self, info: dict) -> str | None:
        """Extract termination reason from Isaac Lab info dict for env 0."""
        try:
            if info.get("time_outs", [False])[0]:
                return "TIMEOUT"
            terminated = info.get("terminated", None)
            if terminated is not None and terminated[0]:
                # Per-env flags are in termination_manager.get_term() — live (num_envs,) bool tensor.
                # info["log"] entries are scalars (episode mean across all envs), not per-env arrays.
                term_mgr = self.env.unwrapped.termination_manager
                if term_mgr.get_term("bad_orientation")[0]:
                    return "BAD_ORIENTATION"
                if term_mgr.get_term("patrol_complete")[0]:
                    return "GOAL_COMPLETE"
                return "TERMINATED"
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        self._step_count = 0
        self._ep_reward_sum = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        self._last_reward = float(reward) if np.isscalar(reward) else float(reward[0])
        self._ep_reward_sum += self._last_reward
        self._last_dist = self._get_dist_to_goal()

        def to_bool(x):
            import torch
            if isinstance(x, torch.Tensor):
                return x.cpu().numpy()
            return x
        done = np.logical_or(to_bool(terminated), to_bool(truncated))
        # Only track env 0 (the one displayed in the video) — np.any(done) would fire on
        # resets from other envs, flooding the overlay with false "RESET" banners and
        # wiping _ep_reward_sum every step (with 256 envs, some env resets almost every step).
        if bool(done[0]):
            reason = self._parse_term_reason(info)
            if reason is None:
                reason = "RESET"
            self._term_reason = reason
            self._term_frames_left = self._TERM_DISPLAY_FRAMES
            self._ep_reward_sum = 0.0  # reset for next episode
            # Capture raw info keys/values for video display
            lines = []
            for k, v in info.items():
                try:
                    val = v[0] if hasattr(v, '__getitem__') else v
                    lines.append(f"{k}: {val}")
                except Exception:
                    lines.append(f"{k}: ?")
            self._info_debug_str = " | ".join(lines)

        return obs, reward, terminated, truncated, info

    def render(self):
        frame = self.env.render()
        if frame is None:
            return frame

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        h, w = frame.shape[:2]

        # Semi-transparent dark band at top
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        band = ImageDraw.Draw(overlay)
        band.rectangle([(0, 0), (w, 100)], fill=(0, 0, 0, 140))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        # --- Top-left info ---
        stage = self._get_curriculum_stage()
        stage_str = f"Stage {stage}" if stage > 0 else "Stage ?"
        dist_str = f"{self._last_dist:.2f}m" if self._last_dist >= 0 else "?"
        line1 = f"Step {self._step_count:4d}  |  {stage_str}  |  dist {dist_str}"
        line2 = f"reward {self._last_reward:+.3f}"
        line3 = f"ep_reward  {self._ep_reward_sum:+.1f}"
        line4 = f"contact  {self._get_contact_force():.2f}N"
        draw.text((8, 4), line1, font=self._font, fill=(220, 220, 220))
        draw.text((8, 26), line2, font=self._font, fill=(180, 255, 180))
        draw.text((8, 48), line3, font=self._font, fill=(180, 220, 255))
        draw.text((8, 72), line4, font=self._font, fill=(255, 180, 80))

        # --- Episode end banner ---
        if self._term_frames_left > 0:
            alpha = min(1.0, self._term_frames_left / 10.0)  # fade out over last 10 frames
            color_map = {
                "TIMEOUT":         (255, 200, 50),
                "BAD_ORIENTATION": (255, 80, 80),
                "GOAL_COMPLETE":   (80, 255, 120),
                "TERMINATED":      (255, 140, 0),
                "RESET":           (200, 200, 200),
            }
            color = color_map.get(self._term_reason, (255, 255, 255))
            r, g, b = color

            # Full-screen dark overlay
            banner = Image.new("RGBA", img.size, (0, 0, 0, int(180 * alpha)))
            img = Image.alpha_composite(img.convert("RGBA"), banner).convert("RGB")
            draw = ImageDraw.Draw(img)

            # Big "EPISODE END" header
            header = "EPISODE END"
            hbbox = draw.textbbox((0, 0), header, font=self._font_large)
            hw = hbbox[2] - hbbox[0]
            hx = (w - hw) // 2
            hy = h // 2 - 50
            draw.text((hx + 3, hy + 3), header, font=self._font_large, fill=(0, 0, 0))
            draw.text((hx, hy), header, font=self._font_large, fill=(int(r * alpha), int(g * alpha), int(b * alpha)))

            # Reason underneath
            reason_text = f"[ {self._term_reason} ]"
            rbbox = draw.textbbox((0, 0), reason_text, font=self._font_large)
            rw = rbbox[2] - rbbox[0]
            rx = (w - rw) // 2
            ry = hy + 45
            draw.text((rx + 2, ry + 2), reason_text, font=self._font_large, fill=(0, 0, 0))
            draw.text((rx, ry), reason_text, font=self._font_large, fill=(int(r * alpha), int(g * alpha), int(b * alpha)))

            # Raw info dict for debugging
            if self._info_debug_str:
                draw.text((8, h - 24), self._info_debug_str[:120], font=self._font, fill=(200, 200, 100))

            self._term_frames_left -= 1

        return np.array(img)



class PatrolMetricsCallback(BaseCallback):
    """Logs patrol-specific metrics to TensorBoard every rollout.

    - patrol/goals_per_episode: mean goals reached per completed episode
    - patrol/collision_rate: fraction of env-steps with contact force > 1N
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._completed_goals: list[float] = []
        self._collision_steps = 0
        self._total_steps = 0

    def _get_isaac_env(self):
        try:
            return self.training_env.unwrapped
        except Exception:
            return None

    def _on_step(self) -> bool:
        isaac_env = self._get_isaac_env()
        if isaac_env is None:
            return True

        dones = self.locals.get("dones")
        if dones is None:
            return True

        self._total_steps += len(dones)

        # Collision rate: count env-steps where chassis contact force exceeds reward threshold
        try:
            sensor = isaac_env.scene.sensors["contact_sensor"]
            # net_forces_w_history shape: (num_envs, history_length, num_bodies, 3)
            # Take norm over xyz, max over history and bodies to get worst-case per env
            forces = sensor.data.net_forces_w_history
            max_force = torch.norm(forces, dim=-1).max(dim=-1).values.max(dim=-1).values  # (num_envs,)
            self._collision_steps += int((max_force > 1.0).sum().item())
        except Exception:
            pass

        # Goals per episode: snapshot command_counter at episode end before env auto-resets
        try:
            cmd_term = isaac_env.command_manager.get_term("pose_command")
            for i, done in enumerate(dones):
                if bool(done):
                    self._completed_goals.append(float(cmd_term.command_counter[i].item()))
        except Exception:
            pass

        return True

    def _on_rollout_end(self) -> None:
        if self._completed_goals:
            self.logger.record("patrol/goals_per_episode", float(np.mean(self._completed_goals)))
            self._completed_goals.clear()

        if self._total_steps > 0:
            self.logger.record("patrol/collision_rate", self._collision_steps / self._total_steps)
        self._collision_steps = 0
        self._total_steps = 0


class ETACallback(BaseCallback):
    """Prints ETA and progress every N iterations to SLURM logs."""

    def __init__(self, num_envs: int, steps_per_rollout: int, verbose: int = 0):
        super().__init__(verbose)
        self.num_envs = num_envs
        self.steps_per_rollout = steps_per_rollout
        self._start_time = None
        self._total_iters = None

    def _on_training_start(self) -> None:
        self._start_time = time.time()
        self._total_iters = self.locals.get("total_timesteps", 0) // (self.num_envs * self.steps_per_rollout)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self._start_time is None or self._total_iters is None or self._total_iters == 0:
            return
        current_iter = self.num_timesteps // (self.num_envs * self.steps_per_rollout)
        elapsed = time.time() - self._start_time
        secs_per_iter = elapsed / max(current_iter, 1)
        remaining_iters = self._total_iters - current_iter
        eta_secs = remaining_iters * secs_per_iter

        eta_m, eta_s = divmod(int(eta_secs), 60)
        eta_h, eta_m = divmod(eta_m, 60)
        elapsed_m, elapsed_s = divmod(int(elapsed), 60)
        elapsed_h, elapsed_m = divmod(elapsed_m, 60)

        print(
            f"[PROGRESS] Iter {current_iter}/{self._total_iters}"
            f" | Elapsed: {elapsed_h}h {elapsed_m:02d}m"
            f" | ETA: {eta_h}h {eta_m:02d}m {eta_s:02d}s"
            f" | {secs_per_iter:.1f}s/iter"
        )


class BestModelCallback(BaseCallback):
    """Saves best and latest model per curriculum stage.

    Per stage:
      - best_model_stage{N}.zip             — highest ep_rew_mean seen in this stage
      - best_model_stage{N}_vecnormalize.pkl
      - latest_model_stage{N}.zip           — overwritten every rollout (progress view)
      - latest_model_stage{N}_vecnormalize.pkl
    """

    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_reward_per_stage: dict[int, float] = {}
        self._current_stage: int = 0

    def _get_curriculum_stage(self) -> int:
        try:
            # self.training_env is VecNormalize wrapping Sb3VecEnvWrapper.
            # Sb3VecEnvWrapper.unwrapped traverses gym wrapper chain to ManagerBasedRLEnv.
            vec_env = self.training_env
            if hasattr(vec_env, "venv"):
                vec_env = vec_env.venv          # VecNormalize -> Sb3VecEnvWrapper
            isaac_env = vec_env.unwrapped       # Sb3VecEnvWrapper.unwrapped -> ManagerBasedRLEnv
            for term_cfg in isaac_env.curriculum_manager._class_term_cfgs:
                if hasattr(term_cfg.func, "current_stage"):
                    return int(term_cfg.func.current_stage)
        except Exception:
            pass
        return 0

    def _on_step(self) -> bool:
        return True

    def _save(self, name: str) -> None:
        self.model.save(os.path.join(self.save_path, name))
        if isinstance(self.training_env, VecNormalize):
            self.training_env.save(os.path.join(self.save_path, f"{name}_vecnormalize.pkl"))

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) == 0:
            return

        stage = self._get_curriculum_stage()
        reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])

        if stage != self._current_stage:
            print(f"[BEST] Curriculum stage {self._current_stage} -> {stage}, resetting reward threshold")
            self._current_stage = stage

        # Always save latest for this stage (overwritten each rollout)
        self._save(f"latest_model_stage{stage}")

        # Save best for this stage if improved
        if reward > self.best_reward_per_stage.get(stage, -float("inf")):
            self.best_reward_per_stage[stage] = reward
            self._save(f"best_model_stage{stage}")
            print(f"[BEST] Stage {stage} new best! ep_rew_mean={reward:.2f} (step {self.num_timesteps})")

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

# Import nova_carter_patrol to trigger gym.register()
import nova_carter_patrol  # noqa: F401

from nova_carter_patrol.nova_carter_patrol_env_cfg import NovaCarterPatrolFlatEnvCfg


def main():
    # Load agent config from yaml
    import yaml
    agent_cfg_path = os.path.join(os.path.dirname(__file__), "nova_carter_patrol", "agents", "sb3_ppo_cfg.yaml")
    with open(agent_cfg_path) as f:
        agent_cfg = yaml.safe_load(f)

    # Environment config
    env_cfg = NovaCarterPatrolFlatEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # Sync SB3 agent seed with env seed (yaml default is overridden here)
    agent_cfg["seed"] = args_cli.seed

    # Override max iterations if specified
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    # Logging directory
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = os.path.join(os.path.dirname(__file__), "logs", "sb3", "NovaCarter-Patrol")
    log_dir = os.path.abspath(os.path.join(log_root, run_info))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")

    # Save configs
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Process SB3 config (converts string types like 'nn.ReLU' to actual classes)
    n_steps_per_rollout = agent_cfg.get("n_steps", 512)  # save before process_sb3_cfg consumes it
    agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")

    # Create environment
    env = gym.make(
        "Isaac-NovaCarter-Patrol-Flat-v0",
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # Wrap for video recording
    if args_cli.video:
        # Set initial camera view
        env_origins = env.unwrapped.scene.env_origins.cpu().numpy()
        env.unwrapped.sim.set_camera_view(
            eye=(float(env_origins[0, 0]), float(env_origins[0, 1]) - 8.0, 12.0),
            target=(float(env_origins[0, 0]), float(env_origins[0, 1]), 0.0),
        )
        debug_overlay = DebugOverlayWrapper(env)
        env = debug_overlay

        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap for SB3 compatibility
    env = Sb3VecEnvWrapper(env)

    # Remove normalization keys from agent_cfg (not PPO params)
    for key in ["normalize_input", "normalize_value", "clip_obs"]:
        agent_cfg.pop(key, None)

    # Normalize observations only — reward normalization is disabled because
    # with short episodes VecNormalize sees near-zero reward variance, amplifies
    # rewards to extreme values, and causes NaN in PPO's Normal distribution.
    print("[INFO] Applying VecNormalize (obs only, reward norm DISABLED)")
    env = VecNormalize(
        env,
        training=True,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=agent_cfg["gamma"],
    )

    start_time = time.time()

    # Create PPO agent
    agent = PPO(policy_arch, env, verbose=1, tensorboard_log=log_dir, **agent_cfg)
    if args_cli.checkpoint is not None:
        agent = PPO.load(args_cli.checkpoint, env, print_system_info=True, tensorboard_log=log_dir, **agent_cfg)
        # Reset timestep counter so learn() trains for the full requested amount
        # (otherwise SB3 sees num_timesteps >= total_timesteps from the checkpoint and exits)
        agent.num_timesteps = 0


    # Callbacks
    callbacks = [
        PatrolMetricsCallback(),
        ETACallback(num_envs=args_cli.num_envs, steps_per_rollout=n_steps_per_rollout),
        BestModelCallback(save_path=log_dir),
    ]

    # Train
    print(f"[INFO] Training for {int(n_timesteps):,} timesteps ({args_cli.num_envs} envs)...")
    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(
            total_timesteps=n_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

    # Save final model
    agent.save(os.path.join(log_dir, "model"))
    print(f"\n[INFO] Final model saved to: {os.path.join(log_dir, 'model.zip')}")

    # Save normalization stats (needed for evaluation)
    if isinstance(env, VecNormalize):
        env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))
        print(f"[INFO] Normalization stats saved to: {os.path.join(log_dir, 'model_vecnormalize.pkl')}")

    print(f"[INFO] Training time: {time.time() - start_time:.1f}s")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
