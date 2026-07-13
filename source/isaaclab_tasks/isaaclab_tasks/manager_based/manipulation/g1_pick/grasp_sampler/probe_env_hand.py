"""Probe the right-hand coupling inside the real Isaac-G1-Pick-v0 training env.

Commands full finger flexion through the env's own action pipeline
(InspireMimicAction) and reports where the joints actually go.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401  (registers envs)
from isaaclab_tasks.utils import parse_env_cfg

ACT = ["R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_index_proximal_joint",
       "R_middle_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint"]
MIM = ["R_thumb_intermediate_joint", "R_thumb_distal_joint", "R_index_intermediate_joint",
       "R_middle_intermediate_joint", "R_ring_intermediate_joint", "R_pinky_intermediate_joint"]


def main():
    env_cfg = parse_env_cfg("Isaac-G1-Pick-v0", num_envs=1)
    env = gym.make("Isaac-G1-Pick-v0", cfg=env_cfg)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    act_ids = [robot.joint_names.index(n) for n in ACT]
    mim_ids = [robot.joint_names.index(n) for n in MIM]

    action = torch.zeros(1, env.unwrapped.action_manager.total_action_dim, device=env.unwrapped.device)
    for level in (0.4, 0.7, 1.0):
        action[0, 7:13] = level  # hand flexion command (scale 0.5 rad)
        for _ in range(120):
            env.step(action)
        a = robot.data.joint_pos[0, act_ids].cpu().numpy().round(3)
        m = robot.data.joint_pos[0, mim_ids].cpu().numpy().round(3)
        print(f"[probe] cmd={level}: proximal={a} slaves={m}", flush=True)

    import os
    os._exit(0)


if __name__ == "__main__":
    main()
