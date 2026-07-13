"""Collect grasping demonstrations for BC pretraining.

A scripted oracle drives the real Isaac-G1-Pick-v0 environment through its own
action pipeline: differential IK moves the palm through the BODex goal grasp
(pregrasp -> grasp), the fingers close (pregrasp -> grasp -> squeeze), then the
arm lifts. Every step records the policy observation (96-dim) and the ACTION
that produces the commanded joint targets, i.e. a = (q_target - q_default)/scale,
so the dataset is exactly policy-compatible for behavioral cloning.

Episodes are kept only if the env's own success termination fires
(cube above 1.134 m).

Run:
    conda activate env_isaaclab && source _isaac_sim/setup_conda_env.sh
    python .../collect_demos.py --headless --num_envs 256 --episodes 2000
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=2000, help="successful episodes to collect")
parser.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__), "grasp_dataset", "bc_demos.npz"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply, compute_pose_error

ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
HAND_JOINTS = [
    "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_index_proximal_joint",
    "R_middle_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint",
]
ARM_SCALE, HAND_SCALE = 0.3, 0.5
PREGRASP_OFFSET = 0.08     # m back along the palm normal
LIFT_HEIGHT = 0.35         # m above the grasp pose
# phase ids
APPROACH, DESCEND, CLOSE, LIFT = 0, 1, 2, 3
PHASE_BUDGET = {APPROACH: 90, DESCEND: 60, CLOSE: 45, LIFT: 999}
CLOSE_STEPS = 45           # finger interpolation length inside CLOSE


def main():
    env_cfg = parse_env_cfg("Isaac-G1-Pick-v0", num_envs=args_cli.num_envs)
    env = gym.make("Isaac-G1-Pick-v0", cfg=env_cfg)
    obs_dict, _ = env.reset()
    uenv = env.unwrapped
    device = uenv.device
    N = args_cli.num_envs

    robot = uenv.scene["robot"]
    goal_term = uenv._grasp_goal_term  # created by the sample_grasp_goal event
    palm_idx = robot.body_names.index("R_hand_base_link")
    arm_ids = torch.tensor([robot.joint_names.index(n) for n in ARM_JOINTS], device=device)
    hand_ids = torch.tensor([robot.joint_names.index(n) for n in HAND_JOINTS], device=device)
    q_def = robot.data.default_joint_pos.clone()
    fixed_base = robot.is_fixed_base
    jac_body_idx = palm_idx - 1 if fixed_base else palm_idx
    print(f"[demo] fixed_base={fixed_base}, palm body idx={palm_idx}", flush=True)

    phase = torch.zeros(N, dtype=torch.long, device=device)
    phase_step = torch.zeros(N, dtype=torch.long, device=device)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device).expand(N, 3)

    # per-env episode buffers on CPU
    ep_obs = [[] for _ in range(N)]
    ep_act = [[] for _ in range(N)]
    ds_obs, ds_act, ds_len = [], [], []
    n_success, n_fail = 0, 0

    obs = obs_dict["policy"]

    def ik_action(goal_pos, goal_quat):
        """One damped-least-squares IK step toward the goal palm pose -> arm action."""
        palm_pos = robot.data.body_pos_w[:, palm_idx]
        palm_quat = robot.data.body_quat_w[:, palm_idx]
        pos_err, ax_err = compute_pose_error(palm_pos, palm_quat, goal_pos, goal_quat,
                                             rot_error_type="axis_angle")
        err = torch.cat([pos_err, 0.5 * ax_err], dim=1).unsqueeze(-1)          # (N,6,1)
        jac = robot.root_physx_view.get_jacobians()[:, jac_body_idx, :, :]     # (N,6,ndof)
        J = jac[:, :, arm_ids]                                                 # (N,6,7)
        JT = J.transpose(1, 2)
        lam = 0.05
        dq = (JT @ torch.linalg.solve(J @ JT + lam**2 * torch.eye(6, device=device), err)).squeeze(-1)
        q_des = robot.data.joint_pos[:, arm_ids] + torch.clamp(dq, -0.15, 0.15)
        return torch.clamp((q_des - q_def[:, arm_ids]) / ARM_SCALE, -1.0, 1.0)

    step_count = 0
    while n_success < args_cli.episodes:
        goal_pos = goal_term.goal_pos_w
        goal_quat = goal_term.goal_quat_w
        approach = quat_apply(goal_quat, y_axis)         # palm normal in world
        pregrasp_pos = goal_pos - PREGRASP_OFFSET * approach
        lift_pos = goal_pos + torch.tensor([0.0, 0.0, LIFT_HEIGHT], device=device)

        # arm target by phase
        tgt_pos = torch.where(phase.unsqueeze(1) == APPROACH, pregrasp_pos,
                  torch.where(phase.unsqueeze(1) >= LIFT, lift_pos, goal_pos))
        arm_act = ik_action(tgt_pos, goal_quat)

        # finger target by phase
        alpha = (phase_step.float() / CLOSE_STEPS).clamp(0, 1).unsqueeze(1)
        q_pre, q_grasp, q_sq = goal_term.goal_hand_q_pre, goal_term.goal_hand_q, goal_term.goal_hand_q_squeeze
        close_q = torch.where(alpha < 0.6, q_pre + (alpha / 0.6) * (q_grasp - q_pre),
                              q_grasp + ((alpha - 0.6) / 0.4) * (q_sq - q_grasp))
        fing_q = torch.where(phase.unsqueeze(1) < CLOSE, q_pre,
                 torch.where(phase.unsqueeze(1) == CLOSE, close_q, q_sq))
        hand_act = torch.clamp((fing_q - q_def[:, hand_ids]) / HAND_SCALE, -1.0, 1.0)

        action = torch.cat([arm_act, hand_act], dim=1)

        # record BEFORE stepping (obs -> action pairing)
        obs_cpu = obs.cpu().numpy()
        act_cpu = action.cpu().numpy()
        for i in range(N):
            ep_obs[i].append(obs_cpu[i])
            ep_act[i].append(act_cpu[i])

        obs_dict, _, terminated, truncated, _ = env.step(action)
        obs = obs_dict["policy"]
        lifted = uenv.termination_manager.get_term("target_lifted")
        dones = (terminated | truncated)

        # phase advancement
        palm_pos = robot.data.body_pos_w[:, palm_idx]
        err = torch.norm(palm_pos - tgt_pos, dim=1)
        phase_step += 1
        budget = torch.tensor([PHASE_BUDGET[APPROACH], PHASE_BUDGET[DESCEND],
                               PHASE_BUDGET[CLOSE], PHASE_BUDGET[LIFT]], device=device)[phase]
        advance = ((phase == APPROACH) & ((err < 0.025) | (phase_step > budget))) | \
                  ((phase == DESCEND) & ((err < 0.015) | (phase_step > budget))) | \
                  ((phase == CLOSE) & (phase_step > budget))
        phase = torch.where(advance, phase + 1, phase)
        phase_step = torch.where(advance, torch.zeros_like(phase_step), phase_step)

        # flush finished episodes
        done_ids = torch.nonzero(dones).squeeze(-1).tolist()
        for i in done_ids:
            if bool(lifted[i]):
                ds_obs.append(np.array(ep_obs[i], dtype=np.float32))
                ds_act.append(np.array(ep_act[i], dtype=np.float32))
                ds_len.append(len(ep_obs[i]))
                n_success += 1
            else:
                n_fail += 1
            ep_obs[i], ep_act[i] = [], []
            phase[i] = APPROACH
            phase_step[i] = 0

        step_count += 1
        if step_count % 120 == 0:
            tot = max(n_success + n_fail, 1)
            print(f"[demo] steps={step_count} success={n_success} fail={n_fail} "
                  f"rate={n_success/tot:.1%}", flush=True)
        if n_success + n_fail > 40 and n_success == 0 and step_count > 2000:
            print("[demo] oracle never succeeds — aborting for debug", flush=True)
            break

    if ds_obs:
        np.savez(args_cli.out,
                 obs=np.concatenate(ds_obs), act=np.concatenate(ds_act),
                 ep_len=np.array(ds_len))
        tot = max(n_success + n_fail, 1)
        print(f"[demo] saved {n_success} episodes ({sum(ds_len)} transitions, "
              f"success rate {n_success/tot:.1%}) -> {args_cli.out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
