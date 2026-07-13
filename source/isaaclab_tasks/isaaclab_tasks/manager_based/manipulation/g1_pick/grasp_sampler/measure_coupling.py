"""Measure the effective mimic coupling of the g1_pick USD hand in simulation.

Sweeps the 6 actuated right-hand joints through their range via PD targets,
lets the PhysX mimic constraint settle, and prints slave positions vs master.

Run:
    conda activate env_isaaclab && source _isaac_sim/setup_conda_env.sh
    python .../measure_coupling.py --headless
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robot_cfg import G1_INSPIRE_CFG  # noqa: E402

ACT = ["R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_index_proximal_joint",
       "R_middle_proximal_joint", "R_ring_proximal_joint", "R_pinky_proximal_joint"]
MIM = ["R_thumb_intermediate_joint", "R_thumb_distal_joint", "R_index_intermediate_joint",
       "R_middle_intermediate_joint", "R_ring_intermediate_joint", "R_pinky_intermediate_joint"]


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = G1_INSPIRE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot = scene["robot"]
    device = robot.device

    act_ids = torch.tensor([robot.joint_names.index(n) for n in ACT], device=device)
    mim_ids = [robot.joint_names.index(n) for n in MIM]
    default_jp = robot.data.default_joint_pos.clone()
    default_jv = torch.zeros_like(robot.data.default_joint_vel)
    body_ids = torch.tensor([i for i in range(robot.num_joints)
                             if i not in act_ids.tolist() + mim_ids], device=device)

    def settle(q6, steps=360):
        t = torch.tensor(q6, device=device, dtype=torch.float32).unsqueeze(0)
        for _ in range(steps):
            robot.write_joint_state_to_sim(default_jp[:, body_ids], default_jv[:, body_ids],
                                           joint_ids=body_ids)
            robot.set_joint_position_target(t, joint_ids=act_ids)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
        act = robot.data.joint_pos[0, act_ids].cpu().numpy()
        mim = robot.data.joint_pos[0, mim_ids].cpu().numpy()
        return act, mim

    print("[cpl] cmd  = commanded proximal | act = actual | mim = slaves "
          "[th_int, th_dist, idx_int, mid_int, ring_int, pinky_int]", flush=True)
    for level in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        q6 = [0.0, min(level, 0.6) * 0.9, level, level, level, level]
        act, mim = settle(q6)
        print(f"[cpl] cmd={q6[2]:.1f}: act={act.round(3)}  mim={mim.round(3)}", flush=True)
        # effective ratios (guard div-by-zero)
        import numpy as np
        masters = np.array([act[1], act[1], act[2], act[3], act[4], act[5]])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(np.abs(masters) > 0.05, mim / masters, np.nan)
        print(f"[cpl]          effective slave/master ratios: {ratios.round(3)}", flush=True)

    os._exit(0)


if __name__ == "__main__":
    main()
