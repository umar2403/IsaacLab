"""Play a trained g1_pick checkpoint WITH the UltraDexGrasp goal grasp visualized.

Draws, every step, in the viewer / recorded video:
  - a coordinate FRAME at the goal grasp pose (position + orientation from the
    32-grasp library, live-tracked to the cube)
  - a SPHERE at the goal that is GREEN when the palm is within `--reach_thresh`
    of it, RED otherwise — so you can see at a glance whether the policy reaches
    the UltraDexGrasp target
  - a small BLUE sphere at the actual palm (R_hand_base_link)

Also prints the palm→goal distance to the console.

Usage (mirrors play.py; supports the same hydra camera overrides):
  conda activate env_isaaclab && source _isaac_sim/setup_conda_env.sh
  python .../grasp_sampler/play_with_goal_markers.py \
    --task Isaac-G1-Pick-Play-v0 --num_envs 1 --headless --video --video_length 400 \
    --enable_cameras \
    --checkpoint /home/umar/IsaacLab/logs/rsl_rl/g1_pick/2026-07-15_13-49-59/model_6998.pt \
    'env.viewer.origin_type=env' 'env.viewer.eye=[1.5,-1.1,1.4]' 'env.viewer.lookat=[0.35,0.0,1.0]'
"""
import argparse
import os
import sys

# make the rsl_rl helper `cli_args` importable (it lives next to play.py)
_RSL_RL_DIR = os.path.join(os.environ.get("ISAACLAB_PATH", "/home/umar/IsaacLab"),
                           "scripts", "reinforcement_learning", "rsl_rl")
sys.path.append(_RSL_RL_DIR)

from isaaclab.app import AppLauncher

import cli_args  # noqa: E402  (from the rsl_rl scripts dir)

parser = argparse.ArgumentParser(description="Play g1_pick with UltraDexGrasp goal markers.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--task", type=str, default="Isaac-G1-Pick-Play-v0", help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point",
                    help="Name of the RL agent configuration entry point.")
parser.add_argument("--reach_thresh", type=float, default=0.05,
                    help="Palm-goal distance (m) below which the goal marker turns green.")
parser.add_argument("--video_fps", type=int, default=15,
                    help="Frame rate written into the MP4. Lower = slower playback (sim runs ~30 Hz).")
parser.add_argument("--real_time", action="store_true", default=False, help="Run at real-time speed.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args  # hand the rest to hydra

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import time
import numpy as np
import yaml
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import quat_error_magnitude
import isaaclab.sim as sim_utils

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

_HERE = os.path.dirname(os.path.abspath(__file__))
_BODEX = os.path.join(_HERE, "ultradex_repo", "third_party", "BODex_api", "src", "bodex", "content")
_URDF = os.path.join(_BODEX, "assets", "robot", "inspire_hand", "inspire_hand_right.urdf")
_ROBOT_YML = os.path.join(_BODEX, "configs", "robot", "inspire_right.yml")
_VALID_NPZ = os.path.join(_HERE, "grasp_dataset", "cube_5cm_grasps_valid.npz")
_ACT = ["thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint", "index_proximal_joint",
        "middle_proximal_joint", "ring_proximal_joint", "pinky_proximal_joint"]

# ---- EDIT ME: extra rotation applied to the ghost goal hand for eyeballing ----
# Euler angles in degrees applied in the palm-LOCAL frame (order xyz).
# e.g. (0, 0, 90) twists the goal hand 90 deg about its approach/finger axis.
_GHOST_ROT_EULER_DEG = (0.0, 0.0, 0.0)
_GHOST_ROT = R.from_euler("xyz", np.deg2rad(_GHOST_ROT_EULER_DEG)).as_matrix()


class GoalHandGhost:
    """Draws the Inspire hand's collision-sphere silhouette at the goal grasp pose.

    Uses the ~40 collision spheres from inspire_right.yml, forward-kinematics'd to
    the goal finger configuration, anchored at the goal root pose, with the
    URDF->USD calibration applied. Result: a translucent 'ghost hand' at the
    UltraDexGrasp target showing both palm pose AND finger shape.
    """

    def __init__(self):
        self.joints = self._parse_urdf(_URDF)
        cfg = yaml.safe_load(open(_ROBOT_YML))["robot_cfg"]["kinematics"]["collision_spheres"]
        self.spheres = []  # list of (link, center(3,), radius)
        for link, lst in cfg.items():
            for s in lst:
                self.spheres.append((link, np.array(s["center"], float), float(s["radius"])))
        self.radii = np.array([r for _, _, r in self.spheres])
        d = np.load(_VALID_NPZ, allow_pickle=True)
        self.T = d["T_usdbase_urdfbase"] if "T_usdbase_urdfbase" in d else np.eye(4)
        self.n = len(self.spheres)

    @staticmethod
    def _parse_urdf(path):
        joints = {}
        for j in ET.parse(path).getroot().findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
            rpy = np.array([float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()])
            ax = j.find("axis")
            joints[j.get("name")] = dict(
                type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                xyz=xyz, rpy=rpy, axis=np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else None)
        return joints

    def _fk_all(self, q6):
        qmap = dict(zip(_ACT, q6))
        poses = {"base": np.eye(4)}
        changed = True
        while changed:
            changed = False
            for name, j in self.joints.items():
                if j["parent"] in poses and j["child"] not in poses:
                    T = np.eye(4)
                    T[:3, :3] = R.from_euler("xyz", j["rpy"]).as_matrix()
                    T[:3, 3] = j["xyz"]
                    if j["type"] == "revolute":
                        a = j["axis"] / np.linalg.norm(j["axis"])
                        Tj = np.eye(4); Tj[:3, :3] = R.from_rotvec(a * qmap.get(name, 0.0)).as_matrix()
                        T = T @ Tj
                    poses[j["child"]] = poses[j["parent"]] @ T
                    changed = True
        return poses

    def adjusted_goal_mat(self, goal_pos_w, goal_quat_w):
        """Goal pose (4x4) with the editable _GHOST_ROT applied in the palm-local frame."""
        goal = np.eye(4)
        goal[:3, :3] = R.from_quat(goal_quat_w, scalar_first=True).as_matrix() @ _GHOST_ROT
        goal[:3, 3] = goal_pos_w
        return goal

    def world_centers(self, goal_pos_w, goal_quat_w, q6):
        """Return (n,3) world positions of all collision spheres for one goal."""
        link_poses = self._fk_all(np.asarray(q6, float))
        goal = self.adjusted_goal_mat(goal_pos_w, goal_quat_w)
        base = goal @ self.T  # world <- USD-handbase <- URDF-base
        out = np.zeros((self.n, 3))
        for i, (link, c, _) in enumerate(self.spheres):
            Tlink = base @ link_poses.get(link, np.eye(4))
            out[i] = (Tlink @ np.array([c[0], c[1], c[2], 1.0]))[:3]
        return out


def _make_markers():
    # goal pose as an RGB coordinate frame (shows target orientation)
    frame_cfg = FRAME_MARKER_CFG.copy()
    frame_cfg.prim_path = "/Visuals/goal_frame"
    frame_cfg.markers["frame"].scale = (0.10, 0.10, 0.10)
    goal_frame = VisualizationMarkers(frame_cfg)

    # ghost hand: one cyan unit sphere prototype, instanced per collision sphere
    ghost = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/Visuals/goal_hand",
        markers={"s": sim_utils.SphereCfg(
            radius=1.0,  # scaled per-instance to each sphere's radius
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.85, 0.9), opacity=0.55),
        )},
    ))
    return goal_frame, ghost


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        # RecordVideo writes the MP4 at env.metadata["render_fps"]; lowering it
        # makes the same motion play back in slow motion.
        env.metadata["render_fps"] = args_cli.video_fps
        env = gym.wrappers.RecordVideo(env, video_folder=os.path.join(log_dir, "videos", "play_markers"),
                                       step_trigger=lambda step: step == 0,
                                       video_length=args_cli.video_length, disable_logger=True)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    goal_frame, ghost = _make_markers()
    hand_viz = GoalHandGhost()
    ghost_scales = torch.tensor(np.repeat(hand_viz.radii[:, None], 3, axis=1), dtype=torch.float32,
                                device=env.unwrapped.device)
    robot = env.unwrapped.scene["robot"]

    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        # the reward term caches the goal-sampler instance on the env after step 1
        term = getattr(env.unwrapped, "_grasp_goal_term", None)
        if term is not None:
            goal_pos = term.goal_pos_w
            goal_quat = term.goal_quat_w
            palm_pos = robot.data.body_pos_w[:, term._palm_body_idx]
            dist = torch.norm(palm_pos - goal_pos, dim=1)
            # palm -> cube center distance
            cube_pos = env.unwrapped.scene[term._object_name].data.root_pos_w
            dist_cube = torch.norm(palm_pos - cube_pos, dim=1)
            # wrist ORIENTATION error (palm quat vs goal quat)
            palm_quat = robot.data.body_quat_w[:, term._palm_body_idx]
            orient_err = quat_error_magnitude(palm_quat, goal_quat)      # rad
            # FINGER config error (current 6 proximal joints vs goal grasp)
            q_now = robot.data.joint_pos[:, term._hand_joint_ids]
            q_err = torch.norm(q_now - term.goal_hand_q, dim=1)          # rad, L2 over 6 joints
            q_err_per = (q_now - term.goal_hand_q).abs()                 # per-joint |error|

            # ghost hand at the goal (env 0): FK the collision spheres to the goal config
            gp0 = goal_pos[0].cpu().numpy()
            gq0 = goal_quat[0].cpu().numpy()
            centers = hand_viz.world_centers(gp0, gq0, term.goal_hand_q[0].cpu().numpy())
            # frame marker shows the SAME (rotated) orientation as the ghost
            adj_quat = R.from_matrix(hand_viz.adjusted_goal_mat(gp0, gq0)[:3, :3]).as_quat(scalar_first=True)
            goal_frame.visualize(
                translations=goal_pos[:1],
                orientations=torch.tensor(adj_quat, dtype=torch.float32,
                                          device=env.unwrapped.device).unsqueeze(0))
            ghost.visualize(
                translations=torch.tensor(centers, dtype=torch.float32, device=env.unwrapped.device),
                scales=ghost_scales)

            # one-time: distance between the goal pose and the cube center
            if timestep == 0:
                goal_cube = torch.norm(goal_pos - cube_pos, dim=1)[0].item()
                print(f"[goal] goal→cube distance = {goal_cube*100:.1f} cm "
                      f"(fixed offset of the goal grasp above/around the cube)", flush=True)

            if timestep % 15 == 0:
                d0 = dist[0].item()
                dc0 = dist_cube[0].item()
                oe0 = np.degrees(orient_err[0].item())         # wrist orientation error
                qe0 = np.degrees(q_err[0].item())              # finger config error (L2)
                per = np.degrees(q_err_per[0].cpu().numpy())   # [th_yaw,th_pitch,idx,mid,ring,pinky]
                tag = "REACHED" if d0 < args_cli.reach_thresh else "       "
                print(f"[goal] step {timestep:4d}  palm→goal = {d0*100:5.1f} cm   "
                      f"palm→cube = {dc0*100:5.1f} cm   wrist_err = {oe0:5.1f}°   "
                      f"finger_err = {qe0:5.1f}°  {tag}", flush=True)
                print(f"        finger err per joint [th_yaw,th_pitch,idx,mid,ring,pinky] = "
                      f"{np.round(per, 1)}", flush=True)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1

        if args_cli.real_time:
            dt = env.unwrapped.step_dt
            wait = dt - (time.time() - start)
            if wait > 0:
                time.sleep(wait)

    # close the env first so RecordVideo finalizes/writes the MP4 ...
    env.close()
    if args_cli.video:
        print(f"[goal] video written to "
              f"{os.path.join(log_dir, 'videos', 'play_markers')}", flush=True)
    # ... then hard-exit. simulation_app.close() hangs indefinitely in headless
    # mode AFTER the run has finished, which looks like the script freezing.
    os._exit(0)


if __name__ == "__main__":
    main()
    simulation_app.close()
