"""Play a trained g1_pick checkpoint WITH the UltraDexGrasp goal grasp visualized.

Draws, every step, in the viewer / recorded video, for every env:
  - a coordinate FRAME at the goal grasp pose (position + orientation from the
    32-grasp library, live-tracked to the cube)
  - a ghost hand at the GOAL pose: cyan/BLUE collision spheres (the same ones
    `grasp_selection`'s optimizer and `mdp.grasp_reach_reward` use)
  - a second ghost hand at the ROBOT'S ACTUAL live pose/joints: ORANGE spheres,
    same sphere set, FK'd to the real hand instead of the goal -- lets you see at
    a glance how closely the sim hand is actually mimicking the target shape

Also prints the palm→goal distance to the console, and (env 0 only) saves two
diagnostic graphs after the run: palm pose error over the episode, and each
fingertip's distance to its own `grasp_reach_reward` target contact point.

Usage (mirrors play.py; supports the same hydra camera overrides):
  conda activate env_isaaclab && source _isaac_sim/setup_conda_env.sh
  python .../grasp_sampler/play_with_goal_markers.py \
    --task Isaac-G1-Pick-Play-v0 --num_envs 1 --headless --video --video_length 400 \
    --enable_cameras \
    --checkpoint /home/umar/IsaacLab/logs/rsl_rl/g1_pick/2026-07-15_13-49-59/model_6998.pt \
    'env.viewer.origin_type=env' 'env.viewer.eye=[1.5,-1.1,1.4]' 'env.viewer.lookat=[0.35,0.0,1.0]'
"""
import argparse
import json
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
parser.add_argument("--episodes", type=int, default=0,
                    help="Stop after env 0 completes this many episodes (reset/success/failure), "
                         "instead of running to --video_length regardless. Episode boundaries are "
                         "marked with vertical dashed lines on the diagnostic graphs so multiple "
                         "episodes stay readable. 0 = disabled (old behavior: run to "
                         "--video_length). --video_length still applies as an upper bound.")
parser.add_argument("--single_episode", action="store_true", default=False,
                    help="Shorthand for --episodes 1.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.single_episode:
    args_cli.episodes = max(args_cli.episodes, 1)
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
from isaaclab.utils.math import quat_apply, quat_error_magnitude
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

# same cache mdp.grasp_reach_reward reads -- see MDP_REPORT.md Sec 5.2.4
_FINGERTIP_CONTACTS_FILE = os.path.join(os.path.dirname(_HERE), "grasp_selection", "fingertip_contacts.json")
_FINGERTIP_BODIES = ["R_thumb_distal", "R_index_intermediate",
                     "R_middle_intermediate", "R_ring_intermediate", "R_pinky_intermediate"]
_FINGERTIP_ORDER = ["thumb", "index", "middle", "ring", "pinky"]  # matches the cache's order

def _cube_surface_gap(p_world, cube_pos, cube_quat_wxyz, sphere_radii, half_edge=0.025):
    """Real geometric gap between a set of collision-sphere SURFACES and the cube's
    box surface -- negative means the sphere genuinely overlaps the cube (not a camera
    occlusion illusion). Same signed-distance formula grasp_selection/hand_model.py's
    cube_surface_dist uses, just applied to the robot's LIVE simulated pose each step
    instead of an offline synthesis pose.

    p_world: (n,3) sphere centers in world frame.
    cube_pos: (3,), cube_quat_wxyz: (4,) -- the cube's LIVE pose this step.
    sphere_radii: (n,) each sphere's own radius (subtracted so this is a SURFACE gap,
        not a center-to-surface distance).
    Returns: (n,) array, gap[i] < 0 <=> sphere i's surface is inside the cube.
    """
    Rm = R.from_quat(cube_quat_wxyz, scalar_first=True).as_matrix()  # world <- cube-local
    p_local = (p_world - cube_pos) @ Rm  # world point -> cube-local frame
    d = np.abs(p_local) - half_edge
    outside = np.linalg.norm(np.clip(d, 0.0, None), axis=-1)
    inside = np.minimum(np.max(d, axis=-1), 0.0)
    sdf = outside + inside  # >0 outside the box, <=0 inside (center-to-surface, signed)
    return sdf - sphere_radii


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

    # ghost hand at the GOAL: cyan/blue unit sphere prototype, instanced per collision sphere
    ghost = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/Visuals/goal_hand",
        markers={"s": sim_utils.SphereCfg(
            radius=1.0,  # scaled per-instance to each sphere's radius
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.85, 0.9), opacity=0.55),
        )},
    ))
    # same sphere set, but FK'd to the robot's ACTUAL live joint state/pose -- lets you see
    # at a glance how well the real hand's shape matches the goal ghost above.
    ghost_actual = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/Visuals/actual_hand",
        markers={"s": sim_utils.SphereCfg(
            radius=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.1), opacity=0.65),
        )},
    ))
    return goal_frame, ghost, ghost_actual


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

    goal_frame, ghost, ghost_actual = _make_markers()
    hand_viz = GoalHandGhost()
    num_envs = env.unwrapped.num_envs
    # one ghost hand's worth of sphere scales, tiled once per env (constant across steps)
    ghost_scales = torch.tensor(
        np.tile(np.repeat(hand_viz.radii[:, None], 3, axis=1), (num_envs, 1)),
        dtype=torch.float32, device=env.unwrapped.device,
    )
    robot = env.unwrapped.scene["robot"]
    fingertip_body_ids = [robot.body_names.index(n) for n in _FINGERTIP_BODIES]

    # fingertip target contact points (grasp_reach_reward's own cache -- MDP_REPORT.md 5.2.4)
    with open(_FINGERTIP_CONTACTS_FILE) as f:
        _fc = json.load(f)
    assert _fc["fingertip_order"] == _FINGERTIP_ORDER
    fingertip_contacts_obj = torch.tensor(
        _fc["contact_pos_cube_frame"], dtype=torch.float32, device=env.unwrapped.device)  # (5,3)

    # env-0 history for the diagnostic graphs, saved after the loop
    hist_palm_pos_err, hist_palm_orient_err, hist_finger_dist, hist_min_gap = [], [], [], []
    episode_boundaries = []  # step indices (in the logged history) where an episode ended
    episode_count = 0

    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

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

            # ghost hand at the goal, for EVERY env: FK the collision spheres to each
            # env's own goal config. CPU numpy loop per env (GoalHandGhost is a single-
            # hand, single-pose visualizer) -- fine at diagnostic-video env counts.
            goal_pos_np = goal_pos.cpu().numpy()
            goal_quat_np = goal_quat.cpu().numpy()
            goal_hand_q_np = term.goal_hand_q.cpu().numpy()
            # ACTUAL hand: same sphere set, FK'd to the robot's real live pose/joints instead
            actual_palm_pos_np = palm_pos.cpu().numpy()
            actual_palm_quat_np = palm_quat.cpu().numpy()
            actual_q_np = q_now.cpu().numpy()
            all_centers, all_adj_quats, all_actual_centers = [], [], []
            for e in range(num_envs):
                gp_e, gq_e = goal_pos_np[e], goal_quat_np[e]
                all_centers.append(hand_viz.world_centers(gp_e, gq_e, goal_hand_q_np[e]))
                all_adj_quats.append(
                    R.from_matrix(hand_viz.adjusted_goal_mat(gp_e, gq_e)[:3, :3]).as_quat(scalar_first=True)
                )
                all_actual_centers.append(
                    hand_viz.world_centers(actual_palm_pos_np[e], actual_palm_quat_np[e], actual_q_np[e])
                )
            centers = np.concatenate(all_centers, axis=0)  # (num_envs * n_spheres, 3)
            adj_quats = np.stack(all_adj_quats, axis=0)    # (num_envs, 4)
            actual_centers = np.concatenate(all_actual_centers, axis=0)

            goal_frame.visualize(
                translations=goal_pos,
                orientations=torch.tensor(adj_quats, dtype=torch.float32, device=env.unwrapped.device))
            ghost.visualize(
                translations=torch.tensor(centers, dtype=torch.float32, device=env.unwrapped.device),
                scales=ghost_scales)
            ghost_actual.visualize(
                translations=torch.tensor(actual_centers, dtype=torch.float32, device=env.unwrapped.device),
                scales=ghost_scales)

            # fingertip -> its own target contact point (grasp_reach_reward's signal, §5.2.4)
            tips_w = robot.data.body_pos_w[:, fingertip_body_ids]          # (N,5,3)
            n_e = tips_w.shape[0]
            contacts_b = fingertip_contacts_obj.unsqueeze(0).expand(n_e, -1, -1).reshape(-1, 3)
            cube_quat_full = env.unwrapped.scene[term._object_name].data.root_quat_w
            cube_quat_exp = cube_quat_full.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4)
            cube_pos_exp = cube_pos.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 3)
            finger_targets_w = (quat_apply(cube_quat_exp, contacts_b) + cube_pos_exp).view(n_e, 5, 3)
            finger_dist = torch.norm(tips_w - finger_targets_w, dim=-1)    # (N,5)

            # REAL geometric gap between the robot's actual live collision spheres and the
            # cube's box surface -- ground truth for "is the hand mesh genuinely overlapping
            # the cube", not a camera-occlusion illusion. Negative = real interpenetration.
            cube_pos0_np = cube_pos[0].cpu().numpy()
            cube_quat0_np = cube_quat_full[0].cpu().numpy()
            gaps0 = _cube_surface_gap(all_actual_centers[0], cube_pos0_np, cube_quat0_np, hand_viz.radii)
            min_gap0 = float(gaps0.min())

            # env-0 history for the post-run graphs. Skip logging on the step where env 0's
            # episode just ended: IsaacLab auto-resets INSIDE env.step() when done fires, so
            # everything read above already reflects the NEXT episode's reset state, not the
            # terminal moment of this one -- logging it would put a false spike/discontinuity.
            episode_ended_this_step = args_cli.episodes > 0 and bool(dones[0].item())
            if not episode_ended_this_step:
                hist_palm_pos_err.append(dist[0].item())
                hist_palm_orient_err.append(np.degrees(orient_err[0].item()))
                hist_finger_dist.append(finger_dist[0].cpu().numpy() * 100.0)  # cm
                hist_min_gap.append(min_gap0 * 100.0)  # cm
            else:
                episode_count += 1
                episode_boundaries.append(len(hist_palm_pos_err))  # mark where the NEXT ep starts

            if timestep % 15 == 0:
                tag = "PENETRATING" if min_gap0 < 0 else ""
                print(f"[goal] step {timestep:4d}  min hand-sphere<->cube surface gap = "
                      f"{min_gap0*100:+6.2f} cm  {tag}", flush=True)

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

        timestep += 1
        if args_cli.episodes > 0 and dones[0].item() > 0:
            print(f"[goal] env 0 episode {episode_count}/{args_cli.episodes} ended at "
                  f"step {timestep}", flush=True)
            if episode_count >= args_cli.episodes:
                break
        if args_cli.video and timestep == args_cli.video_length:
            break

        if args_cli.real_time:
            dt = env.unwrapped.step_dt
            wait = dt - (time.time() - start)
            if wait > 0:
                time.sleep(wait)

    graph_dir = os.path.join(log_dir, "videos", "play_markers")

    # close the env first so RecordVideo finalizes/writes the MP4 ...
    env.close()
    if args_cli.video:
        print(f"[goal] video written to {graph_dir}", flush=True)

    # env-0 diagnostic graphs: palm pose error + per-fingertip contact-point distance
    if hist_palm_pos_err:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(graph_dir, exist_ok=True)
        steps = np.arange(len(hist_palm_pos_err))
        grasp_idx = int(term.goal_grasp_idx[0].item()) if term is not None else -1
        n_ep = f"{len(episode_boundaries) + 1} episode(s)" if episode_boundaries else "1 episode"

        def _mark_episodes(ax):
            for b in episode_boundaries:
                ax.axvline(b, color="0.4", linestyle="--", linewidth=0.8, alpha=0.7)

        fig, ax1 = plt.subplots(figsize=(11, 5))
        ax1.plot(steps, np.array(hist_palm_pos_err) * 100, color="tab:blue", linewidth=1.2)
        ax1.set_xlabel("control step")
        ax1.set_ylabel("palm position error (cm)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(alpha=0.25)
        _mark_episodes(ax1)
        ax2 = ax1.twinx()
        ax2.plot(steps, hist_palm_orient_err, color="tab:orange", linewidth=1.2)
        ax2.set_ylabel("palm orientation error (deg)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        fig.suptitle(f"Palm (R_hand_base_link) pose error vs. goal grasp #{grasp_idx} -- "
                    f"env 0, {n_ep} (dashed = episode reset)")
        fig.tight_layout()
        p1 = os.path.join(graph_dir, "palm_pose_error.png")
        fig.savefig(p1, dpi=120)
        plt.close(fig)

        fig2, ax = plt.subplots(figsize=(11, 5))
        finger_arr = np.stack(hist_finger_dist, axis=0)  # (T, 5) cm
        for i, name in enumerate(_FINGERTIP_ORDER):
            ax.plot(steps, finger_arr[:, i], label=name, linewidth=1.2)
        ax.set_xlabel("control step")
        ax.set_ylabel("fingertip -> target contact point distance (cm)")
        ax.legend()
        ax.grid(alpha=0.25)
        _mark_episodes(ax)
        ax.set_title(f"Per-fingertip distance to its grasp_reach target contact point "
                    f"(grasp #{grasp_idx}) -- env 0, {n_ep} (dashed = episode reset)")
        fig2.tight_layout()
        p2 = os.path.join(graph_dir, "fingertip_contact_distance.png")
        fig2.savefig(p2, dpi=120)
        plt.close(fig2)

        # REAL geometric penetration check: min gap between the robot's actual live
        # collision spheres and the cube's box surface. <0 = genuine interpenetration
        # of the collision geometry, not a camera-occlusion illusion.
        fig3, ax3 = plt.subplots(figsize=(11, 5))
        gap_arr = np.array(hist_min_gap)
        ax3.plot(steps, gap_arr, color="tab:red", linewidth=1.2)
        ax3.axhline(0.0, color="black", linewidth=1.0)
        ax3.fill_between(steps, gap_arr, 0.0, where=(gap_arr < 0), color="red", alpha=0.25,
                         label="interpenetrating")
        ax3.set_xlabel("control step")
        ax3.set_ylabel("min hand-sphere <-> cube surface gap (cm)")
        ax3.grid(alpha=0.25)
        _mark_episodes(ax3)
        n_pen = int((gap_arr < 0).sum())
        ax3.set_title(f"Min collision-sphere-to-cube surface gap -- env 0, {n_ep} "
                    f"({n_pen}/{len(gap_arr)} steps show real interpenetration) "
                    f"(dashed = episode reset)")
        if n_pen:
            ax3.legend()
        fig3.tight_layout()
        p3 = os.path.join(graph_dir, "hand_cube_penetration.png")
        fig3.savefig(p3, dpi=120)
        plt.close(fig3)

        print(f"[goal] graph written -> {p1}", flush=True)
        print(f"[goal] graph written -> {p2}", flush=True)
        print(f"[goal] graph written -> {p3}", flush=True)
        print(f"[goal] penetration summary: {n_pen}/{len(gap_arr)} steps show min gap < 0 "
              f"(worst = {gap_arr.min():+.2f} cm)", flush=True)

    # ... then hard-exit. simulation_app.close() hangs indefinitely in headless
    # mode AFTER the run has finished, which looks like the script freezing.
    os._exit(0)


if __name__ == "__main__":
    main()
    simulation_app.close()
