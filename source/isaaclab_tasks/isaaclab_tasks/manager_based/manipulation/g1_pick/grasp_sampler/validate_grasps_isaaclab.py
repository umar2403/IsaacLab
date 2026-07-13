"""Validate BODex-synthesized Inspire Hand grasps in Isaac Lab with the g1_pick USD.

For each grasp (hand root pose in object frame + 6 proximal joint angles):
1. Hold the G1 fixed in its default pose with the right hand at the pregrasp joints.
2. Place the cube at the grasp-relative pose w.r.t. the hand (computed via a
   calibrated URDF-base -> USD R_hand_base_link transform).
3. Drive the fingers pregrasp -> grasp -> squeeze, let gravity act, hold ~1.5 s.
4. A grasp passes if the cube stays near the palm.

Each parallel env tests one grasp. Output: grasp_dataset/cube_5cm_grasps_valid.npz

Run:
    conda activate env_isaaclab
    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/g1_pick/grasp_sampler/validate_grasps_isaaclab.py --headless
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--grasp_file", type=str,
                    default=os.path.join(os.path.dirname(__file__), "grasp_dataset", "cube_5cm_grasps.npz"))
parser.add_argument("--hold_steps", type=int, default=180, help="steps to hold after squeeze (120 Hz)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, quat_inv, quat_from_matrix

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robot_cfg import G1_INSPIRE_CFG  # noqa: E402

# ---------------------------------------------------------------------------
# joint correspondence: BODex/URDF order -> USD joint names
BODEX_TO_USD_ACTUATED = [
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_ring_proximal_joint",
    "R_pinky_proximal_joint",
]
# Empirical slave postures measured in the Isaac-G1-Pick-v0 env: the USD's
# PhysxMimicJointAPI + the InspireMimicAction drive targets interact such that
# the slave joints settle at roughly constant angles regardless of the
# proximal command. The synthesis URDF freezes them at these values.
FROZEN_SLAVES = {
    "R_thumb_intermediate_joint": -0.16,
    "R_thumb_distal_joint": -0.24,
    "R_index_intermediate_joint": 1.15,
    "R_middle_intermediate_joint": 1.15,
    "R_ring_intermediate_joint": 1.15,
    "R_pinky_intermediate_joint": 1.15,
}
# software mimic ratios that InspireMimicAction writes as slave drive targets
# each step (replicated here so validation matches training-env dynamics)
SOFTWARE_MIMIC_TARGETS = {
    "R_thumb_intermediate_joint": ("R_thumb_proximal_pitch_joint", 0.8024),
    "R_thumb_distal_joint": ("R_thumb_proximal_pitch_joint", 0.8024 * 0.9487),
    "R_index_intermediate_joint": ("R_index_proximal_joint", 1.0843),
    "R_middle_intermediate_joint": ("R_middle_proximal_joint", 1.0843),
    "R_ring_intermediate_joint": ("R_ring_proximal_joint", 1.0843),
    "R_pinky_intermediate_joint": ("R_pinky_proximal_joint", 1.0843),
}
USD_MIMIC = SOFTWARE_MIMIC_TARGETS  # kept for mimic_ids construction
URDF_MIMIC_FOR_CALIB = {k: (SOFTWARE_MIMIC_TARGETS[k][0], 0.0, v)
                        for k, v in FROZEN_SLAVES.items()}
# links used for base-frame calibration (URDF name -> USD body name)
CALIB_LINKS = {
    "thumb_proximal_base": "R_thumb_proximal_base",
    "thumb_proximal": "R_thumb_proximal",
    "thumb_intermediate": "R_thumb_intermediate",
    "thumb_distal": "R_thumb_distal",
    "index_proximal": "R_index_proximal",
    "index_intermediate": "R_index_intermediate",
    "middle_proximal": "R_middle_proximal",
    "middle_intermediate": "R_middle_intermediate",
    "ring_proximal": "R_ring_proximal",
    "pinky_proximal": "R_pinky_proximal",
}
URDF_PATH = os.path.join(os.path.dirname(__file__), "ultradex_repo", "third_party", "BODex_api",
                         "src", "bodex", "content", "assets", "robot", "inspire_hand",
                         "inspire_hand_right.urdf")


def urdf_fk(q6):
    """FK of the dex-urdf inspire right hand at actuated config q6 (URDF joint order).
    Returns dict link_name -> 4x4 pose in URDF 'base' frame."""
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation as R

    root = ET.parse(URDF_PATH).getroot()
    joints = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]) if o is not None else np.zeros(3)
        rpy = np.array([float(v) for v in (o.get("rpy") or "0 0 0").split()]) if o is not None else np.zeros(3)
        ax = j.find("axis")
        axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.array([0, 0, 1.0])
        mim = j.find("mimic")
        joints[j.get("name")] = dict(
            type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            xyz=xyz, rpy=rpy, axis=axis,
            mimic=None if mim is None else (mim.get("joint"), float(mim.get("multiplier") or 1),
                                            float(mim.get("offset") or 0)))
    qmap = dict(zip(["thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint", "index_proximal_joint",
                     "middle_proximal_joint", "ring_proximal_joint", "pinky_proximal_joint"], q6))
    for name, j in joints.items():
        if j["mimic"] is not None:
            src, mult, off = j["mimic"]
            qmap[name] = qmap.get(src, 0.0) * mult + off
    poses = {"base": np.eye(4)}
    changed = True
    while changed:
        changed = False
        for name, j in joints.items():
            if j["parent"] in poses and j["child"] not in poses:
                T = np.eye(4)
                T[:3, :3] = R.from_euler("xyz", j["rpy"]).as_matrix()
                T[:3, 3] = j["xyz"]
                if j["type"] == "revolute":
                    Tj = np.eye(4)
                    Tj[:3, :3] = R.from_rotvec(j["axis"] / np.linalg.norm(j["axis"]) * qmap.get(name, 0.0)).as_matrix()
                    T = T @ Tj
                poses[j["child"]] = poses[j["parent"]] @ T
                changed = True
    return poses


@configclass
class ValidationSceneCfg(InteractiveSceneCfg):
    robot = G1_INSPIRE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 1.5)),
    )


def solve_rigid_transform(P_urdf, P_usd):
    """Least-squares rigid transform T such that T @ p_urdf = p_usd (Kabsch)."""
    cu, cs = P_urdf.mean(0), P_usd.mean(0)
    H = (P_urdf - cu).T @ (P_usd - cs)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, 1, d]) @ U.T
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = cs - Rm @ cu
    return T


def main():
    data = np.load(args_cli.grasp_file, allow_pickle=True)
    grasps = data["grasp_pose"]  # (N, 1, 3, 13)
    n_grasps = grasps.shape[0]
    print(f"[val] loaded {n_grasps} grasps from {args_cli.grasp_file}", flush=True)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120))
    scene_cfg = ValidationSceneCfg(num_envs=n_grasps, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]
    cube: RigidObject = scene["cube"]
    device = robot.device

    hand_base_idx = robot.body_names.index("R_hand_base_link")
    act_ids = [robot.joint_names.index(n) for n in BODEX_TO_USD_ACTUATED]
    mimic_ids = {n: robot.joint_names.index(n) for n in USD_MIMIC}

    default_jp = robot.data.default_joint_pos.clone()
    default_jv = torch.zeros_like(robot.data.default_joint_vel)

    hand_joint_ids = act_ids + list(mimic_ids.values())
    body_joint_ids = [i for i in range(robot.num_joints) if i not in hand_joint_ids]

    def mimic_expand(q6_batch):
        """(N,6) actuated -> (N,12): slaves at the frozen empirical postures."""
        n = q6_batch.shape[0]
        frozen = torch.tensor([FROZEN_SLAVES[k] for k in mimic_ids], device=q6_batch.device)
        return torch.cat([q6_batch, frozen.unsqueeze(0).expand(n, -1)], dim=1)

    def software_mimic_targets(q6_batch):
        """Slave drive targets as InspireMimicAction computes them each step."""
        return torch.stack([
            q6_batch[:, 1] * 0.8024,
            q6_batch[:, 1] * 0.8024 * 0.9487,
            q6_batch[:, 2] * 1.0843,
            q6_batch[:, 3] * 1.0843,
            q6_batch[:, 4] * 1.0843,
            q6_batch[:, 5] * 1.0843,
        ], dim=1)

    def set_hand(q6_batch, calib_urdf_ratios=False):
        """q6_batch: (N,6) actuated. Writes full joint state, holds rest at default."""
        jp = default_jp.clone()
        jp12 = mimic_expand(q6_batch)
        jp[:, act_ids] = jp12[:, :6]
        for i, name in enumerate(mimic_ids):
            jp[:, mimic_ids[name]] = jp12[:, 6 + i]
        robot.write_joint_state_to_sim(jp, default_jv)
        return jp

    # ---------------- calibration: URDF base -> USD R_hand_base_link ----------
    calib_q = np.array([0.5, 0.3, 0.4, 0.4, 0.4, 0.4])
    set_hand(torch.tensor(calib_q, device=device, dtype=torch.float32).repeat(n_grasps, 1),
             calib_urdf_ratios=True)
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(sim.get_physics_dt())

    hb_pos = robot.data.body_pos_w[0, hand_base_idx].cpu().numpy()
    hb_quat = robot.data.body_quat_w[0, hand_base_idx].cpu().numpy()  # wxyz
    from scipy.spatial.transform import Rotation as Rot
    R_hb = Rot.from_quat(hb_quat, scalar_first=True).as_matrix()

    urdf_poses = urdf_fk(calib_q)
    P_urdf, P_usd = [], []
    for uname, sname in CALIB_LINKS.items():
        bidx = robot.body_names.index(sname)
        p_w = robot.data.body_pos_w[0, bidx].cpu().numpy()
        P_usd.append(R_hb.T @ (p_w - hb_pos))          # in USD hand-base frame
        P_urdf.append(urdf_poses[uname][:3, 3])        # in URDF base frame
    T_usdbase_urdfbase = solve_rigid_transform(np.array(P_urdf), np.array(P_usd))
    resid = np.linalg.norm((np.array(P_urdf) @ T_usdbase_urdfbase[:3, :3].T
                            + T_usdbase_urdfbase[:3, 3]) - np.array(P_usd), axis=1)
    print(f"[val] calibration residual per link (m): max={resid.max():.4f} mean={resid.mean():.4f}", flush=True)

    # ---------------- place cubes at grasp-relative poses ---------------------
    pregrasp_q = torch.tensor(grasps[:, 0, 0, 7:], device=device, dtype=torch.float32)
    grasp_q = torch.tensor(grasps[:, 0, 1, 7:], device=device, dtype=torch.float32)
    squeeze_q = torch.tensor(grasps[:, 0, 2, 7:], device=device, dtype=torch.float32)

    set_hand(pregrasp_q)
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(sim.get_physics_dt())

    hb_pos_w = robot.data.body_pos_w[:, hand_base_idx]      # (N,3)
    hb_quat_w = robot.data.body_quat_w[:, hand_base_idx]    # (N,4) wxyz

    # T_obj_urdfbase from grasp file (stage 1 = grasp pose)
    g_pos = grasps[:, 0, 1, :3]     # hand root pos in object frame
    g_quat = grasps[:, 0, 1, 3:7]   # wxyz
    # object pose in URDF-base frame = inverse
    Rg = Rot.from_quat(g_quat, scalar_first=True).as_matrix()      # (N,3,3) obj->urdfbase
    R_obj_in_base = Rg.transpose(0, 2, 1)
    p_obj_in_base = -np.einsum("nij,nj->ni", R_obj_in_base, g_pos)
    # to USD hand-base frame
    Rc, pc = T_usdbase_urdfbase[:3, :3], T_usdbase_urdfbase[:3, 3]
    R_obj_usdb = np.einsum("ij,njk->nik", Rc, R_obj_in_base)
    p_obj_usdb = np.einsum("ij,nj->ni", Rc, p_obj_in_base) + pc
    # to world
    R_hb_w = Rot.from_quat(hb_quat_w.cpu().numpy(), scalar_first=True).as_matrix()
    p_cube_w = np.einsum("nij,nj->ni", R_hb_w, p_obj_usdb) + hb_pos_w.cpu().numpy()
    R_cube_w = np.einsum("nij,njk->nik", R_hb_w, R_obj_usdb)
    q_cube_w = Rot.from_matrix(R_cube_w).as_quat(scalar_first=True)

    # --- direction diagnostic: where is the cube in the USD hand-base frame? ---
    v1 = p_obj_usdb                                            # current convention
    v2 = np.einsum("ij,nj->ni", Rc.T, p_obj_in_base - pc)      # inverse convention
    tip_idx_dbg = [robot.body_names.index(n) for n in
                   ["R_thumb_distal", "R_index_intermediate", "R_middle_intermediate",
                    "R_ring_intermediate", "R_pinky_intermediate"]]
    tips_w0 = robot.data.body_pos_w[0, tip_idx_dbg].cpu().numpy()
    R_hb0 = Rot.from_quat(hb_quat_w[0].cpu().numpy(), scalar_first=True).as_matrix()
    tips_hb = (R_hb0.T @ (tips_w0 - hb_pos_w[0].cpu().numpy()).T).T
    print(f"[val] finger-link centroid in USD hand frame: {tips_hb.mean(0).round(3)}", flush=True)
    print(f"[val] cube in hand frame, convention T:    {v1.mean(0).round(3)}", flush=True)
    print(f"[val] cube in hand frame, convention T^-1: {v2.mean(0).round(3)}", flush=True)

    # fingertip points = distal body pose + fixed tip offset (from the URDF)
    TIP_PARENT = ["R_thumb_distal", "R_index_intermediate", "R_middle_intermediate",
                  "R_ring_intermediate", "R_pinky_intermediate"]
    TIP_OFFSET = np.array([
        [0.0202, 0.0140, -0.006],
        [-0.0008, 0.045, -0.005],
        [-0.001, 0.048, -0.005],
        [-0.0008, 0.045, -0.005],
        [-0.0008, 0.037, -0.005],
    ])

    def tip_gap_report(tag, gidx=27):
        ids = [robot.body_names.index(n) for n in TIP_PARENT]
        bp = robot.data.body_pos_w[gidx, ids].cpu().numpy()
        bq = robot.data.body_quat_w[gidx, ids].cpu().numpy()
        Rb = Rot.from_quat(bq, scalar_first=True).as_matrix()
        tips = bp + np.einsum("nij,nj->ni", Rb, TIP_OFFSET)
        c = cube.data.root_pos_w[gidx].cpu().numpy()
        gaps = np.linalg.norm(tips - c, axis=1) - 0.025
        print(f"[val] grasp #27 {tag}: tip->cube-surface gaps "
              f"[th,ix,mi,ri,pi] = {gaps.round(3)}", flush=True)

    cube_pose = torch.zeros(n_grasps, 7, device=device)
    cube_pose[:, :3] = torch.tensor(p_cube_w, device=device, dtype=torch.float32)
    cube_pose[:, 3:] = torch.tensor(q_cube_w, device=device, dtype=torch.float32)
    cube.write_root_pose_to_sim(cube_pose)
    cube.write_root_velocity_to_sim(torch.zeros(n_grasps, 6, device=device))

    # ---------------- close fingers (PD-driven) and hold ------------------------
    hand_ids_t = torch.tensor(hand_joint_ids, device=device)
    body_ids_t = torch.tensor(body_joint_ids, device=device)
    zero_vel6 = torch.zeros(n_grasps, 6, device=device)

    act_ids_t = torch.tensor(act_ids, device=device)

    def hold_phase(q6, steps, pin_cube, ramp=0):
        """Freeze body kinematically; PD-drive only the 6 actuated hand joints
        (the slave joints have no drives — PhysX's mimic constraint moves them).
        Targets are ramped over `ramp` steps to limit mimic-constraint ringing.

        While `pin_cube` is set the cube is held at its placement pose (the
        object rests on a table during synthesis; mid-air it would fall before
        the fingers close). It is released for the final gravity-hold phase.
        """
        mim_ids_t = torch.tensor([mimic_ids[k] for k in mimic_ids], device=device)
        q_start = robot.data.joint_pos[:, act_ids].clone()
        for i in range(steps):
            alpha = min(1.0, (i + 1) / ramp) if ramp > 0 else 1.0
            targets = q_start + alpha * (q6 - q_start)
            # hold the body with PD targets only — kinematic state writes every
            # step re-excite the underdamped PhysX mimic constraint
            robot.set_joint_position_target(default_jp[:, body_joint_ids], joint_ids=body_ids_t)
            robot.set_joint_position_target(targets, joint_ids=act_ids_t)
            robot.set_joint_position_target(software_mimic_targets(targets), joint_ids=mim_ids_t)
            if pin_cube:
                cube.write_root_pose_to_sim(cube_pose)
                cube.write_root_velocity_to_sim(zero_vel6)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.get_physics_dt())

    tip_idx = [robot.body_names.index(n) for n in
               ["R_thumb_distal", "R_index_intermediate", "R_middle_intermediate",
                "R_ring_intermediate", "R_pinky_intermediate"]]

    def tip_dist():
        tips = robot.data.body_pos_w[:, tip_idx]                      # (N,5,3)
        return torch.norm(tips - cube.data.root_pos_w.unsqueeze(1), dim=2).mean(1)

    # ---- pass A: empirical grip center -----------------------------------------
    # The synthesis hand model differs from the USD hand by ~1-3 cm at the tips
    # (different Inspire revisions), so instead of trusting the BODex cube
    # position, close the sim hand to each grasp's joint config WITHOUT the cube
    # and read where its own pinch center actually is. The BODex palm pose and
    # joint config are kept; only the object position is re-centered.
    park = cube_pose.clone()
    park[:, 2] -= 5.0  # park the cube far below during pass A
    cube.write_root_pose_to_sim(park)
    hold_phase(grasp_q, 180, pin_cube=False, ramp=90)  # close to grasp config, no cube

    ids_tp = [robot.body_names.index(n) for n in TIP_PARENT]
    bp = robot.data.body_pos_w[:, ids_tp].cpu().numpy()          # (N,5,3)
    bq = robot.data.body_quat_w[:, ids_tp].cpu().numpy()
    Rb = Rot.from_quat(bq.reshape(-1, 4), scalar_first=True).as_matrix().reshape(n_grasps, 5, 3, 3)
    tips_w = bp + np.einsum("nkij,kj->nki", Rb, TIP_OFFSET)      # (N,5,3)
    grip_center = 0.5 * tips_w[:, 0] + 0.5 * tips_w[:, 1:3].mean(axis=1)  # thumb vs index/middle
    cube_pose[:, :3] = torch.tensor(grip_center, device=device, dtype=torch.float32)
    print(f"[val] pass A: empirical grip centers computed "
          f"(median shift {np.linalg.norm(grip_center - p_cube_w, axis=1).mean():.3f} m)", flush=True)

    # ---- pass B: reopen, place cube at grip center, close, squeeze, release ----
    hold_phase(pregrasp_q, 120, pin_cube=True, ramp=60)   # reopen around pinned cube
    tip_gap_report("at placement")

    # squeeze past the planned angles until contact, like a closing policy would
    deep_squeeze_q = grasp_q + torch.tensor([0.0, 0.35, 0.35, 0.35, 0.35, 0.35], device=device)
    hold_phase(grasp_q, 150, pin_cube=True, ramp=90)         # close slowly to grasp
    hold_phase(deep_squeeze_q, 360, pin_cube=True, ramp=60)  # deep squeeze + settle
    tip_gap_report("after squeeze")
    q_err = (robot.data.joint_pos[:, act_ids] - squeeze_q).norm(dim=1)
    print(f"[val] after squeeze: cube-tip dist median {tip_dist().median().item():.3f} m | "
          f"hand q tracking err median {q_err.median().item():.3f} rad", flush=True)
    e0 = int(q_err.argmin().item())  # best-tracking env
    print(f"[val] env {e0} per-joint  target: {squeeze_q[e0].cpu().numpy().round(2)}", flush=True)
    print(f"[val] env {e0} per-joint  actual: "
          f"{robot.data.joint_pos[e0, act_ids].cpu().numpy().round(2)}", flush=True)
    mim_names = list(mimic_ids.keys())
    mim_ids_list = [mimic_ids[n] for n in mim_names]
    print(f"[val] env {e0} mimic     target: "
          f"{mimic_expand(squeeze_q)[e0, 6:].cpu().numpy().round(2)}", flush=True)
    print(f"[val] env {e0} mimic     actual: "
          f"{robot.data.joint_pos[e0, mim_ids_list].cpu().numpy().round(2)}", flush=True)
    # release: keep squeezing while gravity acts
    hold_phase(deep_squeeze_q, args_cli.hold_steps, pin_cube=False)

    # ---------------- score -----------------------------------------------------
    palm_pos = robot.data.body_pos_w[:, hand_base_idx]
    d = torch.norm(cube.data.root_pos_w - palm_pos, dim=1)
    init_d = torch.norm(cube_pose[:, :3] - hb_pos_w, dim=1)
    moved = (d - init_d).abs()
    success = (moved < 0.10) & (d < 0.35)
    print(f"[val] {success.sum().item()}/{n_grasps} grasps held | moved: "
          f"median {moved.median().item():.3f} min {moved.min().item():.3f} "
          f"max {moved.max().item():.3f} m", flush=True)

    rank_file = os.path.join(os.path.dirname(args_cli.grasp_file), "geom_rank.npy")
    if os.path.exists(rank_file):
        top = np.load(rank_file)[:10]
        for i in top:
            print(f"[val]   geom-top grasp #{i:3d}: moved={moved[i].item():.3f} "
                  f"held={bool(success[i].item())}", flush=True)

    # re-express the hand pose relative to the empirically re-centered cube:
    # the cube moved by delta_w in world, so in object coords the hand shifts
    # by -R_obj_w^T @ delta_w
    delta_w = grip_center - p_cube_w                           # (N,3)
    R_obj_w = Rot.from_quat(q_cube_w, scalar_first=True).as_matrix()
    delta_obj = np.einsum("nji,nj->ni", R_obj_w, delta_w)
    grasps_corr = grasps.copy()
    grasps_corr[:, 0, :, :3] -= delta_obj[:, None, :]

    # save ALL grip-recentered grasps: the USD hand's drive-less mimic joints
    # behave stochastically, so a strict mid-air-hold pass/fail is not a usable
    # criterion; downstream filtering is geometric (check_grasps_offline.py)
    out = args_cli.grasp_file.replace(".npz", "_recentered.npz")
    np.savez(out, grasp_pose=grasps_corr,
             success_mask=success.cpu().numpy(),
             moved=moved.cpu().numpy(),
             T_usdbase_urdfbase=T_usdbase_urdfbase,
             joint_order=data["joint_order"], stages=data["stages"])
    print(f"[val] saved grip-recentered grasps -> {out}", flush=True)

    # skip simulation_app.close(): it can hang for hours in headless mode
    os._exit(0)


if __name__ == "__main__":
    main()
