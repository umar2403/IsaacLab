# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-grasp reward terms driven by BODex/UltraDexGrasp-synthesized Inspire Hand grasps.

At every reset, :class:`sample_grasp_goal` transforms the validated grasp library
(object frame) to the current cube pose and selects, per environment, the grasp
whose palm position is closest to the robot's current palm. Reward terms then pull
the palm pose and the six proximal hand joints toward that goal grasp.
"""

from __future__ import annotations

import json
import os
import numpy as np
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_mul, quat_from_matrix, quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg

_DEFAULT_GRASP_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "grasp_sampler", "grasp_dataset", "cube_5cm_grasps_valid.npz",
)

# EDIT ME: rotate every goal grasp about the cube's vertical (robot-frame z) axis.
# Rotates position AND orientation together about the cube center, so the grasp
# stays valid (cube is symmetric under 90 deg z-rotation). -90 = clockwise seen
# from above (robot looking down). Set 0.0 to disable.
_GOAL_YAW_DEG = 0.0

# BODex joint order == policy hand-action order (thumb yaw/pitch, index, middle, ring, pinky)
_HAND_JOINT_NAMES = [
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_ring_proximal_joint",
    "R_pinky_proximal_joint",
]


class sample_grasp_goal(ManagerTermBase):
    """Reset event: assigns each env a goal grasp (palm pose in world + 6 hand joints).

    Stores goals on the class instance; the reward terms below look this term up
    through the event manager.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        grasp_file = cfg.params.get("grasp_file", _DEFAULT_GRASP_FILE)
        data = np.load(grasp_file, allow_pickle=True)
        grasps = data["grasp_pose"]  # (N, 1, 3, 13) stages: pregrasp/grasp/squeeze
        if grasps.shape[0] == 0:
            raise RuntimeError(f"No grasps in {grasp_file}")
        # use the 'grasp' stage (index 1) as the goal
        g = grasps[:, 0, 1, :]  # (N, 13)
        # convert hand root pose from the synthesis URDF base frame to the USD
        # R_hand_base_link frame using the calibration saved by the validator
        if "T_usdbase_urdfbase" in data:
            T = data["T_usdbase_urdfbase"]  # p_usd = R p_urdf + t
            R_c = torch.tensor(T[:3, :3], dtype=torch.float32)
            # T_obj<-usdbase = T_obj<-urdfbase @ inv(T_usdbase<-urdfbase)
            from scipy.spatial.transform import Rotation as Rot
            R_g = Rot.from_quat(g[:, 3:7], scalar_first=True).as_matrix()  # obj<-urdfbase
            R_inv = T[:3, :3].T
            t_inv = -R_inv @ T[:3, 3]
            R_new = R_g @ R_inv
            p_new = g[:, :3] + np.einsum("nij,j->ni", R_g, t_inv)
            quat_new = Rot.from_matrix(R_new).as_quat(scalar_first=True)
            g = np.concatenate([p_new, quat_new, g[:, 7:]], axis=1)

        device = env.device
        self.grasp_pos_obj = torch.tensor(g[:, :3], dtype=torch.float32, device=device)
        self.grasp_quat_obj = torch.tensor(g[:, 3:7], dtype=torch.float32, device=device)
        self.grasp_q = torch.tensor(g[:, 7:], dtype=torch.float32, device=device)

        # rotate only the goal ORIENTATION about the cube's vertical z (see _GOAL_YAW_DEG);
        # position is left unchanged
        if _GOAL_YAW_DEG != 0.0:
            a = torch.deg2rad(torch.tensor(_GOAL_YAW_DEG, device=device))
            qz = torch.tensor([torch.cos(a / 2), 0.0, 0.0, torch.sin(a / 2)], device=device)  # wxyz, about +z
            qz_b = qz.unsqueeze(0).expand(self.grasp_quat_obj.shape[0], 4)
            self.grasp_quat_obj = quat_mul(qz_b, self.grasp_quat_obj)
        # pregrasp/squeeze finger stages (used by the demonstration oracle)
        self.grasp_q_pre = torch.tensor(grasps[:, 0, 0, 7:], dtype=torch.float32, device=device)
        self.grasp_q_squeeze = torch.tensor(grasps[:, 0, 2, 7:], dtype=torch.float32, device=device)
        self.num_grasps = g.shape[0]

        # per-env goal buffers
        n = env.num_envs
        self.goal_pos_w = torch.zeros(n, 3, device=device)
        self.goal_quat_w = torch.zeros(n, 4, device=device)
        self.goal_hand_q = torch.zeros(n, 6, device=device)
        self.goal_hand_q_pre = torch.zeros(n, 6, device=device)
        self.goal_hand_q_squeeze = torch.zeros(n, 6, device=device)
        self.goal_grasp_idx = torch.zeros(n, dtype=torch.long, device=device)
        self._object_name = "target_object"
        self._last_live_update = -1

        robot: Articulation = env.scene["robot"]
        self._palm_body_idx = robot.body_names.index("R_hand_base_link")
        self._hand_joint_ids = [robot.joint_names.index(nm) for nm in _HAND_JOINT_NAMES]

        # distractors present in the scene (for clutter-aware goal selection)
        self._distractor_names = [n for n in (f"distractor_{i}" for i in range(1, 11))
                                  if n in env.scene.keys()]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
        grasp_file: str = "",
        fixed_grasp_idx: int = -1,
        random_selection: bool = False,
    ):
        obj: RigidObject = env.scene[object_cfg.name]
        robot: Articulation = env.scene["robot"]

        obj_pos = obj.data.root_pos_w[env_ids]      # (E,3)
        obj_quat = obj.data.root_quat_w[env_ids]    # (E,4) wxyz

        # single-grasp mode: the same library grasp for every env and episode.
        # This makes the goal a deterministic function of the (observed) cube
        # pose, eliminating the hidden-goal noise of per-episode selection.
        # random_selection: draw a uniform random grasp per episode instead —
        # viable with the max(pose, grasp-gate) palm reward, where the grasp
        # endpoint pays fully regardless of which goal was drawn.
        if fixed_grasp_idx >= 0 or random_selection:
            E = len(env_ids)
            if random_selection:
                best = torch.randint(self.num_grasps, (E,), device=env.device)
            else:
                best = torch.full((E,), fixed_grasp_idx, dtype=torch.long, device=env.device)
            gq = self.grasp_quat_obj[best]
            self.goal_grasp_idx[env_ids] = best
            self._object_name = object_cfg.name
            self.goal_pos_w[env_ids] = quat_apply(obj_quat, self.grasp_pos_obj[best]) + obj_pos
            self.goal_quat_w[env_ids] = quat_mul(obj_quat, gq)
            self.goal_hand_q[env_ids] = self.grasp_q[best]
            self.goal_hand_q_pre[env_ids] = self.grasp_q_pre[best]
            self.goal_hand_q_squeeze[env_ids] = self.grasp_q_squeeze[best]
            return

        palm_pos = robot.data.body_pos_w[env_ids, self._palm_body_idx]  # (E,3)

        E, G = len(env_ids), self.num_grasps
        # grasp palm positions in world: obj pose * grasp pos
        oq = obj_quat.unsqueeze(1).expand(E, G, 4).reshape(-1, 4)
        gp = self.grasp_pos_obj.unsqueeze(0).expand(E, G, 3).reshape(-1, 3)
        cand_pos_w = (quat_apply(oq, gp) + obj_pos.unsqueeze(1).expand(E, G, 3).reshape(-1, 3)).view(E, G, 3)

        # selection score 1: reach cost (distance from current palm)
        d = torch.norm(cand_pos_w - palm_pos.unsqueeze(1), dim=2)  # (E,G)

        # selection score 2: clutter risk — count distractors inside each
        # candidate's approach corridor (segment from 12 cm behind the palm
        # goal, along the palm normal, to the object center)
        score = d.clone()
        if self._distractor_names:
            gq_all = self.grasp_quat_obj.unsqueeze(0).expand(E, G, 4).reshape(-1, 4)
            cand_quat_w = quat_mul(oq, gq_all)
            palm_normal = quat_apply(cand_quat_w, torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(E * G, 3))
            p1 = cand_pos_w.view(-1, 3)                     # palm goal
            p0 = p1 - 0.12 * palm_normal                    # pregrasp end of corridor
            p2 = obj_pos.unsqueeze(1).expand(E, G, 3).reshape(-1, 3)  # object center
            dist_pos = torch.stack(
                [env.scene[n].data.root_pos_w[env_ids] for n in self._distractor_names], dim=1
            )  # (E,D,3)
            on_tray = dist_pos[:, :, 2] > 0.7               # ignore hidden distractors
            dist_pos = dist_pos.unsqueeze(1).expand(E, G, -1, 3).reshape(E * G, -1, 3)
            n_block = torch.zeros(E * G, device=env.device)
            for a, b in ((p0, p1), (p1, p2)):               # two corridor segments
                ab = (b - a).unsqueeze(1)                   # (EG,1,3)
                t = ((dist_pos - a.unsqueeze(1)) * ab).sum(-1) / (ab.pow(2).sum(-1) + 1e-9)
                closest = a.unsqueeze(1) + t.clamp(0, 1).unsqueeze(-1) * ab
                seg_d = torch.norm(dist_pos - closest, dim=-1)  # (EG,D)
                blocked = (seg_d < 0.07) & on_tray.unsqueeze(1).expand(E, G, -1).reshape(E * G, -1)
                n_block += blocked.float().sum(-1)
            # each blocking distractor costs as much as 0.5 m of extra reach
            score += 0.5 * n_block.view(E, G)

        best = score.argmin(dim=1)  # (E,)

        gq = self.grasp_quat_obj[best]  # (E,4)
        self.goal_grasp_idx[env_ids] = best
        self._object_name = object_cfg.name
        self.goal_pos_w[env_ids] = cand_pos_w[torch.arange(E, device=env.device), best]
        self.goal_quat_w[env_ids] = quat_mul(obj_quat, gq)
        self.goal_hand_q[env_ids] = self.grasp_q[best]
        self.goal_hand_q_pre[env_ids] = self.grasp_q_pre[best]
        self.goal_hand_q_squeeze[env_ids] = self.grasp_q_squeeze[best]

    def update_live_goals(self, env: ManagerBasedRLEnv):
        """Re-attach each env's chosen grasp to the object's CURRENT pose.

        The grasps are stored relative to the object, so they remain valid when
        the object is pushed around; without this, the goal would stay frozen at
        the object's spawn pose and the policy would chase empty space after any
        contact moves the object.
        """
        if self._last_live_update == env.common_step_counter:
            return  # already refreshed this step (both reward terms call this)
        self._last_live_update = env.common_step_counter
        obj: RigidObject = env.scene[self._object_name]
        gp = self.grasp_pos_obj[self.goal_grasp_idx]    # (N,3)
        gq = self.grasp_quat_obj[self.goal_grasp_idx]   # (N,4)
        self.goal_pos_w = quat_apply(obj.data.root_quat_w, gp) + obj.data.root_pos_w
        self.goal_quat_w = quat_mul(obj.data.root_quat_w, gq)


def _get_goal_term(env: ManagerBasedRLEnv) -> sample_grasp_goal:
    term = getattr(env, "_grasp_goal_term", None)
    if term is None:
        idx = env.event_manager.active_terms["reset"].index("sample_grasp_goal")
        term = env.event_manager._mode_term_cfgs["reset"][idx].func
        env._grasp_goal_term = term
    return term


def grasp_goal_palm_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pos_std: float = 0.15,
    orient_std: float = 0.6,
    orient_weight: float = 0.5,
    pos_mode: str = "full",
) -> torch.Tensor:
    """POSE guidance toward the UltraDexGrasp goal: palm position + wrist orientation.

    Deliberately does NOT reward grasping — that is owned entirely by
    compute_task_reward's grasp term. This term only pulls the palm toward the
    goal grasp's pose; the task reward then handles closing and lifting.

    pos_mode:
      "full"   - distance to the goal position in all 3 axes.
      "height" - only the VERTICAL (z) offset to the goal. Use this when the
                 task reward already centres the palm horizontally (posture +
                 reach) and the grasp library's unique positional contribution
                 is the correct grasp HEIGHT (~hand-length above the object),
                 which task_reward's posture target undershoots.
    """
    term = _get_goal_term(env)
    term.update_live_goals(env)  # goal follows the object if it gets pushed
    robot: Articulation = env.scene[robot_cfg.name]
    palm_pos = robot.data.body_pos_w[:, term._palm_body_idx]
    palm_quat = robot.data.body_quat_w[:, term._palm_body_idx]
    if pos_mode == "height":
        d_pos = torch.abs(palm_pos[:, 2] - term.goal_pos_w[:, 2])
    else:
        d_pos = torch.norm(palm_pos - term.goal_pos_w, dim=1)
    pos_rew = 1.0 - torch.tanh(d_pos / pos_std)
    ang = quat_error_magnitude(palm_quat, term.goal_quat_w)  # radians, [0, pi]
    orient_rew = 1.0 - torch.tanh(ang / orient_std)
    return (1.0 - orient_weight) * pos_rew + orient_weight * orient_rew


def grasp_goal_hand_config_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    q_std: float = 0.5,
    gate_dist: float = 0.10,
) -> torch.Tensor:
    """Reward for matching the goal hand configuration, gated on palm proximity.

    The gate prevents the policy from curling fingers to the goal pose while the
    hand is still far from the object.
    """
    term = _get_goal_term(env)
    term.update_live_goals(env)  # goal follows the object if it gets pushed
    robot: Articulation = env.scene[robot_cfg.name]
    palm_pos = robot.data.body_pos_w[:, term._palm_body_idx]
    palm_d = torch.norm(palm_pos - term.goal_pos_w, dim=1)
    gate = (1.0 - torch.tanh(palm_d / gate_dist)).clamp(min=0.0)

    q = robot.data.joint_pos[:, term._hand_joint_ids]
    q_err = torch.norm(q - term.goal_hand_q, dim=1)
    return gate * (1.0 - torch.tanh(q_err / q_std))


# Cached fingertip target contact points for the active grasp (see MDP_REPORT.md §5.2.4).
# Computed offline by scratchpad/compute_fingertip_contacts.py from the grasp_selection
# optimizer's own cube_contact() projection -- reused here purely as geometry, not for
# re-selecting a grasp. Cached (not recomputed from the gitignored ultradex_repo/URDF at
# runtime) so training doesn't depend on those assets being present, same idiom as
# grasp_selection/scores.json.
_FINGERTIP_CONTACTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "grasp_selection", "fingertip_contacts.json",
)
# order must match the tracked fingertip bodies [R_thumb_distal, R_index_intermediate,
# R_middle_intermediate, R_ring_intermediate, R_pinky_intermediate] i.e. _RIGHT_HAND_BODIES[1:]
_FINGERTIP_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


def grasp_reach_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    std: float = 0.05,
    contacts_file: str = _FINGERTIP_CONTACTS_FILE,
) -> torch.Tensor:
    """Pulls each of the 5 tracked fingertips toward its OWN target contact point on
    the cube surface for the active grasp, instead of one shared centroid target
    (c.f. compute_task_reward's reach_rew). Dense, ungated. Uses mean-of-tanh (not
    tanh-of-mean) so one badly-placed finger isn't washed out by four good ones --
    see MDP_REPORT.md §5.2.4 for the full derivation and design rationale.

    robot_cfg.body_ids must resolve to the 5 fingertip bodies in the order
    [thumb, index, middle, ring, pinky] (_RIGHT_HAND_BODIES[1:]), matching the
    cached contact points' fingertip_order.
    """
    if not hasattr(env, "_grasp_reach_contacts_obj"):
        with open(contacts_file) as f:
            data = json.load(f)
        assert data["fingertip_order"] == _FINGERTIP_ORDER, (
            f"cached fingertip order {data['fingertip_order']} != expected {_FINGERTIP_ORDER}"
        )
        env._grasp_reach_contacts_obj = torch.tensor(
            data["contact_pos_cube_frame"], dtype=torch.float32, device=env.device
        )  # (5,3), constant, in the cube's own object frame

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    tips_w = robot.data.body_pos_w[:, robot_cfg.body_ids]  # (N,5,3)
    cube_quat = obj.data.root_quat_w                        # (N,4)
    cube_pos = obj.data.root_pos_w                          # (N,3)

    n = tips_w.shape[0]
    contacts_obj = env._grasp_reach_contacts_obj.unsqueeze(0).expand(n, -1, -1).reshape(-1, 3)
    cube_quat_exp = cube_quat.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4)
    cube_pos_exp = cube_pos.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 3)
    targets_w = (quat_apply(cube_quat_exp, contacts_obj) + cube_pos_exp).view(n, 5, 3)

    d = torch.norm(tips_w - targets_w, dim=-1)  # (N,5)
    return (1.0 - torch.tanh(d / std)).mean(dim=-1)
