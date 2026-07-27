# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hand kinematics + cube geometry used by the grasp-selection optimizer.

Implements Sections 1-3 and 6.1 of ``OPTIMIZER_IMPLEMENTATION_SPEC.md``:

* URDF forward kinematics of the Inspire hand (root-relative link poses),
* the cube signed-distance function and the point -> cube-face contact projection,
* loading of the ~40 BODex collision spheres from the robot config YAML.

CPU-only (numpy + scipy), no Isaac Lab / Isaac Sim imports, no GPU.

FK note: the spec's reference implementation uses ``pytorch_kinematics``. This
module ships an equivalent dependency-free numpy chain instead (the URDF only
contains ``revolute`` and ``fixed`` joints), mirroring the FK already used and
trusted by ``grasp_sampler/check_grasps_offline.py``. ``selftest.py`` cross-checks
the fingertip poses against that existing implementation.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

__all__ = [
    "CUBE_HALF_EDGE",
    "URDF_ROOT_LINK",
    "STAGE_PREGRASP",
    "STAGE_GRASP",
    "STAGE_SQUEEZE",
    "ACTUATED_JOINT_NAMES",
    "FINGERTIP_LINKS",
    "FINGERS",
    "LINK_FINGER",
    "DEFAULT_URDF",
    "DEFAULT_ROBOT_YAML",
    "DEFAULT_LIBRARY",
    "HandChain",
    "load_chain",
    "link_transforms",
    "palm_matrix",
    "cube_surface_dist",
    "cube_contact",
    "load_spheres",
    "load_library",
    "fk_tips",
]

# --------------------------------------------------------------------------------------
# Reference constants (spec Section 1.4)
# --------------------------------------------------------------------------------------

CUBE_HALF_EDGE = 0.025  # m; the cube is axis-aligned, centered at the object-frame origin
URDF_ROOT_LINK = "base"

STAGE_PREGRASP = 0
STAGE_GRASP = 1  # the stage the optimizer scores
STAGE_SQUEEZE = 2

# The 6 actuated (policy-controlled) joints, in the library's `joint_order`.
# Matched to the URDF BY ORDER (the URDF uses bare names, the policy uses R_-prefixed).
ACTUATED_JOINT_NAMES = [
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
]

FINGERTIP_LINKS = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

# Sphere-bearing links -> the finger they belong to. "hand_base_link" (the palm) also
# carries spheres but is deliberately absent: palm spheres contribute contacts, but the
# palm is not counted toward the MIN_FINGERS gate.
LINK_FINGER = {
    "thumb_proximal_base": "thumb",
    "thumb_proximal": "thumb",
    "thumb_intermediate": "thumb",
    "thumb_distal": "thumb",
    "index_proximal": "index",
    "index_intermediate": "index",
    "middle_proximal": "middle",
    "middle_intermediate": "middle",
    "ring_proximal": "ring",
    "ring_intermediate": "ring",
    "pinky_proximal": "pinky",
    "pinky_intermediate": "pinky",
}

# --------------------------------------------------------------------------------------
# Default asset locations in this repo
# --------------------------------------------------------------------------------------

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BODEX_CONTENT = os.path.join(
    _TASK_DIR, "grasp_sampler", "ultradex_repo", "third_party", "BODex_api", "src", "bodex", "content"
)

DEFAULT_URDF = os.path.join(_BODEX_CONTENT, "assets", "robot", "inspire_hand", "inspire_hand_right.urdf")
DEFAULT_ROBOT_YAML = os.path.join(_BODEX_CONTENT, "configs", "robot", "inspire_right.yml")
DEFAULT_LIBRARY = os.path.join(_TASK_DIR, "grasp_sampler", "grasp_dataset", "cube_5cm_grasps_valid.npz")


# --------------------------------------------------------------------------------------
# Forward kinematics (spec Section 2)
# --------------------------------------------------------------------------------------


class HandChain:
    """Minimal URDF kinematic chain: root-relative pose of every link.

    Only ``fixed``/``revolute``/``continuous``/``prismatic`` joints are supported,
    which covers the Inspire-hand URDF (its coupled "slave" joints are baked in as
    ``type="fixed"`` at the measured postures, so the 6 actuated angles fully
    determine the hand shape).
    """

    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        root = ET.parse(urdf_path).getroot()

        self.link_names = [l.get("name") for l in root.findall("link")]
        self.joints: dict[str, dict] = {}
        for j in root.findall("joint"):
            origin = j.find("origin")
            xyz = np.zeros(3)
            rpy = np.zeros(3)
            if origin is not None:
                xyz = np.array([float(v) for v in (origin.get("xyz") or "0 0 0").split()])
                rpy = np.array([float(v) for v in (origin.get("rpy") or "0 0 0").split()])
            axis_el = j.find("axis")
            axis = None
            if axis_el is not None:
                axis = np.array([float(v) for v in axis_el.get("xyz").split()])
            self.joints[j.get("name")] = dict(
                type=j.get("type"),
                parent=j.find("parent").get("link"),
                child=j.find("child").get("link"),
                xyz=xyz,
                rpy=rpy,
                axis=axis,
            )

        children = {j["child"] for j in self.joints.values()}
        roots = [l for l in self.link_names if l not in children]
        if len(roots) != 1:
            raise RuntimeError(f"{urdf_path}: expected exactly one root link, found {roots}")
        self.root_link = roots[0]

        # Sanity: the 6 actuated joints must all exist and be movable.
        missing = [n for n in ACTUATED_JOINT_NAMES if n not in self.joints]
        if missing:
            raise RuntimeError(f"{urdf_path}: missing actuated joints {missing}")

    def get_joint_parameter_names(self) -> list[str]:
        """Names of the movable joints, in URDF declaration order."""
        return [n for n, j in self.joints.items() if j["type"] != "fixed"]

    def link_transforms(self, joint_q) -> dict[str, np.ndarray]:
        """``{link_name: 4x4 pose relative to the root link}`` at the 6 actuated angles.

        ``joint_q`` is matched to :data:`ACTUATED_JOINT_NAMES` **by order**; every other
        movable joint (there are none in this URDF) is held at 0.
        """
        joint_q = np.asarray(joint_q, dtype=float).reshape(-1)
        if joint_q.shape[0] != len(ACTUATED_JOINT_NAMES):
            raise ValueError(f"expected {len(ACTUATED_JOINT_NAMES)} joint angles, got {joint_q.shape[0]}")
        qmap = {name: float(joint_q[i]) for i, name in enumerate(ACTUATED_JOINT_NAMES)}

        poses = {self.root_link: np.eye(4)}
        changed = True
        while changed:  # topological sweep; the tree is tiny (18 joints)
            changed = False
            for name, j in self.joints.items():
                if j["parent"] in poses and j["child"] not in poses:
                    T = np.eye(4)
                    T[:3, :3] = R.from_euler("xyz", j["rpy"]).as_matrix()  # URDF rpy = extrinsic xyz
                    T[:3, 3] = j["xyz"]
                    if j["type"] in ("revolute", "continuous"):
                        a = j["axis"] / np.linalg.norm(j["axis"])
                        Tj = np.eye(4)
                        Tj[:3, :3] = R.from_rotvec(a * qmap.get(name, 0.0)).as_matrix()
                        T = T @ Tj
                    elif j["type"] == "prismatic":
                        a = j["axis"] / np.linalg.norm(j["axis"])
                        Tj = np.eye(4)
                        Tj[:3, 3] = a * qmap.get(name, 0.0)
                        T = T @ Tj
                    poses[j["child"]] = poses[j["parent"]] @ T
                    changed = True
        return poses


def load_chain(urdf_path: str = DEFAULT_URDF) -> HandChain:
    """Build the FK chain once (reuse it across grasps)."""
    return HandChain(urdf_path)


def link_transforms(chain: HandChain, joint_q) -> dict[str, np.ndarray]:
    """Functional alias of :meth:`HandChain.link_transforms` (matches the spec's API)."""
    return chain.link_transforms(joint_q)


def palm_matrix(pos, quat_wxyz) -> np.ndarray:
    """4x4 pose of the hand root ("base") link in the cube frame. Quaternion is scalar-first."""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(np.asarray(quat_wxyz, dtype=float), scalar_first=True).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


def fk_tips(chain: HandChain, palm_pos, palm_quat, joint_q) -> np.ndarray:
    """(5,3) fingertip positions in the cube frame, order [thumb, index, middle, ring, pinky]."""
    Tp = palm_matrix(palm_pos, palm_quat)
    L = chain.link_transforms(joint_q)
    return np.array([(Tp @ L[name])[:3, 3] for name in FINGERTIP_LINKS])


# --------------------------------------------------------------------------------------
# Cube geometry (spec Section 3)
# --------------------------------------------------------------------------------------


def cube_surface_dist(p, h: float = CUBE_HALF_EDGE) -> float:
    """Signed distance from a point to the surface of the axis-aligned cube at the origin.

    ``> 0`` outside (Euclidean distance to the nearest surface point), ``<= 0`` inside
    (negative penetration depth).
    """
    d = np.abs(np.asarray(p, dtype=float)) - h
    outside = float(np.linalg.norm(np.clip(d, 0.0, None)))
    inside = float(min(np.max(d), 0.0))
    return outside + inside


def cube_contact(p, h: float = CUBE_HALF_EDGE) -> np.ndarray:
    """Project a point onto the nearest cube face -> ``[contact_pos(3), inward_normal(3)]``.

    The dominant axis (largest ``|p_a|``) picks the face; the normal points INTO the cube.
    """
    p = np.asarray(p, dtype=float).copy()
    k = int(np.argmax(np.abs(p) / h))
    contact = p.copy()
    contact[k] = np.sign(p[k]) * h
    normal = np.zeros(3)
    normal[k] = -np.sign(p[k])
    return np.concatenate([contact, normal])


# --------------------------------------------------------------------------------------
# Collision spheres + grasp library (spec Sections 1.1, 1.3)
# --------------------------------------------------------------------------------------


def load_spheres(robot_yaml_path: str = DEFAULT_ROBOT_YAML) -> dict[str, list[tuple[np.ndarray, float]]]:
    """``{link_name: [(center_in_link_frame(3,), radius), ...]}`` from the BODex robot config."""
    import yaml

    with open(robot_yaml_path) as f:
        cfg = yaml.safe_load(f)["robot_cfg"]["kinematics"]["collision_spheres"]
    return {
        link: [(np.asarray(s["center"], dtype=float), float(s["radius"])) for s in lst]
        for link, lst in cfg.items()
    }


def load_library(library_npz: str = DEFAULT_LIBRARY, stage: int = STAGE_GRASP) -> np.ndarray:
    """``(N, 13)`` grasps at the requested stage: ``[pos(3), quat_wxyz(4), q6(6)]``.

    Read RAW from the synthesis frame: the ``T_usdbase_urdfbase`` calibration stored in
    the same file belongs to the online reward only and must NOT be applied here (spec
    Section 0) — the "axis-aligned cube at the origin" assumption only holds in the raw
    synthesis frame.
    """
    data = np.load(library_npz, allow_pickle=True)
    gp = data["grasp_pose"]  # (N, 1, 3, 13)
    if gp.ndim != 4 or gp.shape[-1] != 13:
        raise RuntimeError(f"{library_npz}: expected grasp_pose (N,1,3,13), got {gp.shape}")
    return np.asarray(gp[:, 0, stage, :], dtype=float)
