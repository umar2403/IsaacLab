# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cache the 5 fingertip target contact points for the active grasp (MDP_REPORT.md
§5.2.4 / the `grasp_reach` reward in `mdp/grasp_goal.py`).

For the active grasp (default: whatever `grasp_selection/scores.json` says is
`best_idx`, currently #12), forward-kinematics the 5 fingertip links via the URDF used
by BODex/UltraDexGrasp synthesis, projects each onto the nearest cube face
(`hand_model.cube_contact` -- the same projection the optimizer itself uses), and
writes the result to `grasp_selection/fingertip_contacts.json`.

Why cache instead of computing this online: `grasp_sampler/ultradex_repo/` (the URDF +
collision-sphere YAML) is gitignored and not present in every checkout -- the exact
same reason `get_optimal_grasp_idx()` falls back to the cached `scores.json` instead of
recomputing from the optimizer every time. `mdp.grasp_reach_reward` only ever reads the
cache produced here; it never needs the URDF at training time.

Run with any Python that has numpy/scipy/pyyaml (no Isaac Lab needed) -- e.g. if your
own checkout is missing `ultradex_repo`, point `--urdf` at another checkout that has it:

    python cache_fingertip_contacts.py --urdf /path/to/inspire_hand_right.urdf

Re-run whenever the grasp library changes or the optimizer's pick changes.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from .hand_model import (
        DEFAULT_LIBRARY,
        DEFAULT_URDF,
        FINGERTIP_LINKS,
        STAGE_GRASP,
        HandChain,
        cube_contact,
        load_library,
    )
except ImportError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hand_model import (
        DEFAULT_LIBRARY,
        DEFAULT_URDF,
        FINGERTIP_LINKS,
        STAGE_GRASP,
        HandChain,
        cube_contact,
        load_library,
    )

_HERE = os.path.dirname(os.path.abspath(__file__))
SCORES_JSON = os.path.join(_HERE, "scores.json")
OUT_DEFAULT = os.path.join(_HERE, "fingertip_contacts.json")
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def compute_contacts(grasp_idx: int, library_npz: str, urdf_path: str) -> dict:
    chain = HandChain(urdf_path)
    grasps = load_library(library_npz, stage=STAGE_GRASP)  # (N,13): pos(3) quat_wxyz(4) q6(6)
    g = grasps[grasp_idx]
    pos, quat, q6 = g[:3], g[3:7], g[7:]

    L = chain.link_transforms(q6)
    Tp = np.eye(4)
    Tp[:3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
    Tp[:3, 3] = pos

    tips_cube = np.array([(Tp @ L[name])[:3, 3] for name in FINGERTIP_LINKS])  # (5,3)
    contacts = np.array([cube_contact(t) for t in tips_cube])  # (5,6) [pos(3), inward_normal(3)]

    return {
        "grasp_idx": grasp_idx,
        "library": os.path.relpath(library_npz, _HERE),
        "joint_order": ["thumb_proximal_yaw", "thumb_proximal_pitch", "index_proximal",
                        "middle_proximal", "ring_proximal", "pinky_proximal"],
        "fingertip_order": FINGERS,
        "raw_tip_pos_cube_frame": tips_cube.round(6).tolist(),
        "contact_pos_cube_frame": contacts[:, :3].round(6).tolist(),
        "contact_inward_normal_cube_frame": contacts[:, 3:].round(6).tolist(),
        "note": "contact_pos is raw_tip_pos projected onto the nearest cube face "
                "(cube_contact); positions are in the cube's own object frame, "
                "urdf-base convention, no T_usdbase_urdfbase applied -- consistent "
                "with how sample_grasp_goal's grasp_pos_obj is stored before its own "
                "calibration step.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grasp-idx", type=int, default=None,
                   help="library row to cache (default: best_idx from scores.json)")
    p.add_argument("--library", default=DEFAULT_LIBRARY)
    p.add_argument("--urdf", default=DEFAULT_URDF,
                   help="Inspire hand URDF used by BODex synthesis. If your checkout is "
                        "missing grasp_sampler/ultradex_repo/, point this at another "
                        "checkout that has it.")
    p.add_argument("--out", default=OUT_DEFAULT)
    args = p.parse_args()

    grasp_idx = args.grasp_idx
    if grasp_idx is None:
        with open(SCORES_JSON) as f:
            grasp_idx = int(json.load(f)["best_idx"])

    payload = compute_contacts(grasp_idx, args.library, args.urdf)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"grasp #{grasp_idx}")
    tips = np.array(payload["raw_tip_pos_cube_frame"])
    contacts = np.array(payload["contact_pos_cube_frame"])
    for i, name in enumerate(FINGERS):
        gap = np.linalg.norm(tips[i] - contacts[i])
        print(f"  {name:>6}: raw tip {tips[i].round(4)}  ->  contact {contacts[i].round(4)}  (gap {gap * 1000:.1f} mm)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
