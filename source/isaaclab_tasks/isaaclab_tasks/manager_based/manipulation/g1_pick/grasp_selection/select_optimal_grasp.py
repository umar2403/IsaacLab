# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Version 1 — fingertip FSWO grasp selection (spec Section 5). REFERENCE ONLY.

Kept because Version 2 reuses its FSWO core verbatim and because reproducing its
failure mode is the justification for Version 2:

    ``cube_contact`` snaps every fingertip onto the nearest cube face and DISCARDS the
    gap, so a fingertip hovering 27 mm off the cube is scored as a genuine contact.
    Version 1 therefore ranks a loose fingertip-hover as "perfect force closure"
    (S ~ 0) above a real wrap.

**Do not use this for training.** Use :mod:`rank_grasps_sphere_fswo` (Version 2).
Run this module to see the difference:

    python select_optimal_grasp.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np

try:
    from .fswo import fswo_score
    from .hand_model import (
        DEFAULT_LIBRARY,
        DEFAULT_URDF,
        FINGERS,
        STAGE_GRASP,
        HandChain,
        cube_contact,
        cube_surface_dist,
        fk_tips,
        load_chain,
        load_library,
    )
except ImportError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fswo import fswo_score
    from hand_model import (
        DEFAULT_LIBRARY,
        DEFAULT_URDF,
        FINGERS,
        STAGE_GRASP,
        HandChain,
        cube_contact,
        cube_surface_dist,
        fk_tips,
        load_chain,
        load_library,
    )

__all__ = ["build_contacts_tips", "select_best_grasp_fingertips"]


def build_contacts_tips(chain: HandChain, grasp13: np.ndarray) -> np.ndarray:
    """(5,6) contacts from the 5 fingertips projected onto the cube."""
    pos, quat, jq = grasp13[:3], grasp13[3:7], grasp13[7:]
    tips = fk_tips(chain, pos, quat, jq)
    return np.array([cube_contact(t) for t in tips])


def select_best_grasp_fingertips(
    library_npz: str = DEFAULT_LIBRARY,
    urdf_path: str = DEFAULT_URDF,
    stage: int = STAGE_GRASP,
    lam: float = 1.0,
) -> tuple[int, list[int], np.ndarray]:
    """``(best_idx, ranked_indices, scores)`` under the original fingertip-only FSWO."""
    chain = load_chain(urdf_path)
    grasps = load_library(library_npz, stage=stage)
    scores = np.array([fswo_score(build_contacts_tips(chain, grasps[i]), lam=lam)
                       for i in range(grasps.shape[0])])
    ranking = list(np.argsort(-scores))
    return int(ranking[0]), [int(i) for i in ranking], scores


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--library", default=DEFAULT_LIBRARY)
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--top", type=int, default=8)
    args = p.parse_args()

    best_idx, ranking, scores = select_best_grasp_fingertips(args.library, args.urdf, lam=args.lam)
    chain = load_chain(args.urdf)
    grasps = load_library(args.library)

    print("VERSION 1 (fingertip FSWO) — reference only, do not ship.\n")
    print(f"{'rank':>4} {'grasp':>5} {'fswo':>10}   fingertip surface gaps [mm]  (thumb,index,middle,ring,pinky)")
    for r, idx in enumerate(ranking[: args.top]):
        tips = fk_tips(chain, grasps[idx][:3], grasps[idx][3:7], grasps[idx][7:])
        gaps = np.array([cube_surface_dist(t) for t in tips]) * 1e3
        print(f"{r:>4} {idx:>5} {scores[idx]:>10.6f}   {np.round(gaps, 1)}")

    tips = fk_tips(chain, grasps[best_idx][:3], grasps[best_idx][3:7], grasps[best_idx][7:])
    gaps = np.array([cube_surface_dist(t) for t in tips])
    print(f"\nbest_idx = {best_idx}  (score {scores[best_idx]:.6f})")
    print("  fingertip gaps to the cube surface:")
    for name, g in zip(FINGERS, gaps):
        print(f"    {name:>6}: {g * 1e3:+7.1f} mm")
    print("\n  ^ note how large these can be: the projection discards the gap, which is exactly")
    print("    the failure Version 2 (rank_grasps_sphere_fswo.py) fixes.")


if __name__ == "__main__":
    main()
