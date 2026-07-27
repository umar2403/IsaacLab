# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Self-tests for the grasp-selection optimizer (spec Sections 3, 4.4, 4.5, 8).

    python selftest.py

Checks, in order:

1. cube SDF + face projection (inward normals, snap to the dominant face),
2. FSWO sanity: antipodal contacts -> 0, three orthogonal faces -> ~ -1,
   invariance to a global rigid transform of the contact set,
3. the ``psd_sqrt``/NNLS reduction against a brute-force constrained SLSQP solve
   on random contact sets (this is the correctness detail of spec Section 4.4 —
   using raw ``Q`` as the design matrix instead of ``L`` fails here),
4. FK cross-check: fingertip positions from this module vs. the independent URDF FK
   already used by ``grasp_sampler/check_grasps_offline.py``,
5. frame conventions on the real library: joint order, stage index, no calibration.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fswo import fswo_score  # noqa: E402
from hand_model import (  # noqa: E402
    ACTUATED_JOINT_NAMES,
    CUBE_HALF_EDGE,
    DEFAULT_LIBRARY,
    DEFAULT_URDF,
    FINGERTIP_LINKS,
    STAGE_GRASP,
    cube_contact,
    cube_surface_dist,
    fk_tips,
    load_chain,
    load_library,
)

TOL = 1e-6
_fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        _fails.append(name)


# --------------------------------------------------------------------------------------
# 1. cube geometry
# --------------------------------------------------------------------------------------
def test_cube_geometry() -> None:
    print("\n1. cube geometry (spec Section 3)")
    h = CUBE_HALF_EDGE
    check("center is -h inside", abs(cube_surface_dist([0, 0, 0]) + h) < TOL)
    check("on a face -> 0", abs(cube_surface_dist([h, 0, 0])) < TOL)
    check("1 cm above a face -> 0.01", abs(cube_surface_dist([0, 0, h + 0.01]) - 0.01) < TOL)
    d = cube_surface_dist([h + 0.03, h + 0.04, 0])  # outside a vertical edge
    check("outside an edge -> Euclidean", abs(d - 0.05) < TOL, f"{d:.4f}")

    c = cube_contact([0.01, 0.002, 0.20])  # dominant axis = +z
    check("projects onto the +z face", abs(c[2] - h) < TOL and abs(c[0] - 0.01) < TOL)
    check("normal points INTO the cube", np.allclose(c[3:], [0, 0, -1]))
    c = cube_contact([-0.4, 0.01, 0.0])
    check("projects onto the -x face", np.allclose(c[:3], [-h, 0.01, 0.0]) and np.allclose(c[3:], [1, 0, 0]))


# --------------------------------------------------------------------------------------
# 2. FSWO sanity
# --------------------------------------------------------------------------------------
def test_fswo_sanity() -> None:
    print("\n2. FSWO sanity (spec Section 4.5)")
    h = CUBE_HALF_EDGE
    antipodal = np.array([
        [h, 0.0, 0.0, -1.0, 0.0, 0.0],
        [-h, 0.0, 0.0, 1.0, 0.0, 0.0],
    ])
    s = fswo_score(antipodal)
    check("antipodal pair -> 0", abs(s) < 1e-9, f"S={s:.2e}")

    orthogonal = np.array([
        [h, 0.0, 0.0, -1.0, 0.0, 0.0],
        [0.0, h, 0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, h, 0.0, 0.0, -1.0],
    ])
    s = fswo_score(orthogonal)
    check("3 orthogonal faces -> ~ -1", abs(s + 1.0) < 0.05, f"S={s:.4f}")

    # invariance to a global rigid transform of the whole contact set
    rng = np.random.default_rng(0)
    pts = rng.normal(scale=0.03, size=(6, 3))
    contacts = np.array([cube_contact(p) for p in pts])
    Rm = R.from_rotvec([0.3, -0.7, 1.1]).as_matrix()
    rot = np.hstack([contacts[:, :3] @ Rm.T, contacts[:, 3:] @ Rm.T])
    check("rotation invariance", abs(fswo_score(contacts) - fswo_score(rot)) < 1e-8)

    # a "hover" set still scores ~0 -> exactly the Version-1 failure mode (spec Section 9)
    hover = np.array([cube_contact(p) for p in [[0.30, 0, 0], [-0.30, 0, 0], [0, 0.30, 0]]])
    check("projection hides the gap (motivates V2)", fswo_score(hover) > -1e-6,
          f"S={fswo_score(hover):.2e} for contacts 27.5 cm off the cube")

    # regression: a near-degenerate contact set makes NNLS return |alpha| ~ 1e11 along a
    # null direction of Q; scoring via alpha^T Q alpha then cancels to a large POSITIVE
    # "score" (observed +3.5e7 on library grasp #18). The score must stay in (-inf, 0].
    rng2 = np.random.default_rng(11)
    worst_pos = -np.inf
    for _ in range(200):
        k = int(rng2.integers(4, 12))
        pts = rng2.normal(scale=0.03, size=(k, 3))
        # duplicate a few points to deliberately make the contact set rank-deficient
        pts[: k // 3] = pts[0]
        worst_pos = max(worst_pos, fswo_score(np.array([cube_contact(p) for p in pts])))
    check("score never positive on degenerate contact sets", worst_pos <= 0.0,
          f"max S={worst_pos:.3e}")


# --------------------------------------------------------------------------------------
# 3. NNLS reduction vs. brute force
# --------------------------------------------------------------------------------------
def _brute_force_min(Q: np.ndarray) -> float:
    """min alpha^T Q alpha s.t. alpha >= 0, max alpha = 1 — by pinning each alpha_j = 1."""
    k = Q.shape[0]
    best = np.inf
    for j in range(k):
        free = [i for i in range(k) if i != j]

        def obj(x, j=j, free=free):
            a = np.zeros(k)
            a[free] = x
            a[j] = 1.0
            return float(a @ Q @ a)

        for x0 in (np.zeros(k - 1), np.full(k - 1, 0.5), np.ones(k - 1)):
            res = minimize(obj, x0, method="SLSQP",
                           bounds=[(0.0, None)] * (k - 1),
                           options={"maxiter": 500, "ftol": 1e-14})
            best = min(best, float(res.fun))
    return best


def test_nnls_reduction() -> None:
    print("\n3. NNLS reduction uses the matrix square root L, not Q (spec Section 4.4)")
    rng = np.random.default_rng(7)
    worst_L, worst_Q = 0.0, 0.0
    for _ in range(15):
        k = int(rng.integers(3, 7))
        pts = rng.normal(scale=0.04, size=(k, 3))
        contacts = np.array([cube_contact(p) for p in pts])
        p, n = contacts[:, :3], contacts[:, 3:]
        tau = np.cross(p, n)
        Q = n @ n.T + tau @ tau.T

        ours = -fswo_score(contacts)
        truth = _brute_force_min(Q)
        worst_L = max(worst_L, ours - truth)  # positive = our answer is worse than the true optimum

        # the WRONG reduction the spec warns about: raw Q as the design matrix
        from scipy.optimize import nnls
        best_wrong = np.inf
        for j in range(k):
            mask = np.arange(k) != j
            x, _ = nnls(Q[:, mask], -Q[:, j])
            a = np.zeros(k)
            a[mask] = x
            a[j] = 1.0
            best_wrong = min(best_wrong, float(a @ Q @ a))
        worst_Q = max(worst_Q, best_wrong - truth)

    check("L-based NNLS matches brute force", worst_L < 1e-6, f"max excess {worst_L:.2e}")
    check("Q-based NNLS is measurably worse (why L matters)", worst_Q > 1e-6,
          f"max excess {worst_Q:.2e}")


# --------------------------------------------------------------------------------------
# 4. FK cross-check against the repo's existing independent implementation
# --------------------------------------------------------------------------------------
def test_fk_against_check_grasps_offline() -> None:
    print("\n4. FK cross-check vs. grasp_sampler/check_grasps_offline.py")
    sampler = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "grasp_sampler")
    if not os.path.exists(DEFAULT_URDF):
        check("URDF available", False, DEFAULT_URDF)
        return
    sys.path.insert(0, sampler)
    try:
        from check_grasps_offline import ACT, TIP_LINKS, fk_tips as ref_fk_tips, parse
    except Exception as exc:  # pragma: no cover
        check("import check_grasps_offline", False, str(exc))
        return

    check("actuated joint order matches", list(ACT) == ACTUATED_JOINT_NAMES)
    check("fingertip link order matches", list(TIP_LINKS) == FINGERTIP_LINKS)

    chain = load_chain(DEFAULT_URDF)
    joints = parse(DEFAULT_URDF)
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(20):
        q6 = rng.uniform(-0.4, 1.4, size=6)
        mine = np.array([chain.link_transforms(q6)[l][:3, 3] for l in FINGERTIP_LINKS])
        ref = ref_fk_tips(joints, q6)
        worst = max(worst, float(np.abs(mine - ref).max()))
    check("fingertip FK identical to the trusted implementation", worst < 1e-9, f"max |diff| {worst:.2e} m")

    # and the palm composition path
    g = load_library(DEFAULT_LIBRARY)[0]
    tips_a = fk_tips(chain, g[:3], g[3:7], g[7:])
    Rm = R.from_quat(g[3:7], scalar_first=True).as_matrix()
    tips_b = (Rm @ ref_fk_tips(joints, g[7:]).T).T + g[:3]
    check("T_palm . T_link composition matches", float(np.abs(tips_a - tips_b).max()) < 1e-9)


# --------------------------------------------------------------------------------------
# 5. library frame conventions
# --------------------------------------------------------------------------------------
def test_library_conventions() -> None:
    print("\n5. library frame conventions (spec Section 8)")
    if not os.path.exists(DEFAULT_LIBRARY):
        check("library available", False, DEFAULT_LIBRARY)
        return
    data = np.load(DEFAULT_LIBRARY, allow_pickle=True)
    gp = data["grasp_pose"]
    check("grasp_pose is (N,1,3,13)", gp.ndim == 4 and gp.shape[1] == 1 and gp.shape[2] == 3 and gp.shape[3] == 13,
          str(gp.shape))
    # joint_order/stages are provenance only — mdp/grasp_goal.py never reads them, and
    # libraries imported from other repos may omit them.
    if "joint_order" in data:
        order = [str(s) for s in data["joint_order"]]
        check("joint_order matches ACTUATED_JOINT_NAMES",
              order == [n.replace("_joint", "") for n in ACTUATED_JOINT_NAMES], str(order))
        stages = [str(s) for s in data["stages"]]
        check(f"stage {STAGE_GRASP} is 'grasp'", stages[STAGE_GRASP] == "grasp", str(stages))
    else:
        print("  [skip] joint_order/stages absent (imported library) — order assumed by convention")

    g = load_library(DEFAULT_LIBRARY)
    check("quaternions are unit", float(np.abs(np.linalg.norm(g[:, 3:7], axis=1) - 1).max()) < 1e-5)
    raw = np.asarray(gp[:, 0, STAGE_GRASP, :], dtype=float)
    check("no calibration applied (raw synthesis frame)", np.allclose(g, raw))


def main() -> int:
    print("grasp_selection self-test")
    test_cube_geometry()
    test_fswo_sanity()
    test_nnls_reduction()
    test_fk_against_check_grasps_offline()
    test_library_conventions()
    print(f"\n{'ALL CHECKS PASSED' if not _fails else 'FAILED: ' + ', '.join(_fails)}")
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
