"""Build the final goal-grasp library for RL reward shaping.

Takes the grip-recentered grasps and keeps those that are:
1. **task-feasible** for the G1 arm and the tray: no fingertip or palm below the
   cube's underside, and a near-top-down approach (see ``TOPDOWN_MAX_DEG``),
2. **mechanically sound**, judged by the grasp-selection optimizer: >= 3 distinct
   fingers with collision spheres actually touching the cube, on opposing faces
   (force-closable). See ``../grasp_selection/`` and ``../OPTIMIZER_IMPLEMENTATION_SPEC.md``.

The survivors are ranked by **wrap quality** ``(n_fingers, n_spheres, fswo)`` and the
top ``MAX_KEEP`` are written out in ascending pool order.

Ranking note: this used to sort by a fingertip-distance heuristic
(``|thumb| + two smallest |finger|`` surface distances) and truncate to 32. That
heuristic measures only 5 points and ignores whether the fingers oppose each other,
so it truncated away genuinely better wraps — e.g. pool grasp #79 (10 contacting
spheres, 38 deg top-down) never made the library, while 9- and 8-sphere grasps did.
Pass ``--rank legacy`` to reproduce the old behaviour exactly.

Output: grasp_dataset/cube_5cm_grasps_valid.npz (format consumed by mdp/grasp_goal.py).
"""
import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

from check_grasps_offline import parse, fk_tips, cube_surface_dist, URDF, HERE

sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'grasp_selection'))
import rank_grasps_sphere_fswo as rk          # noqa: E402
from hand_model import load_chain, load_spheres  # noqa: E402

RECENTERED = os.path.join(HERE, 'grasp_dataset', 'cube_5cm_grasps_recentered.npz')
OUT = os.path.join(HERE, 'grasp_dataset', 'cube_5cm_grasps_valid.npz')
MAX_KEEP = 32
# Keep only near-top-down grasps: approach direction within this angle of
# straight-down, so the palm is (roughly) horizontal facing the cube from above.
# The G1 arm reaches a palm-down top approach far more easily than a vertical
# palm / side approach. Set to 180 to disable the constraint.
TOPDOWN_MAX_DEG = 50.0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rank', choices=('optimizer', 'legacy'), default='optimizer',
                   help="'optimizer' = wrap quality from grasp_selection (default); "
                        "'legacy' = the old fingertip-distance heuristic")
    p.add_argument('--max-keep', type=int, default=MAX_KEEP)
    p.add_argument('--topdown-deg', type=float, default=TOPDOWN_MAX_DEG)
    p.add_argument('--out', default=OUT)
    args = p.parse_args()

    d = np.load(RECENTERED, allow_pickle=True)
    g = d['grasp_pose']  # (N,1,3,13), positions already grip-recentered
    joints = parse(URDF)
    N = g.shape[0]

    chain, spheres = load_chain(), load_spheres()
    topdown_thresh = -np.cos(np.radians(args.topdown_deg))
    keep, sort_keys, detail = [], [], {}
    n_topdown_rejected = n_contact_rejected = 0

    for i in range(N):
        pos, quat, q6 = g[i, 0, 1, :3], g[i, 0, 1, 3:7], g[i, 0, 1, 7:]
        Rm = R.from_quat(quat, scalar_first=True).as_matrix()
        tips_obj = (Rm @ fk_tips(joints, q6).T).T + pos
        sd = cube_surface_dist(tips_obj)
        thumb, fingers = sd[0], sd[1:]

        # ---- task feasibility (unchanged) ------------------------------------------
        # tray compatibility: nothing reaches below the cube underside (z=-0.025
        # in object frame, tray surface right below); palm not below cube center
        tray_ok = tips_obj[:, 2].min() > -0.030 and pos[2] > -0.02
        # top-down = the palm faces DOWN: the hand's finger/approach axis (local z)
        # points downward, so the palm comes onto the cube from above (easy for the
        # G1 arm) rather than sideways with a vertical palm. Measured from the grasp
        # ORIENTATION (not stage displacement, which isn't a reliable approach dir).
        topdown_ok = Rm[2, 2] < topdown_thresh

        if args.rank == 'legacy':
            # old gate: thumb + >=2 fingertips within 20 mm of the surface
            ok = np.abs(thumb) < 0.020 and (np.abs(fingers) < 0.020).sum() >= 2
            key = float(np.abs(thumb) + np.sort(np.abs(fingers))[:2].sum())
            key = -key  # legacy sorted ASCENDING; negate so "larger is better" holds
            info = None
        else:
            # new gate: the optimizer's contact + force-closure test on ~41 spheres
            info = rk.score_grasp_spheres(chain, spheres, g[i, 0, 1, :])
            ok = info['valid']
            key = rk._sort_key(info) if ok else None

        if ok and tray_ok and not topdown_ok:
            n_topdown_rejected += 1
        elif not ok and tray_ok and topdown_ok:
            n_contact_rejected += 1

        if ok and tray_ok and topdown_ok:
            keep.append(i)
            sort_keys.append(key)
            detail[i] = info

    print(f"pool: {N} grasps")
    print(f"  rejected for non-top-down approach (> {args.topdown_deg:g} deg): {n_topdown_rejected}")
    print(f"  rejected by the {args.rank} contact test: {n_contact_rejected}")

    order = sorted(range(len(keep)), key=lambda j: sort_keys[j], reverse=True)
    ranked = [keep[j] for j in order][: args.max_keep]
    kept = sorted(ranked)  # store in ascending pool order; source_indices maps back

    print(f"\nkept {len(kept)}/{N} feasible grasps, ranked by {args.rank}")
    if args.rank == 'optimizer':
        print(f"  best: pool #{ranked[0]} -> row {kept.index(ranked[0])} of the new library")
        print(f"  {'row':>4} {'pool#':>6} {'fingers':>7} {'spheres':>7} {'fswo':>10}")
        for r in ranked[:5]:
            e = detail[r]
            print(f"  {kept.index(r):>4} {r:>6} {e['n_fingers']:>7} {e['n_spheres']:>7} {e['fswo']:>10.2e}")
    print(f"\npool indices kept: {kept}")

    np.savez(args.out,
             grasp_pose=g[kept],
             T_usdbase_urdfbase=d['T_usdbase_urdfbase'],
             joint_order=d['joint_order'], stages=d['stages'],
             source_indices=np.array(kept))
    print(f"goal library -> {args.out}")
    if args.rank == 'optimizer':
        print("re-run ../grasp_selection/rank_grasps_sphere_fswo.py to refresh scores.json")


if __name__ == '__main__':
    main()
