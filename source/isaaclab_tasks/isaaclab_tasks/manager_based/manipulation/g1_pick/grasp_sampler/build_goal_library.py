"""Build the final goal-grasp library for RL reward shaping.

Takes the grip-recentered grasps and keeps those that are:
1. geometrically sound in the synthesis model (thumb + >=2 fingers near surface),
2. compatible with a cube resting on the tray (no fingertip or palm below the
   cube's underside, approach not from below).

Output: grasp_dataset/cube_5cm_grasps_valid.npz (format consumed by
mdp/grasp_goal.py).
"""
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

from check_grasps_offline import parse, fk_tips, cube_surface_dist, URDF, HERE

RECENTERED = os.path.join(HERE, 'grasp_dataset', 'cube_5cm_grasps_recentered.npz')
OUT = os.path.join(HERE, 'grasp_dataset', 'cube_5cm_grasps_valid.npz')
MAX_KEEP = 32
# Keep only near-top-down grasps: approach direction within this angle of
# straight-down, so the palm is (roughly) horizontal facing the cube from above.
# The G1 arm reaches a palm-down top approach far more easily than a vertical
# palm / side approach. Set to 180 to disable the constraint.
TOPDOWN_MAX_DEG = 50.0


def main():
    d = np.load(RECENTERED, allow_pickle=True)
    g = d['grasp_pose']  # (N,1,3,13), positions already grip-recentered
    joints = parse(URDF)
    N = g.shape[0]

    topdown_thresh = -np.cos(np.radians(TOPDOWN_MAX_DEG))
    keep, scores = [], []
    n_topdown_rejected = 0
    for i in range(N):
        pre, pos, quat, q6 = g[i, 0, 0, :3], g[i, 0, 1, :3], g[i, 0, 1, 3:7], g[i, 0, 1, 7:]
        Rm = R.from_quat(quat, scalar_first=True).as_matrix()
        tips_obj = (Rm @ fk_tips(joints, q6).T).T + pos
        sd = cube_surface_dist(tips_obj)
        thumb, fingers = sd[0], sd[1:]

        geom_ok = np.abs(thumb) < 0.020 and (np.abs(fingers) < 0.020).sum() >= 2
        # tray compatibility: nothing reaches below the cube underside (z=-0.025
        # in object frame, tray surface right below); palm not below cube center
        tray_ok = tips_obj[:, 2].min() > -0.030 and pos[2] > -0.02
        # top-down = the palm faces DOWN: the hand's finger/approach axis (local z)
        # points downward, so the palm comes onto the cube from above (easy for the
        # G1 arm) rather than sideways with a vertical palm. Measured from the grasp
        # ORIENTATION (not stage displacement, which isn't a reliable approach dir).
        finger_axis_world_z = Rm[2, 2]        # world-z component of the hand local-z
        topdown_ok = finger_axis_world_z < topdown_thresh
        if geom_ok and tray_ok and not topdown_ok:
            n_topdown_rejected += 1
        if geom_ok and tray_ok and topdown_ok:
            keep.append(i)
            scores.append(np.abs(thumb) + np.sort(np.abs(fingers))[:2].sum())
    print(f"(rejected {n_topdown_rejected} geometrically-valid grasps for non-top-down approach)")

    order = np.argsort(scores)
    keep = [keep[j] for j in order][:MAX_KEEP]
    print(f"kept {len(keep)}/{N} grasps: {keep}")

    np.savez(OUT,
             grasp_pose=g[keep],
             T_usdbase_urdfbase=d['T_usdbase_urdfbase'],
             joint_order=d['joint_order'], stages=d['stages'],
             source_indices=np.array(keep))
    print(f"goal library -> {OUT}")


if __name__ == '__main__':
    main()
