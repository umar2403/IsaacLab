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


def main():
    d = np.load(RECENTERED, allow_pickle=True)
    g = d['grasp_pose']  # (N,1,3,13), positions already grip-recentered
    joints = parse(URDF)
    N = g.shape[0]

    keep, scores = [], []
    for i in range(N):
        pos, quat, q6 = g[i, 0, 1, :3], g[i, 0, 1, 3:7], g[i, 0, 1, 7:]
        Rm = R.from_quat(quat, scalar_first=True).as_matrix()
        tips_obj = (Rm @ fk_tips(joints, q6).T).T + pos
        sd = cube_surface_dist(tips_obj)
        thumb, fingers = sd[0], sd[1:]

        geom_ok = np.abs(thumb) < 0.020 and (np.abs(fingers) < 0.020).sum() >= 2
        # tray compatibility: nothing reaches below the cube underside (z=-0.025
        # in object frame, tray surface right below); palm not below cube center
        tray_ok = tips_obj[:, 2].min() > -0.030 and pos[2] > -0.02
        if geom_ok and tray_ok:
            keep.append(i)
            scores.append(np.abs(thumb) + np.sort(np.abs(fingers))[:2].sum())

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
