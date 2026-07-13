"""Offline geometric check of synthesized grasps (no simulator needed).

For each grasp, computes fingertip positions via URDF FK (frozen-slave model),
transforms them into the object frame, and measures distance to the 5cm cube's
surface. Good grasps have contact fingertips at ~0 surface distance with the
thumb opposing at least one finger. Ranks and reports.
"""
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, 'ultradex_repo', 'third_party', 'BODex_api', 'src', 'bodex',
                    'content', 'assets', 'robot', 'inspire_hand', 'inspire_hand_right.urdf')
NPZ = os.path.join(HERE, 'grasp_dataset', 'cube_5cm_grasps.npz')
HALF = 0.025  # half cube edge

TIP_LINKS = ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip']
ACT = ['thumb_proximal_yaw_joint', 'thumb_proximal_pitch_joint', 'index_proximal_joint',
       'middle_proximal_joint', 'ring_proximal_joint', 'pinky_proximal_joint']


def parse(urdf):
    joints = {}
    for j in ET.parse(urdf).getroot().findall('joint'):
        o = j.find('origin')
        xyz = np.array([float(v) for v in (o.get('xyz') or '0 0 0').split()])
        rpy = np.array([float(v) for v in (o.get('rpy') or '0 0 0').split()])
        ax = j.find('axis')
        joints[j.get('name')] = dict(
            type=j.get('type'), parent=j.find('parent').get('link'),
            child=j.find('child').get('link'), xyz=xyz, rpy=rpy,
            axis=np.array([float(v) for v in ax.get('xyz').split()]) if ax is not None else None)
    return joints


def fk_tips(joints, q6):
    qmap = dict(zip(ACT, q6))
    poses = {'base': np.eye(4)}
    changed = True
    while changed:
        changed = False
        for name, j in joints.items():
            if j['parent'] in poses and j['child'] not in poses:
                T = np.eye(4)
                T[:3, :3] = R.from_euler('xyz', j['rpy']).as_matrix()
                T[:3, 3] = j['xyz']
                if j['type'] == 'revolute':
                    Tj = np.eye(4)
                    a = j['axis'] / np.linalg.norm(j['axis'])
                    Tj[:3, :3] = R.from_rotvec(a * qmap.get(name, 0.0)).as_matrix()
                    T = T @ Tj
                poses[j['child']] = poses[j['parent']] @ T
                changed = True
    return np.array([poses[l][:3, 3] for l in TIP_LINKS])


def cube_surface_dist(p):
    """Signed distance from point(s) to the cube surface (object frame)."""
    q = np.abs(p) - HALF
    outside = np.linalg.norm(np.maximum(q, 0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0)
    return outside + inside


def main():
    joints = parse(URDF)
    d = np.load(NPZ, allow_pickle=True)
    g = d['grasp_pose']  # (N,1,3,13)
    N = g.shape[0]
    scores = np.zeros(N)
    detail = []
    for i in range(N):
        pos, quat, q6 = g[i, 0, 1, :3], g[i, 0, 1, 3:7], g[i, 0, 1, 7:]
        Rm = R.from_quat(quat, scalar_first=True).as_matrix()
        tips_obj = (Rm @ fk_tips(joints, q6).T).T + pos  # tips in object frame
        sd = cube_surface_dist(tips_obj)
        thumb, fingers = sd[0], sd[1:]
        close_fingers = int((np.abs(fingers) < 0.012).sum())
        contact_ok = np.abs(thumb) < 0.012 and close_fingers >= 2
        scores[i] = np.abs(thumb) + np.abs(fingers).min()
        detail.append((i, contact_ok, thumb, fingers))
    order = np.argsort(scores)
    n_ok = sum(1 for _, ok, _, _ in detail if ok)
    print(f"grasps with thumb+>=2 fingers within 12mm of surface: {n_ok}/{N}")
    print("top 10 by (|thumb sd| + min |finger sd|):")
    for i in order[:10]:
        idx, ok, th, fg = detail[i]
        print(f"  #{idx:3d} ok={ok}  thumb={th:+.3f}  fingers={np.round(fg, 3)}")
    np.save(os.path.join(HERE, 'grasp_dataset', 'geom_rank.npy'), order)
    np.save(os.path.join(HERE, 'grasp_dataset', 'geom_ok.npy'),
            np.array([ok for _, ok, _, _ in detail]))


if __name__ == '__main__':
    main()
