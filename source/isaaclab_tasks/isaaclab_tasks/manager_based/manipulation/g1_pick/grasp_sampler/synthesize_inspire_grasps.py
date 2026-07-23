"""Synthesize Inspire Hand grasps for the g1_pick target cube using BODex
(via the UltraDexGrasp pipeline).

Run from this directory with the `ultradex` conda env:

    cd grasp_sampler
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
    ~/miniconda3/envs/ultradex/bin/python synthesize_inspire_grasps.py

Output: grasp_dataset/cube_5cm_grasps.npz with array `grasp_pose` of shape
(N, 1, 3, 13): N grasps x 1 hand x 3 stages (pregrasp, grasp, squeeze) x
(root pos xyz + root quat wxyz + 6 hand joints).
Hand joint order matches the g1_pick policy:
[thumb_yaw, thumb_pitch, index, middle, ring, pinky] proximal joints.
Poses are in the object frame (object centered at origin, resting on a table).
"""
import os
import sys

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ultradex_repo')
sys.path.insert(0, REPO)

from util.bodex_util import GraspSynthesizer  # noqa: E402

CUBE_ASSET = os.path.join(REPO, 'asset', 'object_mesh', 'cube')
CUBE_SIZE = 0.05  # unit cube mesh scaled to 5 cm
NUM_GRASP = 300
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'grasp_dataset')


def main():
    synthesizer = GraspSynthesizer(hand=1, num_grasp=NUM_GRASP, hand_type='inspire', dof=6)
    # object at origin, identity orientation (quat wxyz)
    grasp_pose = synthesizer.synthesize_grasp(CUBE_ASSET, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], CUBE_SIZE)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'cube_5cm_grasps.npz')
    np.savez(
        out_path,
        grasp_pose=grasp_pose,
        joint_order=np.array(['thumb_proximal_yaw', 'thumb_proximal_pitch', 'index_proximal',
                              'middle_proximal', 'ring_proximal', 'pinky_proximal']),
        stages=np.array(['pregrasp', 'grasp', 'squeeze']),
        object_size=CUBE_SIZE,
    )
    print(f"saved {grasp_pose.shape[0]} grasps -> {out_path}")
    print(f"shape: {grasp_pose.shape}  (N, hands, stages, 7+dof)")


if __name__ == '__main__':
    main()
