# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
import os

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

_USD_PATH = os.path.join(
    _CURRENT_DIR,
    "g1_with_hands_final.usd"
)

G1_INSPIRE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            # Was 1.0. Tried 5.0 first (measured ~2.8cm steady-state finger/cube overlap under
            # continuous PD closing force) -- that overcorrected: it visibly launched the cube
            # away from the hand for long stretches (explosive separation from injecting too
            # much corrective velocity), which is worse than the original overlap and also a
            # policy trained at 1.0 now facing very different contact dynamics. Settled on a
            # gentler bump instead, paired with more solver iterations below (which improve
            # contact accuracy without adding velocity/momentum, a safer lever).
            max_depenetration_velocity=2.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=48,  # was 32 -- more accurate contact resolution
            solver_velocity_iteration_count=4,
            # fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.1, 0.0, 0.74),
        joint_pos={
            # ── Legs ──
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            
            # ── Waist ──
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,

            # ── RIGHT Arm (Converted from your UI visually posed degrees) ──
            "right_shoulder_pitch_joint": -0.583,  # -33.39 deg
            "right_shoulder_roll_joint": -0.864,   # -49.49 deg
            "right_shoulder_yaw_joint": 0.426,     # 24.39 deg
            "right_elbow_joint": 0.370,            # 21.21 deg
            "right_wrist_roll_joint": -0.410,      # -23.49 deg
            "right_wrist_pitch_joint": 0.000,      # 0.01 deg (rounded to 0)
            "right_wrist_yaw_joint": -0.103,       # -5.9 deg
            
            # ── LEFT Arm ──
            "left_shoulder_pitch_joint": 1.5,
            "left_shoulder_roll_joint": 0.5,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,

            # ── Inspire Right Hand (ALL 12 JOINTS EXPLICITLY INITIALIZED) ──
            "R_thumb_proximal_yaw_joint": 0.7, 
            "R_thumb_proximal_pitch_joint": 0.0,
            "R_thumb_intermediate_joint": 0.0, # <-- Newly added!
            "R_thumb_distal_joint": 0.0,       # <-- Newly added!
            
            "R_index_proximal_joint": 0.0,
            "R_index_intermediate_joint": 0.0, # <-- Newly added!
            
            "R_middle_proximal_joint": 0.0,
            "R_middle_intermediate_joint": 0.0, # <-- Newly added!
            
            "R_ring_proximal_joint": 0.0,
            "R_ring_intermediate_joint": 0.0, # <-- Newly added!
            
            "R_pinky_proximal_joint": 0.0,
            "R_pinky_intermediate_joint": 0.0, # <-- Newly added!
            
            # ── Let's quickly do the left hand too, just to be safe ──
            "L_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint",
                ".*_knee_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            ],
            effort_limit_sim=300,
            stiffness={
                ".*_hip_yaw_joint": 150.0, ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0, ".*_knee_joint": 200.0,
                "waist_.*_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0, ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0, ".*_knee_joint": 5.0,
                "waist_.*_joint": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=20,
            stiffness=20.0,
            damping=2.0,
        ),
        "arms": ImplicitActuatorCfg(
                    joint_names_expr=[
                        ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
                        ".*_elbow_joint", 
                        ".*_wrist_roll_joint", ".*_wrist_pitch_joint", ".*_wrist_yaw_joint" # Merged!
                    ],
                    effort_limit_sim=300.0,  # Plenty of torque to hold the heavy Inspire hand
                    stiffness=300.0,        # Rock solid crane
                    damping=30.0,           # No vibrations
                    armature=0.001,
                ),
        "left_hand": ImplicitActuatorCfg(
            joint_names_expr=["L_.*_joint"], # Uses USD Prefix
            effort_limit_sim=10.0,
            stiffness=10.0,
            damping=1.0,
        ),
        "right_hand": ImplicitActuatorCfg(
            joint_names_expr=["R_.*_joint"], # Uses USD Prefix
            effort_limit_sim=30.0,
            stiffness=100.0,
            damping=0.5,
        ),
    },
)