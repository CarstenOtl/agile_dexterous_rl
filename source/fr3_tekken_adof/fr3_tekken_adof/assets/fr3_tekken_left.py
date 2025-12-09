# SPDX-License-Identifier: BSD-3-Clause
"""
Configuration for the FR3 Tekken ADoF left robot as an ArticulationCfg.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Path to the robot USD (models live under omniisaacgymenvs/models)


from fr3_tekken_adof.assets import ISAACLAB_ASSETS_DATA_DIR 


FR3_TEK_LEFT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/robots/fr3_tekken_adof/fr3_tekkenadof_left.usd",
        # usd_path=str(ROBOT_USD_PATH),
        activate_contact_sensors=True, # enable contact sensors if any are defined, for dexsuite tasks
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=10,
            solver_velocity_iteration_count=4,
            # fixed_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "fr3_joint1": 0.0,
            "fr3_joint2": 0.0,
            "fr3_joint3": 0.0,
            "fr3_joint4": -0.4,
            "fr3_joint5": 0.0,
            "fr3_joint6": 0.6,
            "fr3_joint7": 0.0,
            ### ADOF joint limits
            # Thumb Rot: [-0.34 , 0.34] ==> [-20 , 20] deg
            # MCP Pitch: [0 , 1.22] ==> [0 , ~70 deg]
            # MCP Yaw: [-0.26 , 0.26] ==> [-15 deg , 15 deg]
            # PIP: [0 , 1.57] ==> [0 , 90 deg]
            "revolute_Thumb_rot": 0.0,
            "revolute_thumb_mcp_pitch": 0.0,
            "revolute_thumb_mcp_yaw": 0.0,
            "revolute_thumb_pip": 0.0,
            # # "revolute_thumb_dip": 0.0,
            "revolute_index_mcp_pitch": 0.0,
            "revolute_index_mcp_yaw": 0.0,
            "revolute_index_pip": 0.0,
            # # "revolute_index_dip": 0.0,
            "revolute_middle_mcp_pitch":0.0,
            "revolute_middle_mcp_yaw": 0.0,
            "revolute_middle_pip": 0.0,
            # # "revolute_middle_dip": 0.0,
            "revolute_ring_mcp_pitch": 0.,
            "revolute_ring_mcp_yaw": 0.0,
            "revolute_ring_pip": 0.0,
            # # "revolute_ring_dip": 0.0,
            "revolute_pinky_mcp_pitch": 0.0,
            "revolute_pinky_mcp_yaw": 0.0,
            "revolute_pinky_pip": 0.0,
            # # "revolute_pinky_dip": 0.0,
        },
    ),
    actuators={
        "franka_arm": ImplicitActuatorCfg(
            joint_names_expr=[r"fr3_joint[1-7]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=None,
            stiffness=60000.0,
            damping=6000.0,
        ),
        "thumb_rot": ImplicitActuatorCfg(
            joint_names_expr=["revolute_Thumb_rot"],
            effort_limit_sim=10.0,
            velocity_limit_sim=100.0,
            stiffness=500.0,
            damping=100.0006,
        ),
        "mcp_pitch": ImplicitActuatorCfg(
            joint_names_expr=[r"revolute_.*_mcp_pitch"],
            effort_limit_sim=10.0,
            velocity_limit_sim=300.0,
            stiffness=50.814,
            damping=30.00073,
        ),
        "mcp_yaw": ImplicitActuatorCfg(
            joint_names_expr=[r"revolute_.*_mcp_yaw"],
            effort_limit_sim=10.0,
            velocity_limit_sim=300.0,
            stiffness=50.2861,
            damping=30.00022,
        ),
        "pip": ImplicitActuatorCfg(
            joint_names_expr=[r"revolute_.*_pip"],
            effort_limit_sim=10.0,
            velocity_limit_sim=300.0,
            stiffness=50.25291,
            damping=30.0002,
        ),
        # "dip": ImplicitActuatorCfg(
        #     joint_names_expr=[r"revolute_.*_dip.*"],
        #     effort_limit_sim=10.0,
        #     velocity_limit_sim=8.0,
        #     stiffness=0.25291,
        #     damping=0.0002,
        # ),
    },
)

# Create a variant of the config with explicit actuators for stability
FR3_TEK_LEFT_STABLE = FR3_TEK_LEFT_CONFIG.replace(
    actuators={
        # Arm joints (Franka 7-DoF) – moderate PD gains
        "arm_pd": ImplicitActuatorCfg(
            joint_names_expr=[r"fr3_joint[1-7]"],  # adapt if your naming differs
            effort_limit_sim=120.0,
            velocity_limit_sim=5.0,
            stiffness=60000.0,
            damping=600.0,
        ),
        # Hand / Tekken joints – low stiffness, decent damping so they don't go crazy
        "hand_pd": ImplicitActuatorCfg(
            joint_names_expr=[
                r"revolute_Thumb_rot",
                r"revolute_.*_mcp_pitch",
                r"revolute_.*_mcp_yaw",
                r"revolute_.*_pip",
            ],
            effort_limit_sim=30.0,
            velocity_limit_sim=300.0,
            stiffness=50.0,
            damping=20.0,
            # friction=0.01,
            # armature=0.001,
        ),
    }
)

# __all__ = ["FR3_TEK_LEFT_CONFIG", "FR3_TEK_LEFT_STABLE", "ROBOT_USD_PATH"]
