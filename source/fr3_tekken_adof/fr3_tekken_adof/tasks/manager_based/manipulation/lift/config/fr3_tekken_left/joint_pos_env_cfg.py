# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from fr3_tekken_adof.tasks.manager_based.manipulation.lift import mdp
from fr3_tekken_adof.tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
# from omniisaacgymenvs.robots.articulations.fr3_tekken_left import FR3_TEK_LEFT_STABLE  # isort: skip
from fr3_tekken_adof.assets.fr3_tekken_left import FR3_TEK_LEFT_STABLE  # isort: skip


@configclass
class Fr3TekkenLeftCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set FR3 + Tekken left hand as robot
        self.scene.robot = FR3_TEK_LEFT_STABLE.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for FR3 arm + Tekken ADoF hand
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[r"fr3_joint[1-7]"], scale=0.2, use_default_offset=True
        )
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "revolute_Thumb_rot",
                "revolute_thumb_mcp_pitch",
                "revolute_thumb_mcp_yaw",
                "revolute_thumb_pip",
                "revolute_index_mcp_pitch",
                "revolute_index_mcp_yaw",
                "revolute_index_pip",
                "revolute_middle_mcp_pitch",
                "revolute_middle_mcp_yaw",
                "revolute_middle_pip",
                "revolute_ring_mcp_pitch",
                "revolute_ring_mcp_yaw",
                "revolute_ring_pip",
                "revolute_pinky_mcp_pitch",
                "revolute_pinky_mcp_yaw",
                "revolute_pinky_pip",
            ],
            scale=1.0, 
            use_default_offset=True,
        )
        # Set the body name for the end effector
        # TODO: 
        # - check if fr3_link7 is indeed the correct end effector link should be link8 or tekken_adof_baselink
        self.commands.object_pose.body_name = "fr3_link7"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/fr3_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fr3_link7",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )


@configclass
class Fr3TekkenLeftCubeLiftEnvCfg_PLAY(Fr3TekkenLeftCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
