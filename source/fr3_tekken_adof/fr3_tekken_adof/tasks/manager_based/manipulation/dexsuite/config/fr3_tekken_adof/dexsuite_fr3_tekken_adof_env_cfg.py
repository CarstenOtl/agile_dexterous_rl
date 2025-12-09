# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# from isaaclab_assets.robots import KUKA_ALLEGRO_CFG
from fr3_tekken_adof.assets.fr3_tekken_left import FR3_TEK_LEFT_STABLE  # isort: skip

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp


@configclass
class Fr3TekkenAdofRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class Fr3TekkenAdofReorientRewardCfg(dexsuite.RewardsCfg):

    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb.
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=3.5,
        params={"threshold": 1.0},
    )

    lift_height = RewTerm(
        func=mdp.lift_height,
        weight=3,
        params={"lift_height": 0.06}
    )

    


@configclass
class Fr3TekkenAdofMixinCfg:
    rewards: Fr3TekkenAdofReorientRewardCfg = Fr3TekkenAdofReorientRewardCfg()
    actions: Fr3TekkenAdofRelJointPosActionCfg = Fr3TekkenAdofRelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "palm_link"
        self.scene.robot = FR3_TEK_LEFT_STABLE.replace(prim_path="{ENV_REGEX_NS}/Robot")
        finger_tip_body_list = [
            "Index_Distal_Phalanx", 
            "Middle_Distal_Phalanx", 
            "Ring_Distal_Phalanx", 
            "Pinky_Distal_Phalanx", 
            "Thumb_Distal_Phalanx",
        ]
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/tekken_left_adof/"+link_name ,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["base_link", ".*_Distal_Phalanx"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["base_link", ".*_Distal_Phalanx"])


@configclass
class DexsuiteFr3TekkenAdofReorientEnvCfg(Fr3TekkenAdofMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class DexsuiteFr3TekkenAdofReorientEnvCfg_PLAY(Fr3TekkenAdofMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    pass


@configclass
class DexsuiteFr3TekkenAdofLiftEnvCfg(Fr3TekkenAdofMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteFr3TekkenAdofLiftEnvCfg_PLAY(Fr3TekkenAdofMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass
