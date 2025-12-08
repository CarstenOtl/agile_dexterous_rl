"""Test script to spawn FR3 Tekken ADoF left robot with ManagerBasedRLEnv."""


import argparse
import sys
import math
from pathlib import Path
import torch
from typing import Optional

# ensure package is importable when run directly
_THIS_DIR = Path(__file__).resolve()
_PKG_DIR = _THIS_DIR.parents[1]
_REPO_DIR = _PKG_DIR.parent
for _p in (_PKG_DIR, _REPO_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Spawn FR3 Tekken ADoF left.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--mode",
    type=str,
    choices=["pinky", "fingers", "defaults"],
    default="pinky",
    help="Motion mode: pinky only (default), open/close all fingers with thumb rot oscillation, or reset to defaults.",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Print commanded and observed joint values each step.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch app first
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from omniisaacgymenvs.robots.articulations.fr3_tekken_left import FR3_TEK_LEFT_CONFIG
from omniisaacgymenvs.robots.articulations.fr3_tekken_left import FR3_TEK_LEFT_STABLE


def _resolve_entity_position(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return mean body position (or root) of an entity in env coordinates."""
    asset = env.scene[asset_cfg.name]
    body_names = getattr(asset_cfg, "body_names", None) or []
    if hasattr(asset.data, "body_names") and hasattr(asset.data, "body_pos_w") and body_names:
        indices = [asset.data.body_names.index(name) for name in body_names if name in asset.data.body_names]
        if indices:
            return asset.data.body_pos_w[:, indices, :].mean(dim=1) - env.scene.env_origins
    return asset.data.root_pos_w - env.scene.env_origins


def hand_to_prism_distance(env, hand_cfg: SceneEntityCfg, prism_cfg: SceneEntityCfg) -> torch.Tensor:
    """L2 distance between the hand (or robot root fallback) and prism."""
    hand_pos = _resolve_entity_position(env, hand_cfg)
    prism_pos = _resolve_entity_position(env, prism_cfg)
    return torch.norm(hand_pos - prism_pos, dim=-1)


def prism_height_above_table(env, prism_cfg: SceneEntityCfg, table_height: float = 0.0) -> torch.Tensor:
    """Height of the prism root above a table plane."""
    prism_pos = _resolve_entity_position(env, prism_cfg)
    return prism_pos[:, 2] - table_height


def _upright_vector(quat: torch.Tensor) -> torch.Tensor:
    """Return world-space up vector of an XYZW quaternion."""
    up = torch.tensor((0.0, 0.0, 1.0), device=quat.device, dtype=quat.dtype)
    q_vec = quat[..., :3]
    w = quat[..., 3:4]
    uv = torch.cross(q_vec, up.expand_as(q_vec), dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return up + 2.0 * (w * uv + uuv)


def prism_upright_alignment(env, prism_cfg: SceneEntityCfg) -> torch.Tensor:
    """Dot product of prism up axis with world up (1 means fully upright)."""
    up_vec = _upright_vector(env.scene[prism_cfg.name].data.root_quat_w)
    return up_vec[:, 2]


def prism_angular_velocity_l2(env, prism_cfg: SceneEntityCfg) -> torch.Tensor:
    """Angular velocity magnitude of the prism."""
    ang_vel = env.scene[prism_cfg.name].data.root_ang_vel_w
    return torch.norm(ang_vel, dim=-1)


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Simple scene with ground, light, and the robot."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
    )
    
    # Rigid Object
    cuboid_cfg = RigidObjectCfg(
        prim_path="/World/prism",
        spawn=sim_utils.CuboidCfg(
            size = (0.05, 0.05, 0.1),
            color = (0.8, 0.1, 0.1),
            rigid_props = sim_utils.RigidBodyPropertiesCfg(),
            mass_props = sim_utils.MassPropertiesCfg(),
            collision_props = sim_utils.CollisionPropertiesCfg(),
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1), metallic=0.2),
        ), 
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.05),
        ),
    )
    cuboid_object = RigidObject(cfg=cuboid_cfg)

    # Robot
    robot: ArticulationCfg = FR3_TEK_LEFT_STABLE.replace(prim_path="{ENV_REGEX_NS}/robot")
    

@configclass
class ObservationsCfg:
    """Minimal observations to satisfy ManagerBasedEnvCfg."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action manager: joint position targets for arm, fingers, and thumb rotation."""

    arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
        ],
        scale=1.0,
        use_default_offset=True,
    )

    finger_mcp_pitch = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "revolute_index_mcp_pitch",
            "revolute_middle_mcp_pitch",
            "revolute_ring_mcp_pitch",
            "revolute_pinky_mcp_pitch",
        ],
        scale=1.0,
        use_default_offset=True,
    )

    finger_pip = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "revolute_index_pip",
            "revolute_middle_pip",
            "revolute_ring_pip",
            "revolute_pinky_pip",
        ],
        scale=1.0,
        use_default_offset=True,
    )

    thumb_rot = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["revolute_Thumb_rot"],
        scale=1.0,
        use_default_offset=True,
    )

@configclass
class EventCfg:
    """ configuration for events.""" 

    #reset fr3 
    reset_fr3_position = EventTerm(
        func=mdp.reset_articulation_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", 
                joint_names=[
                    "fr3_joint1",
                    "fr3_joint2",
                    "fr3_joint3",
                    "fr3_joint4",
                    "fr3_joint5",
                    "fr3_joint6",
                    "fr3_joint7",
                ]),
            "position_range": (-0.2 , 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )
    
    #reset ADoF hand
    reset_hand_position = EventTerm(
        func=mdp.reset_articulation_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot",
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
                ]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )   


@configclass
class RewardsCfg:
    """ configuration for rewards.""" 
    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
   
    # (3) Encourage hand to move towards a "closed" posture (coarse grasp shaping)
    hand_closure = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-0.2,  # negative weight since mdp.joint_pos_target_l2 returns a positive distance
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "revolute_index_mcp_pitch",
                    "revolute_middle_mcp_pitch",
                    "revolute_ring_mcp_pitch",
                    "revolute_pinky_mcp_pitch",
                    "revolute_index_pip",
                    "revolute_middle_pip",
                    "revolute_ring_pip",
                    "revolute_pinky_pip",
                ],
            ),
            # "closed" target (rad) – tune this based on your joint limits
            "target": 0.9,
        },
    )
    
    # (4) Penalize high joint velocities (all robot joints)
    joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    hand_to_prism = RewTerm(
        func=hand_to_prism_distance,
        weight=-1.0,
        params={"hand_cfg": SceneEntityCfg("robot"), "prism_cfg": SceneEntityCfg("prism")},
    )

    prism_height = RewTerm(
        func=prism_height_above_table,
        weight=0.5,
        params={"prism_cfg": SceneEntityCfg("prism"), "table_height": 0.0},
    )

    prism_upright = RewTerm(
        func=prism_upright_alignment,
        weight=0.25,
        params={"prism_cfg": SceneEntityCfg("prism")},
    )

    prism_angular_stability = RewTerm(
        func=prism_angular_velocity_l2,
        weight=-0.05,
        params={"prism_cfg": SceneEntityCfg("prism")},
    )


@configclass
class TerminationsCfg:
    """ configuration for terminations.""" 
    
    # (1) time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # (2) failure condition
    # TODO: add failure conditions: 
    # - object dropped (height below threshold)
    # - excessive joint limits violation
    # - robot out of bounds
    # failure = DoneTerm(func=mdp.is_terminated, done_if_true=True)
    
## Environment config

@configclass
class Fr3Tekken_EnvCfg(ManagerBasedRLEnvCfg):
    """Minimal env config just to spawn the robot."""

    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    # Basic Settings
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP Settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5
        # 100 Hz physics
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.5, 0.0, 1.5)
        # self.viewer.lookat = (0.0, 0.0, 0.5)
        self.sim.device = "cuda:0"
        
