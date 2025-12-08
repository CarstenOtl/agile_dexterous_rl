# SPDX-License-Identifier: BSD-3-Clause
"""
Spawn the FR3 Tekken ADoF (left) robot into a physics scene without managers,
so you can inspect it with the Physics Inspector.
"""

import argparse
import sys
import math
from pathlib import Path
import torch
from typing import Optional

# # ensure package is importable when run directly
# _THIS_DIR = Path(__file__).resolve()
# _PKG_DIR = _THIS_DIR.parents[1]
# _REPO_DIR = _PKG_DIR.parent
# for _p in (_PKG_DIR, _REPO_DIR):
#     _s = str(_p)
#     if _s not in sys.path:
#         sys.path.insert(0, _s)


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

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch app first
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# from omniisaacgymenvs.robots.articulations.fr3_tekken_left import FR3_TEK_LEFT_CONFIG
# from omniisaacgymenvs.robots.articulations.fr3_tekken_left import FR3_TEK_LEFT_STABLE

from fr3_tekken_adof.assets.fr3_tekken_left import FR3_TEK_LEFT_STABLE


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Simple scene with ground, light, and the robot."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
    )
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

    # arm = mdp.JointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #         "fr3_joint1",
    #         "fr3_joint2",
    #         "fr3_joint3",
    #         "fr3_joint4",
    #         "fr3_joint5",
    #         "fr3_joint6",
    #         "fr3_joint7",
    #     ],
    #     scale=1.0,
    #     use_default_offset=True,
    # )

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
class EnvCfg(ManagerBasedEnvCfg):
    """Minimal env config just to spawn the robot."""

    scene: SceneCfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        self.decimation = 1
        # 100 Hz physics
        self.sim.dt = 0.005
        self.sim.render_interval = 1
        self.sim.device = args_cli.device
        self.viewer.eye = (2.5, 0.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        self.sim.device = "cpu"
        
        


def main():
    env_cfg = EnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)

    # action dimensions from manager
    mcp_joints = [
        "revolute_index_mcp_pitch",
        "revolute_middle_mcp_pitch",
        "revolute_ring_mcp_pitch",
        "revolute_pinky_mcp_pitch",
    ]
    pip_joints = [
        "revolute_index_pip",
        "revolute_middle_pip",
        "revolute_ring_pip",
        "revolute_pinky_pip",
    ]
    thumb_joints = ["revolute_Thumb_rot"]

    # compute action dimension and offsets
    mcp_offset = 0
    pip_offset = mcp_offset + len(mcp_joints) #4
    thumb_offset = pip_offset + len(pip_joints) #8
    action_dim = getattr(env.action_manager, "total_action_dim", thumb_offset + len(thumb_joints))

    # create action tensor
    actions = torch.zeros((env.num_envs, action_dim), device=env.device)
    print(f"[INFO] Action manager: dim={action_dim}, terms={getattr(env.action_manager, 'term_names', [])}")

    # oscillate finger joints within [0, 60] deg around a 30 deg offset
    offset = math.radians(30.0)
    amp = math.radians(30.0)
    freq_hz = 0.5
    step_dt = env_cfg.decimation * env_cfg.sim.dt

    # indices into action vector for pinky entries
    pinky_mcp_idx = mcp_joints.index("revolute_pinky_mcp_pitch")
    pinky_pip_idx = pip_joints.index("revolute_pinky_pip")

    # joint-name to index for observation printing
    robot_joint_names = env.scene["robot"].data.joint_names
    obs_indices = {
        name: robot_joint_names.index(name) for name in mcp_joints + pip_joints + thumb_joints if name in robot_joint_names
    }

    # freeze base: capture default root pose and zero velocity
    frozen_root_pose = env.scene["robot"].data.default_root_state[:, :7].clone()
    frozen_root_vel = torch.zeros_like(env.scene["robot"].data.default_root_state[:, 7:])

    step = 0
    
    if args_cli.debug:
        print(f"[Debug] Action Tensor before sim start: {actions}")
    
    print("[INFO] Mode:", args_cli.mode)

    while simulation_app.is_running():
        if step % 500 == 0:
            env.reset()
            step = 0
            print("[INFO] Resetting environment...")
        if actions.numel() > 0:
            # lock the base/root each step
            # env.scene["robot"].write_root_pose_to_sim(frozen_root_pose)
            # env.scene["robot"].write_root_velocity_to_sim(frozen_root_vel)
            t = step * step_dt
            # compute commands
            cmd_rad = offset + amp * math.sin(2.0 * math.pi * freq_hz * t)
            thumb_cmd = math.radians(15.0) * math.sin(2.0 * math.pi * freq_hz * t)

            actions.zero_()
            if args_cli.mode == "defaults":
                # hold all actuated finger joints (MCP/PIP and thumb rot) at zero offsets
                actions[:, mcp_offset : mcp_offset + len(mcp_joints)] = 0.0
                actions[:, pip_offset : pip_offset + len(pip_joints)] = 0.0
                actions[:, thumb_offset] = 0.0
            elif args_cli.mode == "pinky":
                actions[:, mcp_offset + pinky_mcp_idx] = cmd_rad
                actions[:, pip_offset + pinky_pip_idx] = cmd_rad
            else:
                # fingers mode: open/close all non-thumb fingers on MCP/PIP, thumb rot oscillates
                actions[:, mcp_offset : mcp_offset + len(mcp_joints)] = cmd_rad
                actions[:, pip_offset : pip_offset + len(pip_joints)] = cmd_rad
                actions[:, thumb_offset] = thumb_cmd

            if args_cli.debug:
                print(f"[Debug] Action Tensor: {actions}")

            env.step(actions)
            if args_cli.debug:
                # print commanded and observed joint values for env 0
                cmd_readout = []
                for i, name in enumerate(mcp_joints):
                    cmd_readout.append((name, actions[0, mcp_offset + i].item()))
                for i, name in enumerate(pip_joints):
                    cmd_readout.append((name, actions[0, pip_offset + i].item()))
                cmd_readout.append((thumb_joints[0], actions[0, thumb_offset].item()))
                obs_readout = []
                for name, idx in obs_indices.items():
                    obs_readout.append((name, env.scene["robot"].data.joint_pos[0, idx].item()))
                cmd_str = ", ".join(f"{n}: {v:+.3f}" for n, v in cmd_readout)
                obs_str = ", ".join(f"{n}: {v:+.3f}" for n, v in obs_readout)
                print(f"[CMD] {cmd_str}")
                print(f"[OBS] {obs_str}\n")
        else:
            env.step(None)
        step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
