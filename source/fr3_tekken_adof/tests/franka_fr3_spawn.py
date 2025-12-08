# SPDX-License-Identifier: BSD-3-Clause
"""
Minimal script to spawn the Franka FR3 arm (no hand) using ManagerBasedEnv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ensure package import when running directly
_THIS_DIR = Path(__file__).resolve()
_PKG_DIR = _THIS_DIR.parents[1]
_REPO_DIR = _PKG_DIR.parent
for _p in (_PKG_DIR, _REPO_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Spawn Franka FR3 arm.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch app early
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

from omniisaacgymenvs.robots.articulations.franka_fr3 import FRANKA_FR3_CONFIG


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Scene with ground, light, and the FR3 arm."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot: ArticulationCfg = FRANKA_FR3_CONFIG.replace(prim_path="{ENV_REGEX_NS}/robot")


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
    """No actions applied; zero-dim placeholder."""

    pass


@configclass
class EnvCfg(ManagerBasedEnvCfg):
    """Minimal env config just to spawn the robot."""

    scene: SceneCfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        self.decimation = 1
        self.sim.dt = 0.005
        self.sim.render_interval = 1
        self.sim.device = args_cli.device
        self.viewer.eye = (2.5, 0.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


def main():
    env_cfg = EnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)

    action_dim = getattr(env.action_manager, "action_dim", 0)
    actions = torch.zeros((env.num_envs, action_dim), device=env.device)
    step = 0

    while simulation_app.is_running():
        if step % 500 == 0:
            env.reset()
            step = 0
            print("[INFO] Resetting environment...")
        env.step(actions)
        step += 1

    env.close()


if __name__ == "__main__":
    import torch

    main()
    simulation_app.close()
