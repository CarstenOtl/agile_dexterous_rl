# SPDX-License-Identifier: BSD-3-Clause
"""
Minimal script to spawn the FR3 Tekken ADoF (left) robot in a scene.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import torch

# add package to path for direct execution
_THIS_DIR = Path(__file__).resolve()
_PKG_DIR = _THIS_DIR.parents[1]
_REPO_DIR = _PKG_DIR.parent
for _p in (_PKG_DIR, _REPO_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Spawn FR3 Tekken ADoF (left) robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch app early (tutorial pattern)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from omniisaacgymenvs.robots.articulations.fr3_tekken_left import FR3_TEK_LEFT_CONFIG


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Scene with ground, light, and FR3 Tekken ADoF robot."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot: ArticulationCfg = FR3_TEK_LEFT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ObservationsCfg:
    """Minimal observation group to satisfy ManagerBasedEnvCfg."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """No-op actions (zero-dim) to satisfy ManagerBasedEnvCfg."""

    pass


@configclass
class EnvCfg(ManagerBasedEnvCfg):
    """Minimal environment just to spawn the robot."""

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

    # Use configured physics dt and decimation for timing (SimulationContext doesn’t expose dt attribute).
    sim_dt = env_cfg.sim.dt * env_cfg.decimation
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
    main()
    simulation_app.close()
