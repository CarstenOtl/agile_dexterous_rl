# SPDX-License-Identifier: BSD-3-Clause
"""
Spawn the FR3 Tekken ADoF (left) robot into a scene for inspection only.
Mirrors the minimal InteractiveScene pattern from tutorials/02_scene/create_scene.py.
"""

import argparse
import sys
from pathlib import Path

import torch

# # ensure package is importable when run directly
# _THIS_DIR = Path(__file__).resolve()
# _PKG_DIR = _THIS_DIR.parents[1]
# _REPO_DIR = _PKG_DIR.parent
# for _p in (_PKG_DIR, _REPO_DIR):
#     _s = str(_p)
#     if _s not in sys.path:
#         sys.path.insert(0, _s)

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Spawn FR3 Tekken ADoF left.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# parser.add_argument(
#     "--device",
#     type=str,
#     default="cpu",
#     help="Device to run the simulation on.",
# )
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch app first
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

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


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """Runs a minimal simulation loop; no actions applied."""
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            # reset root pose/vel around env origins
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # reset joints to defaults
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print("[INFO] Resetting robot state...")

        # Write any pending data (none by default) and step sim
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01, render_interval=1)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 1.5], [0.0, 0.0, 0.5])

    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] Scene setup complete.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
