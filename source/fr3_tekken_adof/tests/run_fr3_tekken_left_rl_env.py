# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the cartpole balancing task.

.. code-block:: bash

    python omniisaacgymenvs/tests/run_fr3_tekken_left_rl_env.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

# ensure package is importable when run directly
_THIS_DIR = Path(__file__).resolve()
_PKG_DIR = _THIS_DIR.parents[1]
_REPO_DIR = _PKG_DIR.parent
for _p in (_PKG_DIR, _REPO_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--task", type=str, default="Isaac-Lift-Cube-fr3-tekken-left-v0", help="Name of the registered gym environment."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import gymnasium as gym

from isaaclab_tasks.utils import parse_env_cfg

from omniisaacgymenvs.robots.articulations.fr3_tekken_left import FR3_TEK_LEFT_CONFIG
from omniisaacgymenvs.robots.articulations.fr3_tekken_left import FR3_TEK_LEFT_STABLE

import omniisaacgymenvs.tasks.manager_based.manipulation.lift.config.fr3_tekken_left  # noqa: F401

if args_cli.debug:
    print("[Debug] Fr3 Tekken configs loaded:", FR3_TEK_LEFT_CONFIG, FR3_TEK_LEFT_STABLE)


def main():
    """Main function."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
