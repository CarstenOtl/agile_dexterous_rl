# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

""" Dextra Franka Tekken environments.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# State Observation
gym.register(
    id="Tekken-Isaac-Dexsuite-fr3-tekken-adof-Reorient-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_fr3_tekken_adof_env_cfg:DexsuiteFr3TekkenAdofReorientEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteFr3TekkenAdofPPORunnerCfg",
    },
)

gym.register(
    id="Tekken-Isaac-Dexsuite-fr3-tekken-adof-Reorient-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_fr3_tekken_adof_env_cfg:DexsuiteFr3TekkenAdofReorientEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteFr3TekkenAdofPPORunnerCfg",
    },
)

# Dexsuite Lift Environments
gym.register(
    id="Tekken-Isaac-Dexsuite-fr3-tekken-adof-Lift-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_fr3_tekken_adof_env_cfg:DexsuiteFr3TekkenAdofLiftEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteFr3TekkenAdofPPORunnerCfg",
    },
)


gym.register(
    id="Tekken-Isaac-Dexsuite-fr3-tekken-adof-Lift-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_fr3_tekken_adof_env_cfg:DexsuiteFr3TekkenAdofLiftEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteFr3TekkenAdofPPORunnerCfg",
    },
)
