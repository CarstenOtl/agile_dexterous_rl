# Description

Repo for Semester/Master Thesis on dexterous manipulation using reinforcement learning in Isaaclab using the agile hands

# Version

- isaaclab 2.3.0
- isaacsim 5.1.0

# Dependencies

- rl_games

# Install

0. follow installation guide of isaaclab to install (recommended use of conda env)
0. activate isaaclab conda env
0. clone this repo
0. cd into the repo `cd agile_dexterous_rl`
0. cd into source folder `cd source/fr3_tekken_adof`
0. pip install -e .
0. sanity check: `pip list | grep -i fr3`

# Usage

## listing custom environments for agile hand + fr3

```
python scripts/environments/list_envs.py
```

all isaaclab predefined envs are also in here but commented out in the listing script.
