import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import pandas as pd
import ray
from ray import tune
from ray.rllib.algorithms.a3c.a3c import A3C
from ray.tune.registry import register_env
from gym.spaces import Discrete
#from ray.rllib.algorithms.a3c.a3c_tf_policy import A3CTF2Policy
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
from gym import spaces
import numpy as np
import sumo_rl
import traci

from stable_baselines3.dqn.policies import DQNPolicy

if __name__ == '__main__':
    ray.init()

    register_env("4x4grid", lambda _: PettingZooEnv(sumo_rl.env(net_file='nets/4x4/4x4.net.xml',
                                                                route_file='nets/4x4/4x4c1c2c1c2.rou.xml',
                                                                out_csv_name='outputs/4x4grid/a3c',
                                                                use_gui=False,
                                                                num_seconds=80000)))

    trainer = A3C(env="4x4grid", config={

        "multiagent": {
            "policies": {
                '0': ("MlpPolicy", spaces.Box(low=np.zeros(11), high=np.ones(11)), spaces.Discrete(2), {})
            },
            "policy_mapping_fn": (lambda id: '0')  # Traffic lights are always controlled by this policy
        },
        "lr": 0.001,
        "no_done_at_end": True
    })
    while True:
        print(trainer.train())  # distributed training step