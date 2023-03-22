import gym
from stable_baselines3.dqn.dqn import DQN
import os
import sys
import gymnasium
sys.modules["gym"] = gymnasium
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
from sumo_rl import SumoEnvironment
import traci


if __name__ == '__main__':

    env = SumoEnvironment(net_file='nets/2x2/2x2.net.xml',
                            route_file='nets/2x2/routes.rou.xml',
                            out_csv_name='outputs/2x2/dqn',
                            single_agent=True,
                            use_gui=False,
                            num_seconds=100000,
                            sumo_warnings=False
                            )

    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.001,
        learning_starts=0,
        train_freq=1,
        target_update_interval=500,
        exploration_initial_eps=0.05,
        exploration_final_eps=0.01,
        verbose=1
    )
    model.learn(total_timesteps=100000)