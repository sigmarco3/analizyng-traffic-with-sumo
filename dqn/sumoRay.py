import argparse
from copy import deepcopy
import logging
import os
import pathlib
from pprint import pformat

import ray
from ray import air, tune

from ray.rllib.algorithms.ppo import ppo
#from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF1POLICY
from ray.rllib.examples.simulators.sumo import marlenvironment
from ray.rllib.utils.test_utils import check_learning_achieved
import sumo_rl
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("ppotrain")

num_agents = 4
net = "nets/2x2grid/2x2.net.xml"
route="nets/2x2grid/2x2test.rou.xml"
n_envs=4
n_steps = 100000
batch_size = 64
learning_rate = 0.001
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("ppotrain")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=10, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=1000000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward",
    type=float,
    default=30000.0,
    help="Reward at which we stop training.",
)
def env_creator():
    return sumo_rl.parallel_env(net_file=net,
                          route_file=route,
                          out_csv_name="outputs/dqn/2x2.csv",
                          use_gui=False,
                          num_seconds=3600,

                          single_agent=False)

if __name__ == "__main__":
    ray.init()
    args = parser.parse_args()
    tune.register_env("sumo_test_env", env_creator)
    # Algorithm.
    #policy_class = ppo.PPOTF1Policy
    config = (
        ppo.PPOConfig()
            .framework("tf")
            .rollouts(
            batch_mode="complete_episodes",
            num_rollout_workers=0,
        )
            .training(
            gamma=0.99,
            lambda_=0.95,
            lr=0.001,
            sgd_minibatch_size=256,
            train_batch_size=4000,
        )
            .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
            .reporting(min_time_s_per_iteration=5)
    )
    env=env_creator()
    env.reset()
    policy_class = "MlpPolicy"
    policies = {}
    for agent in env.aec_env.agents:
        agent_policy_params = {}
        policies[agent] = (
            policy_class,
            env.observation_space,
            env.action_space,
            agent_policy_params,
        )
    config.multi_agent(
        policies=policies,
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: agent_id,
        policies_to_train=["ppo_policy"],
    )
    config.environment("sumo_test_env")
    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }
    # results = tune.Tuner(
    #     "PPO",
    #     param_space=config,
    #     run_config=air.RunConfig(
    #         stop=stop,
    #         verbose=1,
    #         checkpoint_config=air.CheckpointConfig(
    #             checkpoint_frequency=10,
    #         ),
    #     ),
    # ).fit()
    trainer = config.build(env=env)
    trainer.train()
    # And check the results.
    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()