import sys
import gymnasium
sys.modules["gym"] = gymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from gym.envs.registration import register
import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import sumo_rl
from environment.env import SumoEnvironment
import traci
import sumolib

num_agents =4
sumo_binary = sumolib.checkBinary('sumo')
print(sumo_binary)
net = "nets/2x2grid/2x2.net.xml"
route="nets/2x2grid/2x2test.rou.xml"
def make_env():
    def _init():
        return SumoGymEnv(env_name, env_kwargs={"net_file": net, "route_file":route})
    return _init
# class MultiEnv(MultiAgentEnv):
#     def __init__(self, env):
#         self.env = env
#         self.observation_space = env.observation_space
#         self.action_space = env.action_space
#         self.n_agents = num_agents
#
#     def reset(self):
#         return self.env.reset()
#
#     def step(self, actions):
#         obs, reward, done, info = self.env.step(actions)
#         return obs, reward, done, info
def creator():
    return SumoEnvironment(net_file='nets/2x2/2x2.net.xml',
                            route_file=route,
                            out_csv_name='outputs/single_intersection/dqn',
                            single_agent=True,
                            use_gui=False,
                            num_seconds=100000,
                            sumo_warnings=False
                            )
if __name__ == '__main__':
    # Inizializzazione di SUMO e della rete stradale


    env = creator()
    # env1 = ss.pettingzoo_env_to_vec_env_v1(env)
    # env1 = ss.concat_vec_envs_v1(env1, 2, num_cpus=1, base_class='stable_baselines3')
    # env1 = VecMonitor(env1)


    # Caricamento del modello DQN con la libreria stable_baselines3
    model = DQN.load('dqn_singolo')



    # Addestramento del modello per 100.000 step

    # Valutazione del modello addestrato
    obs = env.reset()
    #env.reset()
    for i in range(100):
        action, _states = model.predict(obs)
        obs, rewards, dones,info = env.step(action)
        if dones:
            obs = env.reset()

    # Chiusura di SUMO
    traci.close()
