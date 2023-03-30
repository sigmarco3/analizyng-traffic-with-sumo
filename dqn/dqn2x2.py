import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import traci

# Inizializzazione di SUMO e della rete stradale
traci.start(["sumo", "-c", "path_to_sumocfg_file"])
env = make_vec_env("sumo", env_kwargs={"sumocfg_file_path": "path_to_sumocfg_file"}, n_envs=1)

# Creazione di un modello DQN con la libreria stable_baselines3
model = DQN('MlpPolicy', env, verbose=1)

# Definizione di una funzione di callback per salvare i checkpoint del modello
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./logs/')

# Addestramento del modello per 100.000 step
model.learn(total_timesteps=100000, callback=checkpoint_callback)

# Valutazione del modello addestrato
obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        obs = env.reset()

# Chiusura di SUMO
traci.close()
