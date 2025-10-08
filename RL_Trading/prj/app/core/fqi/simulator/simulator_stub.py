from core.fqi.simulator.dataset_builder import DatasetBuilder
from core.fqi.simulator.simulator import TTFEnv
import numpy as np
from pathlib import Path
import random

import pandas as pd

if __name__ == '__main__':

    data_root = '/home/a2a/a2a/RL_Trading/prj/app/core/data'
    config_path = '/home/a2a/a2a/RL_Trading/prj/app/config/config.yaml'
    persistence = 20 #min
    #data_builder = DatasetBuilder(data_root, config_path, persistence)
    data_path = Path('/home/a2a/a2a/RL_Trading/prj/app/core/data/dataset.parquet')

    env = TTFEnv(data_path=data_path)#


    n_eval_days = 10
    rewards = []
    obs = env.reset()

    for day in range(n_eval_days):
        print(f'Starting day {day}')
        done = False
        day_reward = 0
        ep_len = 0
        while not done:
            # action, _ = model.predict(obs, deterministic=True)
            action = random.choice([-5, 0, 5])


            obs, reward, terminated, _, info = env.step(action)
            done = terminated
            day_reward += reward
            ep_len+=1

        rewards.append(day_reward)

    print(f"Simulation complete. Mean episode reward={np.mean(rewards):.3f}, Return={np.sum(rewards):.3f}%, Episode length: {ep_len}")
