import os
import sys
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.normpath(os.path.join(current_dir, '..', '..', '..', '..', '..','..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..', 'RL_Trading', 'prj', 'app')))


from RL_Trading.prj.app.core.fqi.services.plotter import Plotter
import pandas as pd

if __name__ == '__main__':

    paths = ['/data2/bonetti_a2a/a2a_riccardo/bin/xgb_delta_both_0.0005_pers10_nest250_span1_featstd_optnostd_retrain/',
              '/data2/bonetti_a2a/a2a_riccardo/RL_Trading/results/xgb_delta_both_0.0005_pers10_nest250_span1_featstd_optnostd/']

    #Results_iter3_Validation.csv
    minutes_a = [2]
    minutes_b = [1]

    paths_0 = []
    rewards_0 = []
    for elem in os.listdir(paths[0]):
        if elem.startswith('seed'):
            path = paths[0] + elem + '/Results_iter3_Validation.csv'
            paths_0.append(path)
            df = pd.read_csv(path)
            cum_reward = df['reward_nostd'].sum()
            rewards_0.append(cum_reward)

    paths_1 = []
    rewards_1 = []
    for elem in os.listdir(paths[1]):
        if elem.startswith('seed'):
            path = paths[1] + elem + '/Results_iter5_Validation.csv'
            paths_1.append(path)
            df = pd.read_csv(path)
            cum_reward = df['reward_nostd'].sum()
            rewards_1.append(cum_reward)

    print(rewards_0[np.argsort(rewards_0)[len(rewards_0)//2]])
    print(rewards_1[np.argsort(rewards_1)[len(rewards_1)//2]])

    path_median_0 = paths_0[np.argsort(rewards_0)[len(rewards_0)//2]]
    path_median_1 = paths_1[np.argsort(rewards_1)[len(rewards_1)//2]]

    print(path_median_0)
    print(path_median_1)

    results_a = pd.read_csv(path_median_0)
    results_b = pd.read_csv(path_median_1)

    Plotter.plot_action_diff_rewards(results_a, results_b, minutes_a, minutes_b, 'validation', iteration=5, save_path='/data2/bonetti_a2a/a2a_riccardo/RL_Trading/results/policies_comparison_10min_both_optnostd')