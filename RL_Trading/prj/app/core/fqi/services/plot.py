from RL_Trading.prj.app.core.fqi.services.plotter import Plotter
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap



project_root = f'{os.path.dirname(os.path.abspath(__file__))}/../../../../..'
#study_name = "FQI_1718_19_starting_9"
save_path = f"{project_root}/results/FQI_1718_19_multiseed_double_fqi_optimized_trajectory"#os.path.join(project_root, f"prj/app/core/results/{study_name}")
n_iterations = 4 + 1 #len([f for f in os.listdir(os.path.join(save_path, "seed0")) if f.endswith(".pkl")])
seeds = [int(f.split("d")[1]) for f in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, f)) and f != "tmp"]
for seed in seeds:
    #Plotter.plot_percentage_returns_unfilled_specific_iteration(60, 3, seeds, save_path)
    #plot_percentage_returns_specific_iteration(60, 5, seeds, save_path)


    iteration = 5
    df = pd.read_csv(os.path.join(save_path, f"seed{seed}", f"Results_iter{iteration}_Validation.csv"))
    print(df.head())

    fig, ax = plt.subplots(figsize=(32, 18))
    ax.set_title("Validation return")
    ax.set_ylabel('P&L (% of max allocation)')
    #fig.savefig(os.path.join(save_path, "test.png"))

    trajectories = range(df['trajectory'].max())
    fig, ax = plt.subplots(figsize=(16, 9))
    for traj in trajectories:
        df_traj = df[df['trajectory'] == traj][["reward", "day", "minute"]].reset_index(drop = True)
        df_traj['time'] = (df_traj['minute'] // 60) * 100 + df_traj['minute'] % 60
        df_traj['timestamp'] = pd.to_datetime(df_traj['day'].astype(str) + df_traj['time'].astype(str), format='%Y%m%d%H%M')
        index = df_traj['timestamp']
        ax.scatter(traj, df_traj['reward'].cumsum().iloc[-1])
        ax.set_ylabel('final P&L (% of max allocation)')
        ax.set_xlabel("Trajectory")
        ax.axhline(y=0, color='grey', linestyle='-')
    fig.savefig(os.path.join(save_path, f"test_scatter_seed{seed}_it{iteration}_validation_2020.png"))
