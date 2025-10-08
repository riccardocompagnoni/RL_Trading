import json
import pickle
import shutil
import typing
from joblib import Parallel, delayed
from random import Random
import numpy.typing as npt

import optuna
from optuna.samplers import TPESampler, RandomSampler
from tqdm import tqdm
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import math
import numpy as np
from pathlib import Path
from typing import List
import matplotlib.ticker as mtick
import sys
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates

from core.fqi.simulator.simulator2 import TTFEnv

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.normpath(os.path.join(current_dir, '..', '..', '..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'RL_Trading', 'prj', 'app')))
from RL_Trading.prj.app.core.oamp.config import ConfigOAMP
from RL_Trading.prj.app.core.oamp.oamp import OAMP
from RL_Trading.prj.app.core.fqi.simulator.dataset_builder import DatasetBuilder



class Objective:
    def __init__(self):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = os.path.normpath(os.path.join(current_dir, '..', '..', '..', '..', '..'))

        self.data_root = os.path.join(self.root, 'RL_Trading/prj/app/core/data')
        self.dataset_path = os.path.join(self.data_root, 'dataset.parquet')
        self.config_path = os.path.join(self.root, 'RL_Trading/prj/app/config/config.yaml')
        self.experiment_path = os.path.join(self.root, 'RL_Trading/results/asym_0.025/')
        self.save_path = Path(os.path.join(self.experiment_path, 'oamp_2seeds'))
        self.save_path.mkdir(exist_ok=True)

        self.persistence = 30
        #self.iteration = 5
        self.phase = 'test'
        self.remove_costs = False
        self.concatenate_test = True
        self.no_overnight = False


        #DatasetBuilder(data_root=self.data_root, config_path=self.config_path, years=[2023], persistence=self.persistence).build_dataset()

        self.start_date = np.datetime64('2023-07-01', 'D')
        self.end_date = np.datetime64('2024-10-01', 'D')

        self.date_range = [self.start_date, self.end_date]
        val_test_split_date = np.datetime64('2024-01-10', 'D')
        convergence_split_date = np.datetime64('2024-04-01', 'D')
        self.convergence_split_date = convergence_split_date

        def to_date(dt64):
            return dt64.astype('M8[D]').astype(object)

        phases_dict = {
            'Train': [[to_date(self.start_date), to_date(val_test_split_date - np.timedelta64(1, 'D'))]],
            'Validation': [[to_date(val_test_split_date), to_date(convergence_split_date - np.timedelta64(1, 'D'))]],
            'Test': [[to_date(convergence_split_date), to_date(self.end_date)]]
        }
        print(phases_dict)


        self.seeds = []
        for dir in os.listdir(self.experiment_path):
            for dir in os.listdir(os.path.join(self.experiment_path, dir)):
                if dir.startswith('seed'):
                    self.seeds.append(int(dir.replace('seed', '')))

        self.seeds_labels = [str(seed) for seed in self.seeds]
        seeds_palette = sns.color_palette("tab10", len(self.seeds))
        self.seeds_palette_dict = {str(seed): color for seed, color in zip(self.seeds, seeds_palette)}

        policy_paths = []
        trajectories = []

        #sub = Path(self.experiment_path)
        for sub in Path(self.experiment_path).iterdir():
            params = sub / "parameters_opt.json"
            if not params.exists():
                continue
            params = json.loads(params.read_text())
            it = params["iterations"]

            for seed in sub.glob("seed*"):
                p = seed / f"Policy_iter{it}.pkl"
                policy_paths.append(p)
                trajectories.append(params['trajectory_number'])

        """
        for root, _, files in os.walk(self.experiment_path):
            for fname in files:
                if fname.endswith('.pkl'):
                    policy_paths.append(os.path.join(root, fname))
        """
        dataset_path = Path(os.path.join(self.data_root, 'dataset_full.parquet'))
        env = TTFEnv(data_path=dataset_path, policy_path=None,
                     trajectory_number=22, no_overnight=self.no_overnight, date_range=self.date_range, phases_dates=phases_dict)

        self.test_env =env.clone()
        self.times, self.seeds_allocations, self.seeds_rewards, self.seeds_percentages = self.test_policies(policy_paths=policy_paths, env=env, trajectories=trajectories)
        self.seeds_percentages = np.nan_to_num(self.seeds_percentages)
        self.seeds_rewards = np.nan_to_num(self.seeds_rewards)

        trading_hours = [8, 16]
        self.episode_length = math.ceil(((trading_hours[1] - trading_hours[0]) * 60) / self.persistence)
        print(f'episode length: {self.episode_length}')

        dates = self.times.astype('datetime64[D]')
        self.val_test_index = np.searchsorted(dates, val_test_split_date, side='left')
        self.convergence_index = np.searchsorted(dates, convergence_split_date, side='left')

        self.seeds_allocations, self.seeds_rewards, self.seeds_percentages, self.seeds = self.select_best_seeds(n_seeds=2, original_seeds=5)
        self.seeds_labels = [str(seed) for seed in self.seeds]
        self.seeds_palette_dict = {str(seed): color for seed, color in zip(self.seeds, seeds_palette)}

    def select_best_seeds(self, n_seeds, original_seeds):

        scores = self.seeds_percentages[:, :self.val_test_index].sum(axis=1)
        best_idx = []
        for base in range(0, scores.size, original_seeds):
            group = scores[base:base + original_seeds]
            top = np.argsort(group, kind="stable")[-n_seeds:][::-1]
            best_idx.extend(base + top)

        seeds = [self.seeds[i] for i in best_idx]
        return self.seeds_allocations[best_idx, :], self.seeds_rewards[best_idx, :], self.seeds_percentages[best_idx, :], seeds

    @staticmethod
    def calculate_percentage(df):
        rewards, percentages, pos, entry = [], [], 0, np.nan
        for action, bid, ask in df[['action', 'bid_0', 'ask_0']].values:
            perc = 0
            rew = 0
            if pos and action != pos:
                perc = (bid - 0.00625) / entry - 1 if pos == 5 else -((ask + 0.00625) / entry - 1)
                rew = perc*5*entry
                entry = np.nan
            if action and action != pos:
                entry = (bid - 0.00625) if action == -5 else (ask + 0.00625)
            pos = action
            percentages.append(perc)
            rewards.append(rew)
        return percentages, rewards

    def test_policies(self, env, trajectories, policy_paths=None, actions=None):

        seeds_allocations, seeds_rewards, seeds_percentages = [], [], []

        def env_test_wrapper(original_env, policy, trajectory):
            #env = original_env.clone()
            #env.set_policy(policy)
            with open(policy, 'rb') as load_file:
                policy = pickle.load(load_file)
            return original_env.test(policy=policy, trajectory_number=trajectory, return_full=True)

        def env_test_actions_wrapper(original_env, actions, trajectory):

            return original_env.test(actions=actions, trajectory_number=trajectory, return_full=True)

        if actions is not None:
            actual_n_jobs = min(actions.shape[0], os.cpu_count())
            results = Parallel(n_jobs=actual_n_jobs, backend='loky', verbose=10)(
                delayed(env_test_actions_wrapper)(original_env=env, actions=act) for act in actions
            )

        elif len(policy_paths) >= 3:
            actual_n_jobs = min(len(policy_paths), os.cpu_count())
            results = Parallel(n_jobs=actual_n_jobs, backend='loky', verbose=10)(
                delayed(env_test_wrapper)(original_env=env, policy=p, trajectory=trajectory) for p, trajectory in zip(policy_paths, trajectories)
            )
        else:
            results = [env_test_wrapper(original_env=env, policy=p, trajectory=trajectory) for p, trajectory in zip(policy_paths, trajectories)]

        for (times_i, actions_i, rewards_i, out_i) in results:
            seeds_rewards.append(rewards_i)
            seeds_allocations.append(actions_i)
            seeds_percentages.append(out_i)
            times_last = times_i



        return np.array(times_last), np.array(seeds_allocations), np.array(seeds_rewards), np.array(seeds_percentages)


    def read_results_old(self):

        data_path = os.path.join(self.root, 'RL_Trading/prj/app/core/data/M1_ICE.csv')
        data_df = pd.read_csv(data_path)
        data_df = data_df[['timestamp', 'ask_0', 'bid_0']]
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
        data_df['mid'] = (data_df['ask_0'] + data_df['bid_0'])/2
        data_df['spread'] = data_df['ask_0'] - data_df['bid_0']
        data_df['next_mid'] = data_df['mid'].shift(-self.persistence)
        data_df['next_spread'] = data_df['spread'].shift(-self.persistence)

        results_paths = [[os.path.join(self.experiment_path, f'seed{seed}', f'Results_iter{self.iteration}_{self.phase}.csv')] for seed in self.seeds]

        if self.concatenate_test:
            results_paths = [[result[0], os.path.join(self.experiment_path, 'test_next_year_no_retrain', f'seed{seed}', f'Results_iter{self.iteration}_Test.csv')]
                             for result, seed in zip(results_paths, self.seeds)]



        rewards = []
        allocations = []
        percentages = []

        first=True
        for result in results_paths:
            dfs = []
            for seed in result:
                df = pd.read_csv(seed)
                trajectories = df.trajectory.unique()
                df = df[df.trajectory==trajectories[0]]


                df['date'] = pd.to_datetime(df['day'], format='%Y%m%d')
                delta = pd.to_timedelta(df['minute'], unit='m')
                df['timestamp'] = df['date'] + delta
                df = df[['timestamp', 'action', 'reward_nostd']]


                last_per_day = df.groupby(df['timestamp'].dt.date)['timestamp'].max()
                extra_rows = pd.DataFrame({
                    'timestamp': last_per_day + pd.Timedelta(minutes=self.persistence),
                    'action': 0,
                    'reward_nostd': 0
                })


                df = (pd.concat([df, extra_rows], ignore_index=True)
                           .sort_values('timestamp')
                           .reset_index(drop=True))

                dfs.append(df)

            df = (pd.concat(dfs, ignore_index=True)
                    .sort_values('timestamp')
                    .reset_index(drop=True))

            allocations.append(df.action.to_numpy())

            if first:
                data_df = data_df[data_df['timestamp'].isin(df['timestamp'].unique())]

            df = df.merge(data_df, on='timestamp', how='left')
            df = df.set_index('timestamp')

            if first:
                self.bid_ask = df[['ask_0', 'bid_0']]
                first = False


            if self.remove_costs:
                #df['mid'] = (df['bid_0'] + df['ask_0'])/2
                out, pos, entry = [], 0, np.nan
                for action, mid in df[['action', 'mid']].values:
                    perc = 0
                    if pos and action != pos:
                        perc = mid/entry - 1 if pos == 5 else -(mid/entry - 1)
                        entry = np.nan
                    if action and action != pos:
                        entry = mid
                    pos = action
                    out.append(perc)

            else:
                out, rews = self.calculate_percentage(df)
            
            percentages.append(out)
            rewards.append(rews)

        rewards = np.array(rewards)
        allocations = np.array(allocations)
        percentages = np.array(percentages)
        times = df.index.to_numpy()

        """
        from scipy.ndimage import shift

        for i in range(len(self.seeds)):
            prev_allocation = allocations[i, :]
            curr_allocation = shift(prev_allocation, -1, cval=0)
            prev_mid = df['mid'].to_numpy()
            curr_mid = df['next_mid'].to_numpy()
            prev_spread = df['spread']
            rewards[i, :] = curr_allocation * (curr_mid - prev_mid) - abs(curr_allocation - prev_allocation) * (prev_spread * 0.5 + 0.00625)
           plt.figure(figsize=(15, 8))
        for i in range(10):
            plt.plot(times, np.cumsum(rewards[i, :]), label=f'reward_{i}')
        plt.legend()
        plt.xticks(rotation=90)
        plt.show()

        plt.figure(figsize=(15, 8))
        for i in range(10):
            plt.plot(times, np.cumsum(percentages[i, :]), label=f'percentage_{i}')
        plt.legend()
        plt.xticks(rotation=90)
        plt.show()
        """

        print('Number of nan in percentages: ', len(percentages[np.isnan(percentages)]))
        percentages[np.isnan(percentages)] = 0


        return times, allocations, rewards, percentages

    @staticmethod
    def sample_oamp_params(trial: optuna.Trial):

        params = {
            #'agents_weights_upd_freq': trial.suggest_int('agents_weights_upd_freq', low=1, high=7),
            'agents_sample_freq': trial.suggest_int('agents_sample_freq', low=1, high=7),
            'agents_weights_upd_freq': 1,
            #'agents_sample_freq': 1,
            'loss_fn_window': trial.suggest_int('loss_fn_window', low=15, high=25),
            #'loss_fn_window': -1,
            #'loss_fn_window': 15,
            'lr_ub': trial.suggest_categorical('lr_ub', choices=[0.5, 0.4, 0.3, 0.2]),
            #'lr_ub': 0.5,
            #'action_aggregation_type': trial.suggest_categorical('action_aggregation_type', choices=['threshold', 'max']),
            'action_aggregation_type': 'max'
        }
        if params['action_aggregation_type'] in ['threshold']:
            params.update({
                'action_threshold': trial.suggest_float('action_threshold', low=0.1, high=0.95, log=True)
            })
        return params



    def simulate_oamp(self, params):
        config = ConfigOAMP(params)

        oamp = OAMP(agents_count=len(self.seeds), args=config, episode_length=self.episode_length)

        oamp_actions = []
        agent_history = []
        for i in tqdm(range(len(self.times))):
            if i==self.val_test_index:
                oamp.plot_stats(labels=self.seeds_labels, palette_dict=self.seeds_palette_dict, figsize=(15, 15),
                                save_path=os.path.join(self.save_path, 'oamp_stats_validation.png'))

                agent_history = agent_history + oamp.plot_agent_switches(labels=self.seeds_labels, palette_dict=self.seeds_palette_dict,
                                                 save_path=os.path.join(self.save_path, 'oamp_switches_validation.png'))
                objective_value = oamp.plot_distribution(save_path=self.save_path, ql=5, start_date=None,
                                                         labels=self.seeds_labels, phase='validation')

                oamp = OAMP(agents_count=len(self.seeds), args=config, episode_length=self.episode_length)
            rewards, pnl, actions, time = self.seeds_percentages[:, i], self.seeds_rewards[:, i], self.seeds_allocations[:, i], self.times[i]
            #actions_std = [int(1 + action / 5) for action in actions]
            actions_std = actions
            oamp.step(rewards, pnl, time=time)
            oamp_action = oamp.compute_action(actions_std.astype(int))
            #oamp_actions.append(int(oamp_action * 5 - 5))
            oamp_actions.append(oamp_action)

        self.oamp_actions = np.array(oamp_actions)
        oamp.plot_stats(labels=self.seeds_labels, palette_dict=self.seeds_palette_dict, figsize=(15, 15),
                        save_path=os.path.join(self.save_path, 'oamp_stats_test.png'))
        self.agent_history = agent_history + oamp.plot_agent_switches(labels=self.seeds_labels, palette_dict=self.seeds_palette_dict,
                                                 save_path=os.path.join(self.save_path, 'oamp_switches_test.png'))


        #_, _, _, percentages = self.test_env.test_policy(actions=self.oamp_actions)

        oamp.plot_distribution(save_path=self.save_path, ql=5, start_date=self.convergence_split_date,
                               labels=self.seeds_labels, phase='test')


        """
        J = oamp.get_J(start_date=self.convergence_split_date)[:, 1:]

        labels = self.seeds_labels
        color_palette = sns.color_palette("tab10", len(labels))
        palette_dict = {label: color for label, color in zip(labels, color_palette)}
        colors = [palette_dict[agent] for agent in labels]
        out = pd.DataFrame(J).apply(pd.Series.value_counts, normalize=True).reindex(range(14), fill_value=0).fillna(
            0).sort_index()
        x = out.columns  # length 1000
        series = out.values  # shape (14, 1000)
        plt.figure()
        plt.stackplot(x, *series, colors=colors)  # or add colors=colors
        plt.title("Value Distribution per Column")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        timestep_idx = np.arange(J.shape[1])+self.convergence_index
        actions_J = self.seeds_allocations[J, timestep_idx]
        dataset_path = Path(os.path.join(self.data_root, 'dataset.parquet'))
        sim_env = TTFEnv(data_path=dataset_path, policy_path=None,
                             trajectory_number=5, no_overnight=self.no_overnight, date_range=[self.convergence_split_date, self.end_date])
        times, _, rewards, percentages = self.test_policies(env=sim_env, actions=actions_J)

        cum_rewards = percentages.cumsum(axis=1)  # (n, T)
        ql=5
        ql, q25, median, q75, qh = np.percentile(cum_rewards, [ql, 25, 50, 75, 100 - ql], axis=0)

        plt.figure(figsize=(15, 7))
        plt.plot(times, median, label='Median')
        plt.fill_between(times, ql, qh, alpha=0.3, label='IQR 5-95')
        plt.fill_between(times, q25, q75, alpha=0.3, label='IQR 25-75')
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.xticks(rotation=45)
        plt.xlabel("Time")
        plt.ylabel("Cumulative percent pnl")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.save_path, f'oamp_iqr_simulated_{self.phase}.png'))
        # plt.show()
        plt.close()

        labels = self.seeds_labels
        color_palette = sns.color_palette("tab10", len(labels))
        palette_dict = {label: color for label, color in zip(labels, color_palette)}

        plt.figure(figsize=(15, 7))
        for i in range(self.seeds_percentages.shape[0]):
            plt.plot(times, self.seeds_percentages[i, timestep_idx].cumsum(), color=palette_dict[labels[i]], alpha=0.7)

        # plt.plot(times, median, label='oamp median', linewidth=3)
        # plt.plot(times, q25, label='oamp 25th percentile', linewidth=3)
        # plt.plot(times, q75, label='oamp 75th percentile', linewidth=3)

        plt.plot(times, median, label='Median')
        plt.fill_between(times, ql, qh, alpha=0.3, label='IQR 5-95')
        plt.fill_between(times, q25, q75, alpha=0.3, label='IQR 25-75')
        # plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

        plt.title(f'Cumulative percent pnl ({self.phase})')
        plt.legend()
        plt.xticks(rotation=45)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.grid()
        plt.savefig(os.path.join(self.save_path, f'oamp_percentages_simulated_{self.phase}.png'))
        plt.show()
        plt.close()

        """



        """
            oamp_rewards_val = [self.seeds_rewards[self.agent_history[i], i] for i in range(self.val_test_index)]

            plt.figure(figsize=(15, 7))
            for i in range(len(self.seeds)):
                plt.plot(self.times[:self.val_test_index], np.cumsum(self.seeds_rewards[i, :self.val_test_index]), label=f'{self.seeds[i]}',
                         color=self.seeds_palette_dict[f'{self.seeds[i]}'])
            plt.plot(self.times[:self.val_test_index], np.cumsum(oamp_rewards_val), label='oamp', color='#000000')
            plt.title('Cumulative reward (validation)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid()
            plt.savefig(os.path.join(self.save_path, 'oamp_rewards_validation.png'))
            plt.close()

            oamp_rewards_convergence = [self.seeds_rewards[self.agent_history[i], i] for i in range(self.val_test_index, self.convergence_index)]

            plt.figure(figsize=(15, 7))
            for i in range(len(self.seeds)):
                plt.plot(self.times[self.val_test_index:self.convergence_index], np.cumsum(self.seeds_rewards[i, self.val_test_index:self.convergence_index]), label=f'{self.seeds[i]}',
                         color=self.seeds_palette_dict[f'{self.seeds[i]}'])
            plt.plot(self.times[self.val_test_index:self.convergence_index], np.cumsum(oamp_rewards_convergence), label='oamp', color='#000000')
            plt.title('Cumulative reward (convergence period)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid()
            plt.savefig(os.path.join(self.save_path, 'oamp_rewards_convergence.png'))
            plt.close()

            oamp_rewards_test = [self.seeds_rewards[self.agent_history[i], i] for i in
                                 range(self.convergence_index, len(self.times))]

            plt.figure(figsize=(15, 7))
            for i in range(len(self.seeds)):
                plt.plot(self.times[self.convergence_index:], np.cumsum(self.seeds_rewards[i, self.convergence_index:]),
                         label=f'{self.seeds[i]}',
                         color=self.seeds_palette_dict[f'{self.seeds[i]}'])
            plt.plot(self.times[self.convergence_index:], np.cumsum(oamp_rewards_test), label='oamp', color='#000000')
            plt.title('Cumulative reward (test)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid()
            plt.savefig(os.path.join(self.save_path, 'oamp_rewards_test.png'))
            plt.close()

            oamp_percentages_val = oamp_percentages[:self.val_test_index]
            colors_val = [self.seeds_palette_dict[f'{self.seeds[i]}'] for i in self.agent_history[:self.val_test_index]]

            plt.figure(figsize=(15, 7))
            for i in range(len(self.seeds)):
                plt.plot(self.times[:self.val_test_index],
                         np.cumsum(self.seeds_percentages[i, :self.val_test_index]),
                         label=f'{self.seeds[i]}',
                         color=self.seeds_palette_dict[f'{self.seeds[i]}'],
                         alpha=0.7)

            x = mdates.date2num(self.times[:self.val_test_index])
            y = np.cumsum(oamp_percentages_val)
            points = np.column_stack((x, y))
            segments = np.stack([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, colors=colors_val[:-1], linewidths=3)
            plt.gca().add_collection(lc)

            plt.title('Cumulative percent reward (validation)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            plt.grid()
            plt.savefig(os.path.join(self.save_path, 'oamp_percentages_validation.png'))
            plt.show()
            plt.close()

            oamp_percentages_convergence = [self.seeds_percentages[self.agent_history[i], i] for i in
                                            range(self.val_test_index, self.convergence_index)]
            oamp_percentages_convergence = oamp_percentages[self.val_test_index : self.convergence_index]

            colors_conv = [self.seeds_palette_dict[f'{self.seeds[i]}'] for i in
                           self.agent_history[self.val_test_index:self.convergence_index]]

            plt.figure(figsize=(15, 7))
            for i in range(len(self.seeds)):
                plt.plot(self.times[self.val_test_index:self.convergence_index],
                         np.cumsum(self.seeds_percentages[i, self.val_test_index:self.convergence_index]),
                         label=f'{self.seeds[i]}',
                         color=self.seeds_palette_dict[f'{self.seeds[i]}'],
                         alpha=0.7)

            x = mdates.date2num(self.times[self.val_test_index:self.convergence_index])
            y = np.cumsum(oamp_percentages_convergence)
            points = np.column_stack((x, y))
            segments = np.stack([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, colors=colors_conv[:-1], linewidths=3)
            plt.gca().add_collection(lc)

            plt.title('Cumulative percent reward (convergence period)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            plt.grid()
            plt.savefig(os.path.join(self.save_path, 'oamp_percentages_convergence.png'))
            #plt.show()
            plt.close()

            oamp_percentages_test = [self.seeds_percentages[self.agent_history[i], i] for i in
                                     range(self.convergence_index, len(self.times))]
            oamp_percentages_test = oamp_percentages[self.convergence_index:]

            colors = [self.seeds_palette_dict[f'{self.seeds[i]}'] for i in
                      self.agent_history[self.convergence_index:]]

            plt.figure(figsize=(15, 7))
            for i in range(len(self.seeds)):
                plt.plot(self.times[self.convergence_index:],
                         np.cumsum(self.seeds_percentages[i, self.convergence_index:]),
                         label=f'{self.seeds[i]}',
                         color=self.seeds_palette_dict[f'{self.seeds[i]}'], alpha=0.7)

            x = mdates.date2num(self.times[self.convergence_index:])
            y = np.cumsum(oamp_percentages_test)
            points = np.column_stack((x, y))
            segments = np.stack([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, colors=colors[:-1], linewidths=3)
            plt.gca().add_collection(lc)

            plt.title('Cumulative percent reward (test)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            plt.grid()
            plt.savefig(os.path.join(self.save_path, 'oamp_percentages_test.png'))
            plt.show()
            plt.close()
            """

        return self.agent_history, objective_value


    def __call__(self, trial):
        params = self.sample_oamp_params(trial)

        agent_history, oamp_reward = self.simulate_oamp(params=params)

        return oamp_reward


    def plot_optuna_optimization_history(self, study: optuna.study.Study):

        plots = [
            ("ParamsOptHistory.png", optuna.visualization.plot_optimization_history, {}),
            ("ParamsImportance.png", optuna.visualization.plot_param_importances, {}),
            ("ParamsContour.png", optuna.visualization.plot_contour, {}),
            ("ParamsSlice.png", optuna.visualization.plot_slice, {})
        ]

        os.makedirs(self.save_path, exist_ok=True)
        for filename, fn, kwargs in plots:
            try:
                fig = fn(study, **kwargs)
                fig.write_image(os.path.join(self.save_path, filename))
            except Exception as e:
                print(f"Error in {filename}: {e}")
                for key, _, _ in plots:
                    _path = os.path.join(self.save_path, str(key))
                    shutil.rmtree(_path, ignore_errors=True)


    def _bootstrap_trial(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        best_trials = [t.number for t in study.best_trials]
        print(f'Trial {trial.number} finished')
        #if trial.number > 1:
            #self.plot_optuna_optimization_history(study)

        if trial.number in best_trials:

            with open(os.path.join(self.save_path, 'best_params.json'), 'w') as f:
                json.dump(trial.params, f)

            for phase in ['validation', 'test']:

                src = os.path.join(self.save_path, f"oamp_switches_{phase}.png")
                dst = os.path.join(self.save_path, f"best_oamp_switches_{phase}.png")
                shutil.copy(src, dst)
                src = os.path.join(self.save_path, f"oamp_stats_{phase}.png")
                dst = os.path.join(self.save_path, f"best_oamp_stats_{phase}.png")
                shutil.copy(src, dst)
                src = os.path.join(self.save_path, f"oamp_iqr_{phase}.png")
                dst = os.path.join(self.save_path, f"best_oamp_iqr_{phase}.png")
                shutil.copy(src, dst)
                src = os.path.join(self.save_path, f"oamp_percentages_{phase}.png")
                dst = os.path.join(self.save_path, f"best_oamp_percentages_{phase}.png")
                shutil.copy(src, dst)
                src = os.path.join(self.save_path, f"oamp_sharpe_{phase}.png")
                dst = os.path.join(self.save_path, f"best_oamp_sharpe_{phase}.png")
                shutil.copy(src, dst)



            """
            for phase in ['validation', 'convergence', 'test']:
                src = os.path.join(self.save_path, f"oamp_rewards_{phase}.png")
                dst = os.path.join(self.save_path, f"best_oamp_rewards_{phase}.png")
                shutil.copy(src, dst)
                src = os.path.join(self.save_path, f"oamp_percentages_{phase}.png")
                dst = os.path.join(self.save_path, f"best_oamp_percentages_{phase}.png")
                shutil.copy(src, dst)
            """

            df = pd.DataFrame({
                'day': pd.Series(self.times).dt.strftime('%Y%m%d').astype(int),
                'minute': pd.Series(self.times).dt.hour * 60 + pd.Series(self.times).dt.minute,
                'reward_nostd': [self.seeds_rewards[self.agent_history[i], i] for i in range(len(self.agent_history))],
                'reward_pct': [self.seeds_percentages[self.agent_history[i], i] for i in range(len(self.agent_history))],
                'action': self.oamp_actions,
                'agent_history': self.agent_history
            })

            df.to_csv(os.path.join(self.save_path, 'oamp_log.csv'))



if __name__ == '__main__':
    objective = Objective()
    storage = f'sqlite:///{objective.save_path}/storage.db'
    db_path = os.path.join(objective.save_path, "storage.db")

    if os.path.isfile(db_path):
        os.remove(db_path)
        print(f"Deleted: {db_path}")
    else:
        print(f"No storage.db found at: {db_path}")

    search_space = {
        'loss_fn_window': [i for i in range(5, 30)],
        'agents_weights_upd_freq': [1,2,3,4,5,6,7]
        #'lr_ub': [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

    }
    #study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='maximize', study_name='oamp_tuning', storage=storage)

    study = optuna.create_study(sampler=TPESampler(n_startup_trials=30), direction='maximize', study_name='oamp_tuning', storage=storage)
    study.optimize(objective, n_trials=70, callbacks=[objective._bootstrap_trial])
    objective.plot_optuna_optimization_history(study)





