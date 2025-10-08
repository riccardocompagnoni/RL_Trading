import pickle
import gym
import pandas as pd
import typing
import math
import numpy as np
from pathlib import Path
import os
import sys
import time
import datetime
from matplotlib import pyplot as plt

from core.fqi.services.plotter import Plotter
from core.fqi.trlib.policies.policy import Policy

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.normpath(os.path.join(current_dir, '..', '..', '..', '..', '..','..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..', 'RL_Trading', 'prj', 'app')))

from core.fqi.simulator.dataset_builder import DatasetBuilder

class TTFEnv(gym.Env):
    """Custom Trading Environment."""

    def __init__(self,
                 data_path: typing.Optional[Path]=None,
                 data_builder: typing.Optional[DatasetBuilder]=None,
                 dataset: typing.Optional[pd.DataFrame]=None,
                 trajectory_number: typing.Optional[int] = None,
                 policy_path: typing.Optional[Path]=None,
                 persistence: int = 30,
                 remove_costs: bool = False,
                 no_overnight: bool = True,
                 verbose: bool = False,
                 date_range=None,
                 phases_dates=None,
                 standardization='std'):

        #super(TTFEnv, self).__init__()

        assert dataset is not None or data_path is not None or data_builder is not None, 'Either a dataset, data_path or data_builder needs to be passed'

        if policy_path:
            self.set_policy(policy_path)
        else:
            self.policy=None

        self._actions = [-5, 0, 5]
        self._remove_costs = remove_costs
        self._no_overnight = no_overnight
        self._persistence = persistence
        self._current_state_features = DatasetBuilder.get_state_features()
        self._next_state_features = DatasetBuilder.get_next_state_features()
        self._features = DatasetBuilder.get_state_features(return_allocation=False)
        self._state_dim = len(self._current_state_features)
        self._action_dim = len(self.get_actions())
        self._action_space = gym.spaces.Discrete(self._action_dim)
        self._gamma = 1
        self._trajectory_number = trajectory_number if trajectory_number is not None else np.random.randint(0, self.persistence)
        self._standardization = standardization

        self._verbose = verbose
        if self._verbose:
            if data_path is not None:
                print('Simulator runs on precomputed features')
            else:
                print('Simulator uses DataBuilder')

            if trajectory_number is not None:
                print('Choosing random trajectory number')
            else:
                print(f'Trajectory number={self._trajectory_number}')

            if policy_path is not None:
                print('Using policy')
            else:
                print('Policy not provided. Must provide explicit actions')

            if remove_costs:
                print('Costs are being removed')
            else:
                print('Costs are included')

        if dataset is not None:
            dataset = dataset.copy()
            self._trading_hours = (dataset['time'].dt.hour.min(), dataset['time'].dt.hour.max() + 1)

        else:
            if data_path:
                dataset = pd.read_parquet(data_path)
            else:
                dataset = data_builder.build_dataset()

            if date_range is not None:
                dataset = dataset[(dataset.time.dt.date>=date_range[0]) & (dataset.time.dt.date<=date_range[1])]

            self._trading_hours = (dataset['time'].dt.hour.min(), dataset['time'].dt.hour.max() + 1)

            #self.check_nans(dataset[['time', 'ask_0', 'bid_0']])
            #self.check_minute_coverage(dataset, self._trading_hours)

        self._phases_dates = phases_dates
        self._dataset_full = dataset.copy()
        self._apply_trajectory(self._trajectory_number)

        self._curr_step = 0
        self._allocation = self._actions[1]
        self._entry = np.nan

    def _get_reward(self, prev_spread, prev_mid, curr_mid, prev_allocation, curr_allocation, rolling_std=None, rolling_mean=None):

        cost = abs(curr_allocation - prev_allocation) * (prev_spread * 0.5 + 0.00625)
        pnl = curr_allocation * (curr_mid - prev_mid)
        reward_nostd = pnl - (0. if self._remove_costs else cost)
        reward = pnl - (0. if self._remove_costs else cost)

        if rolling_mean is not None:
            reward = (reward - rolling_mean)
        if rolling_std is not None:
            reward = reward / rolling_std
        #reward = np.where(reward >= 0, reward, reward-0.1*reward**2)

        return reward, cost, pnl, reward_nostd

    def _get_percentage(self, ask, bid, action, allocation=None, entry=None):

        perc = 0
        if allocation is None:
            allocation = self._allocation
        if entry is None:
            entry = self._entry

        if allocation and action != allocation:
            perc = (bid - 0.00625) / entry - 1 if allocation == 5 else -((ask + 0.00625) / entry - 1)
            entry = np.nan
        if action and action != allocation:
            entry = (bid - 0.00625) if action == -5 else (ask + 0.00625)

        return perc, entry

    def build_fqi_dataset(self, phase='Train'):

        date_ranges = self._phases_dates[phase]

        mask = np.logical_or.reduce([
            self._dataset_full['time'].dt.date.between(start, end)
            for start, end in date_ranges
        ])

        df = self._dataset_full[mask].copy()

        actions = np.asarray(self._actions, dtype=np.int8)
        A = actions.size
        N = len(df)

        # Each original row is repeated A*A times
        k = A * A
        out = df.loc[df.index.repeat(k)].copy()

        # Build one block of size k with all (allocation, action) pairs
        # Here: allocation is the "outer" loop, action varies fastest
        alloc_block = np.repeat(actions, A)  # [-5,-5,-5, 0,0,0, 5,5,5] for A=3
        action_block = np.tile(actions, A)  # [-5, 0, 5, -5,0,5, -5,0,5]

        # Tile that block for all N rows
        out['allocation'] = np.tile(alloc_block, N)
        out['action'] = np.tile(action_block, N)

        # Keep memory footprint small for these two columns
        out['allocation'] = out['allocation'].astype('int8')
        out['action'] = out['action'].astype('int8')

        reward, cost, pnl, reward_nostd = self._get_reward(out['spread'], out['mid'], out['next_mid'], out['allocation'], out['action'], out['rolling_std'])
        out['reward'] = reward
        out['cost'] = cost
        out['pnl'] = pnl
        out['reward_nostd'] = reward_nostd

        out = out.dropna(subset=['reward'])


        first_action_time = (pd.to_datetime(f'{self._trading_hours[0]}:00') +
                                 datetime.timedelta(minutes=self.persistence)).time()
        last_action_time = (pd.to_datetime(f'{self._trading_hours[1]}:00') -
                                 datetime.timedelta(minutes=self.persistence)).time()

        last_date = out['time'].dt.normalize().max()
        first_date = out['time'].dt.normalize().max()
        flat_morning = (out['time'].dt.time < first_action_time) & (
                self._no_overnight | (out['time'].dt.normalize() == first_date)
        )
        flat_evening = (out['time'].dt.time > last_action_time) & (
                self._no_overnight | (out['time'].dt.normalize() == last_date)
        )

        mask_bad_morning = (out['allocation'] != 0) & flat_morning
        mask_bad_evening = (out['action'] != 0) & flat_evening
        out = out[~(mask_bad_morning | mask_bad_evening)].reset_index(drop=True)

        if self._no_overnight:
            # minutes + date keys
            minutes = out['time'].dt.hour * 60 + out['time'].dt.minute
            day = out['time'].dt.normalize()

            end_h = self._trading_hours[1]
            last_min = end_h * 60 - self.persistence  # last tradable minute
            prev_min = last_min - self.persistence  # previous action minute

            # pick last-minute rows (evening) and previous-minute rows
            mask_last = minutes.eq(last_min)  # last step rows (action should already be 0 after your filter)
            mask_prev = minutes.eq(prev_min)  # previous step rows (any action)

            if mask_last.any() and mask_prev.any():
                # Build lookup: (date, last_allocation) -> last row index
                last_key = pd.MultiIndex.from_arrays(
                    [day.loc[mask_last].to_numpy(),
                     out.loc[mask_last, 'allocation'].to_numpy()]
                )
                last_idx_lookup = pd.Series(out.index[mask_last].to_numpy(), index=last_key)

                # For prev rows, key is (date, prev_action) so that prev.action == last.allocation
                prev_idx_all = out.index[mask_prev].to_numpy()
                prev_key = pd.MultiIndex.from_arrays(
                    [day.loc[mask_prev].to_numpy(),
                     out.loc[mask_prev, 'action'].to_numpy()]
                )

                # Match prev rows to their corresponding last rows by (date, value)
                matched_last_idx = last_idx_lookup.reindex(prev_key).to_numpy()
                valid = ~pd.isna(matched_last_idx)

                if valid.any():
                    prev_idx = prev_idx_all[valid]
                    last_idx = matched_last_idx[valid].astype(int)

                    cols = ['reward', 'reward_nostd', 'pnl', 'cost']

                    # Add last-evening metrics into the previous-minute row
                    out.loc[prev_idx, cols] = (
                            out.loc[prev_idx, cols].to_numpy() + out.loc[last_idx, cols].to_numpy()
                    )

                    # Zero them on the last-minute rows to avoid double counting
                    out.loc[last_idx, cols] = 0.0

            out = out[~mask_last].reset_index(drop=True)

        out['absorbing_state'] = False
        last_free_action_time = (pd.to_datetime(f'{self._trading_hours[1]}:00') -
                                 2*datetime.timedelta(minutes=self.persistence)).time()
        if self._no_overnight:
            terminal_state = (out['time'].dt.time >= last_free_action_time)

        else:
            terminal_state = (out['time'].dt.time >= last_action_time) & (
                    out['time'].dt.date == out['time'].dt.date.max())

        out.loc[terminal_state, 'absorbing_state'] = True

        return out.sort_values(by=['time', 'allocation', 'action'])


    def clone(self):
        return TTFEnv(dataset=self.dataset, trajectory_number=self._trajectory_number, policy_path=None, persistence=self._persistence, remove_costs=self._remove_costs, no_overnight=self._no_overnight, verbose=self._verbose)

    def get_actions(self):
        return self._actions

    def set_policy(self, policy, is_policy=False):
        if is_policy:
            self.policy=policy
        else:
            with open(policy, 'rb') as load_file:
                self.policy = pickle.load(load_file)


    def _rebuild_caches(self):
        # rebuild NumPy views for the *current* self.dataset
        self._state_np = self.dataset[self._features].to_numpy(copy=True)
        self._time_np = self.dataset['time'].to_numpy(copy=False)
        self._mid_np = self.dataset['mid'].to_numpy(copy=False)
        self._spread_np = self.dataset['spread'].to_numpy(copy=False)
        self._ask_np = self.dataset['ask_0'].to_numpy(copy=False)
        self._bid_np = self.dataset['bid_0'].to_numpy(copy=False)
        if self._standardization=='std':
            #span = 60 * 8 * 1

            #rolling_std = self.dataset['mid'].diff(self.persistence).rolling(window=span,
             #                                                              min_periods=span // 2).std().ffill().bfill()

            self._std_np = self.dataset['rolling_std'].to_numpy(copy=False)

        # next_mid without per-step iloc
        self._next_mid_np = np.empty_like(self._mid_np)
        self._next_mid_np[:-1] = self._mid_np[1:]
        self._next_mid_np[-1] = self._mid_np[-1]  # dummy for last step

        # mark end-of-episode steps once (used for no_overnight)
        n = len(self._mid_np)
        self._overnight_mask = ((np.arange(n) + 1) % self.ep_len) == 0

        # reusable SA buffer
        state_dim = self._state_np.shape[1]
        self._sa_buf = np.empty((len(self._actions), state_dim + 2), dtype=np.float64)

    def _apply_trajectory(self, trajectory_number: int, phase=None):
        """Selects a trajectory offset as a *view* of the full dataset and rebuilds caches."""
        self._trajectory_number = int(trajectory_number)
        self.dataset, self.ep_len = self.restrict_dataset(
            self._dataset_full, self._trajectory_number, self._persistence, self._trading_hours, phase=phase
        )
        self.max_steps = self.dataset.index.max()
        self._rebuild_caches()

    def test(
            self,
            policy: Policy,
            save_csv: bool = False,
            save_plots: bool = False,
            save_root: typing.Optional[str] = None,
            phase: typing.Optional[str] = None,
            iteration: typing.Optional[int] = None,
            trajectory_number: typing.Optional[int] = None,
            trajectory_window: typing.Optional[int] = None,
            filter_method: typing.Optional[str] = None,
            use_estimator_mismatch: bool = False,
            Q_values_diff_threshold: typing.Optional[float] = None,
            return_full: bool = False,
    ):
        if trajectory_number is None:
            trajectories = list(range(0, self._persistence, 10))
        elif trajectory_window is None or trajectory_window == 1:
            trajectories = [trajectory_number]
        else:
            start = trajectory_number - math.floor(trajectory_window / 2)
            end = trajectory_number + math.floor(trajectory_window / 2) + 1
            trajectories = list(range(start, end))

        actions = np.asarray(self._actions, dtype=np.int8)
        prev_alloc_values = actions.copy()
        prev_alloc_to_idx = {int(a): i for i, a in enumerate(self._actions)}
        idx_to_prev_alloc = tuple(int(a) for a in self._actions)

        def precompute_q_cube(state_array: np.ndarray) -> np.ndarray:
            length, state_dim = state_array.shape
            P = len(prev_alloc_values)
            A = len(actions)
            SA = np.empty((length * P * A, state_dim + 2), dtype=np.float64)
            SA[:, :state_dim] = np.repeat(state_array, P * A, axis=0)
            SA[:, state_dim] = np.tile(np.repeat(prev_alloc_values, A), length)
            SA[:, state_dim + 1] = np.tile(np.tile(actions, P), length)
            q_flat = policy.Q.values(SA)
            return q_flat.reshape(length, P, A)

        def run_single_trajectory(traj: int) -> pd.DataFrame:
            ds, episode_len = self.restrict_dataset(
                self._dataset_full, traj, self._persistence, self._trading_hours, phase=phase
            )
            if ds.empty:
                cols = ['day', 'minute', 'trajectory', 'action', 'reward', 'reward_nostd', 'cost', 'pnl',
                        'percentages', 'Q', 'Q_short', 'Q_flat', 'Q_long']
                if use_estimator_mismatch and (Q_values_diff_threshold is not None):
                    cols += ['Q_min', 'Q_max', 'Q_diff']
                return pd.DataFrame(columns=cols)

            state_array = ds[self._features].to_numpy(copy=True)
            time_values = ds['time'].to_numpy(copy=False)
            mid = ds['mid'].to_numpy(copy=False)
            spread = ds['spread'].to_numpy(copy=False)
            ask0 = ds['ask_0'].to_numpy(copy=False)
            bid0 = ds['bid_0'].to_numpy(copy=False)
            if getattr(self, "standardization", None) == 'std' and 'rolling_std' in ds.columns:
                rolling_std = ds['rolling_std'].to_numpy(copy=False)
            else:
                rolling_std = np.full(len(ds), np.nan, dtype=np.float64)

            next_mid = np.empty_like(mid)
            next_mid[:-1] = mid[1:]
            next_mid[-1] = mid[-1]

            n = len(ds)
            is_episode_end = ((np.arange(n) + 1) % episode_len) == 0

            chosen_time = np.empty(n, dtype=time_values.dtype)
            chosen_action = np.empty(n, dtype=np.int8)
            reward_vec = np.empty(n, dtype=np.float64)
            reward_no_std = np.empty(n, dtype=np.float64)
            cost_vec = np.empty(n, dtype=np.float64)
            pnl_vec = np.empty(n, dtype=np.float64)
            percentage_vec = np.zeros(n, dtype=np.float64)
            q_short = np.full(n, np.nan, dtype=np.float64)
            q_flat = np.full(n, np.nan, dtype=np.float64)
            q_long = np.full(n, np.nan, dtype=np.float64)
            q_selected = np.full(n, np.nan, dtype=np.float64)
            q_min = np.full(n, np.nan, dtype=np.float64)
            q_max = np.full(n, np.nan, dtype=np.float64)
            q_diff = np.full(n, np.nan, dtype=np.float64)

            q_cube = precompute_q_cube(state_array)

            allocation_idx = prev_alloc_to_idx[0]
            entry_price = np.nan
            mismatch_active = False
            no_overnight = self._no_overnight

            for t in range(n):
                q_vec = q_cube[t, allocation_idx, :]
                q_short[t], q_flat[t], q_long[t] = float(q_vec[0]), float(q_vec[1]), float(q_vec[2])

                if no_overnight and is_episode_end[t]:
                    greedy_idx = 1
                else:
                    greedy_idx = int(np.argmax(q_vec))

                action_idx = greedy_idx

                if use_estimator_mismatch and (Q_values_diff_threshold is not None):
                    qmn = float(q_vec.min())
                    qmx = float(q_vec.max())
                    qdf = qmx - qmn
                    q_min[t] = qmn
                    q_max[t] = qmx
                    q_diff[t] = qdf
                    apply_filter = False
                    if filter_method == 'first_step':
                        apply_filter = (t == 0 and qdf >= Q_values_diff_threshold)
                    elif filter_method in ('propagate', 'move_flat'):
                        if qdf >= Q_values_diff_threshold:
                            mismatch_active = True
                        apply_filter = mismatch_active
                    if apply_filter:
                        if filter_method == 'propagate':
                            action_idx = allocation_idx
                        else:
                            action_idx = 1

                action_val = int(actions[action_idx])
                alloc_val = int(idx_to_prev_alloc[allocation_idx])

                reward, cost, pnl, reward_ns = self._get_reward(
                    spread[t], mid[t], next_mid[t], alloc_val, action_val, rolling_std=rolling_std[t]
                )
                percentage, entry_price = self._get_percentage(
                    ask0[t], bid0[t], action_val, alloc_val, entry_price
                )

                chosen_time[t] = time_values[t]
                chosen_action[t] = action_val
                reward_vec[t] = reward
                reward_no_std[t] = reward_ns
                cost_vec[t] = cost
                pnl_vec[t] = pnl
                percentage_vec[t] = percentage
                q_selected[t] = float(q_vec[action_idx])

                allocation_idx = prev_alloc_to_idx[action_val]

            ts = pd.to_datetime(pd.Series(chosen_time))
            out = {
                'day': ts.dt.strftime('%Y%m%d').tolist(),
                'minute': (ts.dt.hour * 60 + ts.dt.minute).astype(int).tolist(),
                'trajectory': np.full(n, traj, dtype=int).tolist(),
                'action': chosen_action.tolist(),
                'reward': reward_vec.tolist(),
                'reward_nostd': reward_no_std.tolist(),
                'cost': cost_vec.tolist(),
                'pnl': pnl_vec.tolist(),
                'percentages': percentage_vec.tolist(),
                'Q': q_selected.tolist(),
                'Q_short': q_short.tolist(),
                'Q_flat': q_flat.tolist(),
                'Q_long': q_long.tolist(),
            }
            if use_estimator_mismatch and (Q_values_diff_threshold is not None):
                out.update({'Q_min': q_min.tolist(), 'Q_max': q_max.tolist(), 'Q_diff': q_diff.tolist()})
            return pd.DataFrame(out)

        results = pd.concat([run_single_trajectory(tr) for tr in trajectories], axis=0, ignore_index=True)

        if save_csv:
            assert save_root is not None and phase is not None and iteration is not None
            results.to_csv(os.path.join(save_root, f"Results_iter{iteration}_{phase}.csv"), index=False)
        if save_plots:
            assert save_root is not None and phase is not None and iteration is not None
            Plotter.plot_actions(results, trajectories, phase, iteration, save_root)
            Plotter.plot_actions_weekly(results, trajectories, phase, iteration, save_root)

        if return_full:
            results['datetime'] = pd.to_datetime(results['day'], format='%Y%m%d') + pd.to_timedelta(results['minute'],
                                                                                                    unit='m')
            return results['datetime'].to_numpy(), results['action'].to_numpy(), results['reward_nostd'], results[
                'percentages']

        return float(results.groupby('trajectory')['reward'].sum().mean())

    def test_policy_actions(self, actions):
        self.reset()

        rewards, percentages, allocations, times = [], [], [], []
        for action in actions:
            time, allocation, reward, percentage, is_last_step = self.step(action)

            times.append(time)
            rewards.append(reward)
            allocations.append(allocation)
            percentages.append(percentage)

        return np.array(times), np.array(allocations), np.array(rewards), np.array(percentages)


    def _is_last_step(self):
        return self.max_steps == self._curr_step

    def _get_currstate(self):
        return self.dataset.iloc[self._curr_step]

    def _get_std(self):
        return self.dataset['rolling_std'].iloc[self._curr_step]

    def _get_nextstate(self):
        if self._is_last_step():
            # Dummy row. Could be any other index
            return self.dataset.iloc[0]
        return self.dataset.iloc[self._curr_step+1]


    def _build_sa(self, state, allocation) -> np.ndarray:
        actions = self.policy.actions
        sa = np.zeros((len(actions), len(state)+2))

        for i in range(len(actions)):
            sa[i, 0:len(state)+1] = np.concatenate([state, [allocation]])
            sa[i, len(state)+1] = actions[i]

        return sa

    def reset(self, seed=None, options=None):
        self._curr_step = 0
        self._allocation = self._actions[1]


    def step(self, action: typing.Optional[int]=None):

        is_last_step = self._is_last_step()
        curr_state = self._get_currstate()
        next_state = self._get_nextstate()
        std = self._get_std()

        curr_time, curr_mid, curr_spread, curr_ask, curr_bid, next_mid = (
            curr_state['time'], curr_state['mid'], curr_state['spread'], curr_state['ask_0'], curr_state['bid_0'], next_state['mid'])

        if self.policy:
            if self._no_overnight and (self._curr_step+1)%self.ep_len==0:
                action = self._actions[1]
            else:
                sa = self._build_sa(curr_state[self._current_state_features], self._allocation)
                values = self.policy.Q.values(sa)
                action = self._actions[np.argmax(values)]

        _, _, _, reward = self._get_reward(curr_spread, curr_mid, next_mid, self._allocation, std)
        percentage, entry = self._get_percentage(curr_ask, curr_bid, action)
        self._entry = entry

        self._allocation = action
        self._curr_step += 1

        return curr_time, action, reward, percentage, is_last_step


    @staticmethod
    def check_nans(df: pd.DataFrame):

        for idx, row in df.iterrows():
            for col in df.columns:
                if pd.isna(row[col]):
                    print(f"NaN at time: {row['time']} in column {col}")

    @staticmethod
    def check_minute_coverage(df: pd.DataFrame, trading_hours=(8, 16), timestamp_col="time"):

        start_h, end_h = trading_hours
        ts = df[timestamp_col]

        for day in ts.dt.normalize().unique():
            start = day + pd.Timedelta(hours=start_h)
            end = day + pd.Timedelta(hours=end_h)

            expected = pd.date_range(start=start, end=end, freq="min", inclusive="left")
            present = df.loc[(ts >= start) & (ts < end), timestamp_col].drop_duplicates()

            missing = expected.difference(present)
            for m in missing.sort_values():
                print(f"Missing minute on {m.date()} at {m.strftime('%H:%M')}")


    def restrict_dataset(self, df: pd.DataFrame, trajectory_number, persistence, trading_hours=(8,16), phase=None):

        df["time"] = pd.to_datetime(df["time"])
        if phase is not None:
            date_ranges=self._phases_dates[phase]

            mask = np.logical_or.reduce([
                df['time'].dt.date.between(start, end)  # inclusive on both ends
                for start, end in date_ranges
            ])

            df = df[mask]

        start_h, end_h = trading_hours
        minutes = df["time"].dt.hour * 60 + df["time"].dt.minute
        start_minute = start_h * 60 + trajectory_number
        end_minute = end_h * 60

        # select only minutes inside the window [start_minute, end_minute)
        in_window = (minutes >= start_minute) & (minutes < end_minute)

        # now require alignment with persistence
        aligned = ((minutes - start_minute) % persistence) == 0

        ep_len = math.ceil((end_minute - start_minute) / persistence)

        return df[in_window & aligned].reset_index(drop=True), ep_len

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def current_state_features(self) -> typing.List[str]:
        return self._current_state_features

    @property
    def next_state_features(self) -> typing.List[str]:
        return self._next_state_features

    @property
    def persistence(self) -> int:
        return self._persistence

    @property
    def state_dim(self) -> int:
        return self._state_dim




if __name__ == '__main__':

    from datetime import date
    """
    dates = {
        'Validation': [[date(2022, 1, 1), date(2022, 7, 1)]],
        'Train': [[date(2020, 1, 1), date(2021, 1, 31)], [date(2021, 10, 1), date(2021, 12, 31)]]
    }


    env = TTFEnv(data_path=Path('/home/a2a/a2a/RL_Trading/prj/app/core/data/dataset_full.parquet'),
                 policy_path=Path('/home/a2a/a2a/RL_Trading/results/experts_oamp/3_experts/seed10707/Policy_iter1.pkl'),
                 verbose=True, trajectory_number=2, no_overnight=False, phases_dates=dates)
    print(isinstance(env, gym.Env))
    s = time.time()


    df = env.build_fqi_dataset()
    e = time.time()

    print((e - s), "s")
    
    """

    #env = TTFEnv(data_path=Path('/home/a2a/a2a/RL_Trading/prj/app/core/data/dataset.parquet'), policy_path=Path('/home/a2a/a2a/RL_Trading/results/experts_oamp/3_experts/seed10707/Policy_iter1.pkl'), verbose=True, trajectory_number=2, no_overnight=True)
    from datetime import date
    dates = {
        'Validation': [[date(2023, 1, 1), date(2023, 7, 1)]],
        'Train': [[date(2020, 1, 1), date(2023, 1, 1)]]#, [date(2021, 10, 1), date(2021, 12, 31)]]
    }

    for phase in ['Validation']:
        s = time.time()

        env = TTFEnv(data_path=Path('/home/a2a/a2a/RL_Trading/prj/app/core/data/dataset_full.parquet'), verbose=True, trajectory_number=22, no_overnight=False, phases_dates=dates)

        env.set_policy(Path('/home/a2a/a2a/RL_Trading/results/experts_overnight/1y23_7y23/seed98032/Policy_iter3.pkl'))

        env.reset()

        _ = env.test(policy=env.policy, iteration=1, save_csv=True, trajectory_number=22, trajectory_window=1, save_root='/home/a2a/a2a/RL_Trading/results', phase=phase)
        df = pd.read_csv(f'/home/a2a/a2a/RL_Trading/results/Results_iter1_{phase}.csv')

        #times, allocations, rewards, percentages = env.test_policy_actions(df['action'])

        df['time'] = (df['minute'] // 60) * 100 + df['minute'] % 60
        times = pd.to_datetime(df['day'].astype(str) + df['time'].astype(str), format='%Y%m%d%H%M')

        percentages = df['percentages']
        fig = plt.figure(figsize=(15,7))
        plt.plot(times, np.cumsum(percentages))
        plt.savefig(f'/home/a2a/a2a/RL_Trading/results/fig{phase}.png')
        plt.show()
        print('ok')
        e = time.time()

        print((e - s), "s")
