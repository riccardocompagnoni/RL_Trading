import os
import pickle
from pathlib import Path
import typing
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import polars as pl
import pandas as pd
from datetime import datetime, timedelta, time
from matplotlib import pyplot as plt
import numpy.typing as npt

from core.fqi.simulator.dataset_builder import DatasetBuilder
import warnings
import math
import gc

from RL_Trading.prj.app.core.fqi.trlib.policies.policy import Policy, ImbalancePolicy, TemporalPolicy, LongOnlyPolicy, ShortOnlyPolicy
from core.fqi.simulator.dataset_builder import DatasetBuilder
import numpy as np
from pathlib import Path


def unpack_numpy_data_to_dict(path: Path, keys: list[str]) -> dict:
    data_loaded = np.load(path, allow_pickle=True)
    return {k: data_loaded[k] for k in keys}

def _build_testing_df(df, current_state_features, persistence, policy: Policy, use_estimator_mismatch=False,
                          plot_policy_features: typing.Optional[typing.List[str]] = None) -> typing.Tuple[
        pd.DataFrame, npt.NDArray[int]]:

        action_dim = 3
        closing_hour = 16

        feature_mapping = {
            'mid': 'mid',
            'spread': 'spread',
            'next_mid': 'next_mid',
            'next_spread': 'next_spread',
        }

        if isinstance(policy, ImbalancePolicy):
            df = pd.DataFrame.from_dict({
                'day': df['time'].dt.strftime('%Y%m%d').astype(int),
                'minute': df['minute_of_day'],
                'effective_minute': df['minute_of_day'],
                'time': df['time'],
                feature_mapping['mid']: df[feature_mapping['mid']],
                feature_mapping['spread']: df[feature_mapping['spread']],
                feature_mapping['next_mid']: df[feature_mapping['next_mid']],
                feature_mapping['next_spread']: df[feature_mapping['next_spread']],
                'allocation': df['allocation'],
                'action': df['action'],
                'Q': df.apply(
                    lambda row: 1 if row['BUND_L1-BidSize_0'] > row['BUND_L1-AskSize_0'] else -1 if row[
                                                                                                        'BUND_L1-BidSize_0'] <
                                                                                                    row[
                                                                                                        'BUND_L1-AskSize_0'] else 0,
                    axis=1) * df['action']
            })
        elif isinstance(policy, TemporalPolicy):
            df = pd.DataFrame.from_dict({
                'day': df['time'].dt.strftime('%Y%m%d').astype(int),
                'minute': df['minute_of_day'],
                'effective_minute': df['minute_of_day'],
                'time': df['time'],
                feature_mapping['mid']: df[feature_mapping['mid']],
                feature_mapping['spread']: df[feature_mapping['spread']],
                feature_mapping['next_mid']: df[feature_mapping['next_mid']],
                feature_mapping['next_spread']: df[feature_mapping['next_spread']],
                'allocation': df['allocation'],
                'action': df['action'],
                'Q': df.apply(lambda row: +1 if row['minute_of_day'] <= 10 * 60 + 15 else 0 if row[
                                                                                                         'minute_of_day'] > 10 * 60 + 15 and
                                                                                                     row[
                                                                                                         'minute_of_day'] <= 11 * 60 + 15 else -1 if
                row['minute_of_day'] > 11 * 60 + 15 and row['minute_of_day'] <= 16 * 60 else 1, axis=1) * df[
                         'action'] + 0.1 * (1 - abs(df['action']))

            })
        elif isinstance(policy, LongOnlyPolicy):
            df = pd.DataFrame.from_dict({
                'day': df['time'].dt.strftime('%Y%m%d').astype(int),
                'minute': df['minute_of_day'],
                'effective_minute': df['minute_of_day'],
                'time': df['time'],
                feature_mapping['mid']: df[feature_mapping['mid']],
                feature_mapping['spread']: df[feature_mapping['spread']],
                feature_mapping['next_mid']: df[feature_mapping['next_mid']],
                feature_mapping['next_spread']: df[feature_mapping['next_spread']],
                'allocation': df['allocation'],
                'action': df['action'],
                'Q': +1 * df['action']

            })

        elif isinstance(policy, ShortOnlyPolicy):
            df = pd.DataFrame.from_dict({
                'day': df['time'].dt.strftime('%Y%m%d').astype(int),
                'minute': df['minute_of_day'],
                'effective_minute': df['minute_of_day'],
                'time': df['time'],
                feature_mapping['mid']: df[feature_mapping['mid']],
                feature_mapping['spread']: df[feature_mapping['spread']],
                feature_mapping['next_mid']: df[feature_mapping['next_mid']],
                feature_mapping['next_spread']: df[feature_mapping['next_spread']],
                'allocation': df['allocation'],
                'action': df['action'],
                'Q': -1 * df['action']

            })
        else:

            if use_estimator_mismatch == False:
                df = pd.DataFrame.from_dict({
                    'day': df['time'].dt.strftime('%Y%m%d').astype(int),
                    'minute': df['minute_of_day'],
                    'effective_minute': df['minute_of_day'],
                    'time': df['time'],
                    feature_mapping['mid']: df[feature_mapping['mid']],
                    feature_mapping['spread']: df[feature_mapping['spread']],
                    feature_mapping['next_mid']: df[feature_mapping['next_mid']],
                    feature_mapping['next_spread']: df[feature_mapping['next_spread']],
                    'allocation': df['allocation'],
                    'action': df['action'],
                    'Q': policy.Q.values(df[current_state_features + ['action']].values)
                })
            else:
                df = pd.DataFrame.from_dict({
                    'day': df['time'].dt.strftime('%Y%m%d').astype(int),
                    'minute': df['minute_of_day'],
                    'effective_minute': df['minute_of_day'],
                    'time': df['time'],
                    feature_mapping['mid']: df[feature_mapping['mid']],
                    feature_mapping['spread']: df[feature_mapping['spread']],
                    feature_mapping['next_mid']: df[feature_mapping['next_mid']],
                    feature_mapping['next_spread']: df[feature_mapping['next_spread']],
                    'allocation': df['allocation'],
                    'action': df['action'],
                    'Q': policy.Q.values(df[current_state_features + ['action']].values),
                    'Q_min': policy.Q.values(df[current_state_features + ['action']].values,
                                             get_two_estimations=True)[0],
                    'Q_max': policy.Q.values(df[current_state_features + ['action']].values,
                                             get_two_estimations=True)[1],
                    'Q_diff': policy.Q.values(df[current_state_features + ['action']].values,
                                              get_two_estimations=True)[1] -
                              policy.Q.values(df[current_state_features + ['action']].values,
                                              get_two_estimations=True)[0],
                })

        if plot_policy_features is not None:
            df[plot_policy_features[0]] = df[plot_policy_features[0]]
            df[plot_policy_features[1]] = df[plot_policy_features[1]]

        df.loc[df[feature_mapping['mid']].isna(), ['effective_minute', 'Q']] = np.nan
        columns_to_bfill = [
            'effective_minute', 
            feature_mapping['mid'],
            feature_mapping['spread'],
            feature_mapping['next_mid'],
            feature_mapping['next_spread']
        ]
        tolerance = action_dim * action_dim * (persistence // 2)

        if (tolerance != 0):
            df[columns_to_bfill] = df[columns_to_bfill].bfill(limit=tolerance)

        closing_time = pd.to_datetime(f'{closing_hour}:00')
        first_forced_action_time = (closing_time + timedelta(minutes=-persistence)).time()
        last_free_action_start_time = (closing_time + timedelta(minutes=-2 * persistence)).time()

        reward, cost, pnl, reward_nostd = compute_reward(
            df['time'], df[feature_mapping['spread']], df[feature_mapping['mid']], df[feature_mapping['next_spread']],
            df[feature_mapping['next_mid']], df['allocation'], df['action'], first_forced_action_time, last_free_action_start_time
        )
        df["reward"] = reward
        df["cost"] = cost
        df['pnl'] = pnl
        df['reward_nostd'] = reward_nostd

        missing_q_indices = df.loc[df['Q'].isna(), ['day', 'effective_minute', 'allocation', 'action']]
        df.loc[df['Q'].isna(), 'Q'] = missing_q_indices.merge(df.loc[df['Q'].notna()], how='left')['Q'].values
        # Q can still be nan if it occurred in last step of trading day. In this case, effective_minute is the initial
        # step of the next day in which the initial allocation is forced to 0. So the merge for the other combinations
        # fails.
        # Reward is nan if instead the next mid is missing even considering the tolerance used previously with bfill.
        incomplete_days = df[df['reward'].isna() | df['Q'].isna()]['day'].unique()
        df = df[~df['day'].isin(incomplete_days)]
        complete_days = df['day'].unique() * 1e4
        df['datetime'] = (df['day'] * 1e4 + df['minute']).astype(np.int64)
        df = df.drop(columns=['day', 'minute', 'effective_minute', feature_mapping['mid'], feature_mapping['spread'], feature_mapping['next_mid'], feature_mapping['next_spread']])
        df = df.set_index(['datetime', 'allocation'])
        return df, complete_days


def compute_reward(
        prev_time: pd.Series,
        prev_spread: pd.Series,
        prev_mid: pd.Series,
        curr_spread: pd.Series,
        curr_mid: pd.Series,
        prev_allocation: pd.Series,
        curr_allocation: pd.Series,
        first_forced_action_time : time,
        last_free_action_start_time: time,
        remove_costs: bool=False,
        remove_fixed_costs: bool=False,

) -> (pd.Series, pd.Series):
    """Function that computes reward.

    :param prev_time: time at period start.
    :param prev_spread: spreads at period start.
    :param prev_mid: mid prices at period start.
    :param curr_spread: spreads at period end.
    :param curr_mid: mid prices at period end.
    :param prev_allocation: allocations before portfolio rebalancing.
    :param curr_allocation: allocations after portfolio rebalancing.
    :return the percentage reward from period start to period end.
    """
    # The last action the agent can decide is 2 times persistence before trading closing time. After that time he's
    # forced to close the position. Therefore, the agent's last choice must include the cost to close the position.
    # reward = curr_allocation * (curr_mid - prev_mid) - abs(curr_allocation - prev_allocation) * prev_spread * 0.5
    # forced_action_cost = abs(curr_allocation) * curr_spread * 0.5
    # reward = np.where(prev_time.dt.time >= self.last_free_action_start_time, reward - forced_action_cost, reward)
    # reward = np.where(prev_time.dt.time >= self.first_forced_action_time, 0, reward)

    span = 60 * 8
    # mid_ewm = curr_mid.ewm(span=60*8*5, adjust=False).std()
    mid_std = curr_mid.rolling(
        window=span,
        min_periods=span // 2
    ).std()

    mid_std = mid_std.bfill()

    # mid_ewm = mid_ewm.fillna(curr_mid)

    if remove_costs is True:
        reward = curr_allocation * (curr_mid - prev_mid)
        reward = np.where(prev_time.dt.time >= first_forced_action_time, 0, reward)
        reward_nostd = reward
        pnl = curr_allocation * (curr_mid - prev_mid)
        pnl = np.where(prev_time.dt.time >= first_forced_action_time, 0, pnl)
        cost = 0
    elif remove_fixed_costs is True:
        reward = curr_allocation * (curr_mid - prev_mid) - abs(curr_allocation - prev_allocation) * (
                prev_spread * 0.5)
        cost = abs(curr_allocation - prev_allocation) * (prev_spread * 0.5)
        forced_action_cost = abs(curr_allocation) * (curr_spread * 0.5)
        cost = np.where(prev_time.dt.time >= last_free_action_start_time, cost + forced_action_cost, cost)
        reward = np.where(prev_time.dt.time >= last_free_action_start_time, reward - forced_action_cost,
                          reward)
        reward = np.where(prev_time.dt.time >= first_forced_action_time, 0, reward)
        reward_nostd = reward
        pnl = curr_allocation * (curr_mid - prev_mid)
        pnl = np.where(prev_time.dt.time >= first_forced_action_time, 0, pnl)
    else:
        reward_nostd = curr_allocation * (curr_mid - prev_mid) - abs(curr_allocation - prev_allocation) * (
                    prev_spread * 0.5 + 0.00625)
        reward = (curr_allocation * (curr_mid - prev_mid) - abs(curr_allocation - prev_allocation) * (
                    prev_spread * 0.5 + 0.00625)) / (mid_std + 0.005)
        cost = abs(curr_allocation - prev_allocation) * (prev_spread * 0.5 + 0.00625)
        forced_action_cost = abs(curr_allocation) * (curr_spread * 0.5 + 0.00625)
        cost = np.where(prev_time.dt.time >= last_free_action_start_time, cost + forced_action_cost, cost)
        reward = np.where(prev_time.dt.time >= last_free_action_start_time, reward - forced_action_cost, reward)
        reward = np.where(prev_time.dt.time >= first_forced_action_time, 0, reward)
        pnl = curr_allocation * (curr_mid - prev_mid)
        pnl = np.where(prev_time.dt.time >= first_forced_action_time, 0, pnl)

    return reward, cost, pnl, reward_nostd


def _compute_asset_reward(
        self,
        prev_time: pd.Series,
        prev_spread: pd.Series,
        prev_mid: pd.Series,
        curr_spread: pd.Series,
        curr_mid: pd.Series,
        prev_allocation: pd.Series,
        curr_allocation: pd.Series
):
    reward = curr_allocation * (curr_mid - prev_mid) - abs(curr_allocation - prev_allocation) * (
                prev_spread * 0.5 + 0.00625)
    cost = abs(curr_allocation - prev_allocation) * (prev_spread * 0.5 + 0.00625)
    forced_action_cost = abs(curr_allocation) * (curr_spread * 0.5 + 0.00625)
    cost = np.where(prev_time.dt.time >= self.last_free_action_start_time, cost + forced_action_cost, cost)
    reward = np.where(prev_time.dt.time >= self.last_free_action_start_time, reward - forced_action_cost, reward)
    reward = np.where(prev_time.dt.time >= self.first_forced_action_time, 0, reward)
    pnl = curr_allocation * (curr_mid - prev_mid)
    pnl = np.where(prev_time.dt.time >= self.first_forced_action_time, 0, pnl)
    return reward, cost, pnl


class TTFEnv(gym.Env):
    """Custom Trading Environment."""

    def __init__(self,
                 data_path: typing.Optional[Path]=None,
                 data_builder: typing.Optional[DatasetBuilder]=None,
                 trajectory_number: typing.Optional[int] = None,
                 policy_path: typing.Optional[Path]=None,
                 persistence: int = 30,
                 clip_bound: int = 10,
                 remove_cost: bool = False,
                 eval_mode: bool = False,
                 days_idx: typing.Optional[typing.List[int]] = None,
                 closing_position_gap: int = 0,
                 verbose: bool = False):

        super(TTFEnv, self).__init__()

        assert data_path is not None or data_builder is not None, 'Either a data_path or data_builder needs to be given'

        if data_path:
            dataset = pd.read_parquet(data_path)
            """
            path2 = 'C:/Users/Riccardo/Documents/df_with_combinations.parquet'
            # data_path = 'C:/Users/Riccardo/Documents/df_with_standard.parquet'
            df2 = pd.read_parquet(path2)
            # df2 = df2[(df2['minute_of_day']%60==3) | (df2['minute_of_day']%60==33)]
            df2 = df2[['time', 'reward_nostd']]
            bad_dates = df2[df2.reward_nostd.isna()].time.dt.date.unique()
            del df2
            dataset = dataset[~dataset['time'].dt.date.isin(bad_dates)]
            print('Done')
            """

        else:
            dataset = data_builder.build_dataset()

        state_features = np.array(['traded_quantity',
       'traded_quantity_rolling_mean', 'spread', 'mid', 'spread_std', 'mid_std', 'delta_mid_0',
       'time_to_roll', 'month', 'day_of_week', 'minute_of_day',
       'ratio_mid_date', 'ratio_mid_week', 'ratio_mid_month', 'ratio_spread_date',
       'ratio_spread_week', 'ratio_spread_month',
       'pct_spread', 'pct_delta', 'mid_rsi_8400',
       'mid_macd_7200_15600', 'pct_delta_ewm_18000', 'pct_spread_ewm_18000',
       'pct_delta_std_18000', 'pct_spread_std_18000', 'max_diff_ask_level',
       'max_diff_ask'] +
       [f'delta_mid_{i}' for i in range(1, 5)])


        dataset = pl.from_pandas(dataset)


        market_data = np.array(dataset.select(state_features))
        times = dataset.select('time').to_series().to_numpy()
        mid_prices = dataset.select('mid').to_series().to_numpy()
        spreads = dataset.select('spread').to_series().to_numpy()

        self.data_path = data_path
        #keys = ['market_data', 'state_features', 'times', 'mid_prices', 'spreads']
        #data_dict = unpack_numpy_data_to_dict(data_path, keys)
        #market_data, state_features, times, mid_prices, spreads = [data_dict[key] for key in keys]

        times_series = pl.Series(values=dataset.select('time'), dtype=pl.Datetime, name='datetime')
        unique_days_series = times_series.dt.date().unique().sort()
        #unique_days_series = dataset['time'].dt.date().unique().sort()
        unique_days_idx = np.arange(len(unique_days_series)).tolist()
        times_index_info = times_series.to_frame().with_row_index().with_columns(
            pl.col('datetime').dt.date().replace_strict(old=unique_days_series, new=unique_days_idx,
                                                        return_dtype=pl.UInt32).alias('date_index'),
        )
        days_idx = days_idx or unique_days_idx

        samples_idx = times_index_info.filter(pl.col('date_index').is_in(days_idx))['index'].to_list()
        assert samples_idx == sorted(samples_idx)

        self.market_data = market_data[samples_idx]
        self.times = times[samples_idx]
        self.mid_prices = mid_prices[samples_idx]
        self.spreads = spreads[samples_idx]
        self.state_features = state_features
        self.trajectory_number = trajectory_number
        self.trading_hour = (pl.Series(self.times).dt.hour().min(), pl.Series(self.times).dt.hour().max()+1)
        self.trading_minutes = (self.trading_hour[1] - self.trading_hour[0]) * 60
        self.persistence = persistence
        self.clip_bound = clip_bound

        del market_data, state_features, times, mid_prices, spreads#, data_dict
        gc.collect()

        if policy_path:
            with open(policy_path, 'rb') as load_file:
                self.policy = pickle.load(load_file)
        else:
            self.policy=None


        self.N, self.M = self.market_data.shape

        self.trajectory_upper_bound = min(self.persistence, 60 - closing_position_gap)

        assert self.market_data.ndim == 2, "market_data must be a 2D array"
        # assert self.trading_minutes % self.persistence == 0, "The trading minutes must be divisible by the persistence"
        assert self.clip_bound is None or self.clip_bound > 0, "clip_bound must be greater than 0"
        assert self.state_features.shape == (
            self.M,), "state_features must have the same length as the last dimension of market_data"
        assert self.times.shape == (self.N,), "times must have the same length as the first dimension of market_data"
        assert self.mid_prices.shape == (
            self.N,), "mid_prices must have the same length as the first dimension of market_data"
        assert self.spreads.shape == (
            self.N,), "spreads must have the same length as the first dimension of market_data"
        assert pl.Series(self.times).dt.date().value_counts()[
                   'count'].n_unique() == 1, "All episodes must have the same length"
        assert pl.Series(self.times, dtype=pl.Datetime).is_sorted(), "times must be sorted"
        assert self.trajectory_number is None or 0 <= self.trajectory_number < self.trajectory_upper_bound, f"trajectory_number must be between 0 and {self.trajectory_upper_bound}, or None"

        self.eval_mode = eval_mode
        self.remove_cost = remove_cost
        self.verbose = verbose

        if self.remove_cost:
            warnings.warn('Costs are removed from the reward function.')

        if not self.eval_mode and self.trajectory_number is not None:
            warnings.warn(
                'Trajectory number is set, but the environment is in training mode. Trajectory number should be random (set it to None).')

        assert self.persistence < (self.trading_hour[1] - self.trading_hour[
            0]) * 60, f"Persistence {self.persistence} is too high, it should be less than {(self.trading_hour[1] - self.trading_hour[0]) * 60}"
        self.episode_length = math.ceil(((self.trading_hour[1] - self.trading_hour[0]) * 60) / self.persistence)

        date_first_index_df = pl.Series(self.times).dt.date().alias('date').to_frame().with_row_index().unique(
            subset=['date'], keep='first', maintain_order=True)
        self.nunique_days = date_first_index_df.shape[0]
        self.first_date_row_indexes = date_first_index_df['index'].to_numpy()
        self.daily_minutes = (self.trading_hour[1] - self.trading_hour[0]) * 60
        assert self.persistence <= (
                    self.daily_minutes - 60), f"Give at least 1 hour to close the posistion, {self.persistence} > {self.daily_minutes - 60}"

        self.current_date_index = None
        self.actions = [-5, 0, 5]
        self.n_actions = len(self.actions)
        self.action_ohe_allocation_map = {a: np.eye(self.n_actions, dtype=np.float32)[self.actions.index(a)] for a in
                                          self.actions}

        self.observation_space = spaces.Box(low=-self.clip_bound, high=self.clip_bound,
                                            shape=(self.M + self.n_actions, 1), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_actions)

        self.first_daily_action = None

    def get_maximum_days(self):
        return self.nunique_days -1

    def _get_obs(self):
        return np.concatenate([
            self._get_current_market_state(),
            self.action_ohe_allocation_map[self.allocation]
        ], axis=0, dtype=np.float32).reshape(-1, 1)

    def _get_info(self):
        return {
            'curr_step': self.current_step,
            'step_to_go': self.episode_length - self.current_step
        }

    def _get_current_market_index(self) -> int:
        return self.first_date_row_indexes[self.current_date_index] + \
            (
                self.daily_minutes if self.current_step == self.episode_length else self.current_step * self.persistence + self.current_trajectory_offset)  # last step of the day has no trajectory offset

    def _is_last_forced_action(self):
        return (self.current_step+1)%self.episode_length==0

    def _get_market_state(self, idx: int) -> np.ndarray:
        return self.market_data[idx]

    def _get_current_market_state(self) -> np.ndarray:
        return self._get_market_state(self._get_current_market_index())

    def build_sa(self, state, allocation) -> np.ndarray:
        actions = self.policy.actions
        sa = np.zeros((len(actions), len(state)+2))

        for i in range(len(actions)):
            sa[i, 0:len(state)+1] = np.concatenate([state, [allocation]])
            sa[i, len(state)+1] = actions[i]

        return sa


    def _log_state(self, curr_idx: int, next_idx: int, curr_time: np.datetime64, curr_mid: float, curr_spread: float,
                   curr_allocation: int, next_time: np.datetime64, next_mid: float, next_spread: float,
                   next_allocation: int, reward: float, pnl: float, cost: float):
        python_curr_time = curr_time.astype('M8[ms]').astype('O')
        python_next_time = next_time.astype('M8[ms]').astype('O')
        print(
            f"{curr_idx},{next_idx} - {python_curr_time.strftime('%A')} {python_curr_time.date()} {python_curr_time.time()} -> {python_next_time.time()}: {curr_mid} -> {next_mid}, {curr_spread} -> {next_spread}, {curr_allocation} -> {next_allocation}, {reward}, {pnl}, {cost}")

    # TODO: implement the step method to support None both in the reward information and market states, for now invalid dates are removed
    def step(self, action: typing.Optional[int]=None):
        assert action is not None or self.policy is not None, 'Either a policy must be provided or an explicit action'
        idx = self._get_current_market_index()
        curr_mid, curr_spread, curr_allocation, curr_time = self.mid_prices[idx], self.spreads[idx], self.allocation, self.times[idx]
        if self.policy:
            curr_state = self._get_market_state(idx)
            sa = self.build_sa(curr_state, curr_allocation)
            values = self.policy.Q.values(sa)
            action = self.actions[np.argmax(values)]


        idx_old = idx

        self.current_step += 1

        idx = self._get_current_market_index()

        next_mid, next_spread, next_time = self.mid_prices[idx], self.spreads[idx], self.times[idx]
        reward, pnl, cost = TTFEnv._stock_reward(curr_mid=curr_mid,
                                                   curr_spread=curr_spread,
                                                   curr_allocation=curr_allocation,
                                                   action=action,
                                                   next_mid=next_mid,
                                                   next_spread=next_spread,
                                                   remove_cost=self.remove_cost)
        time = next_time
        self.allocation = action
        truncated = False
        terminated = self._is_last_forced_action()

        if self.verbose:
            self._log_state(curr_idx=idx_old, next_idx=idx, curr_time=curr_time, curr_mid=curr_mid,
                            curr_spread=curr_spread, curr_allocation=curr_allocation, next_time=next_time,
                            next_mid=next_mid, next_spread=next_spread, next_allocation=self.allocation, reward=reward,
                            pnl=pnl, cost=cost)

        if terminated:
            # encode the last step into the current step
            curr_mid, curr_spread, curr_allocation, curr_time = next_mid, next_spread, self.allocation, next_time
            idx_old = idx

            self.current_step += 1

            idx = self._get_current_market_index()
            closing_allocation = 0  # Hold
            next_mid, next_spread, next_time = self.mid_prices[idx], self.spreads[idx], self.times[idx]
            closing_reward, closing_pnl, closing_cost = TTFEnv._stock_reward(
                curr_mid=curr_mid,
                curr_spread=curr_spread,
                curr_allocation=curr_allocation,
                action=closing_allocation,
                next_mid=next_mid,
                next_spread=next_spread,
                remove_cost=self.remove_cost
            )

            closing_position_info_update_dict = {
                'before_closing_allocation': self.allocation,
                #'before_closing_time': curr_time.astype(datetime),
                'before_closing_time': curr_time,
                'before_closing_reward': reward,
                'before_closing_pnl': pnl,
                'before_closing_cost': cost
            }

            reward += closing_reward
            pnl += closing_pnl
            cost += closing_cost
            time = next_time
            self.allocation = closing_allocation

            if self.verbose:
                self._log_state(curr_idx=idx_old, next_idx=idx, curr_time=curr_time, curr_mid=curr_mid,
                                curr_spread=curr_spread, curr_allocation=curr_allocation, next_time=next_time,
                                next_mid=next_mid, next_spread=next_spread, next_allocation=0, reward=closing_reward,
                                pnl=closing_pnl, cost=closing_cost)
            #self.current_step = 0

        observation = self._get_obs()
        info = self._get_info()
        info.update({
            'allocation': self.allocation,
            #'time': time.astype(datetime),
            'time': time,
            'reward': reward,
            'pnl': pnl,
            'cost': cost
        })
        if terminated:
            info.update(closing_position_info_update_dict)

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_trajectory_offset = self.np_random.integers(0, self.trajectory_upper_bound, endpoint=False,
                                                                 dtype=int) if self.trajectory_number is None else self.trajectory_number
        self.allocation = 0  # hold at the beginning

        if self.eval_mode:
            self.current_date_index = self.current_date_index + 1 if self.current_date_index != None else 0
            if self.current_date_index == self.nunique_days:
                self.current_date_index = -1  # Set to zero back
        else:
            self.current_date_index = self.np_random.integers(0, self.nunique_days, endpoint=False, dtype=int)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    @staticmethod
    def _stock_reward(curr_mid: float, curr_allocation: int,
                      curr_spread: float, action: int,
                      next_mid: float, next_spread: float,
                      remove_cost: bool) -> tuple:
        pnl = action * (next_mid - curr_mid)
        cost = (curr_spread * 0.5 + 0.00625) * abs(action - curr_allocation)
        reward = pnl - (0. if remove_cost else cost)
        return reward, pnl, cost

    def render(self, mode='human'):
        pass

if __name__ == '__main__':

    data_root = 'C:/Users/Riccardo/OneDrive - Politecnico di Milano/Webeep/Thesis/RL_Trading/prj/app/core/data'
    config_path = 'C:/Users/Riccardo/OneDrive - Politecnico di Milano/Webeep/Thesis/RL_Trading/prj/app/config/config.yaml'
    persistence = 30  # min
    #data_builder = DatasetBuilder(data_root, config_path, persistence)
    #env = TTFEnv(data_builder=data_builder, eval_mode=True)

    data_path = Path('C:/Users/Riccardo/Documents/df_with_standard.parquet')
    policy_path = Path('C:/Users/Riccardo/OneDrive - Politecnico di Milano/Webeep/Thesis/RL_Trading/results/delta_std_pers30_optnostd_extended_long/seed95575/Policy_iter2.pkl')
    env = TTFEnv(data_path=data_path, policy_path=policy_path, eval_mode=True)
    #env = TTFEnv(data_path=data_path, eval_mode=True)

    n_eval_days = 100
    daily_rewards = []
    rewards = []
    times = []
    obs = env.reset()
    for day in range(n_eval_days):
        print(f'Starting day {day}')
        done = False
        day_reward = 0
        ep_len = 0
        while not done:
            # action, _ = model.predict(obs, deterministic=True)
            #action = np.random.choice([-5, 0, 5])
            action = 5

            obs, reward, terminated, _, info = env.step()
            times.append(info['time'])
            done = terminated
            day_reward += reward
            ep_len += 1
            rewards.append(reward)

        daily_rewards.append(day_reward)

    for time, reward in zip(times, rewards):
        print(f'{time}; {reward}')
    print(len(rewards))
    print(len(np.cumsum(rewards)))

    rew = np.array(rewards, dtype=float)
    # Replace NaN values with 0
    rew[np.isnan(rew)] = 0
    cum_rewards = np.cumsum(rew)

    plt.plot(times, rewards)
    plt.title('Instant reward')
    plt.xticks(rotation=45)
    plt.show()

    plt.plot(times, cum_rewards)
    plt.title('Cumulative reward')
    plt.xticks(rotation=45)

    plt.show()

    print(
        f"Simulation complete. Mean episode reward={np.mean(rewards):.3f}, Return={np.sum(rewards):.3f}, Episode length: {ep_len}")