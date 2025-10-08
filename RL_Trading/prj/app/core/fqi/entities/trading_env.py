import gym
import numpy as np
import numpy.typing as npt
import os
import pandas as pd
import typing
import math
import pickle
from RL_Trading.prj.app.core.fqi.entities.fqi_dataset_builder import FQIDatasetBuilder
from RL_Trading.prj.app.core.fqi.trlib.policies.policy import Policy
from RL_Trading.prj.app.core.fqi.services.plotter import Plotter
from RL_Trading.prj.app.core.fqi.trlib.policies.policy import Policy, ImbalancePolicy, TemporalPolicy, LongOnlyPolicy, ShortOnlyPolicy

class TradingEnv(gym.Env):
    """
    Trading Environment.
    It is requested by FQI, passed as mdp parameter.
    """

    def __init__(self, dataset_builder: FQIDatasetBuilder):
        self._dataset_builder = dataset_builder
        self._df = self._dataset_builder.build_dataset()
        self._df.to_parquet('/home/a2a/a2a/RL_Trading/prj/app/core/data/dataset_old.parquet')
        print(self._df.time.dt.date.unique())
        self._persistence = self._dataset_builder.persistence
        self._years = self._dataset_builder.years
        self._current_state_features = self._dataset_builder.build_current_state_features()
        self._next_state_features =  self._dataset_builder.build_next_state_features()
        self._state_dim = len(self._current_state_features)
        self._action_dim = len(self.get_actions())
        self._action_space = gym.spaces.Discrete(self._action_dim)
        self._gamma = 1

    def _observation(self):
        return

    def _reward(self):
        return

    def _reset(self):
        return

    def step(self):
        return

    def test(
            self,
            policy: Policy,
            save_csv: bool,
            save_plots: bool,
            save_root: typing.Optional[str] = None,
            phase: typing.Optional[str] = None,
            iteration: typing.Optional[int] = None,
            trajectory_number: typing.Optional[int] = None,
            save_iteration_data: bool = False,
            trajectory_window: typing.Optional[int] = None,
            q_threshold: typing.Optional[int] = None,
            filter_method: typing.Optional[str] = None,
            use_estimator_mismatch: bool = False,
            Q_values_diff_threshold: typing.Optional[float] = None,
            plot_policy_features: typing.Optional[typing.List[str]] = None
    ) -> float:

        testing_df, days = self._build_testing_df(policy, use_estimator_mismatch=use_estimator_mismatch, plot_policy_features = plot_policy_features)
        if save_iteration_data == True:
            testing_df.to_csv(os.path.join(save_root, f"testing_df_iteration_{iteration}.csv"))
        results = self._test_parallel(testing_df, days, q_threshold, filter_method, Q_values_diff_threshold=Q_values_diff_threshold, use_estimator_mismatch=use_estimator_mismatch, plot_policy_features = plot_policy_features)
        results['trajectory'] = results['minute'] % self._persistence
        if trajectory_number is not None:
            if trajectory_window is not None and trajectory_window > 1:
                results = results[(results['trajectory'] >= (trajectory_number - math.floor(trajectory_window / 2))) &
                                (results['trajectory'] <= (trajectory_number + math.floor(trajectory_window / 2)))]
            else: 
                results = results[results['trajectory'] == trajectory_number]
        else: #one agent each 10 minutes
            results = results[results['trajectory'].isin(list(range(0, self._persistence, 10)))]

        if save_csv:
            assert save_root is not None
            assert phase is not None
            assert iteration is not None
            csv_filename = f"Results_iter{iteration}_{phase}.csv"
            results.to_csv(os.path.join(save_root, csv_filename), index=False)
        if save_plots:
            assert save_root is not None
            assert phase is not None
            assert iteration is not None
            if trajectory_number is None:
                Plotter.plot_actions(results, list(range(0, self._persistence, 10)), phase, iteration, save_root)
                Plotter.plot_actions_weekly(results, list(range(0, self._persistence,10)), phase, iteration, save_root)

            else:
                if trajectory_window is not None and trajectory_window > 1:
                    Plotter.plot_actions(results, list(range((trajectory_number - math.floor(trajectory_window / 2)), (trajectory_number + math.floor(trajectory_window / 2) + 1))), phase, iteration, save_root)
                    Plotter.plot_actions_weekly(results, list(range((trajectory_number - math.floor(trajectory_window / 2)), (trajectory_number + math.floor(trajectory_window / 2) + 1))), phase, iteration, save_root)

                else:
                    Plotter.plot_actions(results, [trajectory_number], phase, iteration, save_root)
                    Plotter.plot_actions_weekly(results, [trajectory_number], phase, iteration, save_root)

            #Plotter.plot_feature_importances(policy.Q._regressors, self._current_state_features, iteration, save_root)

        if trajectory_number is not None:
            if trajectory_window is not None and trajectory_window > 1:
                return np.mean(results.groupby('trajectory')['reward'].sum())
            else:
                return results['reward'].sum()
        else:
            #TODO return this
            return np.mean(results.groupby('trajectory')['reward'].sum())

    def _test_parallel(self, df: pd.DataFrame, days: npt.NDArray[int], q_threshold: typing.Optional[int] = None,
            filter_method: typing.Optional[str] = None, use_estimator_mismatch = False, Q_values_diff_threshold: typing.Optional[float] = None,  plot_policy_features: typing.Optional[typing.List[str]] = None  ) -> pd.DataFrame:
        # Set timesteps
        n_days = len(days)
        #df['rolling_quantile'] = 0.5 #df['Q_diff'].rolling(10000, center= False).quantile(quantile_cut_on_Q_diff).fillna(+np.inf)
        first_minute = self._df['minute_of_day'].min()
        final_minute = self._df['minute_of_day'].max()
        n_steps = final_minute - first_minute + 1
        # Initialize arrays
        old_allocation = np.zeros(n_days * self._persistence)
        actions = np.zeros((n_days, n_steps))
        rewards = np.zeros((n_days, n_steps))
        rewards_nostd = np.zeros((n_days, n_steps))
        costs = np.zeros((n_days,n_steps))
        pnl = np.zeros((n_days, n_steps))
        q_values = np.zeros((n_days, n_steps))
        q_short = np.zeros((n_days, n_steps))
        q_flat = np.zeros((n_days, n_steps))
        q_long = np.zeros((n_days, n_steps))
        q_min = np.zeros((n_days, n_steps))
        q_max = np.zeros((n_days, n_steps))
        q_diff = np.zeros((n_days, n_steps))
        feature_0 = np.zeros((n_days, n_steps))
        feature_1 = np.zeros((n_days, n_steps))
        # Test policy using a parallelization over trading days
        t = np.arange(self._persistence)
        mask = np.zeros(n_days * self._persistence)
        while t[0] < n_steps:
            # Set the considered timestep (the same for each day)
            time = first_minute + t
            datetimes = np.ravel(np.expand_dims(days, axis=1) + time)
            # Retrieve FQI dataset rows associated with the considered timestep
            current_step = zip(datetimes, old_allocation)
            current_data_idx = list(set(df.index) & set(current_step))
            current_data = df.loc[current_data_idx].sort_values(by=['datetime', 'action']).reset_index()

            if t[0] == 0:
                old_allocation_df = current_data.groupby('datetime').nth(1)

            best_actions_df = self._get_best_actions_df(current_data)

           # if filter_method == 'every_step_flat':
           #     mask = mask + (best_actions_df['Q'].values >= q_threshold)
           #     best_actions_df.loc[mask > 0] = current_data.groupby('datetime').nth(1)[mask > 0].values
           # elif filter_method == 'first_step':
           #     if t[0] == 0:
           #         mask = best_actions_df['Q'].values >= q_threshold
           # elif filter_method == 'every_step_propagate':
           #     mask = best_actions_df['Q'].values >= q_threshold
           #     best_actions_df.loc[mask > 0] = old_allocation_df[mask > 0].values
           # else:
           #     if filter_method != None:
           #         AssertionError(f"Value {filter_method} for filter_method not valid")
            if plot_policy_features is not None:
                feature_0[:, t[0]:t[0]+t_offset] = np.reshape(best_actions_df[plot_policy_features[0]], (n_days, self._persistence))[:, :t_offset]
                feature_1[:, t[0]:t[0]+t_offset] = np.reshape(best_actions_df[plot_policy_features[1]], (n_days, self._persistence))[:, :t_offset]
            if use_estimator_mismatch:
                print(filter_method)
                if filter_method == 'propagate':
                     mask = mask + (best_actions_df['Q_diff'].values >= Q_values_diff_threshold)
                     best_actions_df.loc[mask > 0] = old_allocation_df[mask > 0].values
                elif filter_method == 'move_flat':
                     mask = mask + (best_actions_df['Q_diff'].values >= Q_values_diff_threshold)
                     best_actions_df.loc[mask > 0] = current_data.groupby('datetime').nth(1)[mask > 0].values
                elif filter_method == 'first_step':
                     if t[0] == 0:
                       mask = mask + best_actions_df['Q_diff'].values >= Q_values_diff_threshold
                else:
                     AssertionError(f"Value {filter_method} for filter_method not valid")

            if filter_method == 'first_step':
                best_actions_df.loc[mask > 0] = current_data.groupby('datetime').nth(1)[mask > 0].values
            # Store results
            t_offset = min(self._persistence, max(1, n_steps - t[0]))
            actions[:, t[0]:t[0]+t_offset] = np.reshape(best_actions_df['action'], (n_days, self._persistence))[:, :t_offset]
            rewards[:, t[0]:t[0]+t_offset] = np.reshape(best_actions_df['reward'], (n_days, self._persistence))[:, :t_offset]
            rewards_nostd[:, t[0]:t[0]+t_offset] = np.reshape(best_actions_df['reward_nostd'], (n_days, self._persistence))[:, :t_offset]
            costs[:, t[0]:t[0]+t_offset] = np.reshape(best_actions_df['cost'], (n_days, self._persistence))[:, :t_offset]
            pnl[:, t[0]:t[0]+t_offset] = np.reshape(best_actions_df['pnl'], (n_days, self._persistence))[:, :t_offset]
            q_values[:, t[0]:t[0]+t_offset] = np.reshape(best_actions_df['Q'], (n_days, self._persistence))[:, :t_offset]

            """
            if t[0]+ t_offset>=n_steps:
                q_short[:, t[0]:t[0] + t_offset] = -50*np.ones((n_days, self._persistence))
                q_flat[:, t[0]:t[0] + t_offset] = np.reshape(current_data.groupby('datetime')['Q'].nth(0),
                                                             (n_days, self._persistence))[:, :t_offset]
                q_long[:, t[0]:t[0] + t_offset] = -50*np.ones((n_days, self._persistence))
            else:
            """

            q_short[:, t[0]:t[0]+t_offset] = np.reshape(current_data.groupby('datetime')['Q'].nth(0), (n_days, self._persistence))[:, :t_offset]
            q_flat[:, t[0]:t[0]+t_offset] = np.reshape(current_data.groupby('datetime')['Q'].nth(1), (n_days, self._persistence))[:, :t_offset]
            q_long[:, t[0]:t[0]+t_offset] = np.reshape(current_data.groupby('datetime')['Q'].nth(2), (n_days, self._persistence))[:, :t_offset]
            if use_estimator_mismatch:
                q_min[:, t[0]:t[0] + t_offset] = np.reshape(best_actions_df['Q_min'], (n_days, self._persistence))[:,
                                                    :t_offset]
                q_max[:, t[0]:t[0] + t_offset] = np.reshape(best_actions_df['Q_max'], (n_days, self._persistence))[:,
                                                    :t_offset]

                q_diff[:, t[0]:t[0] + t_offset] = np.reshape(best_actions_df['Q_diff'], (n_days, self._persistence))[:,
                                                    :t_offset]

            # Update the allocations for the next timestep
            old_allocation = best_actions_df['action'].values
            old_allocation_df = best_actions_df
            # Go to next timestep
            t += t_offset
        # Generate results dataset
        datetimes = np.array([str(day + time) for day in days for time in range(first_minute, final_minute + 1)])
        if use_estimator_mismatch == False:
            results = pd.DataFrame.from_dict({
                'day': [f'{d[:8]}' for d in datetimes],
                'minute': [int(d[8:12]) for d in datetimes],
                'action': list(actions.flatten()),
                'reward': list(rewards.flatten()),
                'reward_nostd': list(rewards_nostd.flatten()),
                'cost': list(costs.flatten()),
                'pnl': list(pnl.flatten()),
                'Q': list(q_values.flatten()),
                'Q_short': list(q_short.flatten()),
                'Q_flat': list(q_flat.flatten()),
                'Q_long': list(q_long.flatten()),
            })
        else:
            results = pd.DataFrame.from_dict({
                'day': [f'{d[:8]}' for d in datetimes],
                'minute': [int(d[8:12]) for d in datetimes],
                'action': list(actions.flatten()),
                'reward': list(rewards.flatten()),
                'reward_nostd': list(rewards_nostd.flatten()),
                'cost': list(costs.flatten()),
                'pnl': list(pnl.flatten()),
                'Q': list(q_values.flatten()),
                'Q_short': list(q_short.flatten()),
                'Q_flat': list(q_flat.flatten()),
                'Q_long': list(q_long.flatten()),
                'Q_min': list(q_min.flatten()),
                'Q_max': list(q_max.flatten()),
                'Q_diff': list(q_diff.flatten())
            })
        if plot_policy_features is not None:
            results[plot_policy_features[0]] = list(feature_0.flatten())
            results[plot_policy_features[1]] = list(feature_1.flatten())
        return results

    @staticmethod
    def _get_best_actions_df(current_data: pd.DataFrame) -> pd.DataFrame:
        return current_data.loc[current_data.groupby('datetime')['Q'].idxmax()]

    def _build_testing_df(self, policy: Policy, use_estimator_mismatch=False,
                          plot_policy_features: typing.Optional[typing.List[str]] = None) -> typing.Tuple[
        pd.DataFrame, npt.NDArray[int]]:

        feature_mapping = {
            'mid': 'BUND_mid' if self._dataset_builder.perform_action_on_bund else 'mid',
            'spread': 'BUND_spread' if self._dataset_builder.perform_action_on_bund else 'spread',
            'next_mid': 'next_BUND_mid' if self._dataset_builder.perform_action_on_bund else 'next_mid',
            'next_spread': 'next_BUND_spread' if self._dataset_builder.perform_action_on_bund else 'next_spread',
        }

        if isinstance(policy, ImbalancePolicy):
            df = pd.DataFrame.from_dict({
                'day': self._df['time'].dt.strftime('%Y%m%d').astype(int),
                'minute': self._df['minute_of_day'],
                'effective_minute': self._df['minute_of_day'],
                'time': self._df['time'],
                feature_mapping['mid']: self._df[feature_mapping['mid']],
                feature_mapping['spread']: self._df[feature_mapping['spread']],
                feature_mapping['next_mid']: self._df[feature_mapping['next_mid']],
                feature_mapping['next_spread']: self._df[feature_mapping['next_spread']],
                'allocation': self._df['allocation'],
                'action': self._df['action'],
                'Q': self._df.apply(
                    lambda row: 1 if row['BUND_L1-BidSize_0'] > row['BUND_L1-AskSize_0'] else -1 if row[
                                                                                                        'BUND_L1-BidSize_0'] <
                                                                                                    row[
                                                                                                        'BUND_L1-AskSize_0'] else 0,
                    axis=1) * self._df['action']
            })
        elif isinstance(policy, TemporalPolicy):
            df = pd.DataFrame.from_dict({
                'day': self._df['time'].dt.strftime('%Y%m%d').astype(int),
                'minute': self._df['minute_of_day'],
                'effective_minute': self._df['minute_of_day'],
                'time': self._df['time'],
                feature_mapping['mid']: self._df[feature_mapping['mid']],
                feature_mapping['spread']: self._df[feature_mapping['spread']],
                feature_mapping['next_mid']: self._df[feature_mapping['next_mid']],
                feature_mapping['next_spread']: self._df[feature_mapping['next_spread']],
                'allocation': self._df['allocation'],
                'action': self._df['action'],
                'Q': self._df.apply(lambda row: +1 if row['minute_of_day'] <= 10 * 60 + 15 else 0 if row[
                                                                                                         'minute_of_day'] > 10 * 60 + 15 and
                                                                                                     row[
                                                                                                         'minute_of_day'] <= 11 * 60 + 15 else -1 if
                row['minute_of_day'] > 11 * 60 + 15 and row['minute_of_day'] <= 16 * 60 else 1, axis=1) * self._df[
                         'action'] + 0.1 * (1 - abs(self._df['action']))

            })
        elif isinstance(policy, LongOnlyPolicy):
            df = pd.DataFrame.from_dict({
                'day': self._df['time'].dt.strftime('%Y%m%d').astype(int),
                'minute': self._df['minute_of_day'],
                'effective_minute': self._df['minute_of_day'],
                'time': self._df['time'],
                feature_mapping['mid']: self._df[feature_mapping['mid']],
                feature_mapping['spread']: self._df[feature_mapping['spread']],
                feature_mapping['next_mid']: self._df[feature_mapping['next_mid']],
                feature_mapping['next_spread']: self._df[feature_mapping['next_spread']],
                'allocation': self._df['allocation'],
                'action': self._df['action'],
                'Q': +1 * self._df['action']

            })

        elif isinstance(policy, ShortOnlyPolicy):
            df = pd.DataFrame.from_dict({
                'day': self._df['time'].dt.strftime('%Y%m%d').astype(int),
                'minute': self._df['minute_of_day'],
                'effective_minute': self._df['minute_of_day'],
                'time': self._df['time'],
                feature_mapping['mid']: self._df[feature_mapping['mid']],
                feature_mapping['spread']: self._df[feature_mapping['spread']],
                feature_mapping['next_mid']: self._df[feature_mapping['next_mid']],
                feature_mapping['next_spread']: self._df[feature_mapping['next_spread']],
                'allocation': self._df['allocation'],
                'action': self._df['action'],
                'Q': -1 * self._df['action']

            })
        else:

            if use_estimator_mismatch == False:
                df = pd.DataFrame.from_dict({
                    'day': self._df['time'].dt.strftime('%Y%m%d').astype(int),
                    'minute': self._df['minute_of_day'],
                    'effective_minute': self._df['minute_of_day'],
                    'time': self._df['time'],
                    feature_mapping['mid']: self._df[feature_mapping['mid']],
                    feature_mapping['spread']: self._df[feature_mapping['spread']],
                    feature_mapping['next_mid']: self._df[feature_mapping['next_mid']],
                    feature_mapping['next_spread']: self._df[feature_mapping['next_spread']],
                    'allocation': self._df['allocation'],
                    'action': self._df['action'],
                    'Q': policy.Q.values(self._df[self._current_state_features + ['action']].values)
                })
            else:
                df = pd.DataFrame.from_dict({
                    'day': self._df['time'].dt.strftime('%Y%m%d').astype(int),
                    'minute': self._df['minute_of_day'],
                    'effective_minute': self._df['minute_of_day'],
                    'time': self._df['time'],
                    feature_mapping['mid']: self._df[feature_mapping['mid']],
                    feature_mapping['spread']: self._df[feature_mapping['spread']],
                    feature_mapping['next_mid']: self._df[feature_mapping['next_mid']],
                    feature_mapping['next_spread']: self._df[feature_mapping['next_spread']],
                    'allocation': self._df['allocation'],
                    'action': self._df['action'],
                    'Q': policy.Q.values(self._df[self._current_state_features + ['action']].values),
                    'Q_min': policy.Q.values(self._df[self._current_state_features + ['action']].values,
                                             get_two_estimations=True)[0],
                    'Q_max': policy.Q.values(self._df[self._current_state_features + ['action']].values,
                                             get_two_estimations=True)[1],
                    'Q_diff': policy.Q.values(self._df[self._current_state_features + ['action']].values,
                                              get_two_estimations=True)[1] -
                              policy.Q.values(self._df[self._current_state_features + ['action']].values,
                                              get_two_estimations=True)[0],
                })

        if plot_policy_features is not None:
            df[plot_policy_features[0]] = self._df[plot_policy_features[0]]
            df[plot_policy_features[1]] = self._df[plot_policy_features[1]]

        df.loc[df[feature_mapping['mid']].isna(), ['effective_minute', 'Q']] = np.nan
        columns_to_bfill = [
            'effective_minute', 
            feature_mapping['mid'],
            feature_mapping['spread'],
            feature_mapping['next_mid'],
            feature_mapping['next_spread']
        ]
        tolerance = self._action_dim * self._action_dim * (self._persistence // 2)

        if (tolerance != 0):
            df[columns_to_bfill] = df[columns_to_bfill].bfill(limit=tolerance)

        reward, cost, pnl, reward_nostd = self._dataset_builder.compute_reward(
            df['time'], df[feature_mapping['spread']], df[feature_mapping['mid']], df[feature_mapping['next_spread']],
            df[feature_mapping['next_mid']], df['allocation'], df['action']
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

    def get_actions(self) -> typing.List[int]:
        return self._dataset_builder.actions

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
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def next_state_features(self) -> typing.List[str]:
        return self._next_state_features

    @property
    def persistence(self) -> int:
        return self._persistence

    @property
    def state_dim(self) -> int:
        return self._state_dim
