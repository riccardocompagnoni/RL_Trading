import abc
import datetime
import itertools
import numpy as np
import pandas as pd
import typing
import datetime

#from jupyter_core.migrate import security_file

from RL_Trading.prj.app.core.models.entities.dataset_builder import DatasetBuilder
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FQIDatasetBuilder(DatasetBuilder, abc.ABC):
    """A class to create the dataset used for FQI regression task.

    :param persistence: number of steps the agent keeps the chosen action unchanged.
    :param actions: list of actions the agent can take.
    """
    persistence: int
    asset_name : str
    years: typing.List[int]
    actions: typing.List[int]
    first_forced_action_time: datetime.time = field(init=False)
    last_free_action_start_time: datetime.time = field(init=False)

    def __post_init__(self):
        closing_time = pd.to_datetime(f'{self.closing_hour}:00')
        time = (closing_time + datetime.timedelta(minutes=-self.persistence)).time()
        object.__setattr__(self, 'first_forced_action_time', time)
        time = (closing_time + datetime.timedelta(minutes=-2*self.persistence)).time()
        object.__setattr__(self, 'last_free_action_start_time', time)

    @abc.abstractmethod
    def _build_next_state_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def _build_allocation_action_pairs(self, state_df: pd.DataFrame) -> pd.DataFrame:
        """Function that builds allocation-actions pairs dataset."""
        df = []
        overnight = True


        # Set allocation to 0 at the beginning of the trading day
        first_action_end_time = (pd.to_datetime(f'{self.opening_hour}:00') +
                                 datetime.timedelta(minutes=self.persistence)).time()

        if not overnight:
            for action in self.actions:
                _df = state_df[state_df['time'].dt.time < first_action_end_time].copy()
                _df['allocation'] = 0
                _df['action'] = action
                df.append(_df)
            state_df = state_df.drop(_df.index)
            # Set action to 0 at the end of the trading day

            for allocation in self.actions:
                _df = state_df[(state_df['time'].dt.time >= self.first_forced_action_time)].copy()
                _df['allocation'] = allocation
                _df['action'] = 0
                df.append(_df)
            state_df = state_df.drop(_df.index)
        else:


            for action in self.actions:
                _df = state_df[(state_df['time'].dt.time < first_action_end_time) & (state_df['time'].dt.date==state_df['time'].dt.date.min())].copy()
                _df['allocation'] = 0
                _df['action'] = action
                df.append(_df)
            state_df = state_df.drop(_df.index)

            for allocation in self.actions:
                _df = state_df[(state_df['time'].dt.time >= self.first_forced_action_time) & (state_df['time'].dt.date == state_df['time'].dt.date.max())].copy()
                _df['allocation'] = allocation
                _df['action'] = 0
                df.append(_df)
            state_df = state_df.drop(_df.index)


        # Generate all the other allocation-action pairs
        for allocation in self.actions:
            for action in self.actions:
                _df = state_df.copy()
                _df['allocation'] = allocation
                _df['action'] = action
                df.append(_df)
        df = pd.concat(df)
        # Set absorbing states
        df['absorbing_state'] = False
        if overnight:
            terminal_state = (df['time'].dt.time >= self.last_free_action_start_time)& (df['time'].dt.date == df['time'].dt.date.max())  # & (df['time'].dt.weekday==4)
        else:
            terminal_state = (df['time'].dt.time >= self.last_free_action_start_time)  # & (df['time'].dt.weekday==4)

        df.loc[terminal_state, 'absorbing_state'] = True
        # Return
        return df.sort_values(by=['time', 'allocation', 'action'])

    def _filter_dataset(self, state_df: pd.DataFrame) -> pd.DataFrame:
        df = state_df[state_df['time'].dt.time < self.first_forced_action_time]
        return df


    @staticmethod
    def _clean_df(df: pd.DataFrame, keep_volumes: bool, depth: int = 0) -> pd.DataFrame:
        # Remove useless columns.
#[f'{name}_{level}' for name in ['bid', 'ask'] for level in range(depth)] + \

        #TODO buy, sell, unknown??
        columns_to_drop = \
            [f'{name}_{level}_0' for name in ['bid_quantity', 'ask_quantity'] for level in range(depth)] if keep_volumes is False else []
        # ['buy', 'sell', 'unknown'] + \

        columns_to_keep = [c for c in df.columns if c not in columns_to_drop]
        df = df[columns_to_keep]

        #TODO check what this does exactly

        # Remove columns with unique values
        unique_counts = df.nunique()
        #Enforce not to remove month features when OneHotEncoding is enabled and a whole month is not present in the dataset for some reason e.g. covid month
        columns_to_keep = list(set(unique_counts[unique_counts > 1].index.tolist() + [column for column in columns_to_keep if "month" in column]))
        #df = df[columns_to_keep]


        # Return
        return df

    def compute_reward(
            self,
            prev_time: pd.Series,
            prev_spread: pd.Series,
            prev_mid: pd.Series,
            curr_spread: pd.Series,
            curr_mid: pd.Series,
            prev_allocation: pd.Series,
            curr_allocation: pd.Series
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
        #reward = curr_allocation * (curr_mid - prev_mid) - abs(curr_allocation - prev_allocation) * prev_spread * 0.5
        #forced_action_cost = abs(curr_allocation) * curr_spread * 0.5
        #reward = np.where(prev_time.dt.time >= self.last_free_action_start_time, reward - forced_action_cost, reward)
        #reward = np.where(prev_time.dt.time >= self.first_forced_action_time, 0, reward)

        #span = 60*8
        #mid_ewm = curr_mid.ewm(span=60*8*5, adjust=False).std()
        #mid_std = curr_mid.rolling(
        #    window=span,
        #    min_periods=span // 2
        #).std()

        #mid_std = mid_std.bfill()


        temp_df = pd.DataFrame({
            'prev_time': prev_time,
            'std_col': prev_mid
        })

        to_std = (temp_df.groupby('prev_time')['std_col']
                  .max()
                  .reset_index())


        span = 60*8*1

        rolling_std = to_std['std_col'].diff(self.persistence).rolling(window=span, min_periods=span // 2).std().ffill()
        #rolling = to_std['std_col'].rolling(window=span, min_periods=1).mean()

        time_to_rolling_std = dict(zip(to_std['prev_time'], rolling_std))
        mapped_rolling_std = prev_time.map(time_to_rolling_std)

        rolling_mean = to_std['std_col'].diff(self.persistence).abs().rolling(window=span, min_periods=span // 2).mean().ffill()
        # rolling = to_std['std_col'].rolling(window=span, min_periods=1).mean()

        time_to_rolling_mean = dict(zip(to_std['prev_time'], rolling_mean))
        mapped_rolling_mean = prev_time.map(time_to_rolling_mean)

        #mid_ewm = mid_ewm.fillna(curr_mid)


        if self.remove_costs is True:
            reward_nostd = curr_allocation * (curr_mid - prev_mid)
            reward = ((curr_allocation * (curr_mid - prev_mid)))/(mapped_rolling_std + 0.0005)
            reward = np.where(prev_time.dt.time >= self.first_forced_action_time, 0, reward)
            pnl = curr_allocation * (curr_mid - prev_mid)
            pnl = np.where(prev_time.dt.time >= self.first_forced_action_time, 0, pnl)
            cost = 0
        elif self.remove_fixed_costs is True:
            reward = curr_allocation * (curr_mid - prev_mid) - abs(curr_allocation - prev_allocation) * (
                        prev_spread * 0.5)
            cost = abs(curr_allocation - prev_allocation) * (prev_spread * 0.5)
            forced_action_cost = abs(curr_allocation) * (curr_spread * 0.5)
            cost = np.where(prev_time.dt.time >= self.last_free_action_start_time, cost + forced_action_cost, cost)
            reward = np.where(prev_time.dt.time >= self.last_free_action_start_time, reward - forced_action_cost,
                              reward)
            reward = np.where(prev_time.dt.time >= self.first_forced_action_time, 0, reward)
            reward_nostd = reward
            pnl = curr_allocation * (curr_mid - prev_mid)
            pnl = np.where(prev_time.dt.time >= self.first_forced_action_time, 0, pnl)
        else:
            reward = curr_allocation * (curr_mid - prev_mid) - abs(curr_allocation - prev_allocation) * (prev_spread * 0.5 + 0.00625)
            cost = abs(curr_allocation - prev_allocation) * (prev_spread * 0.5 + 0.00625)

            forced_action_cost = abs(curr_allocation) * (curr_spread * 0.5 + 0.00625)
            cost = np.where(prev_time.dt.time >= self.last_free_action_start_time, cost + forced_action_cost, cost)
            #reward = np.where(prev_time.dt.time >= self.last_free_action_start_time, reward - forced_action_cost, reward)
            #reward = np.where(prev_time.dt.time >= self.first_forced_action_time, 0, reward)
            pnl = curr_allocation * (curr_mid - prev_mid)
            #pnl = np.where(prev_time.dt.time >= self.first_forced_action_time, 0, pnl)

            reward_nostd = reward

            #reward = (reward - mapped_rolling_mean) / (mapped_rolling_std + 0.0005)
            reward = reward / (mapped_rolling_std + 0.0005)
            reward = np.where(reward>=0, reward, 1.2*reward)
        
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
        reward = curr_allocation * (curr_mid - prev_mid) - abs(curr_allocation - prev_allocation) * (prev_spread * 0.5 + 0.00625)
        cost = abs(curr_allocation - prev_allocation) * (prev_spread * 0.5 + 0.00625)
        forced_action_cost = abs(curr_allocation) * (curr_spread * 0.5 + 0.00625)
        cost = np.where(prev_time.dt.time >= self.last_free_action_start_time, cost + forced_action_cost, cost)
        reward = np.where(prev_time.dt.time >= self.last_free_action_start_time, reward - forced_action_cost, reward)
        reward = np.where(prev_time.dt.time >= self.first_forced_action_time, 0, reward)
        pnl = curr_allocation * (curr_mid - prev_mid)
        pnl = np.where(prev_time.dt.time >= self.first_forced_action_time, 0, pnl)
        return reward, cost, pnl
    """
    def compute_reward_for_btp_and_bund(
            self,
            prev_time: pd.Series,
            prev_spread_btp: pd.Series,
            prev_mid_btp: pd.Series,
            curr_spread_btp: pd.Series,
            curr_mid_btp: pd.Series,
            prev_spread_bund: pd.Series,
            prev_mid_bund: pd.Series,
            curr_spread_bund: pd.Series,
            curr_mid_bund: pd.Series,
            prev_allocation: pd.Series,
            curr_allocation: pd.Series
    ):  
        
        curr_allocation_btp = np.select([curr_allocation == -1, curr_allocation == 1],[-1, 1],default=0)
        curr_allocation_bund = np.select([curr_allocation == -2, curr_allocation == 2],[-1, 1],default=0)

        if self.remove_costs is True:
            reward_btp = curr_allocation_btp * (curr_mid_btp - prev_mid_btp)
            reward_btp = np.where(prev_time.dt.time >= self.first_forced_action_time, 0, reward_btp)
            pnl_btp = curr_allocation_btp * (curr_mid_btp - prev_mid_btp)
            pnl_btp = np.where(prev_time.dt.time >= self.first_forced_action_time, 0, pnl_btp)

            reward_bund = curr_allocation_bund * (curr_mid_bund - prev_mid_bund)
            reward_bund = np.where(prev_time.dt.time >= self.first_forced_action_time, 0, reward_bund)
            pnl_bund = curr_allocation_bund * (curr_mid_bund - prev_mid_bund)
            pnl_bund = np.where(prev_time.dt.time >= self.first_forced_action_time, 0, pnl_bund)

            reward = reward_btp + reward_bund
            cost = 0
            pnl = pnl_btp + pnl_bund
        else:

            prev_allocation_btp = np.select([prev_allocation == -1, prev_allocation == 1],[-1, 1],default=0)
            prev_allocation_bund = np.select([prev_allocation == -2, prev_allocation == 2],[-1, 1],default=0)
            
            reward_btp, cost_btp, pnl_btp = self._compute_asset_reward(
                prev_time,prev_spread_btp,prev_mid_btp,curr_spread_btp,curr_mid_btp,prev_allocation_btp,curr_allocation_btp
            )

            reward_bund, cost_bund, pnl_bund = self._compute_asset_reward(
                prev_time,prev_spread_bund,prev_mid_bund,curr_spread_bund,curr_mid_bund,prev_allocation_bund,curr_allocation_bund
            )

            reward = reward_btp + reward_bund
            cost = cost_btp + cost_bund
            pnl = pnl_btp + pnl_bund
        
        return reward, cost, pnl
    """
    @abc.abstractmethod
    def build_current_state_features(self) -> typing.List[str]:
        pass

    @abc.abstractmethod
    def build_next_state_features(self) -> typing.List[str]:
        pass
    
    def _add_LOB_VOL_features(df: pd.DataFrame,number_of_levels: int, target_offset: int) -> pd.DataFrame:
        # To be implemented once we have evaluated results using volumes features in classification task
        return df
    
    def _add_LOB_Price_features(df: pd.DataFrame,number_of_levels: int, target_offset: int) -> pd.DataFrame:
        # To be implemented once we have evaluated results using LOB price features in classification task
        return df
