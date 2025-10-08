import datetime
import pandas as pd
import typing
import itertools
import numpy as np
from matplotlib import pyplot as plt

from RL_Trading.prj.app.core.fqi.entities.fqi_dataset_builder import FQIDatasetBuilder


class IKFQIDatasetBuilder(FQIDatasetBuilder):

    def _add_task_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._add_ratio_features(df)
        #df = self._add_volume_features(df)
        df = self._add_pct_features(df)
        df = self._add_rsi_features(df, ['mid'], [14 *60*10])
        df = self._add_macd_features(df, ['mid'], [(12 *60*10, 26 *60*10, 9*60*10)])
        df = self._add_ewm_features(df, ['pct_delta', 'pct_spread'],[30*60*10])
        df = self._add_std_features(df, ['pct_delta', 'pct_spread'], [30*60*10])
        df = self._add_level_features(df)
        df = self.standardize(df)
        df = self._build_next_state_dataset(df)
        df = self._build_allocation_action_pairs(df)

        reward, cost, pnl, reward_nostd = self.compute_reward(df['time'], df['spread'], df['mid'],
                                       df['next_spread'], df['next_mid'], df['allocation'], df['action'])

        df['reward'] = reward
        df['cost'] = cost
        df['pnl'] = pnl
        df['reward_nostd'] = reward_nostd
        #df = self._filter_datasetdf)

        return df


    def _build_next_state_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        # Time features
        df['next_time_to_roll'] = df['time_to_roll'].shift(-self.persistence)
        #df['next_time_to_auction'] = df['time_to_auction'].shift(-self.persistence)
        df['next_minute_of_day'] = df['minute_of_day'].shift(-self.persistence)
        #TODO still, check why 5 and not 7 days of week
        if self.ohe_temporal_features:
            for i in range(7):
                df[f'next_day_of_week_{i}'] = df[f'day_of_week_{i}'].shift(-self.persistence)
            for i in range(1, 13):
                df[f'next_month_{i}'] = df[f'month_{i}'].shift(-self.persistence)
        else:
            df['next_day_of_week'] = df['day_of_week'].shift(-self.persistence)
            df['next_month'] = df['month'].shift(-self.persistence)
        # Price features
        df['next_mid'] = df['mid'].shift(-self.persistence)
        df['next_spread'] = df['spread'].shift(-self.persistence)
        df['next_mid_std'] = df['mid_std'].shift(-self.persistence)
        df['next_spread_std'] = df['spread_std'].shift(-self.persistence)
        df['next_pct_delta'] = df['pct_delta'].shift(-self.persistence)
        df['next_pct_spread'] = df['pct_spread'].shift(-self.persistence)

        df['next_traded_quantity'] = df['traded_quantity'].shift(-self.persistence)
        df['next_traded_quantity_rolling_mean'] = df['traded_quantity_rolling_mean'].shift(-self.persistence)
        df['next_ratio_mid_date'] = df['ratio_mid_date'].shift(-self.persistence)
        df['next_ratio_mid_week'] = df['ratio_mid_week'].shift(-self.persistence)
        df['next_ratio_mid_month'] = df['ratio_mid_month'].shift(-self.persistence)
        df['next_ratio_spread_date'] = df['ratio_spread_date'].shift(-self.persistence)
        df['next_ratio_spread_week'] = df['ratio_spread_week'].shift(-self.persistence)
        df['next_ratio_spread_month'] = df['ratio_spread_month'].shift(-self.persistence)

        #df['next_total_quantity'] = df['total_quantity'].shift(-self.persistence)
        #df['next_total_bid_ask_imbalance'] = df['total_bid_ask_imbalance'].shift(-self.persistence)
        #df['next_max_bid_quantity'] = df['max_bid_quantity'].shift(-self.persistence)
        #df['next_max_ask_quantity'] = df['max_ask_quantity'].shift(-self.persistence)
        #df['next_max_bid_level'] = df['max_bid_level'].shift(-self.persistence)
        #df['next_max_ask_level'] = df['max_ask_level'].shift(-self.persistence)
        df['next_mid_rsi_8400'] = df['mid_rsi_8400'].shift(-self.persistence)
        df['next_mid_macd_7200_15600'] = df['mid_macd_7200_15600'].shift(-self.persistence)
        df['next_pct_delta_ewm_18000'] = df['pct_delta_ewm_18000'].shift(-self.persistence)
        df['next_pct_spread_ewm_18000'] = df['pct_spread_ewm_18000'].shift(-self.persistence)
        df['next_pct_delta_std_18000'] = df['pct_delta_std_18000'].shift(-self.persistence)
        df['next_pct_spread_std_18000'] = df['pct_spread_std_18000'].shift(-self.persistence)
        df['next_max_diff_ask_level'] = df['max_diff_ask_level'].shift(-self.persistence)
        df['next_max_diff_ask'] = df['max_diff_ask'].shift(-self.persistence)

        #TODO remove
        if self.use_higher_levels:
            df['next_mid_L2'] = df['mid_L2'].shift(-self.persistence)
            df['next_spread_L2'] = df['spread_L2'].shift(-self.persistence)
            df['next_mid_L3'] = df['mid_L3'].shift(-self.persistence)
            df['next_spread_L3'] = df['spread_L3'].shift(-self.persistence)

        if 'imbalance' in df.columns:
            df['next_imbalance'] = df['imbalance'].shift(-self.persistence)

        #TODO check what this does

        # Delta features
        # delta_mid_{j} and next_delta_mid_{j + 1} are equal.
        # The next state window and the current state window overlap and only the delta_mid from current state and next
        # state is inserted since it is the only non-overlapping delta.

        if not self.use_moving_average:
            df['next_delta_mid_0'] = df['delta_mid_0'].shift(-self.persistence)
        else:
            df['next_delta_mid_0_ma_60'] = df['delta_mid_0_ma_60'].shift(-self.persistence)
            df['next_delta_mid_0_ma_180'] = df['delta_mid_0_ma_180'].shift(-self.persistence)

        if self.use_higher_levels:
            df['next_delta_mid_0_L2'] = df['delta_mid_0_L2'].shift(-self.persistence)
            df['next_delta_mid_0_L3'] = df['delta_mid_0_L3'].shift(-self.persistence)
        # Return
        #Volume features
        #TODO what is this?
        """
        if self.volume_features:
            for win_size in [30, 60, 120, 240, 360]:
                df[f'next_L1-BidSize_window_{win_size}_sum'] = df[f'L1-BidSize_window_{win_size}_sum'].shift(-self.persistence)
                df[f'next_L1-AskSize_window_{win_size}_sum'] = df[f'L1-AskSize_window_{win_size}_sum'].shift(-self.persistence)
                df[f'next_L1-BidSize_window_{win_size}_min'] = df[f'L1-BidSize_window_{win_size}_min'].shift(-self.persistence)
                df[f'next_L1-AskSize_window_{win_size}_min'] = df[f'L1-AskSize_window_{win_size}_min'].shift(-self.persistence)
                df[f'next_L1-BidSize_window_{win_size}_max'] = df[f'L1-BidSize_window_{win_size}_max'].shift(-self.persistence)
                df[f'next_L1-AskSize_window_{win_size}_max'] = df[f'L1-AskSize_window_{win_size}_max'].shift(-self.persistence)

        """

        if self.volume_history_size > 0:
            for d in range(1, self.number_of_levels):
                df[f'next_bid_quantity_{d}_0'] = df[f'bid_quantity_{d}_0'].shift(-self.persistence)
                df[f'next_ask_quantity_{d}_0'] = df[f'ask_quantity_{d}_0'].shift(-self.persistence)
                #df[f'next_L{d}-Bid-mid_price_spread_0'] = df[f'L{d}-Bid-mid_price_spread_0'].shift(-self.persistence)
                #df[f'next_L{d}-Ask-mid_price_spread_0'] = df[f'L{d}-Ask-mid_price_spread_0'].shift(-self.persistence)
        
        return df

    @staticmethod
    def _read_auctions_datasets(start_date: datetime.date, file_name: str) -> pd.DataFrame:
        """
        This method returns a dataset containing auctions occuring after start_date relating to BTPs with a maturity of
        around 10 years
        """
        df = pd.read_csv(file_name, parse_dates=['date'])
        df = df[df['date'].dt.date >= start_date]
        df = df[df['type'] == "BTP"]
        df = df[(df['maturity'] > 9) & (df['maturity'] < 11)]
        return df

    def build_current_state_features(self) -> typing.List[str]:

        curr_state_features = (['traded_quantity',
       'traded_quantity_rolling_mean', 'spread', 'mid', 'spread_std', 'mid_std', 'delta_mid_0',
       'time_to_roll', 'month', 'day_of_week', 'minute_of_day',
       'ratio_mid_date', 'ratio_mid_week', 'ratio_mid_month', 'ratio_spread_date',
       'ratio_spread_week', 'ratio_spread_month',
       #'total_quantity', 'total_bid_ask_imbalance', 'max_bid_quantity', 'max_ask_quantity', 'max_bid_level','max_ask_level',
       'pct_spread', 'pct_delta', 'mid_rsi_8400',
       'mid_macd_7200_15600', 'pct_delta_ewm_18000', 'pct_spread_ewm_18000',
       'pct_delta_std_18000', 'pct_spread_std_18000', 'max_diff_ask_level',
       'max_diff_ask'] +
       [f'delta_mid_{i}' for i in range(1, self.number_of_deltas) if not self.use_moving_average]+
       ['allocation'])

        return curr_state_features

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:

        temp_df = pd.DataFrame({
            'time': df['time'],
            'std_col': df['mid']
        })

        to_std = (temp_df.groupby('time')['std_col']
                  .max()
                  .reset_index())

        span = 60*8*1
        rolling_std = to_std['std_col'].diff(self.persistence).rolling(window=span, min_periods=span // 2).std().ffill()
        time_to_rolling_std = dict(zip(to_std['time'], rolling_std))
        mapped_rolling_std = df['time'].map(time_to_rolling_std) + 0.0005


        rolling_mean = to_std['std_col'].diff(self.persistence).abs().rolling(window=span,
                                                                              min_periods=span // 2).mean().ffill()
        time_to_rolling_mean = dict(zip(to_std['time'], rolling_mean))
        mapped_rolling_mean = df['time'].map(time_to_rolling_mean)
        mapped_rolling_mean = 0

        df['mid_std'] = (df['mid']-mapped_rolling_mean)/mapped_rolling_std
        df['spread_std'] = (df['spread']-mapped_rolling_mean)/mapped_rolling_std
        for delta in [f'delta_mid_{i}' for i in range(1, self.number_of_deltas) if not self.use_moving_average]:
            df[delta] = (df[delta]-mapped_rolling_mean)/mapped_rolling_std

        return df


    def build_next_state_features(self) -> typing.List[str]:

        next_state_features =(
        ['next_traded_quantity',
         'next_traded_quantity_rolling_mean',
         'next_spread',
         'next_mid',
         'next_spread_std',
         'next_mid_std',
         'next_delta_mid_0',
         'next_time_to_roll',
         'next_month',
         'next_day_of_week',
         'next_minute_of_day',
         'next_ratio_mid_date',
         'next_ratio_mid_week',
         'next_ratio_mid_month',
         'next_ratio_spread_date',
         'next_ratio_spread_week',
         'next_ratio_spread_month',
         #'next_total_quantity',
         #'next_total_bid_ask_imbalance',
         #'next_max_bid_quantity',
         #'next_max_ask_quantity',
         #'next_max_bid_level',
         #'next_max_ask_level',
         'next_pct_spread',
         'next_pct_delta',
         'next_mid_rsi_8400',
         'next_mid_macd_7200_15600',
         'next_pct_delta_ewm_18000',
         'next_pct_spread_ewm_18000',
         'next_pct_delta_std_18000',
         'next_pct_spread_std_18000',
         'next_max_diff_ask_level',
         'next_max_diff_ask',
        ] + [f'delta_mid_{i}' for i in range(self.number_of_deltas -1) if not self.use_moving_average])

        return next_state_features

    @staticmethod
    def _add_ratio_features(df):

        ratio_columns = ['mid', 'spread']

        df['date'] = df['time'].dt.date
        df['week'] = df['time'].dt.isocalendar().week
        df['year'] = df['time'].dt.year

        for col in ratio_columns:
            week_means = (
                df.groupby(['year', 'week'], sort=False)[col]  # keep original order
                .mean()
                .rename('mean_' + col)
                .reset_index()
                .sort_values(['year', 'week'])  # ensure correct shift
            )

            week_means['prev_week_mean'] = week_means['mean_' + col].shift(1)
            week_means = week_means.drop(columns='mean_' + col).ffill(limit=1)

            df = df.merge(week_means, on=['year', 'week'], how='left')

            month_means = (
                df.groupby(['year', 'month'], sort=False)[col]  # keep original order
                .mean()
                .rename('mean_' + col)
                .reset_index()
                .sort_values(['year', 'month'])  # ensure correct shift
            )

            month_means['prev_month_mean'] = month_means['mean_' + col].shift(1)
            month_means = month_means.drop(columns='mean_' + col).ffill(limit=1)

            df = df.merge(month_means, on=['year', 'month'], how='left')

            first_per_date = df.groupby('date')[col].transform('first').replace(0.0, 1e-8)
            df[f'ratio_{col}_date'] = df[col].div(first_per_date)

            df[f'ratio_{col}_week'] = (
                df[col].div(df['prev_week_mean'])
                .replace([np.inf, -np.inf, np.nan], 1.0)
            )

            df[f'ratio_{col}_month'] = (
                df[col].div(df['prev_month_mean'])
                .replace([np.inf, -np.inf, np.nan], 1.0)
            )

            df = df.drop(columns=['prev_week_mean', 'prev_month_mean'])


        df = df.drop(columns=['year', 'date', 'week'])

        return df


    def _add_volume_features(self, df):

        bid_cols = [f"bid_quantity_{i}_0" for i in range(self.number_of_levels)]
        ask_cols = [f"ask_quantity_{i}_0" for i in range(self.number_of_levels)]

        df["total_bid_quantity"] = df[bid_cols].sum(axis=1)
        df["total_ask_quantity"] = df[ask_cols].sum(axis=1)

        df["total_quantity"] = df["total_bid_quantity"] + df["total_ask_quantity"]
        df["total_bid_ask_imbalance"] = (
            (df["total_bid_quantity"] - df["total_ask_quantity"])
            .div(df["total_quantity"].replace(0., 1.))
        )

        df["max_bid_quantity"] = df[bid_cols].max(axis=1)
        df["max_ask_quantity"] = df[ask_cols].max(axis=1)

        df["max_bid_level"] = df[bid_cols].values.argmax(axis=1)
        df["max_ask_level"] = df[ask_cols].values.argmax(axis=1)

        df = df.drop(columns=["total_bid_quantity", "total_ask_quantity"])

        return df

    @staticmethod
    def _add_pct_features(df):

        df['pct_spread'] = df['spread'].div(df['ask_0'])
        df['pct_delta'] = df['delta_mid_0'].div(df['mid'])

        return df

    @staticmethod
    def _add_rsi_features(df, cols, windows):

        result_df = df.copy()

        for col in cols:
            result_df[f'pct_change_{col}'] = result_df[col].pct_change().fillna(0)
            for window in windows:
                result_df[f'{col}_{window}_gain'] = result_df[f'pct_change_{col}'].clip(lower=0).ewm(span=window).mean()

                result_df[f'{col}_{window}_loss'] = result_df[f'pct_change_{col}'].clip(upper=0).abs().ewm(
                    span=window).mean()

                result_df[f'{col}_{window}_rs'] = result_df[f'{col}_{window}_gain'] / (
                            result_df[f'{col}_{window}_loss'] + 1e-12)

                result_df[f'{col}_rsi_{window}'] = 100 - (100 / (1 + result_df[f'{col}_{window}_rs']))

        to_be_deleted = [col for col in result_df.columns if
                         col.endswith(('_rs', '_gain', '_loss')) or col.startswith('pct_change')]

        result_df = result_df.drop(columns=to_be_deleted)

        return result_df

    @staticmethod
    def _add_macd_features(df, cols, spans):


        for i in range(len(spans)):
            for col in cols:
                for j in range(2):
                    df[f'{col}_ewm_{i}_{spans[i][j]}'] = df[col].ewm(span=spans[i][j]).mean()

        for i in range(len(spans)):
            for col in cols:
                df[f'{col}_macd_{spans[i][0]}_{spans[i][1]}'] = (
                        df[f'{col}_ewm_{i}_{spans[i][0]}'] - df[f'{col}_ewm_{i}_{spans[i][1]}']
                )


        to_drop = [f'{col}_ewm_{i}_{spans[i][j]}' for col in cols for i in range(len(spans)) for j in range(2)]
        df = df.drop(columns=to_drop)

        return df


    @staticmethod
    def _add_ewm_features(df, cols, spans):

        for col in cols:
            for span in spans:
                df[f'{col}_ewm_{span}'] = df[col].ewm(
                    span=span,
                    adjust=True,
                    ignore_na=False
                ).mean()

        return df

    @staticmethod
    def _add_std_features(df, cols, spans):

        for col in cols:
            for span in spans:
                # Calculate rolling standard deviation with minimum periods
                df[f'{col}_std_{span}'] = df[col].rolling(
                    window=span,
                    min_periods=span // 2
                ).std()


        return df

    @staticmethod
    def _add_level_features(df):
        diffs = {}
        for i in range(9):
            curr_col = f'ask_{i}'
            next_col = f'ask_{i + 1}'
            diffs[curr_col] = df[next_col] - df[curr_col]

        diffs_df = pd.DataFrame(diffs)
        df['max_diff_ask_level'] = diffs_df.idxmax(axis=1).fillna('ask_0').str.extract(r'ask_(\d+)').astype(int)
        df['max_diff_ask'] = diffs_df.max(axis=1).fillna(0)

        return df
