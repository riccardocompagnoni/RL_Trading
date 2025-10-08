import datetime
import itertools
import numpy as np
import numpy.typing as npt
import pandas as pd
import typing
import yaml

class DatasetBuilder:


    def __init__(self,
                data_root: str,
                config_path: str,
                persistence: int,
                mids_window: int=1,
                mids_offset: int=15,
                steps_tolerance: int=5,
                number_of_deltas: int=5,
                opening_hour: int=8,
                closing_hour: int=16,
                ohe_temporal_features: bool=False,
                volume_features: bool=False,
                lob_price_features: bool=False,
                volume_history_size: int=0,
                number_of_levels: int=10,
                use_higher_levels: bool=False,
                keep_deltas_of_non_operating_hours: bool=True,
                missing_values_handling: typing.Optional[str]=None,
                use_moving_average: bool=False,
                trades_venue: str='ICE',
                years: typing.List[int]=[2020, 2021, 2022],
    ):
        self.years = years
        self.data_root = data_root
        self.config_path = config_path
        self.mids_window = mids_window
        self.mids_offset = mids_offset
        self.steps_tolerance = steps_tolerance
        self.number_of_deltas = number_of_deltas
        self.opening_hour = opening_hour
        self.closing_hour = closing_hour
        self.ohe_temporal_features = ohe_temporal_features
        self.volume_features = volume_features
        self.volume_history_size = volume_history_size
        self.number_of_levels = number_of_levels
        self.use_higher_levels = use_higher_levels
        self.lob_price_features = lob_price_features
        self.use_moving_average = use_moving_average
        self.trades_venue = trades_venue
        self.persistence = persistence
        self.keep_deltas_of_non_operating_hours = keep_deltas_of_non_operating_hours
        self.missing_values_handling = missing_values_handling

        with open(self.config_path, "r") as file:
            dict_config = yaml.safe_load(file)

        self.lob_file_paths = [f'{self.data_root}/M1_ICE_{y}.csv' for y in self.years]
        self.trades_file_paths = [f'{self.data_root}/M1_trades.csv']
        self.anomalous_dates = dict_config['anomalous_dates']



    def build_dataset(self) -> pd.DataFrame:
        df = self._read_LOB_datasets()
        df = self._remove_dates(df, self.anomalous_dates)
        original_dates = df.index.date
        df = self._fill_time_grid(df)
        df = self._merge_trades_features(df)
        df = self._add_common_features(df)
        if self.missing_values_handling is not None:
            df = self.fill_nan(df)
        if self.volume_features:
            df = self._add_volume_features(df)
        df = self._add_ratio_features(df)
        df = self._add_pct_features(df)
        df = self._add_rsi_features(df, ['mid'], [14 * 60 * 10])
        df = self._add_macd_features(df, ['mid'], [(12 * 60 * 10, 26 * 60 * 10, 9 * 60 * 10)])
        df = self._add_ewm_features(df, ['pct_delta', 'pct_spread'], [30 * 60 * 10])
        df = self._add_std_features(df, ['pct_delta', 'pct_spread'], [30 * 60 * 10])
        df = self._add_level_features(df)
        df = self._add_standardized_features(df)
        df = self._build_next_state_dataset(df)
        df = self._clean_df(df, keep_bidask0=True, keep_volumes=self.volume_history_size > 0, depth=self.number_of_levels)
        df = self._remove_dates_final(df, self.anomalous_dates)

        df = df[df['time'].dt.date.isin(original_dates)]
        #df = df[df['time'].dt.date != datetime.date(2024, 6, 14)]

        df.to_parquet('/home/a2a/a2a/RL_Trading/prj/app/core/data/dataset_full.parquet')
        print('Done')
        #df.to_parquet('C:/Users/Riccardo/Documents/df_simulator.parquet')
        return df

    def _build_next_state_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        # Time features
        df['next_time_to_roll'] = df['time_to_roll'].shift(-self.persistence)
        # df['next_time_to_auction'] = df['time_to_auction'].shift(-self.persistence)
        df['next_minute_of_day'] = df['minute_of_day'].shift(-self.persistence)
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

        # df['next_total_quantity'] = df['total_quantity'].shift(-self.persistence)
        # df['next_total_bid_ask_imbalance'] = df['total_bid_ask_imbalance'].shift(-self.persistence)
        # df['next_max_bid_quantity'] = df['max_bid_quantity'].shift(-self.persistence)
        # df['next_max_ask_quantity'] = df['max_ask_quantity'].shift(-self.persistence)
        # df['next_max_bid_level'] = df['max_bid_level'].shift(-self.persistence)
        # df['next_max_ask_level'] = df['max_ask_level'].shift(-self.persistence)
        df['next_mid_rsi_8400'] = df['mid_rsi_8400'].shift(-self.persistence)
        df['next_mid_macd_7200_15600'] = df['mid_macd_7200_15600'].shift(-self.persistence)
        df['next_pct_delta_ewm_18000'] = df['pct_delta_ewm_18000'].shift(-self.persistence)
        df['next_pct_spread_ewm_18000'] = df['pct_spread_ewm_18000'].shift(-self.persistence)
        df['next_pct_delta_std_18000'] = df['pct_delta_std_18000'].shift(-self.persistence)
        df['next_pct_spread_std_18000'] = df['pct_spread_std_18000'].shift(-self.persistence)
        df['next_max_diff_ask_level'] = df['max_diff_ask_level'].shift(-self.persistence)
        df['next_max_diff_ask'] = df['max_diff_ask'].shift(-self.persistence)

        # TODO remove
        if self.use_higher_levels:
            df['next_mid_L2'] = df['mid_L2'].shift(-self.persistence)
            df['next_spread_L2'] = df['spread_L2'].shift(-self.persistence)
            df['next_mid_L3'] = df['mid_L3'].shift(-self.persistence)
            df['next_spread_L3'] = df['spread_L3'].shift(-self.persistence)

        if 'imbalance' in df.columns:
            df['next_imbalance'] = df['imbalance'].shift(-self.persistence)

        # TODO check what this does

        # Delta features
        # delta_mid_{j} and next_delta_mid_{j + 1} are equal.
        # The next state window and the current state window overlap and only the delta_mid from current state and next
        # state is inserted since it is the only non-overlapping delta.


        if not self.use_moving_average:
            for i in range(1, self.number_of_deltas):
                df[f'next_delta_mid_{i}'] = df[f'delta_mid_{i}'].shift(-self.persistence)
            df['next_delta_mid_0'] = df['delta_mid_0'].shift(-self.persistence)
        else:
            df['next_delta_mid_0_ma_60'] = df['delta_mid_0_ma_60'].shift(-self.persistence)
            df['next_delta_mid_0_ma_180'] = df['delta_mid_0_ma_180'].shift(-self.persistence)

        if self.use_higher_levels:
            df['next_delta_mid_0_L2'] = df['delta_mid_0_L2'].shift(-self.persistence)
            df['next_delta_mid_0_L3'] = df['delta_mid_0_L3'].shift(-self.persistence)
        # Return
        # Volume features
        # TODO what is this?
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
                # df[f'next_L{d}-Bid-mid_price_spread_0'] = df[f'L{d}-Bid-mid_price_spread_0'].shift(-self.persistence)
                # df[f'next_L{d}-Ask-mid_price_spread_0'] = df[f'L{d}-Ask-mid_price_spread_0'].shift(-self.persistence)

        return df

    def _add_standardized_features(self, df: pd.DataFrame) -> pd.DataFrame:

        temp_df = pd.DataFrame({
            'time': df['time'],
            'std_col': df['mid']
        })

        to_std = (temp_df.groupby('time')['std_col']
                  .max()
                  .reset_index())

        span = 60*8*1
        rolling_std = to_std['std_col'].diff(self.persistence).rolling(window=span, min_periods=span // 2).std().ffill().bfill()
        time_to_rolling_std = dict(zip(to_std['time'], rolling_std))
        mapped_rolling_std = df['time'].map(time_to_rolling_std) + 0.0005
        df['rolling_std'] = mapped_rolling_std


        rolling_mean = to_std['std_col'].diff(self.persistence).abs().rolling(window=span,
                                                                              min_periods=span // 2).mean().ffill().bfill()
        time_to_rolling_mean = dict(zip(to_std['time'], rolling_mean))
        mapped_rolling_mean = df['time'].map(time_to_rolling_mean)
        df['rolling_mean'] = mapped_rolling_mean

        df['mid_std'] = (df['mid'])/mapped_rolling_std
        df['spread_std'] = (df['spread'])/mapped_rolling_std
        for delta in [f'delta_mid_{i}' for i in range(1, self.number_of_deltas) if not self.use_moving_average]:
            df[delta] = (df[delta])/mapped_rolling_std

        return df

    @staticmethod
    def _clean_df(df: pd.DataFrame, keep_bidask0: bool, keep_volumes: bool, depth: int = 0) -> pd.DataFrame:
        # Remove useless columns.
        columns_to_drop = \
            [f'{name}_{level}' for name in ['bid', 'ask'] for level in range(depth)] + \
            [f'{name}_{level}_0' for name in ['bid_quantity', 'ask_quantity'] for level in range(depth)] if keep_volumes is False else []

        columns_to_keep = [c for c in df.columns if c not in columns_to_drop]
        if keep_bidask0:
            columns_to_keep.extend(['ask_0', 'bid_0'])
        df = df[columns_to_keep]

        unique_counts = df.nunique()
        #Enforce not to remove month features when OneHotEncoding is enabled and a whole month is not present in the dataset for some reason e.g. covid month
        columns_to_keep = list(set(unique_counts[unique_counts > 1].index.tolist() + [column for column in columns_to_keep if "month" in column]))
        df = df[columns_to_keep]
        # Return
        return df


    def fill_nan(self, df) -> pd.DataFrame: #avoiding filling mid price to drop days removed as rollover days, anomalous (performed by the Trainer)
        if self.missing_values_handling == 'ffill':
            df = pd.concat([df.pop('mid'), df.ffill().fillna(-1)], axis=1)
        elif self.missing_values_handling == 'constant':
            df = pd.concat([df.pop('mid'), df.fillna(-1)], axis=1)
        else:
            raise ValueError(f"{self.missing_values_handling} is not a valid missing value handling")
        return df

    def _read_LOB_datasets(self) -> pd.DataFrame:
        # The relevant subset of columns consists of the timestamp, the best bid and ask prices, and the bid and ask
        # order size of the first 4 levels of the LOB.

        columns_to_keep = \
            ['timestamp'] + ['prod'] + \
            [f'{name}_{level}' for name in ['bid_quantity','ask_quantity'] for level in range(self.number_of_levels)] + \
            [f'{name}_{level}' for name in ['bid','ask'] for level in range(self.number_of_levels)]

        # Concatenate all csv files into a single DataFrame, keeping only the relevant columns.
        df = pd.concat([pd.read_csv(f, usecols=columns_to_keep, parse_dates=['timestamp']) for f in self.lob_file_paths])
        df = df.set_index('timestamp')
        df.index = df.index.floor('min')
        original_name = [f'{name}_{level}' for name in ['bid_quantity','ask_quantity'] for level in range(self.number_of_levels)]
        rename_dict = {col: f"{col}_0" for col in original_name}
        df = df.rename(columns=rename_dict)

        return df


    def _fill_time_grid(self, df: pd.DataFrame) -> pd.DataFrame:
        # Insert the missing rows so that the dataset contains all the complete trading days (i.e., on a time grid
        # covering every minute from 8am to 7pm).
        df.index = df.index.floor('min')
        date_range = pd.bdate_range(df.index[0].date(), df.index[-1].date())
        if self.keep_deltas_of_non_operating_hours:
            time_range = pd.timedelta_range(f'07:00:00', f'17:00:00', freq=pd.offsets.Minute(1))
        else:
            time_range = pd.timedelta_range(f'0{self.opening_hour}:00:00', f'{self.closing_hour}:00:00', freq=pd.offsets.Minute(1))
        datetime_range = [date + time for (date, time) in itertools.product(date_range, time_range)]
        df = df.join(pd.DataFrame({'timestamp': datetime_range}).set_index('timestamp'), how='right')

        df['prod'] = (
            df.groupby(df.index.to_period('M'))['prod']
            .transform(lambda s: s.dropna().iloc[0] if s.notna().any() else pd.NA)
        )
        return df

    def _merge_trades_features(self, df: pd.DataFrame) -> pd.DataFrame:


        if len(self.trades_file_paths) > 0:

            loaded = pd.read_csv(self.trades_file_paths[0])

            if self.trades_venue.lower() != 'all':
                loaded = loaded[loaded['venueCode'] == self.trades_venue]

            loaded['timestamp'] = pd.to_datetime(loaded['timestamp'])
            loaded = loaded.set_index('timestamp')
            trades_per_minute = loaded.resample('min').sum()
            trades_per_minute.rename(columns={'quantity': 'traded_quantity'}, inplace=True)

            trades_per_minute = trades_per_minute[['traded_quantity']]

            df = df.merge(
                trades_per_minute,
                how='left',
                left_index=True,
                right_index=True,
            )

            df['traded_quantity'] = df['traded_quantity'].fillna(0)
            df['traded_quantity_rolling_mean'] = df['traded_quantity'].rolling(
                window=14 *60*10,
                min_periods=0
            ).mean()

        return df

    def _add_common_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add price and size features.
        df['spread'] = df['ask_0'] - df['bid_0']
        df['mid'] = (df['ask_0'] + df['bid_0']) / 2
        #df['mid_roll'] = df['mid'].rolling(self.mids_offset).mean()

        # Calculate delta mid prices.
        if self.mids_window == 1:
            # Delta is calculated between the current mid and the mid price of the previous mids_offset-th minute.
            # If the starting mid is NaN, the previous available mid price is considered up to steps_tolerance minutes before.

            if self.steps_tolerance != 0:
                df['delta_mid_0'] = df['mid'] - df['mid'].ffill(limit=self.steps_tolerance).shift(self.mids_offset)#) / df['mid'].ffill(limit=self.steps_tolerance).shift(self.mids_offset)
            else:
                df['delta_mid_0'] = df['mid'] - df['mid'].shift(self.mids_offset)#) / df['mid'].ffill(limit=self.steps_tolerance).shift(self.mids_offset)

           # df = df[~df.index.duplicated(keep='first')]

            #TODO check this false
            if self.use_moving_average:
                df['delta_mid_0_ma_60'] = df['delta_mid_0'].rolling(window=60, min_periods=1).mean()
                df['delta_mid_0_ma_180'] = df['delta_mid_0'].rolling(window=180, min_periods=1).mean()

            if self.use_higher_levels:
                df['delta_mid_0_L2'] = df['mid_L2'] - df['mid_L2'].ffill(limit=self.steps_tolerance).shift(self.mids_offset)
                df['delta_mid_0_L3'] = df['mid_L3'] - df['mid_L3'].ffill(limit=self.steps_tolerance).shift(self.mids_offset)

        else:
            # Delta is calculated between the current mid and the average mid price of a window starting mids_offset +
            # mids_window minutes earlier and ending mids_offset minutes earlier.
            # The maximum number of NaN values allowed in the window is equal to steps_tolerance.
            # Take the mean if there is at least one value not NaN (min_periods = 1)

            df['delta_mid_0'] = df['mid'] - \
                                df['mid'].rolling(window=self.mids_window, min_periods=1).mean().shift(self.mids_offset)
            
            if self.use_moving_average:
                df['delta_mid_0_ma_60'] = df['delta_mid_0'].rolling(window=60, min_periods=1).mean()
                df['delta_mid_0_ma_180'] = df['delta_mid_0'].rolling(window=180, min_periods=1).mean()

            if self.use_higher_levels:
                df['delta_mid_0_L2'] = df['mid_L2'] - \
                                    df['mid_L2'].rolling(window=self.mids_window, min_periods=1).mean().shift(self.mids_offset)
                df['delta_mid_0_L3'] = df['mid_L3'] - \
                                    df['mid_L3'].rolling(window=self.mids_window, min_periods=1).mean().shift(self.mids_offset)


        for i in range(1, self.number_of_deltas):
            if not self.use_moving_average:
                df[f'delta_mid_{i}'] = df['delta_mid_0'].shift(i * self.mids_offset)
            else: 
                df[f'delta_mid_{i}_ma_60'] = df['delta_mid_0_ma_60'].shift(i * self.mids_offset)
                df[f'delta_mid_{i}_ma_180'] = df['delta_mid_0_ma_180'].shift(i * self.mids_offset)
                # da aggiungere in fqi_dataset_builder_ik le features di ma nello stato gestendo la condizione delta_mid vs delta_mid_ma

            if self.use_higher_levels:
                df[f'delta_mid_{i}_L2'] = df['delta_mid_0_L2'].shift(i * self.mids_offset)
                df[f'delta_mid_{i}_L3'] = df['delta_mid_0_L3'].shift(i * self.mids_offset)


        if self.volume_history_size > 0:
            for d in range(self.number_of_levels):
                for i in range(self.volume_history_size):
                    df[f'bid_quantity_{d}_{i}'] = df[f'bid_quantity_{d}_0'].shift(i * self.mids_offset)
                    df[f'ask_quantity_{d}_{i}'] = df[f'ask_quantity_{d}_0'].shift(i * self.mids_offset)
            df = df.copy()

        df = df[(df.index.hour >= self.opening_hour) & (df.index.hour < self.closing_hour)]
        
        df = self._add_time_features(df)

        df.drop(['prod'], axis=1, inplace=True)
        
        return df
    
    def _add_time_features(self,df: pd.DataFrame) -> pd.DataFrame:

        df = df.reset_index().rename(columns={'timestamp': 'time'})

        df['time_to_roll'] = df.time.dt.daysinmonth - df.time.dt.day
        df['month'] = df['time'].dt.month
        df['day_of_week'] = df['time'].dt.dayofweek
        df['minute_of_day'] = df['time'].dt.hour * 60 + df['time'].dt.minute
        if self.ohe_temporal_features:
            #Handle possiblity of missing month or day of week in dataset (e.g. covid, one entire month is removed)
            df['month'] = df['month'].astype(pd.CategoricalDtype(categories=list(range(1, 13))))
            df['day_of_week'] = df['day_of_week'].astype(pd.CategoricalDtype(categories=list(range(7))))
            # Use one-hot encoding for day of week and month features.
            df = pd.get_dummies(df, columns=['day_of_week', 'month'])

        return df

    @staticmethod
    def _remove_dates(df: pd.DataFrame, dates: typing.List[datetime.date]) -> pd.DataFrame:
        return df[~df.index.floor('D').isin(pd.to_datetime(dates))]

    @staticmethod
    def _remove_dates_final(df: pd.DataFrame, dates: typing.List[datetime.date]) -> pd.DataFrame:
        return df[~df['time'].dt.date.isin(pd.to_datetime(dates))]

    @staticmethod
    def _calculate_time_to_dates(df: pd.DataFrame, dates: npt.NDArray[datetime.date]) -> typing.List[int]:
        """
        For each record of the dataset, this method calculates the number of days between the record's date and
        its supremum in the date array (i.e., the least element greater than or equal to).
        """
        times_to_date = []
        start_date = df.index[0].date()
        # Loop through sorted unique dates.
        for date in np.sort(np.unique(dates)):
            # Add one day to include all timestamps of the final date. Otherwise, only the 00:00 row of that date would
            # be included. Note that doing so will also include midnight of the following day, but this never happens as
            # the last trading time of each day is in late afternoon.
            end_date = date + datetime.timedelta(days=1)
            # Locate records in the dataset between the previous auction date the and current one (i.e., the records set
            # for which the value of end_date is the supremum).
            df_dates = df.loc[start_date:end_date].index.date
            # Calculate the differences in days until the supremum date for the current record block.
            times_to_date.extend((date - df_dates).astype('timedelta64[D]').astype(int))
            # Go to the next auction date.
            start_date = end_date
        return times_to_date

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

    @staticmethod
    def get_state_features(return_allocation=True):
        features = ['traded_quantity',
       'traded_quantity_rolling_mean', 'spread', 'mid', 'spread_std', 'mid_std', 'delta_mid_0',
       'time_to_roll', 'month', 'day_of_week', 'minute_of_day',
       'ratio_mid_date', 'ratio_mid_week', 'ratio_mid_month', 'ratio_spread_date',
       'ratio_spread_week', 'ratio_spread_month',
       'pct_spread', 'pct_delta', 'mid_rsi_8400',
       'mid_macd_7200_15600', 'pct_delta_ewm_18000', 'pct_spread_ewm_18000',
       'pct_delta_std_18000', 'pct_spread_std_18000', 'max_diff_ask_level',
       'max_diff_ask'] + [f'delta_mid_{i}' for i in range(1, 5)]

        if return_allocation:
            features = features + ['allocation']

        return features

    @staticmethod
    def get_next_state_features():
        return ['next_' + feat for feat in DatasetBuilder.get_state_features() if (feat!='allocation' and not (feat.startswith('delta_mid') and not feat.endswith('0')))] + [f'delta_mid_{i}' for i in range(1, 5)]


