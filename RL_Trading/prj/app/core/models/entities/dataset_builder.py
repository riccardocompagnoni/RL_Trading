import abc
import datetime
import itertools
import numpy as np
import numpy.typing as npt
import pandas as pd
import typing
from dataclasses import dataclass
from pandas.tseries.offsets import BDay
from pandas.tseries.offsets import DateOffset
from pandas.tseries.offsets import Day


@dataclass(frozen=True)
class DatasetBuilder(abc.ABC):
    """A class to create the dataset used for some modeling task.

    :param lob_file_paths: paths of LOB csv files.
    :param trades_file_paths: paths of trades csv files.
    :param auctions_file_path: path of auctions csv file.
    :param anomalous_dates: list of anomalous dates to be removed from the original dataset.
    :param roll_dates: futures roll dates.
    :param roll_date_offset: number of days before roll to be removed from the original dataset.
    :param mids_window: number of records to consider in the calculation of the average mid used as starting point in
     the delta mid feature.
    :param mids_offset: time difference between previous and current mid in the delta mid calculation expressed as the
     number of records.
    :param steps_tolerance: maximum number of NaN records tolerated in the mids window. If mids_window is 1, the
     parameter defines the maximum number of NaN records tolerated before mids_offset (i.e., if the starting mid is NaN,
     the last available mid in a lookback window of steps_tolerance size is used).
    :param opening_hour: first hour of trading day.
    :param closing_hour: last hour of trading day.
    :param ohe_temporal_features: apply one-hot encoding to temporal features.
    :param volume_features: add features about volume
    """
    lob_file_paths: typing.List[str]
    trades_file_paths: typing.List[str]
    auctions_file_path: str
    anomalous_dates: typing.List[datetime.date]
    roll_dates: typing.List[datetime.date]
    roll_date_offset: int
    mids_window: int
    mids_offset: int
    steps_tolerance: int
    number_of_deltas: int
    number_of_deltas_bund: int
    opening_hour: int
    closing_hour: int
    ohe_temporal_features: bool
    volume_features: bool
    lob_price_features: bool
    skip_conte_I: bool
    skip_covid: bool
    events: typing.Dict[str, datetime.date]
    volume_history_size: int
    volume_history_size_bund: int
    number_of_levels: int
    number_of_levels_bund: int
    use_higher_levels: bool
    keep_deltas_of_non_operating_hours: bool
    remove_costs: bool
    remove_fixed_costs: bool
    missing_values_handling: typing.Optional[str]
    use_moving_average: bool
    perform_action_on_bund: bool
    perform_action_on_btp_bund: bool

    def build_dataset(self) -> pd.DataFrame:
        df = self._read_LOB_datasets()
        original_dates = df.index.date
        #TODO skipped this
        #df = self._manage_rolls(df)
        #TODO update dates
        df = self._remove_dates(df, self.anomalous_dates)
        df = self._filter_negative_spreads(df)


        #TODO is this rollover?
        #df = self.(df)
        df = self._fill_time_grid(df)
        #df = self._filter_events(df, self.skip_conte_I, self.skip_covid, self.events)
        df = self._merge_trades_features(df)
        df = self._add_common_features(df)
        if self.missing_values_handling is not None:
            #TODO check this
            df = self.fill_nan(df)
        df = self._add_task_specific_features(df)
        df = self._clean_df(df, self.volume_history_size > 0, self.number_of_levels)
        df = self._remove_dates_final(df, self.anomalous_dates)

        df = df[df['time'].dt.date.isin(original_dates)]
        #df.to_parquet('/home/a2a/a2a/RL_Trading/results/df_test_reward.parquet')
        #print('Done')
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

        #TODO modified this
        columns_to_keep = \
            ['timestamp'] + ['prod'] + \
            [f'{name}_{level}' for name in ['bid_quantity','ask_quantity'] for level in range(self.number_of_levels)] + \
            [f'{name}_{level}' for name in ['bid','ask'] for level in range(self.number_of_levels)]

        # Concatenate all csv files into a single DataFrame, keeping only the relevant columns.
        #TODO adjust file paths
        df = pd.concat([pd.read_csv(f, usecols=columns_to_keep, parse_dates=['timestamp']) for f in self.lob_file_paths])
        df = df.set_index('timestamp')
        df.index = df.index.floor('min')
        #TODO why does he rename with _0?
        original_name = [f'{name}_{level}' for name in ['bid_quantity','ask_quantity'] for level in range(self.number_of_levels)]
        rename_dict = {col: f"{col}_0" for col in original_name}
        df = df.rename(columns=rename_dict)

        return df


    def _manage_rolls(self, df: pd.DataFrame) -> pd.DataFrame:
        """This method calculates the time to next roll and removes the roll period from the dataset."""
        # Calculate days until the next roll. At the roll date the value is 0.
        df.loc[:, 'time_to_roll'] = self._calculate_time_to_dates(df, np.array(self.roll_dates))
        # Remove the roll period from the DataFrame.
        dates_to_remove = [
            d - pd.tseries.offsets.BDay(lag)
            for (lag, d) in itertools.product(range(self.roll_date_offset + 1), self.roll_dates)
        ]
        df = self._remove_dates(df, dates_to_remove)
        return df

    def _manage_auctions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method calculates for each row in the DataFrame the days until the next auction.
        """
        auction_df = self._read_auctions_datasets(df.index[0].date(), self.auctions_file_path)
        df.loc[:, 'time_to_auction'] = self._calculate_time_to_dates(df, auction_df['date'].dt.date.values)
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
            #TODO add venue filter

            loaded = pd.read_csv(self.trades_file_paths[0])

            #if self.filter_venue:
            loaded = loaded[loaded['venueCode'] == 'ICE']

            loaded['timestamp'] = pd.to_datetime(loaded['timestamp'])
            loaded = loaded.set_index('timestamp')
            trades_per_minute = loaded.resample('min').sum()
            trades_per_minute.rename(columns={'quantity': 'traded_quantity'}, inplace=True)

            trades_per_minute = trades_per_minute[['traded_quantity']]

            df = df.merge(
                trades_per_minute,
                #on=['year', 'month', 'day', 'hour', 'minute'],
                how='left',
                left_index=True,
                right_index=True,
                #suffixes=('', '_right')
            )

            #df = df.drop(columns=['timestamp_right'])

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
        #TODO check to make this false
        if self.use_higher_levels:
            df['spread_L2'] = df['L2-AskPrice'] - df['L2-BidPrice']
            df['mid_L2'] = (df['L2-AskPrice'] + df['L2-BidPrice']) / 2
            df['spread_L3'] = df['L2-AskPrice'] - df['L2-BidPrice']
            df['mid_L3'] = (df['L2-AskPrice'] + df['L2-BidPrice']) / 2
        #TODO remove?
        if 'buy' in df.columns:
            df['imbalance'] = df['buy'].fillna(0).rolling(60, min_periods=1).sum() - \
                              df['sell'].fillna(0).rolling(60, min_periods=1).sum()
        # Calculate delta mid prices.
        #TODO check this is set to one to avoid else
        if self.mids_window == 1:
            # Delta is calculated between the current mid and the mid price of the previous mids_offset-th minute.
            # If the starting mid is NaN, the previous available mid price is considered up to steps_tolerance minutes
            # before.
           # if self.keep_deltas_of_non_operating_hours is False:
           #     filtered_rows = df[df.index.hour == self.closing_hour]
           # else:
           #     filtered_rows = df[df.index.hour == 19]
           # duplicated_rows = filtered_rows.loc[np.repeat(filtered_rows.index, self.mids_offset)]
           # df = pd.concat([df, duplicated_rows]).sort_index()
            #TODO Shouldn't it use the ffill also at the beginning?
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
            # TODO: conviene fare la media centrata nell'offset?
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

        #if self.volume_history_size > 0 or self.volume_features:
        #    for d in range(1, self.number_of_levels + 1):
        #        df[f'L{d}-Bid-mid_price_spread_0'] = df['L1-BidPrice'] - df[f'L{d}-BidPrice']
        #        df[f'L{d}-Ask-mid_price_spread_0'] = df[f'L{d}-AskPrice'] - df['L1-AskPrice']

        #TODO check which volume history to use
        if self.volume_history_size > 0:
            for d in range(self.number_of_levels):
                for i in range(self.volume_history_size):
                    df[f'bid_quantity_{d}_{i}'] = df[f'bid_quantity_{d}_0'].shift(i * self.mids_offset)
                    df[f'ask_quantity_{d}_{i}'] = df[f'ask_quantity_{d}_0'].shift(i * self.mids_offset)
            df = df.copy()
        # TODO check how to add the volume and price features
        if self.volume_features:

           df = self._add_LOB_VOL_features(df, self.number_of_levels, self.mids_offset)

        if self.lob_price_features:
            df = self._add_LOB_Price_features(df,self.number_of_levels, self.mids_offset)

        #TODO make closing hour exclusive!!!!!!

        # Keep only times between opening and closing hours.
        df = df[(df.index.hour >= self.opening_hour) & (df.index.hour < self.closing_hour)]
        
        df = self._add_time_features(df)

        df.drop(['prod'], axis=1, inplace=True)
        
        return df
    
    def _add_time_features(self,df: pd.DataFrame) -> pd.DataFrame:

        df = df.reset_index().rename(columns={'timestamp': 'time'})
        #df['time_to_roll'] = (
        #    pd.to_datetime(df['prod'], format='%b-%y')
        #    .sub(Day(1))
        #    .dt.date.sub(df['time'].dt.date).apply(lambda x:x.days)
        #)
        df['time_to_roll'] = df.time.dt.daysinmonth - df.time.dt.day

        df['month'] = df['time'].dt.month
        df['day_of_week'] = df['time'].dt.dayofweek
        df['minute_of_day'] = df['time'].dt.hour * 60 + df['time'].dt.minute
        #TODO check whether 7 days of the week is ok? (changed from 6 to 7)
        if self.ohe_temporal_features:
            #Handle possiblity of missing month or day of week in dataset (e.g. covid, one entire month is removed)
            df['month'] = df['month'].astype(pd.CategoricalDtype(categories=list(range(1, 13))))
            df['day_of_week'] = df['day_of_week'].astype(pd.CategoricalDtype(categories=list(range(7))))
            # Use one-hot encoding for day of week and month features.
            df = pd.get_dummies(df, columns=['day_of_week', 'month'])

        return df

    @abc.abstractmethod
    def _add_task_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    @abc.abstractmethod
    def _clean_df(df: pd.DataFrame, keep_volumes: int, depth: int = 0) -> pd.DataFrame:
        pass

    @staticmethod
    @abc.abstractmethod
    def _read_auctions_datasets(start_date: datetime.date, file_name: str) -> pd.DataFrame:
        pass
    
    @staticmethod
    @abc.abstractmethod
    def _add_LOB_VOL_features(df: pd.DataFrame,number_of_levels: int, target_offset: int) -> pd.DataFrame:
        pass

    @staticmethod
    @abc.abstractmethod
    def _add_LOB_Price_features(df: pd.DataFrame,number_of_levels: int, target_offset: int) -> pd.DataFrame:
        pass

    @staticmethod
    def _remove_dates(df: pd.DataFrame, dates: typing.List[datetime.date]) -> pd.DataFrame:
        return df[~df.index.floor('D').isin(pd.to_datetime(dates))]

    @staticmethod
    def _remove_dates_final(df: pd.DataFrame, dates: typing.List[datetime.date]) -> pd.DataFrame:
        return df[~df['time'].dt.date.isin(pd.to_datetime(dates))]


    @staticmethod
    def _filter_events(df: pd.DataFrame, skip_conte_I: bool, skip_covid: bool, events: typing.Dict[str,  datetime.date]) -> pd.DataFrame:

        if skip_conte_I:
            df = df[~df.index.floor('D').isin(pd.date_range(events['conte_I_period'][0], events['conte_I_period'][1], freq=BDay()))]

        if skip_covid:
            df = df[~df.index.floor('D').isin(pd.date_range(events['covid_period'][0], events['covid_period'][1], freq=BDay()))]

        return df

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
    def _filter_negative_spreads(df):

        dates = [datetime.datetime(2020, 3, 2), datetime.datetime(2022, 7, 11), datetime.datetime(2022, 10, 5),
                 datetime.datetime(2024, 6, 14)]

        df = df[~df.index.floor('D').isin(dates)]
        return df
