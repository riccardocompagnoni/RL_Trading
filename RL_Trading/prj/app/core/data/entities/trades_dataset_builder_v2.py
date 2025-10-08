import numpy as np
import os
import pandas as pd
import typing
from RL_Trading.prj.app.core.data.services.utils import extract_gzip, extract_zip


class TradesDatasetBuilderV2():

    def create_dataset(self, year: int, zip_root: str, ric_filters: typing.List[str]) -> pd.DataFrame:
        results = [pd.DataFrame() for _ in range(len(ric_filters))]
        for month in range(1, 13):
            try:
                # Extract csv from compressed files
                to_be_removed = []
                for f in os.listdir(zip_root):
                    if f.startswith(f'{year}-{month:02d}'):
                        if f.endswith('.zip'):
                            to_be_removed.append(extract_zip(zip_root, f))
                        elif f.endswith('.gz'):
                            to_be_removed.append(extract_gzip(zip_root, f))
                if to_be_removed:
                    for i, ric_filter in enumerate(ric_filters):
                        # Create DataFrame from csv files
                        dtf = self._read_csv(zip_root)
                        dtf = self._clean_data(dtf, ric_filter)
                        dtf = self._resample(dtf)
                        if len(dtf) > 0:
                            results[i] = pd.concat([results[i], dtf.fillna(0).reset_index()], axis=0)
                        # Release file binding
                        del dtf
            finally:
                for f in to_be_removed:
                    os.remove(f)
        return results

    @staticmethod
    def _read_csv(csv_root: str) -> pd.DataFrame:
        # List all the files in the folder
        data_files = [os.path.join(csv_root, f) for f in os.listdir(csv_root) if f.endswith(".csv")]
        df_merged = pd.concat([pd.read_csv(f) for f in data_files], axis=0)
        return df_merged

    @staticmethod
    def _clean_data(dtf: pd.DataFrame, ric_filter: str) -> pd.DataFrame:
        # Filter only the relevant instrument/type.
        dtf = dtf.loc[(dtf["#RIC"] == ric_filter) & (dtf["Type"] == "Trade")]
        dtf = dtf.dropna(subset=["Volume", "Price"])
        # Determine buy trades and sell trades
        dtf['buy_trade'] = dtf['Qualifiers'].str.contains('BID')
        dtf['sell_trade'] = dtf['Qualifiers'].str.contains('ASK')
        mask = (dtf['Qualifiers'].str.contains('BID') == False) & (dtf['Qualifiers'].str.contains('ASK') == False)
        dtf['buy_trade'] = np.where(mask, dtf['Price'] >= dtf['Ask Price'], dtf['buy_trade'])
        dtf['sell_trade'] = np.where(mask, dtf['Price'] <= dtf['Bid Price'], dtf['sell_trade'])
        dtf = dtf[['Date-Time', 'Volume', 'buy_trade', 'sell_trade']]
        return dtf

    @staticmethod
    def _resample(dtf: pd.DataFrame) -> pd.DataFrame:
        dtf['side'] = np.where(dtf['buy_trade'], "buy", np.where(dtf['sell_trade'], "sell", "unknown"))
        dtf['Date-Time'] = pd.to_datetime(dtf['Date-Time'], utc=True).dt.tz_convert('Europe/Rome')
        dtf = dtf.groupby([pd.Grouper(key='Date-Time', freq='1min'), 'side'])['Volume'].sum()
        dtf = dtf.reset_index()
        dtf = dtf.pivot_table(values='Volume', index='Date-Time', columns=['side'], aggfunc="sum", fill_value=0)
        dtf.columns.name = None  # Remove the name of the columns index
        return dtf
