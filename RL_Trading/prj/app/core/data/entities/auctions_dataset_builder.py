import datetime
import numpy as np
import os
import pandas as pd
import typing
from dataclasses import dataclass


@dataclass(frozen=True)
class AuctionsDatasetBuilder():
    MEF_COLUMNS_MAP: typing.ClassVar[typing.Dict[int, str]] = {
        0: 'date',
        # 1: 'value_date',
        2: 'ISIN',
        # 3: 'tranche_number',
        4: 'ordinary_auction',
        5: 'description',
        6: 'maturity',
        # 7: 'indexing_coefficient',
        8: 'type',
        9: 'amount_offered',
        10: 'amount_offered_min',
        11: 'amount_offered_max',
        12: 'amount_tendered',
        13: 'amount_allotted',
        14: 'average_price',
        15: 'average_yield_BOT',
        16: 'average_yield',
        17: 'bidders_number'
    }

    def build_dataset(
            self,
            auctions_source: str,
            bbg_auctions_config: typing.Optional[typing.Dict] = None,
            mef_data_root: typing.Optional[str] = None
    ) -> pd.DataFrame:
        if auctions_source == "Bloomberg":
            return self._create_dataset_from_bloomberg(bbg_auctions_config)
        elif auctions_source == "MEF":
            return self._create_dataset_from_mef(mef_data_root)
        else:
            raise AttributeError(f"Invalid auctions_source {auctions_source}")

    def _create_dataset_from_bloomberg(self, auctions_config: typing.Dict) -> pd.DataFrame:
        from RL_Trading.prj.app.core.data.entities.auction import Auction
        df = pd.DataFrame()
        start_date = auctions_config["bdh_start_date"]
        for key, config in auctions_config["auction_types"].items():
            auction = Auction(
                bid_to_cover=config['tickers']['bid_to_cover'],
                average_yield=config['tickers']['average_yield'],
                average_price=config['tickers']['average_price'],
                amount_tendered=config['tickers']['amount_tendered'],
                amount_allotted=config['tickers']['amount_allotted']
            )
            _df = auction.retrieve_historical_data(start_date)
            if not _df.empty:
                if 'corrections' in config:
                    _df = auction.clean_data(_df, config['corrections'])
                _df['type'] = key
                df = pd.concat([df, _df], axis=0)
        df = df.sort_index().reset_index().rename(columns={'index': 'date'})
        return df

    def _create_dataset_from_mef(self, xlsx_root: str) -> pd.DataFrame:
        dfs = []
        for f in os.listdir(xlsx_root):
            if f.endswith('.xlsx') | f.endswith('.xls'):
                xlsx_file_path = os.path.join(xlsx_root, f)
                dfs.append(pd.read_excel(
                    xlsx_file_path,
                    skiprows=[0, 1, 2, 3],
                    header=None,
                    usecols=list(self.MEF_COLUMNS_MAP.keys()))
                )
        df = pd.concat(dfs, axis=0)
        df.columns = list(self.MEF_COLUMNS_MAP.values())
        df = df[df['ordinary_auction'] == 'O']
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df['maturity'] = (pd.to_datetime(df['maturity'], format='%Y-%m-%d') - df['date']).dt.days / 365
        df['average_yield'] = np.where(df['average_yield'] != 0, df['average_yield'], df['average_yield_BOT'])
        df['bid_to_cover'] = df['amount_tendered'] / df['amount_allotted']
        df = df.drop(columns=['ordinary_auction', 'average_yield_BOT'])
        df = df.sort_values(by=['date', 'maturity']).reset_index(drop=True)
        return df
