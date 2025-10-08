import datetime
import pandas as pd
import typing
from dataclasses import dataclass
from xbbg import blp


@dataclass(frozen=True)
class Auction():
    bid_to_cover: str
    amount_tendered: str
    amount_allotted: str
    average_yield: str
    average_price: str

    def retrieve_historical_data(
            self,
            start_date: datetime.datetime,
            end_date: typing.Optional[datetime.datetime] = None
    ) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.datetime.now()
        frames = [
            blp.bdh(self.bid_to_cover, "PX_LAST", start_date, end_date),
            blp.bdh(self.amount_allotted, "PX_LAST", start_date, end_date),
            blp.bdh(self.average_yield, "PX_LAST", start_date, end_date)
        ]
        for df in frames:
            if not df.empty:
                df.columns = df.columns.droplevel(1)
        df = pd.concat(frames, axis=1)
        df = df.rename(columns={
            self.bid_to_cover: 'bid_to_cover',
            self.amount_allotted: 'amount_allotted',
            self.average_yield: 'average_yield'
        })
        return df

    @staticmethod
    def clean_data(auction_df: pd.DataFrame, corrections: typing.List[typing.Dict]) -> pd.DataFrame:
        df = auction_df.copy()
        for correction in corrections:
            if correction['action'] == 'drop':
                df = df[df.index != correction['date']]
            elif correction['action'] == 'overwrite':
                df.loc[correction['date'], correction['column']] = correction['value']
            else:
                raise ValueError(f"Invalid correction action {correction['action']}.")
        return df