import datetime
import pandas as pd
from RL_Trading.prj.app.core.classification.entities.classification_dataset_builder import ClassificationDatasetBuilder


class IKClassificationDatasetBuilder(ClassificationDatasetBuilder):

    def _add_task_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # df['imbalance'] = \
        #     df[[c for c in df.columns if c.endswith('BidSize')]].sum(axis=1) - \
        #     df[[c for c in df.columns if c.endswith('AskSize')]].sum(axis=1)
        # Calculate the target variable as the difference between the mid price associated with the next target_offset
        # minute and the current mid price. If the ending mid is NaN, the previous available mid price is considered up
        # to steps_tolerance minutes before.
        # TODO: Se il final mid è NaN, si prende l'ultimo mid disponibile fino a steps_tolerance steps precedenti. Nella
        #  versione precedente del codice si andava a prendere il primo mid disponibile successivo fino a steps_tolerance.
        #  Cosa si vuol fare?
        df['delta_mid_target'] = df['mid'].ffill(limit=self.steps_tolerance).shift(-self.target_offset) - df['mid']
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
