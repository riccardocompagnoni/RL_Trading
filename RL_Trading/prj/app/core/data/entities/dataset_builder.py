import abc
import os
import pandas as pd
import pyspark
import pyspark.sql.functions as F
import typing
from RL_Trading.prj.app.core.data.services.read_spark import read_csv
from RL_Trading.prj.app.core.data.services.utils import extract_gzip, extract_zip
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetBuilder(abc.ABC):

    def build_dataset(
            self,
            year: int,
            window_duration: str,
            zip_root: str,
            spark_config_path: str,
            ric_filters: typing.List[str],
            roll_chains: typing.Optional[typing.List[str]] = None,
            relevant_dates: typing.Optional[typing.List[str]] = None
    ) -> pd.DataFrame:
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
                for i, ric_filter in enumerate(ric_filters):
                    # Create Pyspark DataFrame from csv files
                    dtf = read_csv(zip_root, spark_config_path)
                    if roll_chains:
                        dtf = self._clean_data(dtf, ric_filter, roll_chains)
                    else:
                        dtf = self._clean_data(dtf, ric_filter)
                    # Pyspark does not have a built-in function for the resample of the dataframe such as pandas,
                    # so an ad-hoc function has been built.
                    dtf = self._resample(dtf, window_duration)
                    # Remove roll dates
                    dtf = self._filter_business_logics(dtf, relevant_dates)
                    # Convert Pyspark DataFrame to Pandas
                    if dtf.count() > 0:
                        results[i] = pd.concat([results[i], dtf.toPandas()], axis=0)
                    # Release file binding
                    del dtf
            finally:
                for f in to_be_removed:
                    os.remove(f)
        return results

    @staticmethod
    @abc.abstractmethod
    def _clean_data(
            dtf: pyspark.sql.dataframe.DataFrame,
            ric_filter: str
    ) -> pyspark.sql.dataframe.DataFrame:
        pass

    @staticmethod
    @abc.abstractmethod
    def _resample(
            dtf: pyspark.sql.dataframe.DataFrame,
            window_duration: str
    ) -> pyspark.sql.dataframe.DataFrame:
        pass

    @staticmethod
    def _filter_business_logics(
            dtf: pyspark.sql.dataframe.DataFrame,
            relevant_dates: typing.List[str]
    ) -> pyspark.sql.dataframe.DataFrame:
        if relevant_dates is not None:
            # Filter out dates in which the roll of the asset is performed
            dtf = dtf.filter(~F.col("Date-Time").isin(relevant_dates))
        return dtf
