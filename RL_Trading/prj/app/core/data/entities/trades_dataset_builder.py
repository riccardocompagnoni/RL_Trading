import datetime
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T
import typing
from RL_Trading.prj.app.core.data.entities.dataset_builder import DatasetBuilder
from dataclasses import dataclass


@dataclass(frozen=True)
class TradesDatasetBuilder(DatasetBuilder):

    @staticmethod
    def _clean_data(dtf: pyspark.sql.dataframe.DataFrame, ric_filter: str) -> pyspark.sql.dataframe.DataFrame:
        # Filter only the relevant instrument/type.
        dtf = dtf.filter((dtf["#RIC"] == ric_filter) & (dtf["Type"] == "Trade"))
        dtf = dtf.na.drop(subset=["Volume", "Price"])
        # Drop unused columns.
        columns_to_drop = ["#RIC", "Domain", "GMT Offset", "Type", "Market VWAP", "Qualifiers", "Seq No", "Exch Time",
                           "Date", "Bid Price", "Bid Size", "Ask Price", "Ask Size"]
        dtf = dtf.drop(*columns_to_drop)
        # Change the type of Date-Time column to pyspark.sql.types.TimestampType.
        # Beware: the nanoseconds precision (1e-9 seconds) of the original dataset will be lost, since
        # the timestamp type represents a time instant in microsecond precision (1e-6 seconds).
        datetime_format = "yyyy-MM-dd'T'HH:mm:ss.SSSSSSSSS'Z'"
        dtf = dtf.withColumn("Date-Time", F.to_timestamp("Date-Time", datetime_format))
        # Change timezone from UTC to Rome
        dtf = dtf.withColumn("Date-Time", F.from_utc_timestamp("Date-Time", "Europe/Rome"))
        return dtf

    @staticmethod
    def _resample(
            dtf: pyspark.sql.dataframe.DataFrame,
            window_duration: str
    ) -> pyspark.sql.dataframe.DataFrame:
        # Create a window using the window function of pyspark.
        window_spec = F.window("Date-Time", window_duration, window_duration)
        # Then apply it: group by the window function and get aggregate values
        resampled_df = dtf.groupBy(window_spec).agg(
            F.last("Date-Time").alias("Date-Time"),
            (F.sum(dtf["Price"] * dtf["Volume"]) / F.sum(dtf["Volume"])).alias("Price"),
            F.sum("Volume").alias("Volume")
        )
        # Drop the window column that is created in the process.
        resampled_df = resampled_df.drop("window")
        # Change the columns types of the dataframe:
        #   - Date-Time is already a timestamp
        #   - Price column is casted to 4-byte single-precision floating point numbers (i.e., float32)
        #   - Volume column is casted to 2-byte signed integer numbers (ranging from -32768 to 32767)
        casted_columns = [F.col("Date-Time"), F.col("Price").cast(T.FloatType()), F.col("Volume").cast(T.ShortType())]
        # Apply the casting using a select.
        #   The select function in this case is better than a repeated call of the withColumn function since the
        #   transformation occurs for many columns. Indeed, Spark DataFrame is immutable and cannot be modified
        #   inplace. Each operation on a DataFrame results in a new DataFrame. Therefore, every time the withColumn
        #   is called, a new DataFrame is created.
        resampled_df = resampled_df.select(casted_columns)
        return resampled_df


@dataclass(frozen=True)
class RollTradesDatasetBuilder(DatasetBuilder):

    @staticmethod
    def _clean_data(
            dtf: pyspark.sql.dataframe.DataFrame,
            ric_filter: str,
            roll_chains: typing.Dict[datetime.date, typing.List[str]],
    ) -> pyspark.sql.dataframe.DataFrame:
        # Drop missing data
        dtf = dtf.na.drop(subset=["Volume", "Price"])
        # Filter only the relevant type.
        dtf = dtf.filter(dtf["Type"] == "Trade")
        # Drop unused columns.
        columns_to_drop = ["Alias Underlying RIC", "Domain", "GMT Offset", "Type", "Market VWAP", "Qualifiers",
                           "Seq. No.", "Exch Time", "Date", "Open", "High", "Low", "Acc. Volume", "Turnover",
                           "Mid Price", "Percentage Change", "Implied Yield", "Unique Trade Identification",
                           "Aggressive Order Condition", "MMT Classification", "Bid Price", "Bid Size", "Ask Price",
                           "Ask Size"]
        dtf = dtf.drop(*columns_to_drop)
        # Change the type of Date-Time column to pyspark.sql.types.TimestampType.
        # Beware: the nanoseconds precision (1e-9 seconds) of the original dataset will be lost, since
        # the timestamp type represents a time instant in microsecond precision (1e-6 seconds).
        time_format = "HH:mm:ss.SSSSSSSSS"
        # dtf = dtf.withColumn("Exch Time", F.to_timestamp("Exch Timedtf.filter(dtf["#RIC"] == "FBTPH4-M4").collect()", time_format))
        datetime_format = f"yyyy-MM-dd'T'{time_format}'Z'"
        dtf = dtf.withColumn("Date-Time", F.to_timestamp("Date-Time", datetime_format))
        # Change timezone from UTC to Rome
        # dtf = dtf.withColumn("Exch Time", F.from_utc_timestamp("Exch Time", "Europe/Rome"))
        dtf = dtf.withColumn("Date-Time", F.from_utc_timestamp("Date-Time", "Europe/Rome"))
        # Filter only the relevant instrument.
        first_date = dtf.agg(F.min("Date-Time")).collect()[0][0].date()
        last_date = dtf.agg(F.max("Date-Time")).collect()[0][0].date()
        chain_idx = None
        if ric_filter == "FBTPc1-c2":
            chain_idx = 0
        elif ric_filter == "FBTPc2-c3":
            chain_idx = 1
        elif ric_filter == "FBTPc3-c4":
            chain_idx = 2
        prev_date = None
        for date, chain in roll_chains.items():
            if date > first_date:
                ric = chain[chain_idx]
                if prev_date is None:
                    mask = (F.col("Date-Time").cast("date") <= date) & (F.col("#RIC") == ric)
                else:
                    mask = mask | \
                           (F.col("Date-Time").cast("date") > prev_date) & (F.col("Date-Time").cast("date") <= date) & \
                           (F.col("#RIC") == ric)
                prev_date = date
            if date > last_date:
                break
        dtf = dtf.filter(mask)#.drop("#RIC")
        return dtf

    @staticmethod
    def _resample(
            dtf: pyspark.sql.dataframe.DataFrame,
            window_duration: str
    ) -> pyspark.sql.dataframe.DataFrame:
        # Create a window using the window function of pyspark.
        window_spec = F.window("Date-Time", window_duration, window_duration)
        # Then apply it: group by the window function and get aggregate values
        resampled_df = dtf.groupBy(window_spec).agg(
            F.last("Date-Time").alias("Date-Time"),
            (F.sum(dtf["Price"] * dtf["Volume"]) / F.sum(dtf["Volume"])).alias("Price"),
            F.sum("Volume").alias("Volume")
        )
        # Drop the window column that is created in the process.
        resampled_df = resampled_df.drop("window")
        # Change the columns types of the dataframe:
        #   - Date-Time is already a timestamp
        #   - Price column is casted to 4-byte single-precision floating point numbers (i.e., float32)
        #   - Volume column is casted to 2-byte signed integer numbers (ranging from -32768 to 32767)
        casted_columns = [F.col("Date-Time"), F.col("Price").cast(T.FloatType()), F.col("Volume").cast(T.ShortType())]
        # Apply the casting using a select.
        #   The select function in this case is better than a repeated call of the withColumn function since the
        #   transformation occurs for many columns. Indeed, Spark DataFrame is immutable and cannot be modified
        #   inplace. Each operation on a DataFrame results in a new DataFrame. Therefore, every time the withColumn
        #   is called, a new DataFrame is created.
        resampled_df = resampled_df.select(casted_columns)
        return resampled_df
