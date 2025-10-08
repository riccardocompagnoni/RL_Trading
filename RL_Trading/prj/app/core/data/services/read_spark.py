from config.udm.sparkWrapper import SparkConfWrapper

import os
import pyspark
from pyspark.sql import SparkSession


def read_csv(csv_root: str, spark_config_path: str) -> pyspark.sql.dataframe.DataFrame:
    sw = SparkConfWrapper(spark_config_path)
    sw.add_basic()
    conf = sw.sc
    # Building spark session to load the data
    spark = SparkSession.builder.appName(sw.conf.get("basic").get("app.name", "ANAGRAFICA")).master(
        sw.conf.get("basic").get("spark.master", f"local[{conf.get('spark.cores.max')}]")).config(
        conf=conf).getOrCreate()
    # List all the files in the folder
    data_files = [os.path.join(csv_root, f) for f in os.listdir(csv_root) if f.endswith(".csv")]
    # Creation of a merged dataframe from all the files in the folder above
    df_merged = spark.read.format('csv').option('encoding', 'UTF-8').option(
        'delimiter', ',').option('header', True).load(path=data_files)
    return df_merged
