import os
import pandas as pd
import typing
import yaml
import datetime


def auctions_preprocessing(xlsx_root: str, data_root: str, config_path: str):
    from RL_Trading.prj.app.core.data.entities.auctions_dataset_builder import AuctionsDatasetBuilder
    with open(config_path, "r") as file:
        dict_config = yaml.safe_load(file)
    builder = AuctionsDatasetBuilder()
    dtf = builder.build_dataset(
        dict_config['auctions_source'],
        dict_config['bloomberg_auctions_config'],
        xlsx_root,
    )
    if not dtf.empty:
        save_file_path = os.path.join(data_root, "IKA_auctions.csv")
        dtf.to_csv(save_file_path, header=True, index=False)


def lob_preprocessing(
        zip_root: str,
        data_root: str,
        spark_config_path: str,
        config_path: str,
        contract_numbers: typing.List[int],
        years: typing.List[int],
        ric: str,
        file_prefix: str
):
    from RL_Trading.prj.app.core.data.entities.lob_dataset_builder import LOBDatasetBuilder
    with open(config_path, "r") as file:
        dict_config = yaml.safe_load(file)
    builder = LOBDatasetBuilder()
    ric_filters = [f"{ric}c{contract_number}" for contract_number in contract_numbers]
    for year in years:
        dtfs = builder.build_dataset(year, dict_config['window_d_lob'], zip_root, spark_config_path, ric_filters)
        for contract_number, dtf in zip(contract_numbers, dtfs):
            if not dtf.empty:
                save_file_path = os.path.join(data_root, f"{file_prefix}_LOB_{year}_c{contract_number}.csv")
                dtf.sort_values(by=['Date-Time']).to_csv(save_file_path, header=True, index=False)


def roll_lob_preprocessing(
        gzip_root: str,
        data_root: str,
        spark_config_path: str,
        config_path: str,
        ric_suffixes: typing.List[str],
        years: typing.List[int],
):
    from RL_Trading.prj.app.core.data.entities.lob_dataset_builder import RollLOBDatasetBuilder
    with open(config_path, "r") as file:
        dict_config = yaml.safe_load(file)
    builder = RollLOBDatasetBuilder()
    ric_filters = [f"FBTP{ric_suffix}" for ric_suffix in ric_suffixes]
    for year in years:
        dtfs = builder.build_dataset(
            year, dict_config['window_d_lob'], gzip_root, spark_config_path, ric_filters, dict_config["roll_chains"]
        )
        for ric_suffix, dtf in zip(ric_suffixes, dtfs):
            if not dtf.empty:
                save_file_path = os.path.join(data_root, f"IKARoll_LOB_{year}_{ric_suffix}.csv")
                dtf.sort_values(by=['Date-Time']).to_csv(save_file_path, header=True, index=False)


def trades_preprocessing(
        zip_root: str,
        data_root: str,
        spark_config_path: str,
        config_path: str,
        contract_numbers: typing.List[int],
        years: typing.List[int],
        ric: str,
        file_prefix: str
):
    from RL_Trading.prj.app.core.data.entities.trades_dataset_builder import TradesDatasetBuilder
    with open(config_path, "r") as file:
        dict_config = yaml.safe_load(file)
    builder = TradesDatasetBuilder()
    ric_filters = [f"{ric}c{contract_number}" for contract_number in contract_numbers]
    for year in years:
        dtfs = builder.build_dataset(year, dict_config['window_d_trades'], zip_root, spark_config_path, ric_filters)
        for contract_number, dtf in zip(contract_numbers, dtfs):
            if not dtf.empty:
                save_file_path = os.path.join(data_root, f"{file_prefix}_trades_{year}_c{contract_number}.csv")
                dtf.sort_values(by=['Date-Time']).to_csv(save_file_path, header=True, index=False)


def trades_preprocessing_v2(
        zip_root: str,
        data_root: str,
        contract_numbers: typing.List[int],
        years: typing.List[int],
        ric: str,
        file_prefix: str
):
    from RL_Trading.prj.app.core.data.entities.trades_dataset_builder_v2 import TradesDatasetBuilderV2
    builder = TradesDatasetBuilderV2()
    ric_filters = [f"{ric}c{contract_number}" for contract_number in contract_numbers]
    for year in years:
        dtfs = builder.create_dataset(year, zip_root, ric_filters)
        for contract_number, dtf in zip(contract_numbers, dtfs):
            if not dtf.empty:
                save_file_path = os.path.join(data_root, f"{file_prefix}_trades_{year}_c{contract_number}_v2.csv")
                dtf.sort_values(by=['Date-Time']).to_csv(save_file_path, header=True, index=False)


def roll_trades_preprocessing(
        gzip_root: str,
        data_root: str,
        spark_config_path: str,
        config_path: str,
        ric_suffixes: typing.List[str],
        years: typing.List[int],
):
    from core.data.entities.trades_dataset_builder import RollTradesDatasetBuilder
    with open(config_path, "r") as file:
        dict_config = yaml.safe_load(file)
    builder = RollTradesDatasetBuilder()
    ric_filters = [f"FBTP{ric_suffix}" for ric_suffix in ric_suffixes]
    for year in years:
        dtfs = builder.build_dataset(
            year, dict_config['window_d_trades'], gzip_root, spark_config_path, ric_filters, dict_config["roll_chains"]
        )
        for ric_suffix, dtf in zip(ric_suffixes, dtfs):
            if not dtf.empty:
                save_file_path = os.path.join(data_root, f"IKARoll_trades_{year}_{ric_suffix}.csv")
                dtf.sort_values(by=['Date-Time']).to_csv(save_file_path, header=True, index=False)


def fqi_dataset_creation(
        years: typing.List[str],
        add_trade_imbalance: bool,
        roll_date_offset: int,
        mids_window: int,
        mids_offset: int,
        steps_tolerance: int,
        number_of_deltas: int,
        ohe_temporal_features: bool,
        persistence: int,
        data_root: str,
        config_path: str
) -> pd.DataFrame:
    from core.fqi.entities.fqi_dataset_builder_ik import IKFQIDatasetBuilder
    with open(config_path, "r") as file:
        dict_config = yaml.safe_load(file)
    lob_file_paths = [f'{data_root}/IKA_LOB_{year}_c1.csv' for year in years]
    trades_file_paths = []
    if add_trade_imbalance:
        trades_file_paths = [f'{data_root}/IKA_trades_{year}_c1_v2.csv' for year in years if year >= 2019]
    auctions_file_path = f'{data_root}/IKA_auctions.csv'
    df = IKFQIDatasetBuilder(
        lob_file_paths=lob_file_paths,
        trades_file_paths=trades_file_paths,
        auctions_file_path=auctions_file_path,
        anomalous_dates=dict_config['anomalous_dates'],
        roll_dates=list(dict_config['roll_chains'].keys()),
        roll_date_offset=roll_date_offset,
        mids_window=mids_window,
        mids_offset=mids_offset,
        steps_tolerance=steps_tolerance,
        number_of_deltas=number_of_deltas,
        opening_hour=8,
        closing_hour=18,
        ohe_temporal_features=ohe_temporal_features,
        persistence=persistence,
        actions=[-1, 0, 1]
    ).build_dataset()
    return df


def classification_dataset_creation(
        years: typing.List[str],
        add_trade_imbalance: bool,
        roll_date_offset: int,
        mids_window: int,
        mids_offset: int,
        steps_tolerance: int,
        number_of_deltas: int,
        ohe_temporal_features: bool,
        volume_features: bool,
        target_offset: int,
        data_root: str,
        config_path: str,
        skip_covid: bool,
        skip_conte_I: bool,
        events: typing.Dict[str, datetime.date],
        volume_history_size: int,
        number_of_levels: int
) -> pd.DataFrame:
    from core.classification.entities.classification_dataset_builder_ik import IKClassificationDatasetBuilder
    with open(config_path, "r") as file:
        dict_config = yaml.safe_load(file)
    lob_file_paths = [f'{data_root}/IKA_LOB_{year}_c1.csv' for year in years]
    trades_file_paths = []
    if add_trade_imbalance:
        trades_file_paths = [f'{data_root}/IKA_trades_{year}_c1_v2.csv' for year in years if year >= 2019]
    auctions_file_path = f'{data_root}/IKA_auctions.csv'
    dtf = IKClassificationDatasetBuilder(
        lob_file_paths=lob_file_paths,
        trades_file_paths=trades_file_paths,
        auctions_file_path=auctions_file_path,
        anomalous_dates=dict_config['anomalous_dates'],
        roll_dates=list(dict_config['roll_chains'].keys()),
        roll_date_offset=roll_date_offset,
        mids_window=mids_window,
        mids_offset=mids_offset,
        steps_tolerance=steps_tolerance,
        number_of_deltas=number_of_deltas,
        opening_hour=8,
        closing_hour=18,
        skip_covid=skip_covid,
        skip_conte_I=skip_conte_I,
        volume_history_size=volume_history_size,
        number_of_levels=number_of_levels,
        ohe_temporal_features=ohe_temporal_features,
        target_offset=target_offset,
        volume_features=volume_features
    ).build_dataset()
    return dtf

"""
def classification_dataset_bund_creation(
        years: typing.List[str],
        add_trade_imbalance: bool,
        roll_date_offset: int,
        mids_window: int,
        mids_offset: int,
        steps_tolerance: int,
        number_of_deltas: int,
        ohe_temporal_features: bool,
        target_offset: int,
        data_root: str,
        config_path: str
) -> pd.DataFrame:
    from core.classification.entities.classification_dataset_builder_bund import BundDatasetBuilder
    with open(config_path, "r") as file:
        dict_config = yaml.safe_load(file)
    lob_file_paths = [f'{data_root}/RXA_LOB_{year}_c1.csv' for year in years]
    trades_file_paths = []
    if add_trade_imbalance:
        trades_file_paths = [f'{data_root}/RXA_trades_{year}_c1_v2.csv' for year in years if year >= 2019]
    auctions_file_path = f'{data_root}/RXA_auctions.csv'
    dtf = BundDatasetBuilder(
        lob_file_paths=lob_file_paths,
        trades_file_paths=trades_file_paths,
        auctions_file_path=auctions_file_path,
        anomalous_dates=dict_config['anomalous_dates'],
        roll_dates=list(dict_config['roll_chains'].keys()),
        roll_date_offset=roll_date_offset,
        mids_window=mids_window,
        mids_offset=mids_offset,
        steps_tolerance=steps_tolerance,
        number_of_deltas=number_of_deltas,
        opening_hour=9,
        closing_hour=17,
        ohe_temporal_features=ohe_temporal_features,
        target_offset=target_offset
    ).build_dataset()
    return dtf
"""