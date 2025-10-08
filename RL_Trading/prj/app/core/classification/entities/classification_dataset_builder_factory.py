import os
import typing
import yaml
from RL_Trading.prj.app.core.classification.entities.classification_dataset_builder import ClassificationDatasetBuilder


class ClassificationDatasetBuilderFactory:

    @staticmethod
    def create(
            asset_name: str,
            years: typing.List[int],
            roll_date_offset: int,
            mids_window: int,
            mids_offset: int,
            steps_tolerance: int,
            number_of_deltas: int,
            opening_hour: int,
            closing_hour: int,
            ohe_temporal_features: bool,
            target_offset: int,
            add_trade_imbalance: bool,
            volume_features: bool = False,
            lob_price_features: bool = False,
            skip_conte_I: bool = False,
            skip_covid: bool = False,
            volume_history_size: int = 0,
            number_of_levels: int = 5,

    ) -> ClassificationDatasetBuilder:
        if asset_name == "IK":
            from RL_Trading.prj.app.core.classification.entities.classification_dataset_builder_ik import IKClassificationDatasetBuilder
            files_prefix = "IKA"
            Builder = IKClassificationDatasetBuilder
        elif asset_name == "RX":
            from RL_Trading.prj.app.core.classification.entities.classification_dataset_builder_rx import RXClassificationDatasetBuilder
            files_prefix = "RXA"
            Builder = RXClassificationDatasetBuilder
        else:
            raise ValueError("Invalid shape type")
        # Set paths
        project_root = f'{os.path.dirname(os.path.abspath(__file__))}/../../../../..'
        data_root = f'{project_root}/data'
        config_path = f'{project_root}/prj/app/config/config.yaml'
        with open(config_path, "r") as file:
            dict_config = yaml.safe_load(file)
        lob_file_paths = [f'{data_root}/{files_prefix}_LOB_{y}_c1.csv' for y in years]
        trades_file_paths = []
        if add_trade_imbalance:
            trades_file_paths = [f'{data_root}/{files_prefix}_trades_{y}_c1_v2.csv' for y in years if y >= 2019]
        builder = Builder(
            lob_file_paths=lob_file_paths,
            trades_file_paths=trades_file_paths,
            auctions_file_path=f'{data_root}/{files_prefix}_auctions.csv',
            anomalous_dates=dict_config['anomalous_dates'],
            roll_dates=list(dict_config['roll_chains'].keys()),
            roll_date_offset=roll_date_offset,
            mids_window=mids_window,
            mids_offset=mids_offset,
            steps_tolerance=steps_tolerance,
            number_of_deltas=number_of_deltas,
            opening_hour=opening_hour,
            skip_covid=skip_covid,
            volume_features=volume_features,
            lob_price_features=lob_price_features,
            events=dict_config['anomalous_dates'],
            skip_conte_I=skip_conte_I,
            volume_history_size=volume_history_size,
            number_of_levels=number_of_levels,
            closing_hour=closing_hour,
            ohe_temporal_features=ohe_temporal_features,
            target_offset=target_offset
        )
        return builder
