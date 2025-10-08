import datetime
import os
import typing
import yaml
import pandas as pd
from  RL_Trading.prj.app.core.fqi.entities.fqi_dataset_builder import FQIDatasetBuilder


class FQIDatasetBuilderFactory:

    @staticmethod
    def create(
            asset_name: str,
            years: typing.List[int],
            roll_date_offset: int,
            mids_window: int,
            mids_offset: int,
            steps_tolerance: int,
            number_of_deltas: int,
            number_of_deltas_bund: int,
            opening_hour: int,
            closing_hour: int,
            ohe_temporal_features: bool,
            persistence: int,
            actions: typing.List[int],
            add_trade_imbalance: bool,
            volume_features: bool,
            skip_conte_I: bool = False,
            skip_covid: bool = False,
            lob_price_features : bool = False,
            skip_percentile: typing.Optional[float] = None,
            volume_history_size: int = 0,
            volume_history_size_bund: int = 0,
            number_of_levels: int = 5,
            number_of_levels_bund: int = 5,
            use_higher_levels:bool = False,
            phase: str = 'train',
            keep_deltas_of_non_operating_hours:bool = False,
            remove_costs: bool = False,
            remove_fixed_costs: bool = False,
            missing_values_handling: typing.Optional[str] = None,
            use_moving_average = False,
            perform_action_on_bund = False,
            perform_action_on_btp_bund = False
    ) -> FQIDatasetBuilder:
        if asset_name == "IK":
            from  RL_Trading.prj.app.core.fqi.entities.fqi_dataset_builder_ik import IKFQIDatasetBuilder
            files_prefix = "IKA"
            Builder = IKFQIDatasetBuilder
        #TODO commented out this
        #elif asset_name == "RX":
        #    from  RL_Trading.prj.app.core.fqi.entities.fqi_dataset_builder_rx import RXFQIDatasetBuilder
        #    files_prefix = "RXA"
        #    Builder = RXFQIDatasetBuilder
        else:
            raise ValueError("Invalid shape type")
        # Set paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.normpath(os.path.join(current_dir, '..', '..', '..', '..', '..'))
        data_root = f'{project_root}/prj/app/core/data'
        config_path = f'{project_root}/prj/app/config/config.yaml'
        with open(config_path, "r") as file:
            dict_config = yaml.safe_load(file)
        #TODO lob file paths into single file or split into years maybe?
        #lob_file_paths = [f'{data_root}/M1.csv']
        lob_file_paths = [f'{data_root}/M1_ICE_{y}.csv' for y in years]
        trades_file_paths = [f'{data_root}/M1_trades.csv']
        #TODO check trades

        #TODO what does this do?
        if skip_percentile is not None:
            print(f"Skipping Percentiles {skip_percentile} in {phase}")
            anomalous_days = dict_config['anomalous_dates'] + pd.to_datetime(pd.read_csv(f"days_to_remove_{skip_percentile}_{phase}.csv")['0']).dt.date.to_list()
            print(pd.to_datetime(pd.read_csv(f"days_to_remove_{skip_percentile}_{phase}.csv")['0']).dt.date.to_list())
        else:
            anomalous_days = dict_config['anomalous_dates']

        #TODO Remove this?
        if volume_history_size_bund > 0 or number_of_levels_bund > 0:
            # Check on available datasets for BUND
            if ((int(years[0])<2015)):
                raise AttributeError("Bund datasets lower limit year is 2015")    
        
        builder = Builder(
            asset_name = asset_name,
            lob_file_paths=lob_file_paths,
            trades_file_paths=trades_file_paths,
            auctions_file_path=f'{data_root}/{files_prefix}_auctions.csv',
            anomalous_dates=anomalous_days,
            roll_dates=list(dict_config['roll_chains'].keys()),
            roll_date_offset=roll_date_offset,
            mids_window=mids_window,
            mids_offset=mids_offset,
            steps_tolerance=steps_tolerance,
            number_of_deltas=number_of_deltas,
            number_of_deltas_bund=number_of_deltas_bund,
            volume_history_size=volume_history_size,
            volume_history_size_bund=volume_history_size_bund,
            number_of_levels = number_of_levels,
            number_of_levels_bund=number_of_levels_bund,
            opening_hour=opening_hour,
            closing_hour=closing_hour,
            lob_price_features=lob_price_features,
            ohe_temporal_features=ohe_temporal_features,
            persistence=persistence,
            years = years,
            actions=actions,
            volume_features=volume_features,
            events= dict_config['events'],
            skip_conte_I = skip_conte_I,
            skip_covid= skip_covid,
            use_higher_levels = use_higher_levels,
            keep_deltas_of_non_operating_hours = keep_deltas_of_non_operating_hours,
            remove_costs = remove_costs,
            remove_fixed_costs = remove_fixed_costs,
            missing_values_handling=missing_values_handling,
            use_moving_average = use_moving_average,
            perform_action_on_bund = perform_action_on_bund,
            perform_action_on_btp_bund = perform_action_on_btp_bund 
        )
        return builder
