import json
import logging
import os
import typing

from RL_Trading.prj.app.core.fqi.entities.fqi_dataset_builder_factory import FQIDatasetBuilderFactory
from RL_Trading.prj.app.core.fqi.entities.tester import Tester
from RL_Trading.prj.app.core.fqi.entities.trading_env import TradingEnv
from RL_Trading.prj.app.core.models.services.utils import get_logger


def test(study_name: str, years: typing.List[int], evaluate_up_to_2023: bool = False):
    phase = "Test"
    # Set paths
    project_root = f'{os.path.dirname(os.path.abspath(__file__))}/../../../../..'
    save_path = f'{project_root}/results_evaluation/{study_name}'
    os.makedirs(save_path, exist_ok=True)
    server_execution = False
    if server_execution == True:
        read_path = f'/data/intesa/giovanni/results/{study_name}'
    else:
        read_path = f'/Volumes/Volume/experiments/{study_name}'

    seeds = [int(f.split("d")[1]) for f in os.listdir(read_path) if
             os.path.isdir(os.path.join(read_path, f)) and f != "tmp" and f != 'test_next_year']

    print(seeds)
    if evaluate_up_to_2023:
        target_year = 2023
    else:
        target_year = years[0] + 1
    for y in range(years[0], years[0] + 1 + (target_year - years[0])):
        save_path_it = os.path.join(save_path, f"{y}")
        for seed in seeds:
            os.makedirs(os.path.join(save_path_it, f'seed{seed}'), exist_ok=True)
        # Load params
        logger = get_logger(logging.INFO)
        with open(f'{read_path}/features_params.json', 'r') as f:
            features_parameters = json.load(f)
        with open(f'{read_path}/parameters_opt.json', 'r') as f:
            regressor_params = json.load(f)
        fqi_iterations = regressor_params['iterations']
        if 'trajectory_number' not in regressor_params.keys():
            trajectory_number = None
        else:
            trajectory_number = regressor_params['trajectory_number']
        trajectory_window = regressor_params['trajectory_window']
        # Initialize Tester
        dataset_builder = FQIDatasetBuilderFactory.create("IK", [y], **features_parameters)
        tester = Tester(TradingEnv(dataset_builder), phase, logger, save_path_it, 1, read_path)
        tester.test(fqi_iterations, seeds)


if __name__ == '__main__':
    study_name = "FQI_1718_19_multiseed_dfqi_multi_trajectory_test_observed_persistence_60_20_delta_prices_volumes_L3_60_min_offset_9_17"
    evaluate_up_to_2023 = True
    year = [2020]
    test(study_name, year, evaluate_up_to_2023)
