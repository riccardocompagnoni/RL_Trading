import json
import logging
import os
import argparse
import typing

from RL_Trading.prj.app.core.fqi.entities.fqi_dataset_builder_factory import FQIDatasetBuilderFactory
from RL_Trading.prj.app.core.fqi.entities.tester import Tester
from RL_Trading.prj.app.core.fqi.entities.trading_env import TradingEnv
from RL_Trading.prj.app.core.models.services.utils import get_logger


def test(study_name: str, limit: int):
    study_name = 'test_noCosts_1_targetOffset_20DeltasBTP_20DeltaBund_VolsBund_val21'
    phase = "Test"
    # Set paths
    print(f"Processing: {study_name}")
    test_years = [2022, 2023]
    print(f"Test_years: {test_years}")
    project_root = f'{os.path.dirname(os.path.abspath(__file__))}/../../../../..'
    #save_path = f'{project_root}/results/{study_name}'
    n_test = limit - min(test_years) + 1
    server_execution = False
    if server_execution == True:
        save_path = f'/data/intesa/giovanni/test_extensive/{study_name}'
    else:
        save_path = f'{project_root}/results/{study_name}'

    # Load params
    seeds = [int(f.split("d")[1]) for f in os.listdir(save_path) if
             os.path.isdir(os.path.join(save_path, f)) and f != "tmp" and f != 'test_next_year_no_retrain' and f != 'test_next_year_no_retrain_old']
    logger = get_logger(logging.INFO)
    with open(f'{save_path}/features_params.json', 'r') as f:
        features_parameters = json.load(f)
    with open(f'{save_path}/parameters_opt.json', 'r') as f:
        regressor_params = json.load(f)
    fqi_iterations = regressor_params['iterations']
    # trajectory_number = regressor_params['trajectory_number']
    # trajectory_window = regressor_params['trajectory_window']
    # Initialize Tester
    for y in test_years:
        # test_years_loop = [y + n for y in test_years]
        root = os.path.split(save_path)[0]
        target_path = os.path.join(root, study_name, 'test_next_year_no_retrain')
        save_path_test = os.path.join(target_path, f"test_{y}")
        test_years_current = [y]
        os.makedirs(save_path,exist_ok=True)
        for seed in seeds:
            os.makedirs(os.path.join(save_path, f'seed{seed}'),exist_ok=True)
        testing_dataset_builder = FQIDatasetBuilderFactory.create("IK", test_years_current,**features_parameters)
        tester = Tester(TradingEnv(testing_dataset_builder), phase, logger, read_path=save_path, save_path=save_path_test)
        tester.test(fqi_iterations, seeds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study_name')
    args = parser.parse_args()
    limit = 2023
    test(args.study_name, limit)
