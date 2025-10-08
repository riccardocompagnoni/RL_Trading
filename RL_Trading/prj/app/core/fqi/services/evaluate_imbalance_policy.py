import json
import logging
import os
import typing

from RL_Trading.prj.app.core.fqi.entities.fqi_dataset_builder_factory import FQIDatasetBuilderFactory
from RL_Trading.prj.app.core.fqi.entities.tester import Tester
from RL_Trading.prj.app.core.fqi.entities.trading_env import TradingEnv
from RL_Trading.prj.app.core.models.services.utils import get_logger
from RL_Trading.prj.app.core.fqi.trlib.policies.policy import TemporalPolicy, ImbalancePolicy, LongOnlyPolicy, ShortOnlyPolicy


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
        read_path = f'/Volumes/Volume/experiments_alberto/results/{study_name}'

    if evaluate_up_to_2023:
        target_year = 2023
    else:
        target_year = years[0] + 1
    for y in range(years[0], years[0] + 1 + (target_year - years[0])):
        save_path_it = os.path.join(save_path, f"{y}")
        os.makedirs(os.path.join(save_path_it), exist_ok=True)
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
        if 'trajectory_window' in regressor_params.keys():
            trajectory_window = regressor_params['trajectory_window']
        else:
            trajectory_window = 1
        # Initialize Tester
        dataset_builder = FQIDatasetBuilderFactory.create("IK", [y], **features_parameters)
        tester = Tester(TradingEnv(dataset_builder), phase, logger, save_path_it, 1, read_path)
        tester.test_fixed_policy(TemporalPolicy(actions=[-1,0,1]), save_path_it, 'TemporalPolicy')


if __name__ == '__main__':
    study_name = "test_noCosts_1_targetOffset_NoDeltasBund_VolsBund_val19"
    evaluate_up_to_2023 = True
    year = [2015]
    test(study_name, year, evaluate_up_to_2023)
