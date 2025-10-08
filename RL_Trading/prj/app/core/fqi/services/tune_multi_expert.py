import json
import logging
import numpy as np
import math
import os
import typing

from RL_Trading.prj.app.core.fqi.entities.fqi_dataset_builder_factory import FQIDatasetBuilderFactory
from RL_Trading.prj.app.core.fqi.entities.trading_env import TradingEnv
from RL_Trading.prj.app.core.fqi.entities.trainer import Trainer
from RL_Trading.prj.app.core.fqi.entities.tuner import Tuner
from RL_Trading.prj.app.core.models.services.utils import get_logger
from joblib import Parallel, delayed
from RL_Trading.prj.app.core.fqi.services.plotter import Plotter
def tuner_interface(study_name: str,
            fixed_params: typing.Dict,
            hyper_parameters_distributions: typing.List[typing.Dict],
            overwrite_study: bool,
            n_trials: int,
            pruning: bool,
            pruning_startup_trials: int,
            pruning_quantile: bool,
            seeds: typing.List[int],
            trainer: Trainer,
            regressor_type: str,
            logger: logging.Logger,
            save_path: str,
            trajectory_number:int,
            optimize_trajectory_number: bool,
            optimize_trajectory_window: bool,
            test_best_model_so_far: bool = False,
            test_on_next_year_trainer: typing.Optional[Trainer] = None,
            trajectory_window_size: typing.Optional[int] = None,
            n_jobs: int = 1,):

        fixed_params['trajectory_number'] = trajectory_number
        tuner = Tuner(
            study_name, fixed_params, hyper_parameters_distributions, overwrite_study, n_trials, pruning,
            pruning_startup_trials, pruning_quantile, seeds, trainer, regressor_type, logger, save_path,
            optimize_trajectory_number = optimize_trajectory_number, optimize_trajectory_window = optimize_trajectory_window,
            trajectory_window_size=trajectory_window_size, test_best_model_so_far=test_best_model_so_far,
            test_on_next_year_trainer=test_on_next_year_trainer, n_jobs=n_jobs
        )
        tuner.tune()


if __name__ == '__main__':
    # Set paths
    study_name = 'FQI_1718_16_multiseed_dfqi_optimize_trajectory_test_observed_persistence_60_20_delta_prices_9_17_shuffle_days_multiple_experts'
    project_root = f'{os.path.dirname(os.path.abspath(__file__))}/../../../../..'
    config_path = f'{project_root}/prj/app/config/config.yaml'
    server_execution = False
    if server_execution == True:
        save_path = f'/data/trading/results/{study_name}'
    else:
        save_path = f'{project_root}/results/{study_name}'
    os.makedirs(save_path, exist_ok=True)
    # Set params
    #np.random.seed()
    #seeds = [0, 1]
    seeds = []
    for s in range(10):  # generate 10 random seeds
        np.random.seed()
        seeds.append(np.random.randint(100000))
    features_parameters = {
        'roll_date_offset': 1,
        'mids_window': 1,
        'mids_offset': 60,
        'steps_tolerance': 5,
        'number_of_deltas': 20,
        'volume_history_size': 0,
        'number_of_levels': 3,
        'opening_hour': 9,
        'closing_hour': 17,
        'ohe_temporal_features': True,
        'persistence': 60,
        'actions': [-1, 0, 1],
        'add_trade_imbalance': False,
        'volume_features': False,
        'skip_conte_I': False,
        'skip_covid': False,
        'use_higher_levels': False
    }
    with open(f'{save_path}/features_params.json', 'w') as f:
        json.dump(features_parameters, f)
    persistence = features_parameters['persistence']
    regressor_type = 'xgb'  # 'extra' | 'xgb'
    if regressor_type == 'xgb':
        fixed_parameters = {
            'grow_policy': "depthwise",
            'booster': "gbtree",
            'predictor': "auto",
            'verbosity': 0,
            'n_jobs': -1,
            'tree_method': "hist",
            'colsample_bylevel': 1,
            'colsample_bynode': 1,
            'trajectory_window': 1, # default value is 1.
            #'iterations': 1
        }
        hyper_parameters_distributions = [
            #{'name': 'trajectory_number', 'type': 'int', 'min': 0, 'max': persistence - 1, 'step': 1},
            {'name': 'iterations', 'type': 'int', 'min': 1, 'max': 10, 'step': 1},
            {'name': 'min_child_weight', 'type': 'int', 'min': 1, 'max': 10, 'step': 1},
            {'name': 'learning_rate', 'type': 'log', 'min': 0.01, 'max': 0.50}, #reduced
            {'name': 'subsample', 'type': 'float', 'min': 0.5, 'max': 1.0, 'step': 0.05}, #reduced
            {'name': 'colsample_bytree', 'type': 'float', 'min': 0.5, 'max': 1.0, 'step': 0.05},
            {'name': 'n_estimators', 'type': 'int', 'min': 50, 'max': 500, 'step': 50},
            {'name': 'reg_lambda', 'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.05},
            {'name': 'reg_alpha', 'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.05}, #reduced reg_alpha
            {'name': 'max_depth', 'type': 'int', 'min': 3, 'max': 10, 'step': 1},
        ]
    n_trials = 350
    overwrite_study = True
    pruning = True
    optimize_trajectory_number = False
    n_jobs = 1
    n_jobs_outer = 1
    optimize_trajectory_window = False # to be set to True, if you want to use a window of trajectory. It works in combination with fixed_parameters['trajectory_window']
    pruning_startup_trials = 100
    pruning_quantile = 0.3
    train_years = [2017, 2018]
    test_years = [2016]
    swap_Q = True
    double_Q = True
    double_Q_strategy = 'mean'
    shuffle_days = True
    remove_days_Q_values_quantile = False
    Q_values_percentile = None  # 0-1 range or None
    test_best_model_so_far = True

    logger = get_logger(logging.INFO)
    if remove_days_Q_values_quantile:
        logger.info(f"Tuning removing days with Q_values_percentile {Q_values_percentile}")
    training_dataset_builder = FQIDatasetBuilderFactory.create("IK", train_years, **features_parameters, skip_percentile=Q_values_percentile)
    testing_dataset_builder = FQIDatasetBuilderFactory.create("IK", test_years, **features_parameters)
    trainer = Trainer(
        TradingEnv(training_dataset_builder), TradingEnv(testing_dataset_builder),
        regressor_type, double_Q, swap_Q, logger, save_path="", save_plots=True, double_Q_strategy = double_Q_strategy,
        n_jobs= n_jobs, shuffle_days=shuffle_days,
    )
    #Prepare trainer to test on next year
    if test_best_model_so_far:
        logger.info(f"Enabled test of the best configuration found on the Test Set {[train_years[1] + 1]}")
        training_dataset_builder_for_test = training_dataset_builder #FQIDatasetBuilderFactory.create("IK", train_years, **features_parameters)
        testing_dataset_builder_for_test = FQIDatasetBuilderFactory.create("IK", [train_years[1] + 1], **features_parameters)
        test_on_next_year_trainer = Trainer(
            TradingEnv(training_dataset_builder_for_test), TradingEnv(testing_dataset_builder_for_test),
            regressor_type, double_Q, swap_Q, logger, save_path="./test_next_year", save_plots=True,
            double_Q_strategy=double_Q_strategy, trainer_phase = "Test", n_jobs=n_jobs,
            shuffle_days=shuffle_days
        )
    else:
        test_on_next_year_trainer = None

    Parallel(n_jobs=n_jobs_outer, backend='loky', verbose=10)(
        delayed(tuner_interface)
        (study_name=f"{study_name}_trajectory_{t}", fixed_params = fixed_parameters, hyper_parameters_distributions = hyper_parameters_distributions, overwrite_study= overwrite_study, n_trials=n_trials, pruning=pruning, pruning_startup_trials = pruning_startup_trials, pruning_quantile = pruning_quantile, seeds = seeds, trainer = trainer, regressor_type=regressor_type, logger = logger, save_path = os.path.join(save_path, f"{study_name}_trajectory_{t}"), trajectory_number = t, test_best_model_so_far=test_best_model_so_far, test_on_next_year_trainer=test_on_next_year_trainer, n_jobs=n_jobs, optimize_trajectory_number = optimize_trajectory_number, optimize_trajectory_window = optimize_trajectory_window ) for t in range(0, 60, 10))
    #plot ensamble
    paths = []
    for t in range(0, 60, 10):
        paths.append(os.path.join(save_path, f"{study_name}_trajectory_{t}"))
    print(paths)
    #take the minimum number of iterations across all the model trained
    iterations = 10
    for p in paths:
        regressor_params = json.load(open(os.path.join(p, "parameters_opt.json")))
        iterations = min(regressor_params['iterations'], iterations)

    Plotter.plot_percentage_returns_combined('Validation', iterations, seeds, paths, save_path)
