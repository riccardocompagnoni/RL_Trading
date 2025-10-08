import json
import logging
import time
from pathlib import Path

import numpy as np
import math
import os
import sys
from datetime import date, timedelta

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.normpath(os.path.join(current_dir, '..', '..', '..', '..', '..','..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..', 'RL_Trading', 'prj', 'app')))
from RL_Trading.prj.app.core.fqi.entities.fqi_dataset_builder_factory import FQIDatasetBuilderFactory
from RL_Trading.prj.app.core.fqi.entities.trading_env import TradingEnv
from RL_Trading.prj.app.core.fqi.entities.trainer import Trainer
from RL_Trading.prj.app.core.fqi.entities.tuner import Tuner
from RL_Trading.prj.app.core.models.services.utils import get_logger
from RL_Trading.prj.app.core.fqi.entities.tester import Tester
from RL_Trading.prj.app.core.fqi.simulator.simulator2 import TTFEnv



if __name__ == '__main__':

    # Set paths

    study_name = '1y21_7y21'
    print(study_name)
    #time.sleep(60*60*1)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(current_dir, '..', '..', '..', '..', '..'))
    config_path = f'{project_root}/prj/app/config/config.yaml'
    server_execution = False
    if server_execution == True:
        save_path = f'/data/intesa/giovanni/results/asym_0.025/{study_name}'
    else:
        save_path = f'{project_root}/results/quant_0.45/{study_name}'
    os.makedirs(save_path, exist_ok=True)
    dataset_path = Path(f'{project_root}/prj/app/core/data/dataset_full.parquet')

    # Set paramsh
    #np.random.seed()
    #seeds = [0, 1]
    seeds = []
    for s in range(5):  # generate 10 random seeds
        np.random.seed()
        seeds.append(np.random.randint(100000))
    features_parameters = {
        'roll_date_offset': 1,
        'mids_window': 1,
        'mids_offset': 15,
        'steps_tolerance': 5,
        'number_of_deltas': 5,
        'volume_history_size': 0,
        'number_of_levels': 10,
        'number_of_deltas_bund': 0,
        'volume_history_size_bund': 0,
        'number_of_levels_bund': 0,
        'opening_hour': 8,
        'closing_hour': 16,
        'ohe_temporal_features': False,
        'persistence': 30,
        'actions': [-5, 0, 5],
        'add_trade_imbalance': False,
        'volume_features': False,
        'skip_conte_I': False,
        'skip_covid': False,
        'use_higher_levels': False,
        'keep_deltas_of_non_operating_hours': True,
        'remove_costs': False,
        'remove_fixed_costs': False,
        'missing_values_handling': None, # ffill | constant ( filled with -1)
        'use_moving_average': False,
        'perform_action_on_bund': False,
        'perform_action_on_btp_bund': False
    }

    if features_parameters['perform_action_on_btp_bund']:
        features_parameters['actions'] = [-2, -1, 0, 1, 2]
        # features_parameters['actions'] = [(-1,0), (0,-1), (0,0), (1,0), (0,1)]

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
            'objective': "reg:quantileerror",
            'quantile_alpha': 0.45,
            #'iterations': 1
        }
        hyper_parameters_distributions = [
            #{'name': 'trajectory_number', 'type': 'int', 'min': 0, 'max': persistence - 1, 'step': 1},
            {'name': 'trajectory_number', 'type': 'int', 'min': 0 + (math.ceil(fixed_parameters['trajectory_window'] / 2) - 1), 'max': persistence - 1 - (math.ceil(fixed_parameters['trajectory_window'] / 2) - 1), 'step': 1},
            {'name': 'iterations', 'type': 'int', 'min': 7, 'max': 12, 'step': 1},
            {'name': 'filter_method', 'type': 'cat', 'choices': ['propagate', 'move_flat']},
            {'name': 'Q_values_diff_threshold', 'type': 'float', 'min': 0.3, 'max': 1.5, 'step': 0.1},
            {'name': 'min_child_weight', 'type': 'int', 'min': 1, 'max': 10, 'step': 1},
            {'name': 'learning_rate', 'type': 'log', 'min': 0.01, 'max': 0.50}, #reduced
            {'name': 'subsample', 'type': 'float', 'min': 0.5, 'max': 1.0, 'step': 0.05}, #reduced
            {'name': 'colsample_bytree', 'type': 'float', 'min': 0.2, 'max': 1.0, 'step': 0.05},
            {'name': 'n_estimators', 'type': 'int', 'min': 10, 'max': 250, 'step': 10},
            {'name': 'reg_lambda', 'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.05},
            {'name': 'reg_alpha', 'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.05}, #reduced reg_alpha
            {'name': 'max_depth', 'type': 'int', 'min': 1, 'max': 10, 'step': 1},
        ]
    elif regressor_type == 'extra':
        fixed_parameters = {
            #'n_estimators': 300,
            'criterion': 'squared_error',
            'max_depth': None,
            'min_samples_leaf': 1,
            'max_features': 1.0,
            'ccp_alpha': 0,
            'verbose': 0,
            'n_jobs': -1,
            'trajectory_window': 1,  # default value is 1.,
            #'iterations': 5
        }
        hyper_parameters_distributions = [
            # {'name': 'trajectory_number', 'type': 'int', 'min': 0, 'max': persistence - 1, 'step': 1},
            {'name': 'n_estimators', 'type': 'int', 'min': 10, 'max': 510, 'step': 50 },
            {'name': 'trajectory_number', 'type': 'int',
             'min': 0 + (math.ceil(fixed_parameters['trajectory_window'] / 2) - 1),
             'max': persistence - 1 - (math.ceil(fixed_parameters['trajectory_window'] / 2) - 1), 'step': 1},
            {'name': 'iterations', 'type': 'int', 'min': 1, 'max': 4, 'step': 1},
            {'name': 'filter_method', 'type': 'cat', 'choices': ['propagate', 'move_flat']},
            {'name': 'Q_values_diff_threshold', 'type': 'float', 'min': 0.3, 'max': 1.5, 'step': 0.1},
            {'name': 'min_samples_split', 'type': 'int', 'min': 100, 'max': 10000, 'step': 50}, #100
        ]
    else:
        raise ValueError(f"{regressor_type} is not a valid type of regressor")

    n_trials = 150
    overwrite_study = True
    unroll_fqi_iterations = False #True to consider each fqi iteration as a separate trial
    pruning = True
    optimize_trajectory_number = True
    n_jobs = -1
    cvar = None #range 0 - 1, None for not using
    optimize_trajectory_window = False # to be set to True, if you want to use a window of trajectory. It works in combination with fixed_parameters['trajectory_window']
    optimize_Q_value_diff_threshold = False
    pruning_startup_trials = 400
    pruning_quantile = 0.3
    train_years = [2019, 2020] #dummy: train is always 2020 and validation 2021
    test_years = [2021] #validation
    swap_Q = True
    double_Q = True
    double_Q_strategy = 'mean'
    use_estimator_mismatch = False
    shuffle = "minute" #day | minute | week | None, randomized over shuffle type the two estimators
    remove_days_Q_values_quantile = False
    Q_values_quantile = None  # 0-1 range or None
    Q_value_diff_threshold = None if optimize_Q_value_diff_threshold is False and use_estimator_mismatch is True else None
    filter_method = None if optimize_Q_value_diff_threshold is False and use_estimator_mismatch is True else None# propagate | move_flat | first_step
    trajectory_window = fixed_parameters['trajectory_window']
    del fixed_parameters['trajectory_window']

    start = date(2021, 1, 1)
    end = date(2021, 7, 1)

    phases_dict = {
        "Train": [
            [date(2020, 3, 1), start - timedelta(days=1)],  # until day before validation
            [end, date(2023, 7, 1)]  # from end of validation until test start
        ],
        "Validation": [
            [start, end - timedelta(days=1)]  # validation interval
        ],
        "Test": [
            [date(2023, 7, 1), date(2024, 9, 30)]  # fixed test interval
        ]
    }

    #Q_values_diff_threshold = 0.5 #currently used 0.5 as threshold
    retrain_when_testing = False #if true, the model is retrained before testing on a new test year
    test_best_model_so_far = True
    test_only_next_year = True #if this is set to false and test_best_model_so_far is true, the model is tested up to 2023 without retrain
    test_trial = False


    if optimize_trajectory_number is False:
        hyper_parameters_distributions = [d for d in hyper_parameters_distributions if d['name'] != 'trajectory_number']

    if optimize_Q_value_diff_threshold is False:
        hyper_parameters_distributions = [d for d in hyper_parameters_distributions if d['name'] not in ['Q_values_diff_threshold', 'filter_method']]


    """
    logger = get_logger(logging.INFO)
    if remove_days_Q_values_quantile:
        logger.info(f"Tuning removing days with Q_values_percentile {Q_values_quantile}")
    #if Q_values_diff_threshold is not None:
    #    logger.info(f"Tuning removing days with Q_values_quantile {Q_values_diff_threshold} on validation and test")
    training_dataset_builder = FQIDatasetBuilderFactory.create("IK", train_years, **features_parameters)
    testing_dataset_builder = FQIDatasetBuilderFactory.create("IK", test_years, **features_parameters)
    trainer = Trainer(
        TradingEnv(training_dataset_builder), TradingEnv(testing_dataset_builder),
        regressor_type, double_Q, swap_Q, logger, save_path="", save_plots=True, double_Q_strategy = double_Q_strategy,
        n_jobs= n_jobs, shuffle=shuffle, cvar_param=cvar, use_estimator_mismatch=use_estimator_mismatch
    )
    tester = None
    #Prepare trainer to test on next year
    test_on_next_year_trainer = None

    if test_best_model_so_far and test_only_next_year:
        if retrain_when_testing:
            logger.info(f"Enabled test of the best configuration found on the Test Set {[y + 1 for y in test_years]}")
            #training_dataset_builder_for_test = FQIDatasetBuilderFactory.create("IK", [y + 1 for y in train_years], **features_parameters)
            training_dataset_builder_for_test = FQIDatasetBuilderFactory.create("IK", train_years+test_years, **features_parameters)
            testing_dataset_builder_for_test = FQIDatasetBuilderFactory.create("IK", [y + 1 for y in test_years], **features_parameters)
            test_on_next_year_trainer = Trainer(
                TradingEnv(training_dataset_builder_for_test), TradingEnv(testing_dataset_builder_for_test),
                regressor_type, double_Q, swap_Q, logger, save_path="./test_next_year", save_plots=True,
                double_Q_strategy=double_Q_strategy, trainer_phase = "Test", n_jobs=n_jobs,
                shuffle=shuffle, cvar_param=cvar, use_estimator_mismatch=use_estimator_mismatch
            )

        if test_trial:
            testing_dataset_builder_for_test = FQIDatasetBuilderFactory.create("IK", [y + 1 for y in test_years],
                                                                               **features_parameters)

            tester = Tester(TradingEnv(testing_dataset_builder_for_test), "Test", logger,
                            read_path=os.path.join(save_path, 'tmp'),
                            save_path=os.path.join(save_path, "tmp_test_next_year_no_retrain"),
                            n_jobs=n_jobs, use_estimator_mismatch=Q_value_diff_threshold is not None)


    tuner = Tuner(
        study_name, fixed_parameters, hyper_parameters_distributions, overwrite_study, n_trials, pruning,
        pruning_startup_trials, pruning_quantile, seeds, trainer, regressor_type, logger, save_path,
        optimize_trajectory_number,
        optimize_trajectory_window = optimize_trajectory_number and optimize_trajectory_window,
        trajectory_window_size= trajectory_window, test_best_model_so_far=test_best_model_so_far,
        test_on_next_year_trainer=test_on_next_year_trainer, n_jobs = n_jobs, optimize_cvar=cvar is not None, retrain_when_testing = retrain_when_testing, features_parameters=features_parameters, optimize_Q_value_diff_threshold = optimize_Q_value_diff_threshold,
        Q_value_diff_threshold = Q_value_diff_threshold, filter_method = filter_method, test_up_to_year=2022, tester_trial = tester, unroll_fqi_iterations=unroll_fqi_iterations)
    tuner.tune()
    """
    logger = get_logger(logging.INFO)
    if remove_days_Q_values_quantile:
        logger.info(f"Tuning removing days with Q_values_percentile {Q_values_quantile}")
    # if Q_values_diff_threshold is not None:
    #    logger.info(f"Tuning removing days with Q_values_quantile {Q_values_diff_threshold} on validation and test")
    training_dataset_builder = FQIDatasetBuilderFactory.create("IK", train_years, **features_parameters)
    testing_dataset_builder = FQIDatasetBuilderFactory.create("IK", test_years, **features_parameters)

    env = TTFEnv(data_path=dataset_path, persistence=features_parameters['persistence'], remove_costs=features_parameters['remove_costs'], no_overnight=False, phases_dates=phases_dict)

    trainer = Trainer(
        env, env,
        regressor_type, double_Q, swap_Q, logger, save_path="", save_plots=True, double_Q_strategy=double_Q_strategy,
        n_jobs=n_jobs, shuffle=shuffle, cvar_param=cvar, use_estimator_mismatch=use_estimator_mismatch
    )
    tester = None
    # Prepare trainer to test on next year
    test_on_next_year_trainer = None

    if test_best_model_so_far and test_only_next_year:
        if retrain_when_testing:
            logger.info(f"Enabled test of the best configuration found on the Test Set {[y + 1 for y in test_years]}")
            test_on_next_year_trainer = Trainer(
                env,env,
                regressor_type, double_Q, swap_Q, logger, save_path="./test_next_year", save_plots=True,
                double_Q_strategy=double_Q_strategy, trainer_phase="Test", n_jobs=n_jobs,
                shuffle=shuffle, cvar_param=cvar, use_estimator_mismatch=use_estimator_mismatch
            )

        if test_trial:
            tester = Tester(env, "Test", logger,
                            read_path=os.path.join(save_path, 'tmp'),
                            save_path=os.path.join(save_path, "tmp_test_next_year_no_retrain"),
                            n_jobs=n_jobs, use_estimator_mismatch=Q_value_diff_threshold is not None)

    tuner = Tuner(
        study_name, fixed_parameters, hyper_parameters_distributions, overwrite_study, n_trials, pruning,
        pruning_startup_trials, pruning_quantile, seeds, trainer, regressor_type, logger, save_path,
        optimize_trajectory_number,
        optimize_trajectory_window=optimize_trajectory_number and optimize_trajectory_window,
        trajectory_window_size=trajectory_window, test_best_model_so_far=test_best_model_so_far,
        test_on_next_year_trainer=test_on_next_year_trainer, n_jobs=n_jobs, optimize_cvar=cvar is not None,
        retrain_when_testing=retrain_when_testing, features_parameters=features_parameters,
        optimize_Q_value_diff_threshold=optimize_Q_value_diff_threshold,
        Q_value_diff_threshold=Q_value_diff_threshold, filter_method=filter_method, test_up_to_year=2022,
        tester_trial=tester, unroll_fqi_iterations=unroll_fqi_iterations)
    tuner.tune()

