import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.normpath(os.path.join(current_dir, '..', '..', '..', '..', '..','..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..', 'RL_Trading', 'prj', 'app')))

from RL_Trading.prj.app.core.fqi.entities.fqi_dataset_builder_factory import FQIDatasetBuilderFactory
from RL_Trading.prj.app.core.fqi.entities.trading_env import TradingEnv
from RL_Trading.prj.app.core.fqi.entities.trainer import Trainer
from RL_Trading.prj.app.core.models.services.utils import get_logger
from RL_Trading.prj.app.core.fqi.simulator.simulator2 import TTFEnv



def train_on_years(train_years: list, test_years: list, load_configuration: bool):
    # Set paths
    study_name = "old_config_0.05_8y20_2y21"
    old_study = 'experts_overnight/8y20_2y21'
    start = date(2020, 8, 1)
    end = date(2021, 2, 1)
    print(study_name)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(current_dir, '..', '..', '..', '..', '..'))
    config_path = f'{project_root}/prj/app/config/config.yaml'
    server_execution = False
    if server_execution == True:
        save_path = f'/data/intesa/giovanni/results/{study_name}'
    else:
        save_path = f'{project_root}/results/{study_name}'
    os.makedirs(save_path, exist_ok=True)
    old_path = f'{project_root}/results/{old_study}'
    seeds = []

    if load_configuration: #recovering seeds
        seeds = [int(f.split("d")[1]) for f in os.listdir(old_path) if
                 os.path.isdir(os.path.join(old_path, f)) and f.startswith('seed')]
    else:
        seeds = [14871, 19225, 31583, 42410, 6363, 81312, 89870, 90225, 9121]
        #seeds = []
        #for s in range(10):  # generate 10 random seeds
        #    np.random.seed()
        #    seeds.append(np.random.randint(100000))
    if load_configuration:
        features_parameters = json.load(open(os.path.join(old_path, "features_params.json")))
        #features_parameters = json.load(open(os.path.join( f'/Volumes/Volume/experiments/FQI_1718_16_multiseed_dfqi_optimize_trajectory_test_observed_persistence_60_20_delta_prices_9_17', "features_params.json")))
        #features_parameters['closing_hour'] = 17
        #features_parameters['opening_hour'] = 9
        #features_parameters['mids_window'] = 10
        #features_parameters['number_of_deltas'] = 0
       # features_parameters['volume_history_size'] = 60
       # features_parameters['number_of_levels'] = 3


    else:
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
            'missing_values_handling': None,  # ffill | constant ( filled with -1)
            'use_moving_average': False,
            'perform_action_on_bund': False,
            'perform_action_on_btp_bund': False
        }
    #with open(f'{save_path}/features_params.json', 'w') as f:
    #    json.dump(features_parameters, f)
    #fqi_iterations = 2
    regressor_type = 'xgb'  # 'extra' | 'xgb'
    if regressor_type == 'xgb':
        if load_configuration:
            regressor_params = json.load(open(os.path.join(old_path, "parameters_opt.json")))
            #regressor_params = json.load(open(os.path.join(f'/Volumes/Volume/experiments/FQI_1516_14_multiseed_dfqi_optimize_trajectory_test_observed_persistence_60_20_delta_prices_9_17_3_levels', "parameters_opt.json")))
            #regressor_params['subsample'] =
            #regressor_params['colsample_bytree'] = 0.3
            fqi_iterations = regressor_params['iterations']
            #regressor_params['objective'] = 'reg:quantileerror'
            #regressor_params['quantile_alpha'] = 0.40
        else:

            regressor_params = {"trajectory_number": 17,
             "iterations": 3, "min_child_weight": 9, "learning_rate": 0.12209815509973816, "subsample": 0.6,
             "colsample_bytree": 0.7, "n_estimators": 205, "reg_lambda": 0.75, "reg_alpha": 0.8, "max_depth": 4,
                               'grow_policy': "depthwise",
                               'booster': "gbtree",
                               'predictor': "auto",
                               'verbosity': 0,
                               'n_jobs': -1,
                               'tree_method': "hist",
                               'colsample_bylevel': 1,
                               'colsample_bynode': 1,
                               'trajectory_window': 1,}
            fqi_iterations = regressor_params['iterations']

    elif regressor_type == 'extra':
        if load_configuration:
            regressor_params = json.load(open(os.path.join(old_path, "parameters_opt.json")))
        else:
        #TODO add load come xgb
            regressor_params = {
                'iterations': 3,
                'random_state': 0,
                'n_jobs': -1,
                'min_samples_split': 1000,
                'n_estimators': 300,
                'trajectory_window': 1
            }
        fqi_iterations = 2

    swap_Q = True
    double_Q = True
    double_Q_strategy = 'mean'
    remove_days_Q_values_quantile = False
    n_jobs = -1
    cvar = None
    #TODO check what shuffle does
    shuffle = "minute" #day | minute | week | None, randomized over shuffle type the two estimators
    remove_days_Q_values_quantile_test = False
    Q_values_percentile = None  #0-1 range
    Q_values_diff_threshold = None
    filter_method = None    #'first_step'   propagate | move_flat | first_step
    use_estimator_mismatch = False
    # this parameter is used when we want to skip the training since we are analyzing the effect of removal of training days with Q values particulary high
    skip_training = False
    logger = get_logger(logging.INFO)
    if "trajectory_number" in regressor_params.keys():
        trajectory_number = regressor_params['trajectory_number']
        del regressor_params['trajectory_number']
    else:
        trajectory_number = None
    trajectory_window=1
    #trajectory_window = regressor_params['trajectory_window']
    #del regressor_params['trajectory_window']
    del regressor_params['iterations']



    phases_dict = {
        "Train": [
            [date(2020, 3, 1), start - timedelta(days=1)],  # until day before validation
            [end, date(2023, 7, 1)]  # from end of validation until test start
        ],
        "Validation":  [
            [date(2023, 7, 1), date(2024, 9, 30)]  # fixed test interval
        ]
    }

    dataset_path = Path(f'{project_root}/prj/app/core/data/dataset_full.parquet')
    env = TTFEnv(data_path=dataset_path, persistence=features_parameters['persistence'], remove_costs=features_parameters['remove_costs'], no_overnight=False, phases_dates=phases_dict)


    if skip_training is False:
        #double_Q_strategy_min = False #force to use the mean to estimate the Q values

        trainer = Trainer(
            env, env, regressor_type, double_Q, swap_Q, logger, save_path=save_path, save_plots=True, double_Q_strategy=double_Q_strategy, plot_loss=False, n_jobs=n_jobs,
            shuffle=shuffle, cvar_param=cvar, use_estimator_mismatch = use_estimator_mismatch
        )

        if trajectory_window != 1:
            gains = trainer.train(fqi_iterations, regressor_params, seeds, trajectory_number=trajectory_number, trajectory_window=trajectory_window, filter_method=filter_method, Q_values_diff_threshold=Q_values_diff_threshold, unroll_fqi_iterations=False)
        else:
            gains = trainer.train(fqi_iterations, regressor_params, seeds, trajectory_number=trajectory_number,filter_method=filter_method, Q_values_diff_threshold=Q_values_diff_threshold, unroll_fqi_iterations=False)

        if remove_days_Q_values_quantile:
            Trainer.save_days_to_remove(seeds, save_path, percentile=Q_values_percentile)
        if remove_days_Q_values_quantile_test:
            Trainer.save_days_to_remove_test(seeds, save_path, percentile=Q_values_percentile)
    else:
        logger.info("Skipped Training")
        print(Trainer.get_percentile_threshold(seeds, save_path, percentile=Q_values_percentile))
        #Trainer.save_days_to_remove(seeds, "", percentile=Q_values_percentile)

    if remove_days_Q_values_quantile:
        #double_Q_strategy_min = True
        logger.info("Retrain the model without the anomalous days detected from Q values quantiles")
        training_dataset_builder = FQIDatasetBuilderFactory.create("IK", train_years, **features_parameters, skip_percentile=Q_values_percentile)
        if remove_days_Q_values_quantile_test:
            logger.info("Removing days also on test")
            testing_dataset_builder = FQIDatasetBuilderFactory.create("IK", test_years, **features_parameters, skip_percentile=Q_values_percentile, phase='test')
        else:
            testing_dataset_builder = FQIDatasetBuilderFactory.create("IK", test_years, **features_parameters)

        trainer = Trainer(
            env, env,
            regressor_type, double_Q, swap_Q, logger, save_path=save_path, save_plots=True,
            double_Q_strategy=double_Q_strategy, plot_loss=False,
            Q_values_quantile=Q_values_percentile, n_jobs= n_jobs,
            shuffle=shuffle, cvar_param=cvar
        )
        if trajectory_window != 1:
            gains = trainer.train(fqi_iterations, regressor_params, seeds,
                                                  trajectory_number=trajectory_number,
                                                  trajectory_window=trajectory_window, removing_detected_days=True, unroll_fqi_iterations=False)
        else:
            gains = trainer.train(fqi_iterations, regressor_params, seeds,
                                                  trajectory_number=trajectory_number, removing_detected_days=True, unroll_fqi_iterations=False)


if __name__ == '__main__':
    manual_years = True
    dataset_years = list(range(2010, 2017))

    if manual_years:
        train_years = [2019, 2020]
        test_years = [2021, 2022]
        load_configuration = True
        train_on_years(train_years=train_years, test_years=test_years, load_configuration=load_configuration)
    else:
        size_s = 3
        for cut_year in range(size_s, len(dataset_years)):
            for prev_years in range(cut_year):
                train_years = dataset_years[prev_years: cut_year]
                test_years = [dataset_years[cut_year]]
                print(train_years)
                print(test_years)
                train_on_years(train_years=train_years, test_years=test_years, load_configuration=False)