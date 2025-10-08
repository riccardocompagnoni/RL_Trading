import json
import logging
import os
import numpy as np
import argparse

from RL_Trading.prj.app.core.fqi.entities.fqi_dataset_builder_factory import FQIDatasetBuilderFactory
from RL_Trading.prj.app.core.fqi.entities.trading_env import TradingEnv
from RL_Trading.prj.app.core.fqi.entities.trainer import Trainer
from RL_Trading.prj.app.core.models.services.utils import get_logger


def train_on_years(train_years: list, test_years: list, load_configuration: bool, save_path_p:str = "", start_hour: int = 8, horizon: int = 60, number_of_deltas: int = 20, n_jobs:int = 1):
    # Set paths
    study_name = "FQI_1516_17_multiseed_dfqi_optimize_trajectory_test_observed_persistence_10"
    project_root = f'{os.path.dirname(os.path.abspath(__file__))}/../../../../..'
    config_path = f'{project_root}/prj/app/config/config.yaml'
    server_execution = False
    if server_execution == True:
        save_path = f'/data/trading/results/{study_name}'
    else:
        save_path = f'{project_root}/results/{study_name}'
    os.makedirs(save_path, exist_ok=True)
    seeds = []

    if load_configuration: #recovering seeds
        seeds = [int(f.split("d")[1]) for f in os.listdir(save_path) if
                 os.path.isdir(os.path.join(save_path, f)) and f != "tmp" and f != 'test_next_year']
    else:
        seeds = []
        for s in range(10):  # generate 10 random seeds
            np.random.seed()
            seeds.append(np.random.randint(100000))
    if load_configuration:
        features_parameters = json.load(open(os.path.join(save_path, "features_params.json")))
    else:
        features_parameters = {
            'roll_date_offset': 1,
            'mids_window': 1,
            'mids_offset': horizon,
            'steps_tolerance': 5,
            'number_of_deltas': number_of_deltas,
            'opening_hour': start_hour,
            'closing_hour': 18,
            'ohe_temporal_features': True,
            'persistence': horizon,
            'actions': [-1, 0, 1],
            'add_trade_imbalance': False,
            'volume_features': False,
            'skip_conte_I': False,
            'skip_covid': False
        }
        with open(f'{save_path_p}/features_params.json', 'w') as f:
            json.dump(features_parameters, f)
    #fqi_iterations = 2
    regressor_type = 'xgb'  # 'extra' | 'xgb'
    if regressor_type == 'xgb':
        if load_configuration:
            regressor_params = json.load(open(os.path.join(save_path, "parameters_opt.json")))
            fqi_iterations = regressor_params['iterations']
        else:
            regressor_params = {'trajectory_number': 0, 'trajectory_window': 1, 'iterations': 10, 'subsample': 0.75, 'colsample_bytree': 0.75}
            fqi_iterations = regressor_params['iterations']
    elif regressor_type == 'extra':
        regressor_params = {
            'random_state': 0,
            'n_jobs': -1,
            'min_samples_split': 1000,
            'n_estimators': 300
        }
    swap_Q = True
    double_Q = True
    double_Q_strategy = 'mean'
    remove_days_Q_values_quantile = False
    remove_days_Q_values_quantile_test = False
    Q_values_percentile = None  #0-1 range
    # this parameter is used when we want to skip the training since we are analyzing the effect of removal of training days with Q values particulary high
    skip_training = False
    logger = get_logger(logging.INFO)
    trajectory_number = regressor_params['trajectory_number']
    trajectory_window = regressor_params['trajectory_window']
    del regressor_params['trajectory_number']
    del regressor_params['trajectory_window']
    del regressor_params['iterations']
    if skip_training is False:
        #double_Q_strategy_min = False #force to use the mean to estimate the Q values
        training_dataset_builder = FQIDatasetBuilderFactory.create("IK", train_years, **features_parameters)
        testing_dataset_builder = FQIDatasetBuilderFactory.create("IK", test_years, **features_parameters)
        trainer = Trainer(
            TradingEnv(training_dataset_builder), TradingEnv(testing_dataset_builder),
            regressor_type, double_Q, swap_Q, logger, save_path=save_path_p, save_plots=True, double_Q_strategy=double_Q_strategy, plot_loss=True, n_jobs=n_jobs
        )

        if trajectory_window != 1:
            gains = trainer.train(fqi_iterations, regressor_params, seeds, trajectory_number=trajectory_number, trajectory_window=trajectory_window)
        else:
            gains = trainer.train(fqi_iterations, regressor_params, seeds, trajectory_number=trajectory_number)

        if remove_days_Q_values_quantile:
            Trainer.save_days_to_remove(seeds, save_path_p, percentile=Q_values_percentile)
        if remove_days_Q_values_quantile_test:
            Trainer.save_days_to_remove_test(seeds, save_path_p, percentile=Q_values_percentile)
    else:
        logger.info("Skipped Training")
        print(Trainer.get_percentile_threshold(seeds, save_path_p, percentile=Q_values_percentile))
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
            TradingEnv(training_dataset_builder), TradingEnv(testing_dataset_builder),
            regressor_type, double_Q, swap_Q, logger, save_path=save_path_p, save_plots=True,
            double_Q_strategy=double_Q_strategy, plot_loss=True,
            Q_values_quantile=Q_values_percentile,
        )
        if trajectory_window != 1:
            gains = trainer.train(fqi_iterations, regressor_params, seeds,
                                                  trajectory_number=trajectory_number,
                                                  trajectory_window=trajectory_window, removing_detected_days=True)
        else:
            gains = trainer.train(fqi_iterations, regressor_params, seeds,
                                                  trajectory_number=trajectory_number, removing_detected_days=True)


if __name__ == '__main__':
    manual_years = True
    dataset_years = list(range(2010, 2017))

    parser = argparse.ArgumentParser()
    parser.add_argument('--start_hour', type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--number_of_deltas", type=int)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--train_years", nargs='+', type=int)
    parser.add_argument("--test_years", nargs='+', type=int)
    parser.add_argument("--n_jobs", type=int)
    args = parser.parse_args()

    if manual_years:
        train_years = args.train_years
        test_years = args.test_years

        print(f'Train:{train_years}')
        print(f'Test:{test_years}')
        load_configuration = False
        train_on_years(train_years=train_years, test_years=test_years, load_configuration=load_configuration, save_path_p=args.save_path, start_hour = args.start_hour, horizon = args.horizon , number_of_deltas=args.number_of_deltas, n_jobs = args.n_jobs)
    else:
        size_s = 3
        for cut_year in range(size_s, len(dataset_years)):
            for prev_years in range(cut_year):
                train_years = dataset_years[prev_years: cut_year]
                test_years = [dataset_years[cut_year]]
                print(train_years)
                print(test_years)
                train_on_years(train_years=train_years, test_years=test_years, load_configuration=False)
