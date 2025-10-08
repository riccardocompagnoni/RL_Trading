import pickle
import optuna
import json
import matplotlib.pyplot as plt
import logging
import numpy as np
import os
from RL_Trading.prj.app.core.fqi.entities.fqi_dataset_builder_factory import FQIDatasetBuilderFactory
from RL_Trading.prj.app.core.fqi.entities.trading_env import TradingEnv
from RL_Trading.prj.app.core.fqi.entities.trainer import Trainer
from RL_Trading.prj.app.core.models.services.utils import get_logger

study_name = "FQI_1718_19_multiseed_dfqi_optimized_trajectory_ohe_no_skip_test_observed"
storage_url = f"sqlite:////Users/giovannidispoto/Desktop/PhD/new_version/influence/RL_Trading/results/{study_name}/optuna_study.db"  # Example for SQLite
best_iterations_json = json.load(open(f"/Users/giovannidispoto/Desktop/PhD/new_version/influence/RL_Trading/results/{study_name}/validation_test_performances.json", "r"))
project_root = f'{os.path.dirname(os.path.abspath(__file__))}/../../../../..'
server_execution = False
if server_execution == True:
    save_path = f'/data/trading/results/{study_name}'
else:
    save_path = f'{project_root}/results/{study_name}'
def train_on_years(train_years: list, test_years: list, load_configuration: bool, params):
    # Set paths
    config_path = f'{project_root}/prj/app/config/config.yaml'
    os.makedirs(save_path, exist_ok=True)
    seeds = []

    if load_configuration:  # recovering seeds
        seeds = [int(f.split("d")[1]) for f in os.listdir(save_path) if
                 os.path.isdir(os.path.join(save_path, f)) and f != "tmp" and f != 'test_next_year']
    else:
        seeds = []
        for s in range(10):  # generate 10 random seeds
            np.random.seed()
            seeds.append(np.random.randint(100000))

    features_parameters = json.load(open(os.path.join(save_path, "features_params.json")))
    regressor_type = 'xgb'  # 'extra' | 'xgb'
    if regressor_type == 'xgb':
        if load_configuration:
            regressor_params = params
            fqi_iterations = regressor_params['iterations']
        # else:
        #    regressor_params = {'trajectory_number': 44, 'iterations': 1, 'min_child_weight': 5, 'learning_rate': 0.21090880087569627, 'subsample': 0.75, 'colsample_bytree': 1.0, 'n_estimators': 400, 'reg_lambda': 0.25, 'reg_alpha': 1.0, 'max_depth': 3}

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
    Q_values_percentile = None  # 0-1 range
    # this parameter is used when we want to skip the training since we are analyzing the effect of removal of training days with Q values particulary high
    skip_training = False
    logger = get_logger(logging.INFO)
    trajectory_number = regressor_params['trajectory_number']
    if 'trajectory_window' not in regressor_params.keys():
        trajectory_window = 1
    else:
        trajectory_window = regressor_params['trajectory_window']
    del regressor_params['trajectory_number']
    del regressor_params['iterations']
    if skip_training is False:
        # double_Q_strategy_min = False #force to use the mean to estimate the Q values
        training_dataset_builder = FQIDatasetBuilderFactory.create("IK", train_years, **features_parameters)
        testing_dataset_builder = FQIDatasetBuilderFactory.create("IK", test_years, **features_parameters)
        trainer = Trainer(
            TradingEnv(training_dataset_builder), TradingEnv(testing_dataset_builder),
            regressor_type, double_Q, swap_Q, logger, save_path="", save_plots=True,
            double_Q_strategy=double_Q_strategy, plot_loss=True, n_jobs=1
        )

        if trajectory_window != 1:
            gains = trainer.train(fqi_iterations, regressor_params, seeds, trajectory_number=trajectory_number,
                                  trajectory_window=trajectory_window)
        else:
            gains = trainer.train(fqi_iterations, regressor_params, seeds, trajectory_number=trajectory_number)

        if remove_days_Q_values_quantile:
            Trainer.save_days_to_remove(seeds, "", percentile=Q_values_percentile)
        if remove_days_Q_values_quantile_test:
            Trainer.save_days_to_remove_test(seeds, "", percentile=Q_values_percentile)
    else:
        logger.info("Skipped Training")
        print(Trainer.get_percentile_threshold(seeds, "", percentile=Q_values_percentile))
        # Trainer.save_days_to_remove(seeds, "", percentile=Q_values_percentile)

    if remove_days_Q_values_quantile:
        # double_Q_strategy_min = True
        logger.info("Retrain the model without the anomalous days detected from Q values quantiles")
        training_dataset_builder = FQIDatasetBuilderFactory.create("IK", train_years, **features_parameters,
                                                                   skip_percentile=Q_values_percentile)
        if remove_days_Q_values_quantile_test:
            logger.info("Removing days also on test")
            testing_dataset_builder = FQIDatasetBuilderFactory.create("IK", test_years, **features_parameters,
                                                                      skip_percentile=Q_values_percentile, phase='test')
        else:
            testing_dataset_builder = FQIDatasetBuilderFactory.create("IK", test_years, **features_parameters)

        trainer = Trainer(
            TradingEnv(training_dataset_builder), TradingEnv(testing_dataset_builder),
            regressor_type, double_Q, swap_Q, logger, save_path="", save_plots=True,
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


    return gains

trials_n = best_iterations_json['trial_number']
# Load the study
study = optuna.load_study(study_name=study_name, storage=storage_url)

print(study)

# Retrieve trials
trials = study.trials

trial_n_params = {}
trial_n_params['trial'] = []
trial_n_params['params'] = []
trial_n_params['objective_value'] = []

# Print hyperparameters for each trial
for trial in trials:
   # if trial.value != None and (float(trial.value) >= 10 and float(trial.value) < 11):
   if trial.value != None:
        if trial.number in trials_n:
            trial_n_params['trial'].append(trial.number)
            trial_n_params['params'].append(trial.params)
            trial_n_params['objective_value'].append(trial.value)
            print(f"Trial number: {trial.number}")
            print(f"Trial state: {trial.state}")
            print(f"Hyperparameters: {trial.params}")
            print(f"Objective value: {trial.value}")
            print("--------------")

print(trial_n_params)

validation_test_performances = {}
validation_test_performances['trial_number'] = []
validation_test_performances["optimization_value"] = []
validation_test_performances['test_value'] = []

for i in range(len(trial_n_params['trial'])):
    print(f"Training trial {trial_n_params['trial'][i]}")
    gains = train_on_years([2018, 2019], [2020], load_configuration=True, params=trial_n_params['params'][i])

    validation_test_performances['trial_number'].append(trial_n_params['trial'][i])
    validation_test_performances["optimization_value"].append(trial_n_params['objective_value'][i])
    validation_test_performances['test_value'].append(gains[-1])

    if len(validation_test_performances['trial_number']) > 0:
        plt.figure()
        plt.title("Performance Comparison Validation/Test")
        plt.plot(range(len(validation_test_performances['optimization_value'])),
                 validation_test_performances['optimization_value'],
                 label="Validation Performances")
        plt.plot(range(len(validation_test_performances['test_value'])),
                 validation_test_performances['test_value'], label="Test Performances")
        plt.xlabel("Iterations (Only best iterations are reported)")
        plt.ylabel("P&L (% of max allocation)")
        plt.legend()
        plt.savefig(f'{save_path}/PerformanceHistory.png')
        plt.close()
        json.dump(validation_test_performances, open(f"{save_path}/validation_test_performances_2.json", "w"))

