import datetime
import logging
import numpy as np
import pandas as pd

from RL_Trading.prj.app.core.classification.entities.loss_function import WeightedCost, OverlapCost
from RL_Trading.prj.app.core.classification.entities.classification_dataset_builder_factory import ClassificationDatasetBuilderFactory
from RL_Trading.prj.app.core.classification.entities.tuner import Tuner
from core.models.services.utils import get_logger

"""
features_parameters = {
        'roll_date_offset': 1,
        'mids_window': 1,
        'mids_offset': 10,
        'steps_tolerance': 5,
        'number_of_deltas': 20,
        'volume_history_size': 0,
        'number_of_levels': 5,
        'opening_hour': 8,
        'closing_hour': 18,
        'ohe_temporal_features': True,
        'persistence': 10,
        'actions': [-1, 0, 1],
        'add_trade_imbalance': False,
        'volume_features': False,
        'skip_conte_I': False,
        'skip_covid': False
    }"""

def build_classification_dataset() -> pd.DataFrame:
    years = [2016, 2017, 2018]
    features_parameters = {
        'roll_date_offset': 1,
        'mids_window': 1,
        'mids_offset': 10,
        'steps_tolerance': 5,
        'number_of_deltas': 0,
        'volume_history_size': 20,
        'number_of_levels': 5,
        'opening_hour': 9,
        'closing_hour': 18,
        'ohe_temporal_features': True,
        'target_offset': 10,
        'add_trade_imbalance': False,
        'volume_features': False,
        'skip_conte_I': False,
        'skip_covid': False
    }
    dtf = ClassificationDatasetBuilderFactory.create("IK", years, **features_parameters).build_dataset()
    #dtf = dtf[dtf['time'].dt.date >= datetime.date(2018, month=1, day=1)]
    dtf = dtf.dropna(subset='delta_mid_target')
    return dtf


if __name__ == '__main__':
    np.random.seed()
    classifier_type = 'xgb'  # 'extra' | 'xgb'
    if classifier_type == 'xgb':
        fixed_parameters = {
            'objective': "multi:softmax",
            'random_state': 0,
            'verbosity': 0,
            'n_jobs': -1,
            'tree_method': "hist",
        }
        hyper_parameters_distributions = [
            {'name': 'min_child_weight', 'type': 'int', 'min': 1, 'max': 10, 'step': 1},
            {'name': 'learning_rate', 'type': 'log', 'min': 0.01, 'max': 0.50},
            {'name': 'subsample', 'type': 'float', 'min': 0.5, 'max': 1.0, 'step': 0.05},
            {'name': 'colsample_bytree', 'type': 'float', 'min': 0.5, 'max': 1.0, 'step': 0.05},
            {'name': 'n_estimators', 'type': 'int', 'min': 50, 'max': 500, 'step': 50},
            {'name': 'reg_lambda', 'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.05},
            {'name': 'reg_alpha', 'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.05},
            {'name': 'max_depth', 'type': 'int', 'min': 3, 'max': 10, 'step': 1},
        ]
    elif classifier_type == 'extra':
        fixed_parameters = {
            'criterion': "gini",
            'random_state': 0,
            'verbose': 0,
            'max_features': "sqrt",
            'bootstrap': False,
            'oob_score': False,
            'n_jobs': -1,
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0,
            'warm_start': False,
            #'monotonic_cst': None
        }
        hyper_parameters_distributions = [
            {'name': 'min_samples_split', 'type': 'int', 'min': 10, 'max': 100, 'step': 10},
            {'name': 'min_samples_leaf', 'type': 'int', 'min': 1, 'max': 10, 'step': 1},
            {'name': 'min_weight_fraction_leaf', 'type': 'float', 'min': 0.0, 'max': 0.5, 'step': 0.05},
            {'name': 'n_estimators', 'type': 'int', 'min': 50, 'max': 500, 'step': 50},
            {'name': 'max_depth', 'type': 'int', 'min': 3, 'max': 10, 'step': 1},
            {'name': 'ccp_alpha', 'type': 'float', 'min': 0.0, 'max': 1.0, 'step': 0.05},
        ]
    study_name = f'IKA_test'
    df = build_classification_dataset()
    #df = df[(df.time.dt.year >= 2010) & (df.time.dt.year <= 2020)]
    cost_matrix = np.array([
        [0.0, 0.1, 1.0],
        [0.1, 0.0, 0.1],
        [1.0, 0.1, 0.0]
    ])
    loss_function = WeightedCost(cost_matrix)
    logger = get_logger(logging.INFO)  # logging.INFO
    overwrite_study = True
    n_trials = 300
    bootstrap_rounds = 6
    num_training_years = 2
    early_stopping_rounds = None #5
    tuner = Tuner(
        study_name,
        classifier_type,
        fixed_parameters,
        hyper_parameters_distributions,
        overwrite_study,
        n_trials,
        bootstrap_rounds,
        df,
        num_training_years,
        early_stopping_rounds,
        cost_matrix,
        loss_function,
        logger
    )
    tuner.tune()
