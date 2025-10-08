import datetime
import json
import logging
import numpy as np
import os
import pandas as pd
import shutil

from RL_Trading.prj.app.core.classification.entities.loss_function import WeightedCost, OverlapCost
from RL_Trading.prj.app.core.classification.entities.classification_dataset_builder_factory import ClassificationDatasetBuilderFactory
from RL_Trading.prj.app.core.classification.entities.trainer import Trainer
from RL_Trading.prj.app.core.models.services.utils import get_logger
from core.fqi.services.read_only_handler import handleRemoveReadonly


def build_classification_dataset() -> pd.DataFrame:
    years = [2016, 2017, 2018]
    features_parameters = {
        'roll_date_offset': 1,
        'mids_window': 1,
        'mids_offset': 10,
        'steps_tolerance': 5,
        'number_of_deltas': 20,
        'volume_history_size': 0,
        'number_of_levels': 10,
        'opening_hour': 8,
        'closing_hour': 18,
        'ohe_temporal_features': True,
        'target_offset': 10,
        'add_trade_imbalance': False,
        'volume_features': True,
        'lob_price_features': False,
        'skip_conte_I': False,
        'skip_covid': False
    }
    dtf = ClassificationDatasetBuilderFactory.create("IK", years, **features_parameters).build_dataset()
    dtf = dtf.dropna(subset='delta_mid_target')
    return dtf


if __name__ == '__main__':
    df = build_classification_dataset()
    # Define cost matrix
    cost_matrix = np.array([
        [0.0, 0.1, 1.0],
        [0.1, 0.0, 0.1],
        [1.0, 0.1, 0.0]
    ])
    loss_function = WeightedCost(cost_matrix)
    # Initialize trainer
    logger = get_logger(logging.DEBUG)
    classifier_type = 'xgb'  # 'extra' | 'xgb'
    trainer = Trainer(df, classifier_type, cost_matrix, loss_function, logger)
    # Get optimized parameters
    study_name = 'test_Volumes_Prices_LOB_target10'
    project_root = f'{os.path.dirname(os.path.abspath(__file__))}/../../../../..'
    study_path = f'{project_root}/results/{study_name}'
    with open(f'{study_path}/parameters_opt.json', 'r') as f:
        classifier_params = json.load(f)
    # Train model
    num_training_years = 2
    bootstrap_rounds = 2
    early_stopping_rounds = None #5
    save_path = f'{study_path}/model'
    if os.path.exists(save_path):
        shutil.rmtree(save_path, onerror=handleRemoveReadonly)
    os.makedirs(save_path)
    cost = trainer.train(bootstrap_rounds, classifier_params, num_training_years, early_stopping_rounds, save_path)
