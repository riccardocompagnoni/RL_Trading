import logging
import numpy as np
import numpy.typing as npt
import pandas as pd
import typing
import xgboost

from RL_Trading.prj.app.core.classification.entities.loss_function import LossFunction
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier


class Trainer(object):

    def __init__(
            self,
            df: pd.DataFrame,
            classifier_type: str,
            cost_matrix: npt.NDArray[npt.NDArray[float]],
            loss_function: LossFunction,
            logger: logging.Logger,
    ):
        self._df = df
        if classifier_type == 'xgb':
            self._classifier = xgboost.XGBClassifier
        elif classifier_type == 'extra':
            self._classifier = ExtraTreesClassifier
            self._df = self._df.ffill().bfill()
        else:
            raise AttributeError(f"Invalid classifier {classifier_type}.")
        self._cost_matrix = cost_matrix
        class_weights_dict = dict(zip([0, 1, 2], cost_matrix.sum(axis=0)))
        self._loss_function = loss_function
        self._class_labels = list(class_weights_dict.keys())
        self._class_weights_dict = class_weights_dict
        self._logger = logger

    def train(
            self,
            bootstrap_rounds: int,
            classifier_params: typing.Dict,
            num_training_years: int,
            early_stopping_rounds: typing.Optional[int] = None,
            save_path: typing.Optional[str] = None
    ):
        if early_stopping_rounds is None:
            classifier = self._classifier(**classifier_params)
        else:
            classifier = self._classifier(**classifier_params, early_stopping_rounds=early_stopping_rounds)
        years = self._df['time'].dt.year.unique()
        num_classes = len(self._class_labels)
        new_cost_function = False
        confusion_matrices_valid = []
        self._logger.info(f'Training started with params {classifier_params}.')
        for i in range(len(years) - num_training_years):
            # Get 3 years training set and 1 year validation set.
            train_years = years[i:i+num_training_years]
            valid_year = years[i+num_training_years]
            file_prefix = f"{''.join([f'{y % 100}' for y in train_years])}_{valid_year % 100}"
            train_df = self._df[self._df['time'].dt.year.isin(train_years)].drop(columns='time')
            valid_df = self._df[self._df['time'].dt.year == valid_year].drop(columns='time')
            X_valid = valid_df.drop(columns='delta_mid_target')
            y_valid = valid_df['delta_mid_target']
            # Boostrap subsample from the orginal sets.
            for j in range(bootstrap_rounds):
                self._logger.info(f'Train {train_years}, Validation {valid_year}, Bootstrap {j+1} of {bootstrap_rounds}.')
                # Sample blocks of 60 consecutive rows (i.e., 1 hour).
                block_size = 60*24
                # Calculate the number of blocks to sample to obtain a boostrapped dataset with a length equal to the
                # original training set.
                n_blocks = int(np.ceil(len(train_df) / block_size))
                # Create a list of [0, 1, ..., block_size] of size n_blocks.
                nexts = np.repeat([np.arange(0, block_size)], n_blocks, axis=0)
                # Calculate the starting index of the last block of size block_size.
                last_block = len(train_df) - block_size
                # Get the starting index of the blocks by sampling them from the range [0, last_block).
                blocks = np.random.randint(0, last_block, (n_blocks, 1))
                # Generate block indices by summing blocks and nexts (from [0, 1, ..., block_size] to
                # [idx, idx + 1, ... idx + block_size], where idx is the sampled starting index of the block).
                # Concatenate sampled blocks and keep the first len(train_df) records to get the same number of records
                # as the original (the last block could be shorter).
                train_idxs = (blocks + nexts).ravel()[:len(train_df)]
                # Get the boostrapped training set and its labels.
                X_train = train_df.iloc[train_idxs].drop(columns='delta_mid_target')
                y_train = train_df.iloc[train_idxs]['delta_mid_target']
                # Transform y into quantile-based labels to get balanced dataset.
                #TODO check se la definizione delle classi debba avvenire a livello di bootstrap dataset
                y_train_c, b = pd.qcut(y_train, num_classes, retbins=True, labels=self._class_labels)
                b[0], b[-1] = -np.inf, np.inf
                self._logger.debug(f'Splitting points: {b[1]:,.3f}, {b[2]:,.3f}.')
                # and use the same cutpoints to label the validation set.
                y_valid_c = pd.cut(y_valid, bins=b, labels=self._class_labels)
                # Fit the model.
                if early_stopping_rounds is None:
                    classifier.fit(X_train, y_train_c, sample_weight=y_train_c.map(self._class_weights_dict))
                else:
                    classifier.fit(X_train, y_train_c, sample_weight=y_train_c.map(self._class_weights_dict),
                                   eval_set=[(X_valid, y_valid_c)], sample_weight_eval_set=[y_valid_c.map(self._class_weights_dict)])
                confusion_matrix_valid = confusion_matrix(y_valid_c, classifier.predict(X_valid),
                                                          labels=self._class_labels, normalize="true")
                confusion_matrices_valid.append(confusion_matrix_valid)
                if save_path is not None:
                    # Save feature importances.
                    feature_importances = pd.Series(classifier.feature_importances_, index=X_train.columns)
                    feature_importances.to_csv(f'{save_path}/{file_prefix}_{j}_fi.txt', header=False)
                    # Save confusion matrices for the current bootstrapped dataset.
                    confusion_matrix_train = confusion_matrix(y_train_c, classifier.predict(X_train), labels=self._class_labels, normalize="true")
                    np.savetxt(f'{save_path}/{file_prefix}_{j}_train_cm.txt', confusion_matrix_train)
                    np.savetxt(f'{save_path}/{file_prefix}_{j}_valid_cm.txt', confusion_matrix_valid)
        cost = self._loss_function.calculate_cost(confusion_matrices_valid)
        self._logger.info(f'Training ended with cost {cost:,.5f}.')
        return cost
