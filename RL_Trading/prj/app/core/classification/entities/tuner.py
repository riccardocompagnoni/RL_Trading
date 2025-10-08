import json
import logging
import numpy.typing as npt
import optuna
import os
import pandas as pd
import typing

from munch import DefaultMunch
from RL_Trading.prj.app.core.classification.entities.loss_function import LossFunction
from RL_Trading.prj.app.core.models.entities.hyper_parameter import HyperParameter
from RL_Trading.prj.app.core.classification.entities.trainer import Trainer


class Tuner:

    def __init__(
            self,
            study_name: str,
            classifier_type: str,
            fixed_parameters: typing.Dict,
            hyper_parameters_distributions: typing.List[typing.Dict],
            overwrite_study: bool,
            n_trials: int,
            bootstrap_rounds: int,
            df: pd.DataFrame,
            num_training_years: int,
            early_stopping_rounds: int,
            cost_matrix: npt.NDArray[npt.NDArray[float]],
            loss_function: LossFunction,
            logger: logging.Logger
    ):
        # Enable propagation and disable the Optuna’s default handler to prevent double logging since the root logger
        # has been configured.
        self._logger = logger
        optuna.logging.enable_propagation()
        optuna.logging.disable_default_handler()
        # Set params
        self._n_trials = n_trials
        self._bootstrap_rounds = bootstrap_rounds
        self._num_training_years = num_training_years
        self._early_stopping_rounds = early_stopping_rounds
        self._fixed_classifier_params = fixed_parameters
        # Create Trainer instance
        self._trainer = Trainer(df, classifier_type, cost_matrix, loss_function, self._logger)
        # Set path
        self._study_name = study_name
        self._root_path = f'../../results/{self._study_name}'
        os.makedirs(self._root_path, exist_ok=True)
        # Set hyper parameters
        self._hyper_parameters = [HyperParameter(DefaultMunch.fromDict(d)) for d in hyper_parameters_distributions]
        # Create Optuna study
        storage = f'sqlite:///{self._root_path}/optuna_study.db'
        if overwrite_study:
            self._delete_study(storage)
        self.study = optuna.create_study(
            study_name=self._study_name,
            storage=storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(multivariate=True),
            direction='minimize'
        )

    def _delete_study(self, storage: str):
        study_path = storage.replace('sqlite:///', '')
        if os.path.exists(study_path):
            os.remove(study_path)
            self._logger.info(f'Study {self._study_name} deleted from storage.')

    def tune(self):
        # Optimizing Hyper-parameters
        self.study.optimize(self._objective, n_trials=self._n_trials, callbacks=[self._optimize_callback])

    def _objective(self, trial: optuna.trial.Trial) -> float:
        # Suggesting hyper-parameters
        classifier_params = self._suggest_params(trial)
        # Training classification model
        cost = self._trainer.train(
            self._bootstrap_rounds,
            classifier_params,
            self._num_training_years,
            self._early_stopping_rounds,
            save_path=self._root_path
        )
        if trial.number > 1:
            # Plot optimization results
            self._save_plots()
        return cost

    def _suggest_params(self, trial: optuna.trial.Trial) -> typing.Dict:
        classifier_params = self._fixed_classifier_params.copy()
        # Suggest hyper parameters
        for hyper_parameter in self._hyper_parameters:
            classifier_params[hyper_parameter.name] = hyper_parameter.suggest_value(trial)
        return classifier_params

    def _optimize_callback(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if study.best_trial.number >= trial.number:
            optimal_params = {**self._fixed_classifier_params, **self.study.best_params}
            with open(f'{self._root_path}/parameters_opt.json', 'w') as f:
                json.dump(optimal_params, f)

    def _save_plots(self):
        # Plot optimization results
        fig = optuna.visualization.plot_optimization_history(self.study)
        fig.write_image(f'{self._root_path}/ParamsOptHistory.png')
        fig = optuna.visualization.plot_param_importances(self.study)
        fig.write_image(f'{self._root_path}/ParamsImportance.png')
        fig = optuna.visualization.plot_contour(self.study)
        fig.write_image(f'{self._root_path}/ParamsContour.png', width=3000, height=1750)
        fig = optuna.visualization.plot_slice(self.study)
        fig.write_image(f'{self._root_path}/ParamsSlice.png')
