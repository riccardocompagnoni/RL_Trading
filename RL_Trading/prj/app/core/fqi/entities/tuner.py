import json
import logging
import math
import numpy as np
import optuna
import os
import pandas as pd
import pickle
import shutil
import typing

from munch import DefaultMunch
import matplotlib.pyplot as plt

from RL_Trading.prj.app.core.fqi.entities.fqi_dataset_builder_factory import FQIDatasetBuilderFactory
from RL_Trading.prj.app.core.fqi.entities.trading_env import TradingEnv
from RL_Trading.prj.app.core.models.entities.hyper_parameter import HyperParameter
from RL_Trading.prj.app.core.fqi.entities.tester import Tester
from RL_Trading.prj.app.core.fqi.entities.trainer import Trainer
from RL_Trading.prj.app.core.fqi.services.plotter import Plotter
from core.fqi.services.read_only_handler import handleRemoveReadonly


class Tuner:

    def __init__(
            self,
            study_name: str,
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
            optimize_trajectory_number: bool,
            optimize_trajectory_window: bool,
            test_best_model_so_far: bool = False,
            tester_trial: typing.Optional[Tester] = None,
            test_on_next_year_trainer: typing.Optional[Trainer] = None,
            trajectory_window_size: typing.Optional[int] = None,
            n_jobs: int = 1,
            optimize_cvar: bool = False,
            retrain_when_testing: bool = False,
            features_parameters: typing.Optional[typing.Dict] = None,
            test_up_to_year: int = 2023,
            optimize_Q_value_diff_threshold : bool = False,
            Q_value_diff_threshold: typing.Optional[float] = None,
            filter_method: typing.Optional[str] = None,
            unroll_fqi_iterations = False

    ):
        # Enable propagation and disable the Optuna’s default handler to prevent double logging since the root logger
        # has been configured.
        self._logger = logger
        optuna.logging.enable_propagation()
        optuna.logging.disable_default_handler()
        # Set params
        self._n_trials = n_trials
        self._test_up_to_year = test_up_to_year
        self._Q_value_diff_threshold = Q_value_diff_threshold
        self._pruning = pruning
        self._filter_method = filter_method
        self._unroll_fqi_iterations = unroll_fqi_iterations
        self._features_parameters = features_parameters
        self._pruning_startup_trials = pruning_startup_trials
        self._pruning_quantile = pruning_quantile
        self._optimize_Q_value_diff_threshold = optimize_Q_value_diff_threshold
        self._fixed_params = fixed_params
        self._optimize_trajectory_number = optimize_trajectory_number
        self._optimize_trajectory_window = optimize_trajectory_window
        self._optimize_cvar = optimize_cvar
        if self._optimize_trajectory_window:
            self._trajectory_window_size = trajectory_window_size
        self._seeds = seeds
        self._validation_test_performances = {"trial_number": [], "optimization_value": [], "test_value": []}
        self._test_best_model_so_far = test_best_model_so_far
        self._retrain_when_testing = retrain_when_testing
        self._tester_trial = tester_trial
        # Set path
        self._study_name = study_name
        self._root_path = save_path
        os.makedirs(self._root_path, exist_ok=True)
        self._tmp_res_save_path = os.path.join(self._root_path, 'tmp')
        self._test_best_model_so_far_save_path = os.path.join(self._root_path, 'test_next_year')
        if tester_trial:
            self._tmp_tester_trial_path = os.path.join(self._root_path, "tmp_test_next_year_no_retrain")
            self._tester_trial_path = os.path.join(self._root_path, "test_next_year_no_retrain")
            os.makedirs(self._tmp_tester_trial_path, exist_ok=True)
            os.makedirs(self._tester_trial_path, exist_ok=True)
            self._validation_test_performances_complete = {"trial_number": [], "optimization_value": [], "test_value": [], 'std': []}

        # Create Trainer instance
        self._trainer = trainer
        self._trainer.save_path = self._tmp_res_save_path
        self._test_on_next_year_trainer = test_on_next_year_trainer
        self._n_jobs = n_jobs
        if test_best_model_so_far and self._test_on_next_year_trainer is not None:
            self._test_on_next_year_trainer.save_path = self._test_best_model_so_far_save_path
        # Set hyper-parameters
        self._hyper_parameters = [HyperParameter(DefaultMunch.fromDict(d)) for d in hyper_parameters_distributions]

        # Number of trials counting fqi iterations only once
        self._actual_trials = 0
        # Create Optuna study
        storage = f'sqlite:///{self._root_path}/optuna_study.db'
        if overwrite_study:
            self._delete_study(storage)
        elif len(os.listdir(self._root_path)) > 0:
            # Load used seeds
            self._logger.info('Recovering seeds from available study.')
            seeds = [int(f.split("d")[1]) for f in os.listdir(self._root_path) if
                     os.path.isdir(os.path.join(self._root_path, f)) and f.startswith("seed")]
            self._seeds = seeds
            self._logger.info(f'Recovered seeds: {self._seeds}.')
        if self._optimize_Q_value_diff_threshold and self._Q_value_diff_threshold is not None:
            logger.warning(f"Enabled optimization of the Q_value_diff_threshold, the threshold specified {self._Q_value_diff_threshold} will be ignored")
        self.study = optuna.create_study(
            study_name=self._study_name,
            storage=storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(n_startup_trials=50, multivariate=True),
            direction='maximize'
        )
        # Set gains history for pruning
        self._gains_history = [[]]
        for hyper_parameter in self._hyper_parameters:
            if hyper_parameter.name == 'iterations':
                self._gains_history = [[] for _ in range(hyper_parameter.distribution.high)]
                break
        if self._pruning:
            trials_df = self.study.trials_dataframe(attrs=('value', 'params', 'state'))
            if len(trials_df) > 0:
                trials_df = trials_df[trials_df['state'] == 'COMPLETE'][['params_iterations', 'value']]
                for i in range(min(trials_df['params_iterations'].max(), len(self._gains_history))):
                    self._gains_history[i] = trials_df[trials_df['params_iterations'] == i + 1]['value'].to_list()

    def _delete_study(self, storage: str):
        study_path = storage.replace('sqlite:///', '')
        if os.path.exists(study_path):
            os.remove(study_path)
            self._logger.info(f'Study {self._study_name} deleted from storage.')

    def tune(self):
        self.study.optimize(self._objective, n_trials=self._n_trials, callbacks=[self._bootstrap_trial])
        if os.path.exists(self._tmp_res_save_path):
            shutil.rmtree(self._tmp_res_save_path, onerror=handleRemoveReadonly)
        #if os.path.exists(self._test_best_model_so_far_save_path):
        #    shutil.rmtree(self._test_best_model_so_far_save_path, onerror=handleRemoveReadonly)

    def _objective(self, trial: optuna.trial.Trial) -> float:
        self._actual_trials += 1

        # Clean working folder
        if os.path.exists(self._tmp_res_save_path):
            shutil.rmtree(self._tmp_res_save_path, onerror=handleRemoveReadonly)
        os.makedirs(self._tmp_res_save_path)
        # Suggesting hyper-parameters
        regression_params, fqi_iterations, trajectory_number, filter_method, Q_values_diff_threshold = self._suggest_params(trial)

        if self._optimize_Q_value_diff_threshold is False and self._Q_value_diff_threshold is not None:
            Q_values_diff_threshold = self._Q_value_diff_threshold
            filter_method = self._filter_method

        pruning_min_values = None
        pruning = self._pruning and self._actual_trials >= self._pruning_startup_trials
        if pruning:
            pruning_min_values = np.ones(len(self._gains_history)) * -np.inf
            for i, gains in enumerate(self._gains_history):
                if gains:
                    pruning_min_values[i] = np.quantile(gains, self._pruning_quantile)
        # Training FQI model
        quantile = self._fixed_params['quantile_alpha'] if 'quantile_alpha' in self._fixed_params.keys() else None
        if self._optimize_trajectory_window:
            gains = self._trainer.train(fqi_iterations, regression_params, self._seeds, pruning, pruning_min_values, trajectory_number, trajectory_window = self._fixed_params['trajectory_window'], filter_method=filter_method, Q_values_diff_threshold = Q_values_diff_threshold, unroll_fqi_iterations=self._unroll_fqi_iterations, quantile=quantile)
        else:
            gains = self._trainer.train(fqi_iterations, regression_params, self._seeds, pruning, pruning_min_values, trajectory_number, filter_method=filter_method, Q_values_diff_threshold=Q_values_diff_threshold, unroll_fqi_iterations=self._unroll_fqi_iterations, quantile=quantile)

        # Saving intermediate iterations objective values
        if self._unroll_fqi_iterations:
            for it in range(fqi_iterations):
                self._gains_history[it].append(gains[it])
            trial.set_user_attr("intermediate_values", gains[:fqi_iterations - 1])
        # Plot optimization results
        return gains[-1]

    def _suggest_params(self, trial: optuna.trial.Trial) -> typing.Tuple[typing.Dict, int, typing.Optional[int], typing.Optional[int], typing.Optional[int]]:
        # Start from fixed parameters
        regressor_params = self._fixed_params.copy()
        # Suggest hyper parameters
        for hyper_parameter in self._hyper_parameters:
            suggested_value = hyper_parameter.suggest_value(trial)
            if hyper_parameter.name in regressor_params:
                raise AttributeError(f'{hyper_parameter.name} already fixed.')
            regressor_params[hyper_parameter.name] = suggested_value
        # Return regressor parameters and iteration separately
        if self._optimize_trajectory_number:
            if self._optimize_Q_value_diff_threshold:
                return regressor_params, regressor_params.pop('iterations'), regressor_params.pop('trajectory_number'), regressor_params.pop('filter_method'), regressor_params.pop('Q_values_diff_threshold')
            else:
                return regressor_params, regressor_params.pop('iterations'), regressor_params.pop('trajectory_number'), None, None
        else:
            if self._optimize_Q_value_diff_threshold:
                return regressor_params, regressor_params.pop('iterations'), None, regressor_params.pop('filter_method'), regressor_params.pop('Q_values_diff_threshold')
            else:
                return regressor_params, regressor_params.pop('iterations'), None, None, None

    def _bootstrap_trial(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        quantile = self._fixed_params['quantile_alpha'] if 'quantile_alpha' in self._fixed_params.keys() else None

        if trial.state != optuna.trial.TrialState.PRUNED:
            if self._unroll_fqi_iterations:
                for it in range(len(trial.user_attrs['intermediate_values'])):
                    # Set trial parameters
                    params = trial.params.copy()
                    params['iterations'] = it + 1
                    # Set trial parameters distribution
                    distributions = {}
                    for hyper_parameter in self._hyper_parameters:
                        if not hyper_parameter.is_fixed():
                            distributions[hyper_parameter.name] = hyper_parameter.distribution
                    # Set trial objective value
                    value = trial.user_attrs['intermediate_values'][it]
                    # Create trial and add trial
                    bootstrap_trial = optuna.trial.create_trial(params=params, distributions=distributions, value=value)
                    study.add_trial(bootstrap_trial)

            if trial.number > 1:
                self._save_plots()

            if self._tester_trial:
                params = {**self._fixed_params, **trial.params}
                regressor_params = params.copy()
                fqi_iterations = params['iterations']
                if 'filter_method' in regressor_params:
                    filter_method = regressor_params['filter_method']
                    Q_values_diff_threshold = regressor_params['Q_values_diff_threshold']
                    del regressor_params['filter_method']
                    del regressor_params['Q_values_diff_threshold']
                else:
                    if self._optimize_Q_value_diff_threshold is False and self._Q_value_diff_threshold is not None:
                        Q_values_diff_threshold = self._Q_value_diff_threshold
                        filter_method = self._filter_method
                    else:
                        filter_method = None
                        Q_values_diff_threshold = None
                self._logger.info(f"Evaluating Algorithm up to {self._test_up_to_year} without retrain")
                assert len(self._trainer._testing_env._dataset_builder.years) == 1

                gains, last_gains = self._tester_trial.test(fqi_iterations, self._seeds, Q_values_diff_threshold=Q_values_diff_threshold,
                            filter_method=filter_method, quantile=quantile)

                Plotter.plot_percentage_returns("Test", fqi_iterations, self._seeds, self._tmp_tester_trial_path)
                Plotter.plot_percentage_pl("Test", fqi_iterations, self._seeds, self._tmp_tester_trial_path,
                                           self._trainer._training_env.persistence, True)
                Plotter.plot_seeds_percentage("Test", fqi_iterations, self._seeds, self._tmp_tester_trial_path,
                                           self._trainer._training_env.persistence, False)

                Plotter.plot_best_q_values("Test", fqi_iterations, self._seeds, self._tmp_tester_trial_path)

                self._validation_test_performances_complete['trial_number'].append(trial.number)
                self._validation_test_performances_complete["optimization_value"].append(trial.value)
                self._validation_test_performances_complete['test_value'].append(gains[-1])
                self._validation_test_performances_complete['std'].append(np.std(last_gains))

                if len(self._validation_test_performances_complete['trial_number']) > 0:
                    plt.figure()
                    plt.title("Performance Comparison Validation/Test (no retrain)")
                    plt.plot(range(len(self._validation_test_performances_complete['optimization_value'])),
                             self._validation_test_performances_complete['optimization_value'],
                             label="Validation Performances")
                    plt.plot(range(len(self._validation_test_performances_complete['test_value'])),
                             self._validation_test_performances_complete['test_value'], label="Test Performances")
                    plt.xlabel("Iterations (Only best iterations are reported)")
                    plt.ylabel("P&L")
                    plt.legend()
                    plt.savefig(f'{self._root_path}/PerformanceHistory(NoRetrain).png')
                    plt.close()

                    plt.figure()
                    plt.title("Validation performance vs. std dev")
                    plt.plot(range(len(self._validation_test_performances_complete['optimization_value'])),
                             self._validation_test_performances_complete['optimization_value'],
                             label="Validation Performances")
                    plt.plot(range(len(self._validation_test_performances_complete['std'])),
                             self._validation_test_performances_complete['std'], label="Standard deviation")
                    plt.xlabel("Trials")
                    plt.ylabel("P&L")
                    plt.legend()
                    plt.savefig(f'{self._root_path}/StdHistory.png')
                    plt.close()
                    json.dump(self._validation_test_performances_complete,
                              open(f"{self._root_path}/validation_test_performances_complete.json", "w"))


            # Replace training output if current trial is the new best trial
            if study.best_trial.number >= trial.number:
                for f in os.listdir(self._root_path):
                    if f.startswith('seed'):
                        shutil.rmtree(os.path.join(self._root_path, f), onerror=handleRemoveReadonly)
                shutil.copytree(self._tmp_res_save_path, self._root_path, dirs_exist_ok=True)
                if self._tester_trial:
                    shutil.copytree(self._tmp_tester_trial_path, self._tester_trial_path, dirs_exist_ok=True)

                optimal_params = {**self._fixed_params, **self.study.best_params}
                with open(f'{self._root_path}/parameters_opt.json', 'w') as f:
                    json.dump(optimal_params, f)
                # Test and plot results on training set
                self._logger.info('Testing on training set the new best trial.')
                fqi_iterations = optimal_params['iterations']
                if self._optimize_trajectory_number:
                    trajectory_number = optimal_params['trajectory_number']
                if self._optimize_Q_value_diff_threshold:
                    Q_values_diff_threshold = optimal_params['Q_values_diff_threshold']
                    filter_method = optimal_params['filter_method']
                else:
                    if self._optimize_Q_value_diff_threshold is False and self._Q_value_diff_threshold is not None:
                        Q_values_diff_threshold = self._Q_value_diff_threshold
                        filter_method = self._filter_method
                    else:
                        Q_values_diff_threshold = None
                        filter_method = None
                phase = "Train"
                if self._optimize_cvar:
                    tester = Tester(self._trainer._training_env, phase, self._logger, self._root_path, n_jobs=self._n_jobs, use_estimator_mismatch=Q_values_diff_threshold != None)
                    tester.test(fqi_iterations, self._trainer.get_seeds_cvar(), Q_values_diff_threshold=Q_values_diff_threshold, filter_method=filter_method, unroll_fqi_iterations=self._unroll_fqi_iterations, quantile=quantile)
                    Plotter.plot_percentage_returns(phase, fqi_iterations, self._trainer.get_seeds_cvar(), self._root_path)
                    Plotter.plot_percentage_pl(phase, fqi_iterations, self._trainer.get_seeds_cvar(), self._root_path,
                                               self._trainer._training_env.persistence, True)
                    Plotter.plot_seeds_percentage(phase, fqi_iterations, self._trainer.get_seeds_cvar(), self._root_path,
                                               self._trainer._training_env.persistence, False)

                    Plotter.plot_best_q_values(phase, fqi_iterations, self._trainer.get_seeds_cvar(), self._root_path)
                else:
                    tester = Tester(self._trainer._training_env, phase, self._logger, self._root_path,
                                    n_jobs=self._n_jobs, use_estimator_mismatch=Q_values_diff_threshold is not None)
                    tester.test(fqi_iterations, self._seeds, Q_values_diff_threshold=Q_values_diff_threshold, filter_method=filter_method, unroll_fqi_iterations=self._unroll_fqi_iterations, quantile=quantile)
                    if self._unroll_fqi_iterations:
                        Plotter.plot_percentage_returns(phase, fqi_iterations, self._seeds, self._root_path)
                        Plotter.plot_percentage_pl(phase, fqi_iterations, self._seeds, self._root_path,
                                                   self._trainer._training_env._persistence, True)
                        Plotter.plot_seeds_percentage(phase, fqi_iterations, self._seeds, self._root_path,
                                                   self._trainer._training_env._persistence, False)
                        Plotter.plot_best_q_values(phase, fqi_iterations, self._seeds, self._root_path)
                # Plot results on Validation set
                self._logger.info('Saving validation plots for the new best trial.')
                for i in range(fqi_iterations):
                    iteration = i + 1
                    if self._unroll_fqi_iterations or iteration == fqi_iterations:
                        for seed in self._seeds:
                            save_path = os.path.join(self._root_path, f'seed{seed}')
                            df = pd.read_csv(os.path.join(save_path, f'Results_iter{iteration}_Validation.csv'))
                            if self._optimize_trajectory_number:
                                if self._optimize_trajectory_window:
                                    #df = df[df['trajectory'] == optimal_params['trajectory_number']]
                                    df = df[(df['trajectory'] >= (
                                                optimal_params['trajectory_number'] - math.floor(self._trajectory_window_size / 2))) &
                                                      (df['trajectory'] <= (optimal_params['trajectory_number'] + math.floor(
                                                          self._trajectory_window_size / 2)))]
                                    Plotter.plot_actions(df, list(range(optimal_params['trajectory_number'] - math.floor(self._trajectory_window_size / 2), optimal_params['trajectory_number'] + 1 + math.floor(
                                                          self._trajectory_window_size / 2))), 'Validation', iteration, save_path)
                                    Plotter.plot_actions_weekly(df, list(range(optimal_params['trajectory_number'] - math.floor(self._trajectory_window_size / 2), optimal_params['trajectory_number'] + 1 + math.floor(
                                                          self._trajectory_window_size / 2))), 'Validation', iteration, save_path)
                                else:
                                    df = df[df['trajectory'] == optimal_params['trajectory_number']]
                                    Plotter.plot_actions(df, [trajectory_number], 'Validation', iteration, save_path)
                                    Plotter.plot_actions_weekly(df, [trajectory_number], 'Validation', iteration, save_path)

                            else:
                                persistence = self._trainer._training_env.persistence
                                Plotter.plot_actions(df, list(range(0, persistence, 10)), 'Validation', iteration, save_path)
                                Plotter.plot_actions_weekly(df, list(range(0, persistence, 10)), 'Validation', iteration, save_path)
                            with open(os.path.join(save_path, f'Policy_iter{iteration}.pkl'), 'rb') as load_file:
                                policy = pickle.load(load_file)
                            if policy.Q.double_Q:
                                regressors = [policy.Q._regressors1, policy.Q._regressors2]
                            else:
                                regressors = [policy.Q._regressors]
                            """
                            Plotter.plot_feature_importances(
                                regressors, self._trainer._training_env.current_state_features, iteration, save_path
                            )
                            """
                self._logger.info('Plots saved.')

                if self._test_best_model_so_far:
                    if self._test_on_next_year_trainer is None:
                        if self._retrain_when_testing: #create trainer
                            self._logger.info(f"Evaluating Algorithm up to {self._test_up_to_year} with retrain")
                            assert len(self._trainer._testing_env._dataset_builder.years) == 1
                            current_validation_year = self._trainer._testing_env._dataset_builder.years[0]
                            for i in range(self._test_up_to_year - current_validation_year):
                                train_years_current = [current_validation_year + i - 1, current_validation_year + i]
                                test_years_current = [current_validation_year + i + 1]
                                self._logger.info(f'Test phase: training on {train_years_current} and testing on {test_years_current}')
                                training_dataset_builder = FQIDatasetBuilderFactory.create(self._trainer._training_env._dataset_builder.asset_name, train_years_current, **self._features_parameters)
                                testing_dataset_builder = FQIDatasetBuilderFactory.create(self._trainer._training_env._dataset_builder.asset_name, test_years_current, **self._features_parameters)
                                trainer = Trainer(
                                    TradingEnv(training_dataset_builder), TradingEnv(testing_dataset_builder),
                                    self._trainer._regressor_type, self._trainer._double_Q, self._trainer._swap_Q, self._trainer._logger, save_path=os.path.join(self._root_path, "test_next_year_retrain", f'{current_validation_year + i + 1}'), save_plots=self._trainer._save_plots,
                                    double_Q_strategy=self._trainer._double_Q_strategy,
                                    n_jobs=self._trainer._n_jobs, shuffle_days=self._trainer.shuffle_days, cvar_param=self._trainer._cvar_param,
                                    use_estimator_mismatch=self._trainer._use_estimator_mismatch, trainer_phase='Test'
                                )
                                if self._optimize_trajectory_number:
                                    regressor_params = optimal_params.copy()
                                    t_n = regressor_params['trajectory_number']
                                    it_n = regressor_params['iterations']
                                    if 'filter_method' in regressor_params:
                                        filter_method = regressor_params['filter_method']
                                        Q_values_diff_threshold = regressor_params['Q_values_diff_threshold']
                                        del regressor_params['filter_method']
                                        del regressor_params['Q_values_diff_threshold']
                                    else:
                                        if self._optimize_Q_value_diff_threshold is False and self._Q_value_diff_threshold is not None:
                                            Q_values_diff_threshold = self._Q_value_diff_threshold
                                            filter_method = self._filter_method
                                        else:
                                            filter_method = None
                                            Q_values_diff_threshold = None
                                    del regressor_params['trajectory_number']
                                    del regressor_params['iterations']
                                    if self._optimize_trajectory_window:
                                        if self._optimize_cvar:
                                            gains = trainer.train(it_n, regressor_params,
                                                                                          self._trainer.get_seeds_cvar(),
                                                                                          trajectory_number=t_n,
                                                                                          trajectory_window=
                                                                                          self._fixed_params[
                                                                                              'trajectory_window'],
                                                                                          Q_values_diff_threshold=Q_values_diff_threshold,
                                                                                          filter_method=filter_method
                                                                                            )
                                        else:
                                            gains = trainer.train(it_n, regressor_params,
                                                                                          self._seeds,
                                                                                          trajectory_number=t_n,
                                                                                          trajectory_window=
                                                                                          self._fixed_params[
                                                                                              'trajectory_window'],
                                                                                          Q_values_diff_threshold=Q_values_diff_threshold,
                                                                                          filter_method=filter_method)
                                    else:
                                        if self._optimize_cvar:
                                            gains = trainer.train(it_n, regressor_params,
                                                                                          self._trainer.get_seeds_cvar(),
                                                                                          trajectory_number=t_n,
                                                                                          Q_values_diff_threshold=Q_values_diff_threshold,
                                                                                          filter_method=filter_method)
                                        else:
                                            gains = trainer.train(it_n, regressor_params,
                                                                                          self._seeds,
                                                                                          trajectory_number=t_n,
                                                                                          Q_values_diff_threshold=Q_values_diff_threshold,
                                                                                          filter_method=filter_method)
                                else:
                                    regressor_params = optimal_params.copy()
                                    del regressor_params['iterations']
                                    if 'filter_method' in regressor_params:
                                        filter_method = regressor_params['filter_method']
                                        Q_values_diff_threshold = regressor_params['Q_values_diff_threshold']
                                        del regressor_params['filter_method']
                                        del regressor_params['Q_values_diff_threshold']
                                    else:
                                        if self._optimize_Q_value_diff_threshold is False and self._Q_value_diff_threshold is not None:
                                            Q_values_diff_threshold = self._Q_value_diff_threshold
                                            filter_method = self._filter_method
                                        else:
                                            filter_method = None
                                            Q_values_diff_threshold = None
                                    if self._optimize_cvar:
                                        gains = trainer.train(fqi_iterations, regressor_params,
                                                                                      self._trainer.get_seeds_cvar(), Q_values_diff_threshold=Q_values_diff_threshold, filter_method=filter_method)
                                    else:
                                        gains = trainer.train(fqi_iterations, regressor_params,
                                                                                      self._seeds, Q_values_diff_threshold=Q_values_diff_threshold, filter_method=filter_method)


                        else: #only use the policy on the next year by means of Tester class
                            regressor_params = optimal_params.copy()
                            if 'filter_method' in regressor_params:
                                filter_method = regressor_params['filter_method']
                                Q_values_diff_threshold = regressor_params['Q_values_diff_threshold']
                                del regressor_params['filter_method']
                                del regressor_params['Q_values_diff_threshold']
                            else:
                                if self._optimize_Q_value_diff_threshold is False and self._Q_value_diff_threshold is not None:
                                    Q_values_diff_threshold = self._Q_value_diff_threshold
                                    filter_method = self._filter_method
                                else:
                                    filter_method = None
                                    Q_values_diff_threshold = None
                            self._logger.info(f"Evaluating Algorithm up to {self._test_up_to_year} without retrain")
                            """
                            #assert len(self._trainer._testing_env._dataset_builder.years) == 1
                            #current_validation_year = self._trainer._testing_env._dataset_builder.years[0]
                            for i in range(self._test_up_to_year - current_validation_year):
                                #test_years_current = [current_validation_year + i + 1]
                                #testing_dataset_builder = FQIDatasetBuilderFactory.create(self._trainer._training_env._dataset_builder.asset_name, test_years_current,
                                #                                                          **self._features_parameters)
                                tester = Tester(TradingEnv(testing_dataset_builder), "Test", self._logger, read_path=self._root_path, save_path=os.path.join(self._root_path, "test_next_year_no_retrain", f'{current_validation_year + i + 1}'),
                                                n_jobs=self._n_jobs, use_estimator_mismatch=Q_values_diff_threshold is not None)
                                tester.test(fqi_iterations, self._seeds, Q_values_diff_threshold=Q_values_diff_threshold, filter_method=filter_method)
                                Plotter.plot_percentage_returns("Test", fqi_iterations, self._seeds, os.path.join(self._root_path, "test_next_year_no_retrain", f'{current_validation_year + i + 1}'))
                                Plotter.plot_percentage_pl("Test", fqi_iterations, self._seeds, os.path.join(self._root_path, "test_next_year_no_retrain", f'{current_validation_year + i + 1}'),
                                                           self._trainer._training_env.persistence, True)
                                Plotter.plot_seeds_percentage("Test", fqi_iterations, self._seeds,
                                                           os.path.join(self._root_path, "test_next_year_no_retrain",
                                                                        f'{current_validation_year + i + 1}'),
                                                           self._trainer._training_env.persistence, False)
                                Plotter.plot_best_q_values("Test", fqi_iterations, self._seeds, os.path.join(self._root_path, "test_next_year_no_retrain", f'{current_validation_year + i + 1}'))
                            """

                            tester = Tester(self._trainer._testing_env, "Test", self._logger,
                                            read_path=self._root_path,
                                            save_path=os.path.join(self._root_path, "test_no_retrain"),
                                            n_jobs=self._n_jobs,
                                            use_estimator_mismatch=Q_values_diff_threshold is not None)
                            tester.test(fqi_iterations, self._seeds,
                                        Q_values_diff_threshold=Q_values_diff_threshold,
                                        filter_method=filter_method, unroll_fqi_iterations=self._unroll_fqi_iterations, quantile=quantile)
                            if self._unroll_fqi_iterations:
                                Plotter.plot_percentage_returns("Test", fqi_iterations, self._seeds,
                                                                os.path.join(self._root_path,
                                                                             "test_no_retrain"))
                                Plotter.plot_percentage_pl("Test", fqi_iterations, self._seeds,
                                                           os.path.join(self._root_path, "test_no_retrain"),
                                                           self._trainer._training_env.persistence, True)
                                Plotter.plot_seeds_percentage("Test", fqi_iterations, self._seeds,
                                                              os.path.join(self._root_path, "test_no_retrain"),
                                                              self._trainer._training_env.persistence, False)
                                Plotter.plot_best_q_values("Test", fqi_iterations, self._seeds,
                                                           os.path.join(self._root_path, "test_no_retrain"))


                    else: #evaluate using the trainer passed
                        self._logger.info("Evaluating Algorithm on Test set using the Trainer")
                        #Check performance reached on a real test set.
                        if self._optimize_trajectory_number:
                            regressor_params = optimal_params.copy()
                            t_n = regressor_params['trajectory_number']
                            it_n = regressor_params['iterations']
                            if 'filter_method' in regressor_params:
                                filter_method = regressor_params['filter_method']
                                Q_values_diff_threshold = regressor_params['Q_values_diff_threshold']
                                del regressor_params['filter_method']
                                del regressor_params['Q_values_diff_threshold']
                            else:
                                filter_method = None
                                Q_values_diff_threshold = None
                            del regressor_params['trajectory_number']
                            del regressor_params['iterations']
                            if self._optimize_trajectory_window:
                                if self._optimize_cvar:
                                    gains = self._test_on_next_year_trainer.train(it_n, regressor_params, self._trainer.get_seeds_cvar(),
                                                                                  trajectory_number=t_n,
                                                                                  trajectory_window=self._fixed_params[
                                                                                      'trajectory_window'],
                                                                                  filter_method=filter_method,
                                                                                  Q_values_diff_threshold=Q_values_diff_threshold)
                                else:
                                    gains = self._test_on_next_year_trainer.train(it_n, regressor_params,
                                                                                  self._seeds,
                                                                                  trajectory_number=t_n,
                                                                                  trajectory_window=self._fixed_params[
                                                                                      'trajectory_window'],
                                                                                  filter_method=filter_method,
                                                                                  Q_values_diff_threshold=Q_values_diff_threshold
                                                                                  )
                            else:
                                if self._optimize_cvar:
                                    gains = self._test_on_next_year_trainer.train(it_n, regressor_params, self._trainer.get_seeds_cvar(),
                                                                                  trajectory_number=t_n, filter_method=filter_method,
                                                                                  Q_values_diff_threshold=Q_values_diff_threshold)
                                else:
                                    gains = self._test_on_next_year_trainer.train(it_n, regressor_params, self._seeds,
                                                                                  trajectory_number=t_n,  filter_method=filter_method,
                                                                                  Q_values_diff_threshold=Q_values_diff_threshold)
                        else:
                            regressor_params = optimal_params.copy()
                            del regressor_params['iterations']
                            if 'filter_method' in regressor_params:
                                filter_method = regressor_params['filter_method']
                                Q_values_diff_threshold = regressor_params['Q_values_diff_threshold']
                                del regressor_params['filter_method']
                                del regressor_params['Q_values_diff_threshold']
                            else:
                                filter_method = None
                                Q_values_diff_threshold = None
                            if self._optimize_cvar:
                                gains = self._test_on_next_year_trainer.train(fqi_iterations, regressor_params, self._trainer.get_seeds_cvar(), filter_method=filter_method,
                                                                                  Q_values_diff_threshold=Q_values_diff_threshold)
                            else:
                                gains = self._test_on_next_year_trainer.train(fqi_iterations, regressor_params, self._seeds, filter_method=filter_method,
                                                                                  Q_values_diff_threshold=Q_values_diff_threshold)


                        # Plot performances on Validation and Test to see how it goes over time

                        self._validation_test_performances['trial_number'].append(study.best_trial.number)
                        self._validation_test_performances["optimization_value"].append(study.best_trial.value)
                        self._validation_test_performances['test_value'].append(gains[-1])

                        if len(self._validation_test_performances['trial_number']) > 0:
                            plt.figure()
                            plt.title("Performance Comparison Validation/Test")
                            plt.plot(range(len(self._validation_test_performances['optimization_value'])), self._validation_test_performances['optimization_value'],
                                     label="Validation Performances")
                            plt.plot(range(len(self._validation_test_performances['test_value'])), self._validation_test_performances['test_value'], label="Test Performances")
                            plt.xlabel("Iterations (Only best iterations are reported)")
                            plt.ylabel("P&L (% of max allocation)")
                            plt.legend()
                            plt.savefig(f'{self._root_path}/PerformanceHistory.png')
                            plt.close()
                            json.dump(self._validation_test_performances, open(f"{self._root_path}/validation_test_performances.json", "w"))







    def _save_plots(self):
        # Plot optimization results
        fig = optuna.visualization.plot_optimization_history(self.study)
        fig.write_image(f'{self._root_path}/ParamsOptHistory.png')
        try:
            fig = optuna.visualization.plot_param_importances(self.study)
            fig.write_image(f'{self._root_path}/ParamsImportance.png')
        except Exception as e:
            print(e)
        fig = optuna.visualization.plot_contour(self.study)
        fig.write_image(f'{self._root_path}/ParamsContour.png', width=3000, height=1750)
        fig = optuna.visualization.plot_slice(self.study)
        fig.write_image(f'{self._root_path}/ParamsSlice.png')
        fig = optuna.visualization.plot_timeline(self.study)
        fig.write_image(f'{self._root_path}/Timeline.png')


        #Plot performances on Validation and Test to see how it goes over time
        #if self._test_best_model_so_far:
        #    trials = self.study.trials
        #    data = {"trial_number":[], "optimization_value": [], "test_value" : []}

        #    for trial in trials:
        #        if trial.state == optuna.trial.TrialState.COMPLETE and "test_performances" in trial.user_attrs.keys():
        #            data['trial_number'].append(trial.number)
        #            data["optimization_value"].append(trial.value)
        #            data['test_value'].append(np.mean(trial.user_attributes['test_performances']))

        #    if len(data['trial_number']) > 0:
        #        plt.title("Performance Comparison Validation/Test")
        #        plt.plot(range(len(data['optimization_value'])), data['optimization_value'], label="Validation Performances")
        #        plt.plot(range(len(data['test_value'])), data['test_value'], label="Test Performances")
        #        plt.xlabel("Iterations (Only best iterations are reported)")
        #        plt.ylabel("P&L (% of max allocation)")
        #        plt.savefig(f'{self._root_path}/PerformanceHistory_old.png')



