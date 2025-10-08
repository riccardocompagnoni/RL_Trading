import datetime
import logging.handlers
import multiprocessing
import numpy as np
import numpy.typing as npt
import pandas as pd
import sys
import math
import optuna
import os
import pickle
import random
import typing
import datetime
import xgboost
from joblib import Parallel, delayed, parallel_backend
from sklearn.ensemble import ExtraTreesRegressor
from scipy.stats import bootstrap

from RL_Trading.prj.app.core.fqi.entities.trading_env import TradingEnv
from RL_Trading.prj.app.core.fqi.trlib.algorithms.reinforcement.fqi import FQI
from RL_Trading.prj.app.core.fqi.trlib.policies.qfunction import ZeroQ
from RL_Trading.prj.app.core.fqi.trlib.policies.valuebased import EpsilonGreedy
from RL_Trading.prj.app.core.fqi.services.plotter import Plotter
from RL_Trading.prj.app.core.models.services.utils import get_logger



class Trainer(object):

    def __init__(
            self,
            training_env,
            testing_env,
            regressor_type: str,
            double_Q: bool,
            swap_Q: bool,
            logger: logging.Logger,
            save_path: str,
            save_plots: bool,
            double_Q_strategy: str = 'mean',
            shuffle: typing.Optional[str] = None,
            plot_loss: typing.Optional[bool] = False,
            Q_values_quantile: typing.Optional[float] = None,
            trainer_phase: str = 'train',
            n_jobs: int = 1,
            cvar_param: typing.Optional[float] = None,
            use_estimator_mismatch: bool = False,

    ):
        self._logger = logger
        self._save_path = save_path
        self._save_plots = save_plots
        self._gains = None
        self._seeds = None
        self._plot_loss = plot_loss
        self._cvar_param = cvar_param
        self._use_estimator_mismatch = use_estimator_mismatch
        # Set environments
        self._training_env = training_env
        self._training_df = self._training_env.build_fqi_dataset()
        self._days_list = self._training_df["time"].dt.date.unique()
        self._week_list = (self._training_df["time"].dt.isocalendar()['year'].astype(str) + "_" +self._training_df["time"].dt.isocalendar()['week'].astype(str)).unique()
        self._testing_env = testing_env
        self._pi = EpsilonGreedy(self._training_env.get_actions(), ZeroQ(), epsilon=0)
        self._double_Q = double_Q
        self._shuffle = shuffle
        self._swap_Q = swap_Q
        self._seeds_cvar = []
        self._Q_values_quantile = Q_values_quantile
        self._double_Q_strategy = double_Q_strategy
        self._trainer_phase = trainer_phase
        self._n_jobs = n_jobs
        self._regressor_type = regressor_type
        # Set regressor
        if regressor_type == 'xgb':
            self._regressor = xgboost.Booster#.XGBRegressor
        elif regressor_type == 'extra':
            self._regressor = ExtraTreesRegressor
        else:
            raise AttributeError(f"Invalid regressor {regressor_type}.")


    def train(
            self,
            fqi_iterations: int,
            regressor_params: typing.Dict,
            seeds: typing.List[int],
            pruning: typing.Optional[bool] = False,
            pruning_min_values: typing.Optional[npt.NDArray[float]] = None,
            trajectory_number: typing.Optional[int] = None,
            save_iteration_data: typing.Optional[bool] = False,
            trajectory_window: typing.Optional[int] = None,
            saving_day_with_values_over_percentile: typing.Optional[bool] = None,
            removing_detected_days: typing.Optional[bool] = None,
            Q_values_diff_threshold: typing.Optional[float] = None,
            filter_method: typing.Optional[str] = None,
            unroll_fqi_iterations = True,
            quantile=None
    ) -> typing.List[float]:


        if self._n_jobs == -1 or self._n_jobs > len(seeds):
            actual_n_jobs = len(seeds)
        else:
            actual_n_jobs = self._n_jobs

        with parallel_backend('threading'):
            gains = Parallel(n_jobs = actual_n_jobs, backend='loky', verbose=10)(delayed(self._train_with_seed)(fqi_iterations, regressor_params, seed, pruning, pruning_min_values,
                                       trajectory_number, save_iteration_data, trajectory_window=trajectory_window,
                                       removing_detected_days=removing_detected_days, Q_values_diff_threshold = Q_values_diff_threshold, filter_method = filter_method, unroll_fqi_iterations=unroll_fqi_iterations) for seed in seeds)

        if self._cvar_param is not None:
            self._gains = gains
            self._seeds = seeds

            combined = list(zip(gains, seeds))

            # Sort the combined list by the first array's values
            combined.sort(key=lambda x: x[0], reverse = True)

            # Separate the sorted tuples back into two arrays
            gains, seeds = zip(*combined)

            cvar_values = math.floor(len(seeds) * self._cvar_param)
            self._seeds_cvar = list(seeds[:cvar_values])
        #if removing_detected_days:
        #    phase = f"Training_removing_percentile_{self._Q_values_quantile}"
        #else:
        #    phase = "Training"

        #Plotter.plot_percentage_returns(phase, fqi_iterations, seeds, self._save_path)
        if removing_detected_days is not None:
            if self._trainer_phase == 'train':
                phase = f"Validation_removing_percentile_{self._Q_values_quantile}"
            else:
                phase = f"Test_removing_percentile_{self._Q_values_quantile}"

        else:
            if self._trainer_phase == 'train':
                phase = "Validation"
            else:
                phase = "Test"
        if self._cvar_param is not None:
            Plotter.plot_percentage_returns(phase, fqi_iterations, seeds[:cvar_values], self._save_path)
            Plotter.plot_percentage_pl(phase, fqi_iterations, seeds[:cvar_values], self._save_path,
                                       self._testing_env._persistence, True)
            Plotter.plot_seeds_percentage(phase, fqi_iterations, seeds[:cvar_values], self._save_path,
                                       self._testing_env._persistence, False)

            # Plotter.plot_costs_pnl(phase, fqi_iterations, seeds, self._save_path)
            Plotter.plot_best_q_values(phase, fqi_iterations, seeds[:cvar_values], self._save_path)
            for trajectory in range(0, self._training_env._persistence, 10):
                Plotter.plot_percentage_returns_single_trajectory(phase, fqi_iterations, seeds[:cvar_values], self._save_path,
                                                                  trajectory)

        else:
            if unroll_fqi_iterations:
                Plotter.plot_percentage_returns(phase, fqi_iterations, seeds, self._save_path)
                Plotter.plot_percentage_pl(phase, fqi_iterations, seeds, self._save_path,
                                           self._testing_env._persistence, True)
                Plotter.plot_seeds_percentage(phase, fqi_iterations, seeds, self._save_path,
                                           self._testing_env._persistence, False)

                if trajectory_number == None: #Optimizing over 6 trajectories list(range(0, persistence, 10))
                    for trajectory in range(0, self._training_env._persistence, 10):
                        Plotter.plot_percentage_returns_single_trajectory(phase, fqi_iterations, seeds, self._save_path, trajectory_number=trajectory)
                #Plotter.plot_costs_pnl(phase, fqi_iterations, seeds, self._save_path)
                Plotter.plot_best_q_values(phase, fqi_iterations, seeds, self._save_path)
            else:
                Plotter.plot_seeds_percentage(phase, 1, seeds, self._save_path,
                                              self._testing_env._persistence, False, labels=[str(fqi_iterations)])

        #if trajectory_number is not None:
        #    Plotter.plot_q_values(phase, fqi_iterations, trajectory_number, seeds,  self._save_path)
        if self._cvar_param is not None:
            if saving_day_with_values_over_percentile is not None and saving_day_with_values_over_percentile:
                self.save_days_to_remove(seeds[:cvar_values], save_iteration_data, self._Q_values_quantile)
        else:
            if saving_day_with_values_over_percentile is not None and saving_day_with_values_over_percentile:
                self.save_days_to_remove(seeds, save_iteration_data, self._Q_values_quantile)

        gains = gains if self._cvar_param is None else gains[:cvar_values]

        if quantile is not None:

            q = np.atleast_1d(quantile).astype(float)
            n_res = 10_000

            res = bootstrap(
                (gains,),
                statistic=lambda d, axis: np.quantile(d, q, axis=axis),
                axis=0,
                n_resamples=n_res,
                method="basic",
                random_state=np.random.default_rng(),
            )
            boot = res.bootstrap_distribution

            return np.median(boot, axis=2)[0].tolist()

        return np.mean(gains, axis=0).tolist()


    def get_seeds_cvar(self):
        return self._seeds_cvar

    @staticmethod
    def save_days_to_remove(seeds, save_path, percentile):
        days = []
        for seed in seeds:
            _df = pd.read_csv(os.path.join(f'seed{seed}/Results_iter1_Training.csv'))
            days.append(_df[_df['Q'] >= _df['Q'].quantile(q=percentile)].day.unique().tolist())
        days = list([str(d) for dset in days for d in dset])
        days = pd.DataFrame(days).drop_duplicates()
        days = days.sort_values(0).reset_index(drop=True)
        days[0] = pd.to_datetime(days[0])
        days.to_csv(f"days_to_remove_{percentile}_train.csv", index=False)
    @staticmethod
    def get_percentile_threshold(seeds, save_path, iterations ,percentile):
        thresholds = {}
        for seed in seeds:
            thresholds[seed] = {}
            for iteration in range(iterations):
                _df = pd.read_csv(os.path.join(f'{save_path}/seed{seed}/Results_iter{iteration+1}_Training.csv'))
                thresholds[seed][iteration+1] =_df['Q'].quantile(q=percentile)
        return thresholds
    @staticmethod
    def save_days_to_remove_test(seeds, save_path, percentile):
        days = []
        for seed in seeds:
            _df = pd.read_csv(os.path.join(f'seed{seed}/Results_iter1_Validation.csv'))
            days.append(_df[_df['Q'] >= _df['Q'].quantile(q=percentile)].day.unique().tolist())
        days = list([str(d) for dset in days for d in dset])
        days = pd.DataFrame(days).drop_duplicates()
        days = days.sort_values(0).reset_index(drop=True)
        days[0] = pd.to_datetime(days[0])
        days.to_csv(f"days_to_remove_{percentile}_test.csv", index=False)
    def _train_with_seed(
            self,
            fqi_iterations: int,
            regressor_params: typing.Dict,
            seed: int,
            pruning: bool,
            pruning_min_values: typing.Union[npt.NDArray[float], None],
            trajectory_number: typing.Optional[int] = None,
            save_iteration_data: typing.Optional[bool] = False,
            trajectory_window: typing.Optional[int] = None,
            removing_detected_days: typing.Optional[bool] = False,
            Q_values_diff_threshold: typing.Optional[float] = None,
            filter_method: typing.Optional[str] = None,
            unroll_fqi_iterations = False
    ) -> typing.List[float]:

        if self._n_jobs > 1 or self._n_jobs == -1: #workaround to get the logging work in parallel execution
            logger_ = get_logger(logging.INFO)
        else:
            logger_ = self._logger

        save_path = os.path.join(self._save_path, f'seed{seed}')
        os.makedirs(save_path, exist_ok=True)
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        regressor_params['random_state'] = seed
        algorithm = FQI(
            mdp=self._training_env,
            policy=self._pi,
            actions=self._training_env.get_actions(),
            batch_size=1,
            max_iterations=fqi_iterations,
            double_Q=self._double_Q,
            swap_Q=self._swap_Q,
            double_Q_strategy = self._double_Q_strategy,
            regressor_type=self._regressor,
            **regressor_params
        )
        logger_.info(f"Algorithm set (seed = {seed}).")
        # ======== FQI iterations ========
        gains = []
        if self._shuffle is not None and self._double_Q:  # only select days once then same for each iteration
            if self._shuffle == 'day':
                days_mask = np.ones(self._days_list.shape[0]).astype(bool)
                days_mask[: self._days_list.shape[0] // 2] = 0
                np.random.shuffle(days_mask)
                days_first_estimator = self._days_list[days_mask]
                mask = np.ones(self._training_df.shape[0]).astype(bool)
                temp_df = self._training_df.copy()
                temp_df['row'] = np.arange(len(temp_df))
                temp_df['date'] = temp_df['time'].dt.date
                df_sampled = temp_df.groupby('date').filter(lambda x: x["date"].values[0] in days_first_estimator)
                mask[df_sampled['row']] = 0
            elif self._shuffle == 'week':
                week_mask = np.ones(self._week_list.shape[0]).astype(bool)
                week_mask[: self._week_list.shape[0] // 2] = 0
                np.random.shuffle(week_mask)
                weeks_first_estimator = self._week_list[week_mask]
                mask = np.ones(self._training_df.shape[0]).astype(bool)
                temp_df = self._training_df.copy()
                temp_df['row'] = np.arange(len(temp_df))
                temp_df['week'] = temp_df["time"].dt.isocalendar()['year'].astype(str) + "_" + temp_df["time"].dt.isocalendar()['week'].astype(str)
                df_sampled = temp_df.groupby('week').filter(lambda x: x["week"].values[0] in weeks_first_estimator)
                mask[df_sampled['row']] = 0
            else:
                if self._shuffle != 'minute':
                    raise ValueError(f"The shuffle value {self._shuffle} is not a valid input")



        for i in range(fqi_iterations):
            iteration = i + 1
            logger_.info(f"ITERATION {iteration}: Algorithm started.")
            fit_params = {}
            if self._double_Q: #Random shuffle the traning set and split between the two set of estimators
                if self._shuffle == 'minute':
                    mask = np.ones(self._training_df.shape[0]).astype(bool)
                    mask[: self._training_df.shape[0]//2] = 0
                    np.random.shuffle(mask)
                fit_params['double_mask'] = mask

            # Training
            if self._plot_loss:
                # Training
                regressors1_loss, regressors2_loss = algorithm._iter(
                    self._training_df[self._training_env.current_state_features + ['action']].to_numpy(dtype=np.float32),
                    self._training_df['reward'].to_numpy(dtype=np.float32),
                    self._training_df[self._training_env.next_state_features + ['action']].to_numpy(dtype=np.float32),
                    self._training_df['absorbing_state'].to_numpy(dtype=bool),
                    self._plot_loss,
                    **fit_params
                )
            else:
                algorithm._iter(
                    self._training_df[self._training_env.current_state_features + ['action']].to_numpy(dtype=np.float32),
                    self._training_df['reward'].to_numpy(dtype=np.float32),
                    self._training_df[self._training_env.next_state_features + ['action']].to_numpy(dtype=np.float32),
                    self._training_df['absorbing_state'].to_numpy(dtype=bool),
                    self._plot_loss,
                    **fit_params
                )
            logger_.info(f"ITERATION {iteration}: Algorithm trained.")
            if self._plot_loss:
                if self._double_Q:
                    sa = self._training_df[['time', 'action']]
                    loss_combine = {}
                    for a in self._training_env.get_actions():
                        sa1 = sa[(sa['action'] == a) & (mask)].copy()
                        sa2 = sa[(sa['action'] == a) & (mask == False)].copy()
                        sa1.loc[:, 'loss'] = regressors1_loss[a]
                        sa2.loc[:, 'loss'] = regressors2_loss[a]
                        sa_c = pd.concat([sa1, sa2])
                        sa_c = sa_c.sort_values('time')
                        dates = sa_c['time']
                        loss_combine[a] = sa_c['loss']
                    # NotImplemented("Not available plot of the loss for Double FQI")
                    Plotter.plot_loss(loss_combine, dates, iteration, save_path)
                else:
                    Plotter.plot_loss(regressors1_loss,
                                      self._training_df[['time', 'allocation']].drop_duplicates()['time'],
                                      iteration, save_path)
            if removing_detected_days is not None and removing_detected_days:
                phase = f"Training_removing_percentile_{self._Q_values_quantile}"

                gain = self._training_env.test(
                    policy=algorithm._policy,
                    save_csv=True,
                    save_plots=self._save_plots,
                    save_root=save_path,
                    phase=phase,
                    iteration=iteration,
                    trajectory_number=trajectory_number,
                    #save_iteration_data=save_iteration_data,
                    trajectory_window=trajectory_window,
                    use_estimator_mismatch=self._use_estimator_mismatch,
                    Q_values_diff_threshold=Q_values_diff_threshold,
                    filter_method=filter_method
                )
                logger_.info(f"ITERATION {iteration}: Algorithm tested on training set (gain = {gain:,.2f}%).")

            #else:
            #    phase = "Training"
            #    # Testing on training
            #    gain = self._training_env.test(
            #        policy=algorithm._policy,
            #        save_csv=True,
            #        save_plots=self._save_plots,
            #        save_root=save_path,
            #        phase=phase,
            #        iteration=iteration,
            #        trajectory_number=trajectory_number,
            #        save_iteration_data=save_iteration_data,
            #        trajectory_window=trajectory_window
            #    )
            #    self._logger.info(f"ITERATION {iteration}: Algorithm tested on training set (gain = {gain:,.2f}%).")
            if algorithm._policy.Q.double_Q:
                regressors = [algorithm._policy.Q._regressors1, algorithm._policy.Q._regressors2]
            else:
                regressors = [algorithm._policy.Q._regressors]
            """
            Plotter.plot_feature_importances(
                regressors, self._training_env.current_state_features, iteration, save_path
            )
            Plotter.plot_feature_importances(
                regressors, self._training_env.current_state_features, iteration, save_path
            )
            """

            if removing_detected_days is not None and removing_detected_days:
                if self._trainer_phase == 'test':
                    phase = f"Test_removing_percentile_{self._Q_values_quantile}"
                else:
                    phase = f"Validation_removing_percentile_{self._Q_values_quantile}"
            else:
                if self._trainer_phase == 'test' or self._trainer_phase == 'Test':
                    phase = "Test"
                else:
                    phase = "Validation"


            if algorithm._policy.Q.double_Q:
                regressors = [algorithm._policy.Q._regressors1, algorithm._policy.Q._regressors2]
            else:
                regressors = [algorithm._policy.Q._regressors]

            """
            Plotter.plot_feature_importances(
                regressors, self._training_env.current_state_features, iteration, save_path
            )
            """

            if unroll_fqi_iterations or iteration==fqi_iterations:
                # Testing
                gain = self._testing_env.test(
                    policy=algorithm._policy,
                    save_csv=True,
                    save_plots=self._save_plots,
                    save_root=save_path,
                    phase=phase,
                    iteration=iteration,
                    trajectory_number=trajectory_number,
                    #save_iteration_data=save_iteration_data,
                    trajectory_window=trajectory_window,
                    use_estimator_mismatch=self._use_estimator_mismatch,
                    Q_values_diff_threshold= Q_values_diff_threshold,
                    filter_method =filter_method
                )

                if self._trainer_phase == 'test':
                    logger_.info(f"ITERATION {iteration}: Algorithm tested on test set (gain = {gain:,.2f}%).")
                else:
                     logger_.info(f"ITERATION {iteration}: Algorithm tested on validation set (gain = {gain:,.2f}%).")
                # Prune
                if pruning and gain < pruning_min_values[i]:
                    raise optuna.TrialPruned()
                gains.append(gain)
                # Saving

                model_name = f'Policy_iter{iteration}.pkl'
                with open(os.path.join(save_path, model_name), 'wb+') as f:
                    pickle.dump(algorithm._policy, f)
                logger_.info(f"ITERATION {iteration}: Algorithm saved.")

        return gains

    @property
    def save_path(self) -> str:
        return self._save_path

    @save_path.setter
    def save_path(self, value: str):
        self._save_path = value



