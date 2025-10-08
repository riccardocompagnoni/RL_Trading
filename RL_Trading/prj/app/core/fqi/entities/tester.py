import datetime
import logging.handlers
from typing import Any

import numpy as np
import os
import multiprocessing
import pickle
from joblib import Parallel, delayed
import typing

from scipy.stats import bootstrap

from RL_Trading.prj.app.core.fqi.entities.trading_env import TradingEnv
from RL_Trading.prj.app.core.fqi.services.plotter import Plotter
from RL_Trading.prj.app.core.models.services.utils import get_logger


class Tester(object):

    def __init__(self, testing_env: TradingEnv, phase: str, logger: logging.Logger, save_path:str, n_jobs:int = 1, read_path: str = None, use_estimator_mismatch: bool = False):
        self._testing_env = testing_env
        self._logger = logger
        self._save_path = save_path
        self._phase = phase
        if read_path is None:
            self._save_path = save_path
            self._read_path = save_path
        else:
            self._save_path = save_path
            self._read_path = read_path
        self._n_jobs = n_jobs
        self._use_estimator_mismatch = use_estimator_mismatch

    def test(self, fqi_iterations: int, seeds: typing.List[int],  trajectory_number: typing.Optional[int] = None, trajectory_window: typing.Optional[int] = None, Q_values_diff_threshold: typing.Optional[float] = None, filter_method: typing.Optional[str] = None, unroll_fqi_iterations=False, quantile=None
            ) -> tuple[Any, Any]:

        if self._n_jobs == -1 or self._n_jobs > len(seeds):
            actual_n_jobs = len(seeds)
        else:
            actual_n_jobs = self._n_jobs


        gains = Parallel(n_jobs = actual_n_jobs, backend='loky', verbose=10)(delayed(self._test_with_seed)(fqi_iterations, seed, trajectory_number, trajectory_window, Q_values_diff_threshold, filter_method, unroll_fqi_iterations) for seed in seeds)

        if unroll_fqi_iterations:
            Plotter.plot_percentage_returns(self._phase, fqi_iterations, seeds, self._save_path)
            Plotter.plot_percentage_pl(self._phase, fqi_iterations, seeds, self._save_path, self._testing_env._persistence, True)
            Plotter.plot_seeds_percentage(self._phase, fqi_iterations, seeds, self._save_path, self._testing_env._persistence, False)
            Plotter.plot_best_q_values(self._phase, fqi_iterations, seeds, self._save_path)
            if trajectory_number is not None:
                Plotter.plot_q_values(self._phase, fqi_iterations, trajectory_number, seeds, self._save_path)

            if trajectory_number is None:
                for trajectory in range(0, self._testing_env._persistence, 10):
                    Plotter.plot_percentage_returns_single_trajectory(self._phase, fqi_iterations, seeds, self._save_path,
                                                                      trajectory)
        else:
            Plotter.plot_seeds_percentage(self._phase, 1, seeds, self._save_path,
                                          self._testing_env._persistence, False, labels=[str(fqi_iterations)])

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

            return np.median(boot, axis=2)[0].tolist(), np.array(gains)[:, -1]

        return np.mean(gains, axis=0).tolist(), np.array(gains)[:, -1]

    def _test_with_seed(self, fqi_iterations: int, seed: int, trajectory_number: typing.Optional[int] = None, trajectory_window: typing.Optional[int] = None, Q_values_diff_threshold: typing.Optional[float] = None, filter_method: typing.Optional[str] = None, unroll_fqi_iterations=False
            ) -> typing.List[float]:
        if self._n_jobs > 1 or self._n_jobs == -1: #workaround to get the logging work in parallel execution
            logger_ = get_logger(logging.INFO)
        else:
            logger_ = self._logger

        save_path = os.path.join(self._save_path, f'seed{seed}')
        os.makedirs(save_path, exist_ok=True)
        read_path = os.path.join(self._read_path, f'seed{seed}')
        logger_.info(f'Algorithm seed = {seed}.')
        # ======== FQI iterations ========
        gains = []
        for i in range(fqi_iterations):

            iteration = i+1
            if unroll_fqi_iterations or iteration==fqi_iterations:
                logger_.info(f"ITERATION {iteration}: Algorithm started.")
                with open(os.path.join(read_path, f'Policy_iter{iteration}.pkl'), 'rb') as load_file:
                    policy = pickle.load(load_file)
                logger_.info(f"ITERATION {iteration}: Algorithm read.")
                # Testing
                gain = self._testing_env.test(
                    policy=policy,
                    save_csv=True,
                    save_plots=True,
                    save_root=save_path,
                    phase=self._phase,
                    iteration=iteration,
                    trajectory_number=trajectory_number,
                    trajectory_window=trajectory_window,
                    Q_values_diff_threshold=Q_values_diff_threshold,
                    filter_method = filter_method,
                    use_estimator_mismatch=self._use_estimator_mismatch
                )
                gains.append(gain)
                logger_.info(f"ITERATION {iteration}: Algorithm tested on {self._phase} set (gain = {gain:,.2f}%).")
        return gains

    def test_fixed_policy(self, policy, save_path, policy_name):
        gain = self._testing_env.test(
            policy=policy,
            save_csv=True,
            save_plots=True,
            save_root=save_path,
            phase=self._phase,
            iteration=1,
            trajectory_number=None,
            trajectory_window=None,
            Q_values_diff_threshold=None,
            filter_method=None,
            use_estimator_mismatch=self._use_estimator_mismatch,
            plot_policy_features=None
        )
        Plotter.plot_percentage_returns_fixed_policy(policy_name, self._phase, self._save_path,)
        self._logger.info(f"Algorithm tested on {self._phase} set (gain = {gain:,.2f}%).")
