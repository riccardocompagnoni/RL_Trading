import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
import pandas as pd
import seaborn as sns
import typing
import xgboost
from scipy.stats import bootstrap
from datetime import datetime

def boot_median_iqr_scipy(a, n_resamples=1000, rng=None):

    if rng is None:
        rng = np.random.default_rng()
    res = bootstrap(
        (a,),
        statistic=lambda d, axis: np.median(d, axis=axis),
        axis=0,
        n_resamples=n_resamples,
        random_state=rng,
        confidence_level=0.90,
        method="basic",
    )

    bootstat = res.bootstrap_distribution

    med = np.median(bootstat, axis=1)
    q5 = np.percentile(bootstat, 5, axis=1)
    q95 = np.percentile(bootstat, 95, axis=1)

    return med, q5, q95


class Plotter(object):

    def __new__(cls, *args, **kwargs):
        raise NotImplemented("This class cannot be constructed. Use static method in order to initialize it.")

    def __init__(self):
        raise NotImplemented("This class cannot be constructed. Use static method in order to initialize it.")

    @staticmethod
    def _read_csv_results(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df['time'] = (df['minute'] // 60) * 100 + df['minute'] % 60
        df['timestamp'] = pd.to_datetime(df['day'].astype(str) + df['time'].astype(str), format='%Y%m%d%H%M')
        return df
    @classmethod
    def plot_percentage_returns_percentiles(cls,
            percentiles: typing.List[float],
            phase: str,
            iterations: int,
            seeds: typing.List[int],
            save_path: str,
            ):

        n_rows = iterations
        fig, axs = plt.subplots(n_rows, 1, figsize=(30, 7 * n_rows))
        # Get a colormap
        cmap = cm.get_cmap('tab10', len(percentiles) + iterations)
        # Plot a line with standard deviations for each FQI iteration
        min_scale = +np.inf
        max_scale = -np.inf
        for j, prc in enumerate(percentiles):
            for i in range(iterations):
                iteration = i + 1
                df = pd.DataFrame()
                df_q = pd.DataFrame()
                for seed in seeds:
                    if prc == 1:
                        _df = cls._read_csv_results(os.path.join(save_path, f'seed{seed}/Results_iter{iteration}_{phase}.csv'))
                    else:
                        _df = cls._read_csv_results(os.path.join(save_path, f'seed{seed}/Results_iter{iteration}_{phase}_removing_percentile_{prc}.csv'))
                    _df['return'] = _df.groupby(by='trajectory')['reward'].cumsum()
                    index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp']
                    data = np.mean(_df.groupby(by='trajectory')['return'].apply(list).to_list(), axis=0)
                    q = np.mean(_df.groupby(by='trajectory')['Q'].apply(list).to_list(), axis=0)
                    df = pd.concat([df, pd.Series(data=data, index=index)], axis=1)
                    df_q = pd.concat([df_q, pd.Series(data=q, index=index)], axis=1)
                # Calculate max and min q values to make a common axis for each Q plots
                if df_q.min().values[0] < min_scale:
                    min_scale = df_q.min().values[0]
                if df_q.max().values[0] > max_scale:
                    max_scale = df_q.max().values[0]
                # Plot returns with standard deviations
                n = df.shape[0]
                x = range(n)
                medians = df.median(axis=1)
                axs.plot(x, medians, color=cmap(i + j), label=f"FQI Iteration {iteration}, percentile {prc}")
                if df.shape[1] > 1:
                    q25 = df.quantile(0.25, axis=1)
                    q75 = df.quantile(0.75, axis=1)
                    axs.fill_between(x, q25, q75, alpha=0.3, color=cmap(i + j), linestyle='--')
                # Plot best Q value with standard deviations

        # Format plots
        axs.set_title(f"{phase} return")
        axs.set_ylabel("Cumulative reward")
        axs.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
        axs.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
        axs.legend()
        axs.margins(x=0)
        axs.xaxis.set_major_locator(mticker.IndexLocator(n // 10, 0))
        axs.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].strftime('%Y-%m-%d')))
        fig.savefig(os.path.join(save_path, f'Returns_CI_{phase}_percentiles_{percentiles}.png'), dpi=200)
        plt.close(fig)
    """
    @classmethod
    def plot_percentage_returns_more_cases_without_Q(
            cls,
            phase: str,
            iterations: int,
            directories: typing.Dict[str, str],
            seeds: typing.Dict[str, typing.List],
            save_path: str,
            label_dictionary:typing.Dict[str, str]
    ):
        n_rows = 1
        fig, axs = plt.subplots(n_rows, 1, figsize=(30, 7 * n_rows))
        # Get a colormap
        cmap = cm.get_cmap('tab10', iterations+len(directories.keys()))
        # Plot a line with standard deviations for each FQI iteration
        j = 0
        for name, directory in directories.items():
            print(name, directory)
            for i in range(iterations):
                iteration = i + 1
                df = pd.DataFrame()
                df_q = pd.DataFrame()
                for seed in seeds[f"seeds_{name}"]:
                    if "results/test" in directory:
                        _df = cls._read_csv_results(os.path.join(directory, f'seed{seed}/Results_iter{iteration}_Validation.csv'))
                    else:
                        _df = cls._read_csv_results(os.path.join(directory, f'seed{seed}/Results_iter{iteration}_{phase}.csv'))

                    if "skip_covid" not in name:
                        to_fill = _df[['day']]
                    else:
                         tmp = to_fill

                    _df['return'] = _df.groupby(by='trajectory')['reward'].cumsum()
                    index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp']
                    data = np.mean(_df.groupby(by='trajectory')['return'].apply(list).to_list(), axis=0)
                    if "skip_covid" not in name:
                        index_full = index
                    df = pd.concat([df, pd.Series(data=data, index=index)], axis=1)

                # Plot returns with standard deviations
                n = df.shape[0]

                x = range(n)
                means = df.mean(axis=1)
                axs.plot(x, means, color=cmap(i+j), label=label_dictionary[name])
                if df.shape[1] > 1:
                    stds = df.std(axis=1)
                    axs.fill_between(x, means - stds, means + stds, alpha=0.3, color=cmap(i+j), linestyle='--')
                j += 1
            # Format plots
        axs.set_title(f"{phase} return")
        axs.set_ylabel("Cumulative reward")
        axs.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
        axs.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
        axs.legend()
        axs.margins(x=0)
        axs.xaxis.set_major_locator(mticker.IndexLocator(n // 10, 0))
        axs.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].strftime('%Y-%m-%d')))
        fig.savefig(os.path.join(save_path, f'Returns_CI_{phase}.png'), dpi=200)
        plt.close(fig)
    """

    @classmethod
    def plot_percentage_returns_nostd_fixed_policy(
            cls,
            policy_name: str,
            phase: str,
            save_path: str,
    ):
        n_rows = 1
        fig, axs = plt.subplots(n_rows, 1, figsize=(30, 7 * n_rows))
        # Get a colormap
        cmap = cm.get_cmap('tab10', 1)
        # Plot a line with standard deviations for each FQI iteration
        min_scale = +np.inf
        max_scale = -np.inf
        for i in range(1):
            iteration = i + 1
            df = pd.DataFrame()
            df_q = pd.DataFrame()
            _df = cls._read_csv_results(os.path.join(save_path, f'Results_iter{iteration}_{phase}.csv'))
            _df['return_nostd'] = _df.groupby(by='trajectory')['reward_nostd'].cumsum()
            index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp']
            data = np.mean(_df.groupby(by='trajectory')['return_nostd'].apply(list).to_list(), axis=0)
            q = np.mean(_df.groupby(by='trajectory')['Q'].apply(list).to_list(), axis=0)
            df = pd.concat([df, pd.Series(data=data, index=index)], axis=1)
            df_q = pd.concat([df_q, pd.Series(data=q, index=index)], axis=1)
            # Plot returns with standard deviations
            n = df.shape[0]
            x = range(n)
            medians = df.median(axis=1)
            axs.plot(x, medians, color=cmap(i))
            if df.shape[1] > 1:
                q25 = df.quantile(0.25, axis=1)
                q75 = df.quantile(0.75, axis=1)
                axs.fill_between(x, q25, q75, alpha=0.3, color=cmap(i), linestyle='--')
            # Plot best Q value with standard deviations

        # Format plots
        axs.set_title(f"{phase} return")
        axs.set_ylabel("Cumulative reward (not std)")
        axs.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
        axs.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
        axs.legend()
        axs.margins(x=0)
        axs.xaxis.set_major_locator(mticker.IndexLocator(n // 10, 0))
        axs.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].strftime('%Y-%m-%d')))
        fig.savefig(os.path.join(save_path, f'Returns_CI_{phase}_{policy_name}_nostd.png'), dpi=200)
        plt.close(fig)

    @classmethod
    def plot_percentage_returns_fixed_policy(
            cls,
            policy_name: str,
            phase: str,
            save_path: str,
    ):
        n_rows = 1
        fig, axs = plt.subplots(n_rows, 1, figsize=(30, 7 * n_rows))
        # Get a colormap
        cmap = cm.get_cmap('tab10', 1)
        # Plot a line with standard deviations for each FQI iteration
        min_scale = +np.inf
        max_scale = -np.inf
        for i in range(1):
            iteration = i + 1
            df = pd.DataFrame()
            df_q = pd.DataFrame()
            _df = cls._read_csv_results(os.path.join(save_path, f'Results_iter{iteration}_{phase}.csv'))
            _df['return'] = _df.groupby(by='trajectory')['reward'].cumsum()
            index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp']
            data = np.mean(_df.groupby(by='trajectory')['return'].apply(list).to_list(), axis=0)
            q = np.mean(_df.groupby(by='trajectory')['Q'].apply(list).to_list(), axis=0)
            df = pd.concat([df, pd.Series(data=data, index=index)], axis=1)
            df_q = pd.concat([df_q, pd.Series(data=q, index=index)], axis=1)
            # Plot returns with standard deviations
            n = df.shape[0]
            x = range(n)
            medians = df.median(axis=1)
            axs.plot(x, medians, color=cmap(i))
            if df.shape[1] > 1:
                q25 = df.quantile(0.25, axis=1)
                q75 = df.quantile(0.75, axis=1)
                axs.fill_between(x, q25, q75, alpha=0.3, color=cmap(i), linestyle='--')
            # Plot best Q value with standard deviations

        # Format plots
        axs.set_title(f"{phase} return")
        axs.set_ylabel("Cumulative reward")
        axs.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
        axs.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
        axs.legend()
        axs.margins(x=0)
        axs.xaxis.set_major_locator(mticker.IndexLocator(n // 10, 0))
        axs.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].strftime('%Y-%m-%d')))
        fig.savefig(os.path.join(save_path, f'Returns_CI_{phase}_{policy_name}.png'), dpi=200)
        plt.close(fig)

        cls.plot_percentage_returns_nostd_fixed_policy(policy_name, phase, save_path)

    @classmethod
    def plot_percentage_returns_nostd_more_cases_without_Q(
            cls,
            phase: str,
            iterations: int,
            directories: typing.Dict[str, str],
            seeds: typing.Dict[str, typing.List],
            save_path: str,
            label_dictionary: typing.Dict[str, str]
    ):
        n_rows = 1
        fig, axs = plt.subplots(n_rows, 1, figsize=(30, 7 * n_rows))
        # Get a colormap
        cmap = cm.get_cmap('tab10', iterations + len(directories.keys()))
        # Plot a line with standard deviations for each FQI iteration
        j = 0
        num_lines = len(directories.items())
        dfs = []
        max_len = 0
        max_index = 0
        for name, directory in directories.items():
            print(name, directory)
            for i in range(1):
                iteration = i + 1
                df = pd.DataFrame()
                for seed in seeds[f"seeds_{name}"]:
                    if "results/test" in directory:
                        _df = cls._read_csv_results(os.path.join(directory,
                                                                 f'seed{seed}/Results_iter{iteration}_Validation.csv'))
                    else:
                        _df = cls._read_csv_results(os.path.join(directory,
                                                                 f'seed{seed}/Results_iter{iteration}_{phase}.csv'))
                    # if "skip_covid" not in name:
                    #     to_fill = _df[['day']]
                    # else:
                    #     tmp = to_fill

                    _df['return_nostd'] = _df.groupby(by='trajectory')['reward_nostd'].cumsum()
                    index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp']
                    data = np.mean(_df.groupby(by='trajectory')['return_nostd'].apply(list).to_list(), axis=0)
                    if "skip_covid" not in name:
                        index_full = index
                    df = pd.concat([df, pd.Series(data=data, index=index)], axis=1)

                # Plot returns with standard deviations
                n = df.shape[0]

                x = range(n)
                medians = df.median(axis=1)
                # axs.plot(x, medians, color=cmap(i+j), label=label_dictionary[name])
                if df.shape[1] > 1:
                    q25 = df.quantile(0.25, axis=1)
                    q75 = df.quantile(0.75, axis=1)
                    # axs.fill_between(x, q25, q75, alpha=0.3, color=cmap(i+j), linestyle='--')
                _df = pd.concat([medians, q25, q75], axis=1)
                _df = _df.set_index(_df.index.round("H"))
                dfs.append(_df)
                if n > max_len:
                    max_len = n
                    max_name = name
                    max_index = j
                j += 1
                # Format plots

        max_df = dfs[max_index]
        index = pd.DataFrame()
        index['timestamp'] = max_df.index
        x = range(max_df.shape[0])
        for j in range(num_lines):
            if j != max_index:
                # join with the max length one
                df = dfs[j]
                df = pd.concat([max_df, df], axis=1)
                df = df.iloc[:, [3, 4, 5]].ffill()  # Now we have 3 columns: median, q25, q75
                dfs[j] = df
        for j in range(num_lines):
            name = list(directories.keys())[j]
            medians = dfs[j].iloc[:, 0]
            q25 = dfs[j].iloc[:, 1]
            q75 = dfs[j].iloc[:, 2]
            axs.plot(x, medians, color=cmap(j), label=label_dictionary[name])
            axs.fill_between(x, q25, q75, alpha=0.3, color=cmap(j), linestyle='--')

        axs.set_title(f"{phase} return")
        axs.set_ylabel("Cumulative reward (not std)")
        axs.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
        axs.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
        axs.legend()
        axs.margins(x=0)
        axs.xaxis.set_major_locator(mticker.IndexLocator(n // 10, 0))
        axs.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].dt.strftime('%Y-%m-%d')))
        fig.savefig(os.path.join(save_path, f'Returns_CI_{phase}_nostd.png'), dpi=200)
        plt.close(fig)

    @classmethod
    def plot_percentage_returns_more_cases_without_Q(
            cls,
            phase: str,
            iterations: int,
            directories: typing.Dict[str, str],
            seeds: typing.Dict[str, typing.List],
            save_path: str,
            label_dictionary: typing.Dict[str, str]
    ):
        n_rows = 1
        fig, axs = plt.subplots(n_rows, 1, figsize=(30, 7 * n_rows))
        # Get a colormap
        cmap = cm.get_cmap('tab10', iterations + len(directories.keys()))
        # Plot a line with standard deviations for each FQI iteration
        j = 0
        num_lines = len(directories.items())
        dfs = []
        max_len = 0
        max_index = 0
        for name, directory in directories.items():
            print(name, directory)
            for i in range(1):
                iteration = i + 1
                df = pd.DataFrame()
                for seed in seeds[f"seeds_{name}"]:
                    if "results/test" in directory:
                        _df = cls._read_csv_results(os.path.join(directory,
                                                                 f'seed{seed}/Results_iter{iteration}_Validation.csv'))
                    else:
                        _df = cls._read_csv_results(os.path.join(directory,
                                                                 f'seed{seed}/Results_iter{iteration}_{phase}.csv'))
                    # if "skip_covid" not in name:
                    #     to_fill = _df[['day']]
                    # else:
                    #     tmp = to_fill

                    _df['return'] = _df.groupby(by='trajectory')['reward'].cumsum()
                    index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp']
                    data = np.mean(_df.groupby(by='trajectory')['return'].apply(list).to_list(), axis=0)
                    if "skip_covid" not in name:
                        index_full = index
                    df = pd.concat([df, pd.Series(data=data, index=index)], axis=1)

                # Plot returns with standard deviations
                n = df.shape[0]

                x = range(n)
                medians = df.median(axis=1)
                # axs.plot(x, medians, color=cmap(i+j), label=label_dictionary[name])
                if df.shape[1] > 1:
                    q25 = df.quantile(0.25, axis=1)
                    q75 = df.quantile(0.75, axis=1)
                    # axs.fill_between(x, q25, q75, alpha=0.3, color=cmap(i+j), linestyle='--')
                _df = pd.concat([medians, q25, q75], axis=1)
                _df = _df.set_index(_df.index.round("H"))
                dfs.append(_df)
                if n > max_len:
                    max_len = n
                    max_name = name
                    max_index = j
                j += 1
                # Format plots

        max_df = dfs[max_index]
        index = pd.DataFrame()
        index['timestamp'] = max_df.index
        x = range(max_df.shape[0])
        for j in range(num_lines):
            if j != max_index:
                # join with the max length one
                df = dfs[j]
                df = pd.concat([max_df, df], axis=1)
                df = df.iloc[:, [3, 4, 5]].ffill()  # Now we have 3 columns: median, q25, q75
                dfs[j] = df
        for j in range(num_lines):
            name = list(directories.keys())[j]
            medians = dfs[j].iloc[:, 0]
            q25 = dfs[j].iloc[:, 1]
            q75 = dfs[j].iloc[:, 2]
            axs.plot(x, medians, color=cmap(j), label=label_dictionary[name])
            axs.fill_between(x, q25, q75, alpha=0.3, color=cmap(j), linestyle='--')

        axs.set_title(f"{phase} return")
        axs.set_ylabel("Cumulative reward")
        axs.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
        axs.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
        axs.legend()
        axs.margins(x=0)
        axs.xaxis.set_major_locator(mticker.IndexLocator(n // 10, 0))
        axs.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].dt.strftime('%Y-%m-%d')))
        fig.savefig(os.path.join(save_path, f'Returns_CI_{phase}.png'), dpi=200)
        plt.close(fig)

        cls.plot_percentage_returns_nostd_more_cases_without_Q(phase, iterations, directories, seeds, save_path, label_dictionary)

    def plot_percentage_returns_nostd_combined(
            cls,
            phase: str,
            iterations: int,
            seeds: typing.List[int],
            save_paths: str,
            output_path:str
    ):
        n_rows = 1
        fig, axs = plt.subplots(1, 1, figsize=(30, 7*n_rows))
        # Get a colormap
        cmap = cm.get_cmap('tab10', iterations)
        # Plot a line with standard deviations for each FQI iteration
        for i in range(iterations):
            iteration = i+1
            df = pd.DataFrame()
            for seed in seeds:
                for save_path in save_paths:
                    _df = cls._read_csv_results(os.path.join(save_path, f'seed{seed}/Results_iter{iteration}_{phase}.csv'))
                    _df['return_nostd'] = _df.groupby(by='trajectory')['reward_nostd'].cumsum()
                    index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp'] - _df.loc[0, 'trajectory']
                    data = np.mean(_df.groupby(by='trajectory')['return_nostd'].apply(list).to_list(), axis=0)
                    df = pd.concat([df, pd.Series(data=data, index=index)], axis=1)

            # Plot returns with standard deviations
            n = df.shape[0]
            x = range(n)
            medians = df.median(axis=1)
            axs.plot(x, medians, color=cmap(i), label=f"FQI Iteration {iteration}")
            if df.shape[1] > 1:
                q25 = df.quantile(0.25, axis=1)
                q75 = df.quantile(0.75, axis=1)
                axs[0].fill_between(x, q25, q75, alpha=0.3, color=cmap(i), linestyle='--')
        # Format plots
        axs.set_title(f"{phase} return")
        axs.set_ylabel("Cumulative reward (not std)")
        axs.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
        axs.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
        axs.legend()
        axs.margins(x=0)
        axs.xaxis.set_major_locator(mticker.IndexLocator(n // 10, 0))
        axs.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].strftime('%Y-%m-%d')))
        fig.savefig(os.path.join(output_path, f'Returns_CI_{phase}_nostd.png'), dpi=200)
        plt.close(fig)


    def plot_percentage_returns_combined(
            cls,
            phase: str,
            iterations: int,
            seeds: typing.List[int],
            save_paths: str,
            output_path:str
    ):
        n_rows = 1
        fig, axs = plt.subplots(1, 1, figsize=(30, 7*n_rows))
        # Get a colormap
        cmap = cm.get_cmap('tab10', iterations)
        # Plot a line with standard deviations for each FQI iteration
        for i in range(iterations):
            iteration = i+1
            df = pd.DataFrame()
            for seed in seeds:
                for save_path in save_paths:
                    _df = cls._read_csv_results(os.path.join(save_path, f'seed{seed}/Results_iter{iteration}_{phase}.csv'))
                    _df['return'] = _df.groupby(by='trajectory')['reward'].cumsum()
                    index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp'] - _df.loc[0, 'trajectory']
                    data = np.mean(_df.groupby(by='trajectory')['return'].apply(list).to_list(), axis=0)
                    df = pd.concat([df, pd.Series(data=data, index=index)], axis=1)

            # Plot returns with standard deviations
            n = df.shape[0]
            x = range(n)
            medians = df.median(axis=1)
            axs.plot(x, medians, color=cmap(i), label=f"FQI Iteration {iteration}")
            if df.shape[1] > 1:
                q25 = df.quantile(0.25, axis=1)
                q75 = df.quantile(0.75, axis=1)
                axs[0].fill_between(x, q25, q75, alpha=0.3, color=cmap(i), linestyle='--')
        # Format plots
        axs.set_title(f"{phase} return")
        axs.set_ylabel("Cumulative reward")
        axs.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
        axs.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
        axs.legend()
        axs.margins(x=0)
        axs.xaxis.set_major_locator(mticker.IndexLocator(n // 10, 0))
        axs.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].strftime('%Y-%m-%d')))
        fig.savefig(os.path.join(output_path, f'Returns_CI_{phase}.png'), dpi=200)
        plt.close(fig)

        cls.plot_percentage_returns_nostd_combined(phase, iterations, seeds, save_paths, output_path)


    @classmethod
    def plot_percentage_returns_nostd(
            cls,
            phase: str,
            iterations: int,
            seeds: typing.List[int],
            save_path: str,
    ):
        n_rows = iterations + 1
        fig, axs = plt.subplots(n_rows, 1, figsize=(30, 7 * n_rows))
        # Get a colormap
        cmap = cm.get_cmap('tab10', iterations)
        # Plot a line with standard deviations for each FQI iteration
        min_scale = +np.inf
        max_scale = -np.inf
        for i in range(iterations):
            iteration = i + 1
            df = pd.DataFrame()
            df_q = pd.DataFrame()
            for seed in seeds:
                _df = cls._read_csv_results(os.path.join(save_path, f'seed{seed}/Results_iter{iteration}_{phase}.csv'))
                _df['return_nostd'] = _df.groupby(by='trajectory')['reward_nostd'].cumsum()
                index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp']
                data = np.mean(_df.groupby(by='trajectory')['return_nostd'].apply(list).to_list(), axis=0)
                q = np.mean(_df.groupby(by='trajectory')['Q'].apply(list).to_list(), axis=0)
                df = pd.concat([df, pd.Series(data=data, index=index)], axis=1)
                df_q = pd.concat([df_q, pd.Series(data=q, index=index)], axis=1)
            # Calculate max and min q values to make a common axis for each Q plots
            if df_q.min().values[0] < min_scale:
                min_scale = df_q.min().values[0]
            if df_q.max().values[0] > max_scale:
                max_scale = df_q.max().values[0]
            # Plot returns with standard deviations
            n = df.shape[0]
            x = range(n)
            medians = df.median(axis=1)
            axs[0].plot(x, medians, color=cmap(i), label=f"FQI Iteration {iteration}")
            if df.shape[1] > 1:
                q25 = df.quantile(0.25, axis=1)
                q75 = df.quantile(0.75, axis=1)
                axs[0].fill_between(x, q25, q75, alpha=0.3, color=cmap(i), linestyle='--')

            # Plot best Q value with IQR
            medians = df_q.median(axis=1)
            axs[i + 1].plot(x, medians, color=cmap(i), label=f"Value estimate iteration {iteration}")
            if df_q.shape[1] > 1:
                q25 = df_q.quantile(0.25, axis=1)
                q75 = df_q.quantile(0.75, axis=1)
                axs[i + 1].fill_between(x, q25, q75, alpha=0.3, color=cmap(i), linestyle='--')
        # Format plots
        axs[0].set_title(f"{phase} return")
        axs[1].set_title(f"{phase} Q function estimate")
        axs[0].set_ylabel("Cumulative reward (not std)")
        axs[0].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
        for ax in axs[1:]:
            ax.set_ylim(ymin=min_scale, ymax=max_scale)
            ax.set_ylabel("Value estimate")
        for ax in axs:
            ax.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
            ax.legend()
            ax.margins(x=0)
            ax.xaxis.set_major_locator(mticker.IndexLocator(n // 10, 0))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].strftime('%Y-%m-%d')))
        fig.savefig(os.path.join(save_path, f'Returns_CI_{phase}_nostd.png'), dpi=200)
        plt.close(fig)

    @classmethod
    def plot_percentage_returns(
            cls,
            phase: str,
            iterations: int,
            seeds: typing.List[int],
            save_path: str,
    ):
        n_rows = iterations + 1
        fig, axs = plt.subplots(n_rows, 1, figsize=(30, 7*n_rows))
        # Get a colormap
        cmap = cm.get_cmap('tab10', iterations)
        # Plot a line with standard deviations for each FQI iteration
        min_scale = +np.inf
        max_scale = -np.inf
        for i in range(iterations):
            iteration = i+1
            df = pd.DataFrame()
            df_q = pd.DataFrame()
            for seed in seeds:
                _df = cls._read_csv_results(os.path.join(save_path, f'seed{seed}/Results_iter{iteration}_{phase}.csv'))
                _df['return'] = _df.groupby(by='trajectory')['reward'].cumsum()
                index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp']
                data = np.mean(_df.groupby(by='trajectory')['return'].apply(list).to_list(), axis=0)
                q = np.mean(_df.groupby(by='trajectory')['Q'].apply(list).to_list(), axis=0)
                df = pd.concat([df, pd.Series(data=data, index=index)], axis=1)
                df_q = pd.concat([df_q, pd.Series(data=q, index=index)], axis=1)
            # Calculate max and min q values to make a common axis for each Q plots
            if df_q.min().values[0] < min_scale:
                min_scale = df_q.min().values[0]
            if df_q.max().values[0] > max_scale:
                max_scale = df_q.max().values[0]
            # Plot returns with standard deviations
            n = df.shape[0]
            x = range(n)
            medians = df.median(axis=1)
            axs[0].plot(x, medians, color=cmap(i), label=f"FQI Iteration {iteration}")
            if df.shape[1] > 1:
                q25 = df.quantile(0.25, axis=1)
                q75 = df.quantile(0.75, axis=1)
                axs[0].fill_between(x, q25, q75, alpha=0.3, color=cmap(i), linestyle='--')

            # Plot best Q value with IQR
            medians = df_q.median(axis=1)
            axs[i + 1].plot(x, medians, color=cmap(i), label=f"Value estimate iteration {iteration}")
            if df_q.shape[1] > 1:
                q25 = df_q.quantile(0.25, axis=1)
                q75 = df_q.quantile(0.75, axis=1)
                axs[i + 1].fill_between(x, q25, q75, alpha=0.3, color=cmap(i), linestyle='--')
        # Format plots
        axs[0].set_title(f"{phase} return")
        axs[1].set_title(f"{phase} Q function estimate")
        axs[0].set_ylabel("Cumulative reward")
        axs[0].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
        for ax in axs[1:]:
            ax.set_ylim(ymin=min_scale, ymax=max_scale)
            ax.set_ylabel("Value estimate")
        for ax in axs:
            ax.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
            ax.legend()
            ax.margins(x=0)
            ax.xaxis.set_major_locator(mticker.IndexLocator(n // 10, 0))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].strftime('%Y-%m-%d')))
        fig.savefig(os.path.join(save_path, f'Returns_CI_{phase}.png'), dpi=200)
        plt.close(fig)

        cls.plot_percentage_returns_nostd(phase, iterations, seeds, save_path)

    @classmethod
    def plot_percentage_returns_nostd_single_trajectory(
            cls,
            phase: str,
            iterations: int,
            seeds: typing.List[int],
            save_path: str,
            trajectory_number: int
    ):
        n_rows = iterations + 1
        fig, axs = plt.subplots(n_rows, 1, figsize=(30, 7 * n_rows))
        # Get a colormap
        cmap = cm.get_cmap('tab10', iterations)
        # Plot a line with standard deviations for each FQI iteration
        min_scale = +np.inf
        max_scale = -np.inf
        for i in range(iterations):
            iteration = i + 1
            df = pd.DataFrame()
            df_q = pd.DataFrame()
            for seed in seeds:
                _df = cls._read_csv_results(os.path.join(save_path, f'seed{seed}/Results_iter{iteration}_{phase}.csv'))
                _df['return_nostd'] = _df.groupby(by='trajectory')['reward_nostd'].cumsum()
                index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp']
                data = _df[_df['trajectory'] == trajectory_number]['return_nostd'].to_list()
                q = _df[_df['trajectory'] == trajectory_number]['Q'].to_list()
                df = pd.concat([df, pd.Series(data=data, index=index)], axis=1)
                df_q = pd.concat([df_q, pd.Series(data=q, index=index)], axis=1)
            # Calculate max and min q values to make a common axis for each Q plots
            if df_q.min().values[0] < min_scale:
                min_scale = df_q.min().values[0]
            if df_q.max().values[0] > max_scale:
                max_scale = df_q.max().values[0]
            # Plot returns with standard deviations
            n = df.shape[0]
            x = range(n)

            medians = df.median(axis=1)
            axs[0].plot(x, medians, color=cmap(i), label=f"FQI Iteration {iteration}")
            if df.shape[1] > 1:
                q25 = df.quantile(0.25, axis=1)
                q75 = df.quantile(0.75, axis=1)
                axs[0].fill_between(x, q25, q75, alpha=0.3, color=cmap(i), linestyle='--')

            # Plot best Q value with IQR
            medians = df_q.median(axis=1)
            axs[i + 1].plot(x, medians, color=cmap(i), label=f"Value estimate iteration {iteration}")
            if df_q.shape[1] > 1:
                q25 = df_q.quantile(0.25, axis=1)
                q75 = df_q.quantile(0.75, axis=1)
                axs[i + 1].fill_between(x, q25, q75, alpha=0.3, color=cmap(i), linestyle='--')

        # Format plots
        axs[0].set_title(f"{phase} return")
        axs[1].set_title(f"{phase} Q function estimate")
        axs[0].set_ylabel("Cumulative reward (not std)")
        axs[0].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
        for ax in axs[1:]:
            ax.set_ylim(ymin=min_scale, ymax=max_scale)
            ax.set_ylabel("Value estimate")
        for ax in axs:
            ax.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
            ax.legend()
            ax.margins(x=0)
            ax.xaxis.set_major_locator(mticker.IndexLocator(n // 10, 0))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].strftime('%Y-%m-%d')))
        fig.savefig(os.path.join(save_path, f'Returns_CI_{phase}_trajectory_{trajectory_number}_nostd.png'), dpi=200)
        plt.close(fig)

    @classmethod
    def plot_percentage_returns_single_trajectory(
            cls,
            phase: str,
            iterations: int,
            seeds: typing.List[int],
            save_path: str,
            trajectory_number: int
    ):
        n_rows = iterations + 1
        fig, axs = plt.subplots(n_rows, 1, figsize=(30, 7 * n_rows))
        # Get a colormap
        cmap = cm.get_cmap('tab10', iterations)
        # Plot a line with standard deviations for each FQI iteration
        min_scale = +np.inf
        max_scale = -np.inf
        for i in range(iterations):
            iteration = i + 1
            df = pd.DataFrame()
            df_q = pd.DataFrame()
            for seed in seeds:
                _df = cls._read_csv_results(os.path.join(save_path, f'seed{seed}/Results_iter{iteration}_{phase}.csv'))
                _df['return'] = _df.groupby(by='trajectory')['reward'].cumsum()
                index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp']
                data = _df[_df['trajectory'] == trajectory_number]['return'].to_list()
                q = _df[_df['trajectory'] == trajectory_number]['Q'].to_list()
                df = pd.concat([df, pd.Series(data=data, index=index)], axis=1)
                df_q = pd.concat([df_q, pd.Series(data=q, index=index)], axis=1)
            # Calculate max and min q values to make a common axis for each Q plots
            if df_q.min().values[0] < min_scale:
                min_scale = df_q.min().values[0]
            if df_q.max().values[0] > max_scale:
                max_scale = df_q.max().values[0]
            # Plot returns with standard deviations
            n = df.shape[0]
            x = range(n)
            medians = df.median(axis=1)
            axs[0].plot(x, medians, color=cmap(i), label=f"FQI Iteration {iteration}")
            if df.shape[1] > 1:
                q25 = df.quantile(0.25, axis=1)
                q75 = df.quantile(0.75, axis=1)
                axs[0].fill_between(x, q25, q75, alpha=0.3, color=cmap(i), linestyle='--')

            # Plot best Q value with IQR
            medians = df_q.median(axis=1)
            axs[i + 1].plot(x, medians, color=cmap(i), label=f"Value estimate iteration {iteration}")
            if df_q.shape[1] > 1:
                q25 = df_q.quantile(0.25, axis=1)
                q75 = df_q.quantile(0.75, axis=1)
                axs[i + 1].fill_between(x, q25, q75, alpha=0.3, color=cmap(i), linestyle='--')

        axs[0].set_title(f"{phase} return")
        axs[1].set_title(f"{phase} Q function estimate")
        axs[0].set_ylabel("Cumulative reward")
        axs[0].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
        for ax in axs[1:]:
            ax.set_ylim(ymin=min_scale, ymax=max_scale)
            ax.set_ylabel("Value estimate")
        for ax in axs:
            ax.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
            ax.legend()
            ax.margins(x=0)
            ax.xaxis.set_major_locator(mticker.IndexLocator(n // 10, 0))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].strftime('%Y-%m-%d')))
        fig.savefig(os.path.join(save_path, f'Returns_CI_{phase}_trajectory_{trajectory_number}.png'), dpi=200)
        plt.close(fig)

        cls.plot_percentage_returns_nostd_single_trajectory(phase, iterations, seeds, save_path, trajectory_number)

    @classmethod
    def plot_percentage_returns_nostd_sum(
            cls,
            phase: str,
            iterations: int,
            seeds: typing.List[int],
            save_path: str,
    ):
        n_rows = iterations + 1
        fig, axs = plt.subplots(n_rows, 1, figsize=(30, 7 * n_rows))
        # Get a colormap
        cmap = cm.get_cmap('tab10', iterations)
        # Plot a line with standard deviations for each FQI iteration
        min_scale = +np.inf
        max_scale = -np.inf
        for i in range(iterations):
            iteration = i + 1
            df = pd.DataFrame()
            df_q = pd.DataFrame()
            for seed in seeds:
                _df = cls._read_csv_results(os.path.join(save_path, f'seed{seed}/Results_iter{iteration}_{phase}.csv'))
                _df['return_nostd'] = _df.groupby(by='trajectory')['reward_nostd'].cumsum()
                index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp']
                # calculate the sum of the returns of the different agent launched at different trajectory numbers in the window
                data = np.sum(_df.groupby(by='trajectory')['return_nostd'].apply(list).to_list(), axis=0)
                q = np.mean(_df.groupby(by='trajectory')['Q'].apply(list).to_list(), axis=0)
                df = pd.concat([df, pd.Series(data=data, index=index)], axis=1)
                df_q = pd.concat([df_q, pd.Series(data=q, index=index)], axis=1)
            # Calculate max and min q values to make a common axis for each Q plots
            if df_q.min().values[0] < min_scale:
                min_scale = df_q.min().values[0]
            if df_q.max().values[0] > max_scale:
                max_scale = df_q.max().values[0]
            # Plot returns with standard deviations
            n = df.shape[0]
            x = range(n)
            medians = df.median(axis=1)
            axs[0].plot(x, medians, color=cmap(i), label=f"FQI Iteration {iteration}")
            if df.shape[1] > 1:
                q25 = df.quantile(0.25, axis=1)
                q75 = df.quantile(0.75, axis=1)
                axs[0].fill_between(x, q25, q75, alpha=0.3, color=cmap(i), linestyle='--')

            # Plot best Q value with IQR
            medians = df_q.median(axis=1)
            axs[i + 1].plot(x, medians, color=cmap(i), label=f"Value estimate iteration {iteration}")
            if df_q.shape[1] > 1:
                q25 = df_q.quantile(0.25, axis=1)
                q75 = df_q.quantile(0.75, axis=1)
                axs[i + 1].fill_between(x, q25, q75, alpha=0.3, color=cmap(i), linestyle='--')
        # Format plots
        axs[0].set_title(f"{phase} return with Trajectory Window")
        axs[1].set_title(f"{phase} Q function estimate")
        axs[0].set_ylabel("Cumulative reward (not std)")
        axs[0].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
        for ax in axs[1:]:
            ax.set_ylim(ymin=min_scale, ymax=max_scale)
            ax.set_ylabel("Value estimate")
        for ax in axs:
            ax.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
            ax.legend()
            ax.margins(x=0)
            ax.xaxis.set_major_locator(mticker.IndexLocator(n // 10, 0))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].strftime('%Y-%m-%d')))
        fig.savefig(os.path.join(save_path, f'Returns_CI_{phase}_nostd.png'), dpi=200)
        plt.close(fig)


    @classmethod
    def plot_percentage_returns_sum(
            cls,
            phase: str,
            iterations: int,
            seeds: typing.List[int],
            save_path: str,
    ):
        n_rows = iterations + 1
        fig, axs = plt.subplots(n_rows, 1, figsize=(30, 7 * n_rows))
        # Get a colormap
        cmap = cm.get_cmap('tab10', iterations)
        # Plot a line with standard deviations for each FQI iteration
        min_scale = +np.inf
        max_scale = -np.inf
        for i in range(iterations):
            iteration = i + 1
            df = pd.DataFrame()
            df_q = pd.DataFrame()
            for seed in seeds:
                _df = cls._read_csv_results(os.path.join(save_path, f'seed{seed}/Results_iter{iteration}_{phase}.csv'))
                _df['return'] = _df.groupby(by='trajectory')['reward'].cumsum()
                index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp']
                #calculate the sum of the returns of the different agent launched at different trajectory numbers in the window
                #TODO divdide by number of trajectories?
                data = np.sum(_df.groupby(by='trajectory')['return'].apply(list).to_list(), axis=0)
                q = np.mean(_df.groupby(by='trajectory')['Q'].apply(list).to_list(), axis=0)
                df = pd.concat([df, pd.Series(data=data, index=index)], axis=1)
                df_q = pd.concat([df_q, pd.Series(data=q, index=index)], axis=1)
            # Calculate max and min q values to make a common axis for each Q plots
            if df_q.min().values[0] < min_scale:
                min_scale = df_q.min().values[0]
            if df_q.max().values[0] > max_scale:
                max_scale = df_q.max().values[0]
            # Plot returns with standard deviations
            n = df.shape[0]
            x = range(n)
            medians = df.median(axis=1)
            axs[0].plot(x, medians, color=cmap(i), label=f"FQI Iteration {iteration}")
            if df.shape[1] > 1:
                q25 = df.quantile(0.25, axis=1)
                q75 = df.quantile(0.75, axis=1)
                axs[0].fill_between(x, q25, q75, alpha=0.3, color=cmap(i), linestyle='--')

            # Plot best Q value with IQR
            medians = df_q.median(axis=1)
            axs[i + 1].plot(x, medians, color=cmap(i), label=f"Value estimate iteration {iteration}")
            if df_q.shape[1] > 1:
                q25 = df_q.quantile(0.25, axis=1)
                q75 = df_q.quantile(0.75, axis=1)
                axs[i + 1].fill_between(x, q25, q75, alpha=0.3, color=cmap(i), linestyle='--')
        # Format plots
        axs[0].set_title(f"{phase} return with Trajectory Window")
        axs[1].set_title(f"{phase} Q function estimate")
        axs[0].set_ylabel("Cumulative reward")
        axs[0].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
        for ax in axs[1:]:
            ax.set_ylim(ymin=min_scale, ymax=max_scale)
            ax.set_ylabel("Value estimate")
        for ax in axs:
            ax.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
            ax.legend()
            ax.margins(x=0)
            ax.xaxis.set_major_locator(mticker.IndexLocator(n // 10, 0))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].strftime('%Y-%m-%d')))
        fig.savefig(os.path.join(save_path, f'Returns_CI_{phase}.png'), dpi=200)
        plt.close(fig)

        cls.plot_percentage_returns_nostd_sum(phase, iterations, seeds, save_path)

    @classmethod
    def plot_bootstrapped(
            cls,
            phase: str,
            iterations: int,
            seeds: typing.List[int],
            save_path: str,
    ):
        n_rows = iterations + 1
        fig, axs = plt.subplots(n_rows, 1, figsize=(30, 7 * n_rows))
        cmap = cm.get_cmap('tab10', iterations)
        min_scale = +np.inf
        max_scale = -np.inf

        for i in range(iterations):
            iteration = i + 1
            df = pd.DataFrame()
            df_q = pd.DataFrame()
            for seed in seeds:
                _df = cls._read_csv_results(os.path.join(save_path, f'seed{seed}/Results_iter{iteration}_{phase}.csv'))
                _df['return_nostd'] = _df.groupby(by='trajectory')['reward_nostd'].cumsum()
                index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp']
                data = np.sum(_df.groupby(by='trajectory')['return_nostd'].apply(list).to_list(), axis=0)
                q = np.mean(_df.groupby(by='trajectory')['Q'].apply(list).to_list(), axis=0)
                df = pd.concat([df, pd.Series(data=data, index=index)], axis=1)
                df_q = pd.concat([df_q, pd.Series(data=q, index=index)], axis=1)
            if df_q.min().values[0] < min_scale:
                min_scale = df_q.min().values[0]
            if df_q.max().values[0] > max_scale:
                max_scale = df_q.max().values[0]

            n = df.shape[0]
            x = range(n)

            med, q25, q75 = boot_median_iqr_scipy(df.to_numpy().T)
            axs[0].plot(x, med, color=cmap(i), label=f"FQI Iteration {iteration}")
            if df.shape[1] > 1:
                axs[0].fill_between(x, q25, q75, alpha=0.3, color=cmap(i), linestyle='--')

            med_q, q25_q, q75_q = boot_median_iqr_scipy(df_q.to_numpy().T)
            axs[i + 1].plot(x, med_q, color=cmap(i), label=f"Value estimate iteration {iteration}")
            if df_q.shape[1] > 1:
                axs[i + 1].fill_between(x, q25_q, q75_q, alpha=0.3, color=cmap(i), linestyle='--')

        axs[0].set_title(f"{phase} return (bootstrapped)")
        axs[1].set_title(f"{phase} Q function estimate")
        axs[0].set_ylabel("Cumulative reward (not std)")
        axs[0].yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
        for ax in axs[1:]:
            ax.set_ylim(ymin=min_scale, ymax=max_scale)
            ax.set_ylabel("Value estimate")
        for ax in axs:
            ax.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
            ax.legend()
            ax.margins(x=0)
            ax.xaxis.set_major_locator(mticker.IndexLocator(n // 10, 0))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].strftime('%Y-%m-%d')))
        fig.savefig(os.path.join(save_path, f'Returns_CI_{phase}_bootstrapped.png'), dpi=200)
        plt.close(fig)

    @classmethod
    def plot_seeds(
            cls,
            phase: str,
            iterations: int,
            seeds: typing.List[int],
            save_path: str,
    ) -> None:

        n_rows = iterations
        fig, axs = plt.subplots(n_rows, 1, figsize=(30, 7 * n_rows))
        if n_rows == 1:
            axs = [axs]

        cmap = plt.cm.get_cmap("tab10", max(10, len(seeds)))

        for i in range(iterations):
            iteration = i + 1
            ax = axs[i]

            for j, seed in enumerate(seeds):
                csv_path = os.path.join(
                    save_path, f"seed{seed}/Results_iter{iteration}_{phase}.csv"
                )

                _df = pd.read_csv(csv_path)
                _df['date'] = pd.to_datetime(_df['day'], format='%Y%m%d')
                delta = pd.to_timedelta(_df['minute'], unit='m')
                _df['timestamp'] = _df['date'] + delta


                _df = (
                    _df.sort_values(["trajectory", "timestamp"])
                    .reset_index(drop=True)
                )

                # Cumulative reward per trajectory
                _df["return_nostd"] = _df.groupby("trajectory")["reward_nostd"].cumsum()

                # Sum across trajectories to get a single curve per seed
                summed_returns = np.sum(
                    _df.groupby("trajectory")["return_nostd"].apply(list).to_list(), axis=0
                )
                # Use the timestamps of the *first* trajectory as the x‑axis
                index = _df[_df["trajectory"] == _df.loc[0, "trajectory"]]["timestamp"].reset_index(drop=True)
                x_vals = np.arange(len(index))  # numeric x‑coords

                # Plot ------------------------------------------------------
                ax.plot(
                    x_vals,
                    summed_returns,
                    label=f"Seed {seed}",
                    color=cmap(j),
                    linewidth=1.6,
                )

                # -----------------------------
                # Per‑subplot formatting ------
                # -----------------------------
            ax.set_title(
                f"{phase.capitalize()} cumulative returns – FQI iteration {iteration}"
            )
            ax.set_ylabel("Cumulative reward (no std)")
            ax.grid(color="gray", linestyle=(0, (5, 5)), linewidth=0.5)
            ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.1f}"))

            # X‑axis ticks: max 10, integer positions ----------------------
            n_points = len(index)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, integer=True))

            # Safe formatter: clip out‑of‑bounds indices --------------------
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda x, pos, idx=index: (
                        idx.iloc[min(max(int(round(x)), 0), len(idx) - 1)].strftime(
                            "%Y-%m-%d"
                        )
                        if 0 <= int(round(x)) < len(idx)
                        else ""
                    )
                )
            )
            ax.margins(x=0)
            ax.legend(title="Random seed", ncol=min(len(seeds), 5))

            # ------------------------------------------------------------------
            # Finalise figure ---------------------------------------------------
            # ------------------------------------------------------------------
        fig.tight_layout()
        output_file = os.path.join(save_path, f"Returns_AllSeeds_{phase}_nostd.png")
        fig.savefig(output_file, dpi=200)
        plt.close(fig)
        print(f"[plot saved to] {output_file}")

    @classmethod
    def plot_seeds_percentage(
            cls,
            phase: str,
            iterations: int,
            seeds: list[int],
            save_path: str,
            persistence: int = 1,
            remove_costs = False,
            labels = None
    ) -> None:
        if labels==None:
            labels = [str(i) for i in range(1, iterations+1)]
        data_path = "/home/a2a/a2a/RL_Trading/prj/app/core/data/M1_ICE.csv"
        #data_path = "/data2/bonetti_a2a/a2a_riccardo/RL_Trading/prj/app/core/data/M1_ICE.csv"
        data_df = pd.read_csv(data_path)[["timestamp", "ask_0", "bid_0"]]
        data_df["timestamp"] = pd.to_datetime(data_df["timestamp"])

        first = True
        n_rows = iterations
        fig, axs = plt.subplots(n_rows, 1, figsize=(30, 7 * n_rows))
        if n_rows == 1:
            axs = [axs]
        cmap = plt.cm.get_cmap("tab10", max(10, len(seeds)))

        for i, iteration in enumerate(range(1, iterations + 1)):
            ax = axs[iteration - 1]
            percentages_df = []

            for seed in seeds:
                file = os.path.join(save_path, f'seed{seed}/Results_iter{labels[i]}_{phase}.csv')

                results_df = pd.read_csv(file)
                trajectories = [results_df.trajectory.unique()[0]]
                results_df['date'] = pd.to_datetime(results_df['day'], format='%Y%m%d')
                delta = pd.to_timedelta(results_df['minute'], unit='m')
                results_df['timestamp'] = results_df['date'] + delta
                results_df = results_df[['timestamp', 'action', 'trajectory']]

                #results_df = results_df[results_df.timestamp>=datetime(2024, 1, 1)]

                traj_percent_df = []
                for trajectory in trajectories:
                    traj_df = results_df[results_df.trajectory == trajectory]

                    last_per_day = traj_df.groupby(traj_df['timestamp'].dt.date)['timestamp'].max()
                    extra_rows = pd.DataFrame({
                        'timestamp': last_per_day + pd.Timedelta(minutes=persistence),
                        'action': 0
                    })
                    """
                    traj_df = (pd.concat([traj_df, extra_rows], ignore_index=True)
                               .sort_values('timestamp')
                               .reset_index(drop=True))
                    """

                    if first:
                        data_df = data_df[data_df['timestamp'].isin(traj_df['timestamp'].unique())]
                        first = False

                    df = traj_df.merge(data_df, on='timestamp', how='left')
                    df = df.set_index('timestamp')

                    if remove_costs:
                        df['mid'] = (df['bid_0'] + df['ask_0'])/2
                        out, pos, entry = [], 0, np.nan
                        for action, mid in df[['action', 'mid']].values:
                            perc = 0
                            if pos and action != pos:
                                perc = mid/entry - 1 if pos == 5 else -(mid/entry - 1)
                                entry = np.nan
                            if action and action != pos:
                                entry = mid
                            pos = action
                            out.append(perc)

                    else:
                        out, pos, entry = [], 0, np.nan
                        for action, bid, ask in df[['action', 'bid_0', 'ask_0']].values:
                            perc = 0
                            if pos and action != pos:
                                perc = (bid - 0.00625) / entry - 1 if pos == 5 else -((ask + 0.00625) / entry - 1)
                                entry = np.nan
                            if action and action != pos:
                                entry = (bid - 0.00625) if action == -5 else (ask + 0.00625)
                            pos = action
                            out.append(perc)
                    traj_percent_df.append(pd.Series(out, index=df.index))
                traj_percent_df = pd.concat(traj_percent_df, axis=1).mean(axis=1)
                percentages_df.append(traj_percent_df)

            seed_pct = pd.concat(percentages_df, axis=1)

            seed_pct = seed_pct.dropna(how="any").cumsum()
            idx = seed_pct.index
            x_vals = np.arange(len(idx))

            for j, col in enumerate(seed_pct.columns):
                ax.plot(x_vals, seed_pct[col], label=f"Seed {seeds[j]}", color=cmap(j), linewidth=1.6)

            ax.set_title(f"{phase.capitalize()} cumulative % returns – FQI iteration {labels[i]}")
            ax.set_ylabel("Cumulative % P&L")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1, 1))
            ax.grid(color="gray", linestyle=(0, (5, 5)), linewidth=0.5)
            ax.margins(x=0)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, integer=True))
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda v, _, idx=idx: idx[int(v)].strftime("%Y-%m-%d") if 0 <= int(v) < len(idx) else ""
                )
            )
            ax.legend(title="Random seed", ncol=min(len(seeds), 5))

        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f"Returns_AllSeeds_{phase}_percentage.png"), dpi=200)
        plt.close(fig)

    @classmethod
    def plot_percentage_pl(cls,
                 phase: str,
                 iterations: int,
                 seeds: typing.List[int],
                 save_path: str,
                 persistence: int,
                 bootstrapped = False,
                 remove_costs = False):
        # TODO change data_path
        data_path = "/home/a2a/a2a/RL_Trading/prj/app/core/data/M1_ICE.csv"
        #data_path = "/data2/bonetti_a2a/a2a_riccardo/RL_Trading/prj/app/core/data/M1_ICE.csv"
        data_df_orig = pd.read_csv(data_path)
        data_df_orig = data_df_orig[['timestamp', 'ask_0', 'bid_0']]
        data_df_orig['timestamp'] = pd.to_datetime(data_df_orig['timestamp'])

        fig, ax = plt.subplots(figsize=(30, 7))
        cmap = cm.get_cmap('tab10', iterations)

        first = True

        for iteration in range(1, iterations + 1):
            percentages_df = []
            for seed in seeds:
                file = os.path.join(save_path, f'seed{seed}/Results_iter{iteration}_{phase}.csv')
                results_df = pd.read_csv(file)
                trajectories = results_df.trajectory.unique()
                results_df['date'] = pd.to_datetime(results_df['day'], format='%Y%m%d')
                delta = pd.to_timedelta(results_df['minute'], unit='m')
                results_df['timestamp'] = results_df['date'] + delta
                results_df = results_df[['timestamp', 'action', 'trajectory']]

                traj_percent_df = []
                for trajectory in trajectories:
                    traj_df = results_df[results_df.trajectory==trajectory]

                    last_per_day = traj_df.groupby(traj_df['timestamp'].dt.date)['timestamp'].max()
                    extra_rows = pd.DataFrame({
                        'timestamp': last_per_day + pd.Timedelta(minutes=persistence),
                        'action': 0
                    })
                    #print(extra_rows)
                    traj_df = (pd.concat([traj_df, extra_rows], ignore_index=True)
                                  .sort_values('timestamp')
                                  .reset_index(drop=True))


                    data_df = data_df_orig[data_df_orig['timestamp'].isin(traj_df['timestamp'].unique())]

                    df = traj_df.merge(data_df, on='timestamp', how='left')
                    df = df.set_index('timestamp')

                    if remove_costs:
                        df['mid'] = (df['bid_0'] + df['ask_0']) / 2
                        out, pos, entry = [], 0, np.nan
                        for action, mid in df[['action', 'mid']].values:
                            perc = 0
                            if pos and action != pos:
                                perc = mid / entry - 1 if pos == 5 else -(mid / entry - 1)
                                entry = np.nan
                            if action and action != pos:
                                entry = mid
                            pos = action
                            out.append(perc)

                    else:
                        out, pos, entry = [], 0, np.nan
                        for action, bid, ask in df[['action', 'bid_0', 'ask_0']].values:
                            perc = 0
                            if pos and action != pos:
                                perc = (bid - 0.00625) / entry - 1 if pos == 5 else -((ask + 0.00625) / entry - 1)
                                entry = np.nan
                            if action and action != pos:
                                entry = (bid - 0.00625) if action == -5 else (ask + 0.00625)
                            pos = action
                            out.append(perc)

                    traj_percent_df.append(pd.Series(out, index=df.index))
                traj_percent_df = pd.concat(traj_percent_df, axis=1).mean(axis=1)
                percentages_df.append(traj_percent_df)


            percent_df = pd.concat(percentages_df, axis=1)
            percent_df = percent_df.dropna(how='any')
            idx = percent_df.index
            x = range(len(percent_df))

            if bootstrapped:
                med, q5, q95 = boot_median_iqr_scipy(percent_df.cumsum().to_numpy().T)
            else:
                med = None

            ax.plot(x, med, color=cmap(iteration - 1), label=f'Iter {iteration}')
            ax.fill_between(x, q5, q95, color=cmap(iteration - 1), alpha=.3)

        ax.set_title(f'Cumulative percentage returns({phase.lower()})')
        ax.set_ylabel('Cumulative % P&L')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1, 1))
        ax.grid(color='gray', linestyle=(0, (5, 5)))
        ax.margins(x=0)
        ax.xaxis.set_major_locator(mticker.IndexLocator(len(idx) // 10, 0))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: idx[int(v)].strftime('%Y-%m-%d')
            if 0 <= int(v) < len(idx) else ''))
        ax.legend()
        suffix = '_bootstrapped' if bootstrapped else ''
        fig.savefig(os.path.join(save_path, f'Cum_percentage_{phase + suffix}.png'), dpi=200)
        plt.close(fig)





    @classmethod
    def plot_best_q_values(
            cls,
            phase: str,
            iterations: int,
            seeds: typing.List[int],
            save_path: str,
    ):
        # Get a colormap
        cmap = cm.get_cmap('tab10', iterations)
        for j, seed in enumerate(seeds):
            n_rows, n_cols = cls._find_n_rows_n_cols(iterations)
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 7*n_rows))
            axs = np.array(axs).flatten()
            seed_path = os.path.join(save_path, f'seed{seed}')
            min_scale = +np.inf
            max_scale = -np.inf
            for i in range(iterations):
                iteration = i + 1
                _df = cls._read_csv_results(os.path.join(seed_path, f'Results_iter{iteration}_{phase}.csv'))
                index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp']
                q = _df.groupby(by='trajectory')['Q'].apply(list).to_list()
                means = np.mean(q, axis=0)
                stds = np.std(q, axis=0)
                n = len(means)
                x = range(n)
                # Calculate max and min q values to make a common axis for each Q plots
                if means.min() < min_scale:
                    min_scale = means.min()
                if means.max() > max_scale:
                    max_scale = means.max()
                axs[i].plot(x, means, color=cmap(i))
                axs[i].fill_between(x, means - stds, means - stds, alpha=0.3, color=cmap(i), linestyle='--')
                axs[i].set_title(f"Iteration {iteration}")
            # Format plots
            for ax in axs:
                ax.set_ylim(ymin=min_scale, ymax=max_scale)
                ax.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
                ax.margins(x=0)
                ax.set_ylabel("Best Q function estimate")
                ax.xaxis.set_major_locator(mticker.IndexLocator(n // 10, 0))
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].strftime('%Y-%m-%d')))
            fig.suptitle(f"{phase} set, seed {seed}")
            fig.tight_layout()
            fig.savefig(os.path.join(seed_path, f'Q_best_{phase}.png'))
            plt.close(fig)

    @classmethod
    def plot_q_values(
            cls,
            phase: str,
            iterations: int,
            trajectory: int,
            seeds: typing.List[int],
            save_path: str,
    ):
        for seed in seeds:
            fig, axs = plt.subplots(3, iterations, figsize=(8*iterations, 21), sharex=True, sharey=True)
            seed_path = os.path.join(save_path, f'seed{seed}')
            min_scale = +np.inf
            max_scale = -np.inf
            for i in range(iterations):
                iteration = i + 1
                _df = cls._read_csv_results(os.path.join(seed_path, f'Results_iter{iteration}_Validation.csv'))
                index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp']
                # Calculate max and min q values to make a common axis for each Q plots
                if _df[['Q_short', 'Q_flat', 'Q_long']].min().min() < min_scale:
                    min_scale = _df[['Q_short', 'Q_flat', 'Q_long']].min().min()
                if _df[['Q_short', 'Q_flat', 'Q_long']].max().max() > max_scale:
                    max_scale = _df[['Q_short', 'Q_flat', 'Q_long']].max().max()
                n = _df.shape[0]
                x = range(n)
                if iterations > 1:
                    _axs = axs[:, i]
                else:
                    _axs = axs
                _axs[0].plot(x, _df['Q_short'])
                _axs[1].plot(x, _df['Q_flat'])
                _axs[2].plot(x, _df['Q_long'])
                for a in [-1, 0, 1]:
                    _axs[a].set_title(f"Action: {a} (iteration {iteration})")
            for ax in axs.flatten():
                ax.set_ylabel("Q function estimate")
                ax.margins(x=0)
                ax.set_ylim(ymin=min_scale, ymax=max_scale)
                ax.xaxis.set_major_locator(mticker.IndexLocator(n // 5, 0))
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].strftime('%Y-%m-%d')))
            fig.suptitle(f"{phase} set, seed {seed}, trajectory {trajectory}")
            fig.tight_layout()
            fig.savefig(os.path.join(seed_path, f'Q_traj{trajectory}_{phase}.png'))
            plt.close(fig)

    @staticmethod
    def plot_percentage_returns_unfilled_specific_iteration(
            specific_iteration: int,
            seeds: typing.List[int],
            save_path: str
	):
        fig, ax = plt.subplots(figsize=(16, 9))
        # Get a colormap
        cmap = cm.get_cmap('tab10', specific_iteration)
        # Plot a line with its confidence interval for each FQI iteration
        for i in [specific_iteration]:
            iteration = i + 1
            df = pd.DataFrame()
            for seed in seeds:
                _df = cls._read_csv_results(os.path.join(save_path, f'seed{seed}/Results_iter{iteration}_Validation.csv'))
                _df['return'] = _df.groupby(by='trajectory')['reward'].cumsum()
                index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp']
                data = np.mean(_df.groupby(by='trajectory')['return'].apply(list).to_list(), axis=0)
                df = pd.concat([df, pd.Series(data=data, index=index)], axis=1)
            n = df.shape[0]
            for c in df.columns:
                ax.plot(range(n), df[c], color=cmap(i))
            # ax.plot(range(n), means, color=cmap(i), label=f'FQI Iteration {iteration}')
            # if df.shape[1] > 1:
            # stds = df.std(axis=1)
            # ax.fill_between(range(n), means - stds, means + stds, alpha=0.3, color=cmap(i), linestyle='--')
            # ax.plot(range(n), df, color=cmap(i), linestyle="--")
        ax.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
        ax.set_title("Validation return")
        ax.set_ylabel("Cumulative reward")
        #ax.legend()
        ax.margins(x=0)
        ax.xaxis.set_major_locator(mticker.IndexLocator(n // 10, 0))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].strftime('%Y-%m-%d')))
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
        fig.savefig(os.path.join(save_path, f'returns_CI_unfilled_iteration_{specific_iteration + 1}.png'), dpi=200)
        plt.close(fig)

    @staticmethod
    def plot_percentage_returns_unfilled(
            iterations: int,
            seeds: typing.List[int],
            save_path: str
	):
        fig, ax = plt.subplots(figsize=(16, 9))
        # Get a colormap
        cmap = cm.get_cmap('tab10', iterations)
        # Plot a line with its confidence interval for each FQI iteration
        for i in range(iterations):
            iteration = i + 1
            df = pd.DataFrame()
            for seed in seeds:
                _df = cls._read_csv_results(os.path.join(save_path, f'seed{seed}/Results_iter{iteration}_Validation.csv'))
                _df['return'] = _df.groupby(by='trajectory')['reward'].cumsum()
                index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp']
                data = np.mean(_df.groupby(by='trajectory')['return'].apply(list).to_list(), axis=0)
                df = pd.concat([df, pd.Series(data=data, index=index)], axis=1)
            n = df.shape[0]
            for c in df.columns:
                ax.plot(range(n), df[c], color=cmap(i))
            # ax.plot(range(n), means, color=cmap(i), label=f'FQI Iteration {iteration}')
            # if df.shape[1] > 1:
            # stds = df.std(axis=1)
            # ax.fill_between(range(n), means - stds, means + stds, alpha=0.3, color=cmap(i), linestyle='--')
            # ax.plot(range(n), df, color=cmap(i), linestyle="--")
        ax.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
        ax.set_title("Validation return")
        ax.set_ylabel("Cumulative reward")
        ax.legend()
        ax.margins(x=0)
        ax.xaxis.set_major_locator(mticker.IndexLocator(n // 10, 0))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].strftime('%Y-%m-%d')))
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
        fig.savefig(os.path.join(save_path, 'returns_CI_unfilled.png'), dpi=200)
        plt.close(fig)

    @staticmethod
    def plot_heatmap(
            results: pd.DataFrame,
            phase: str,
            iteration: int,
            save_path: typing.Optional[str] = None,
            ax: typing.Optional[matplotlib.axes.SubplotBase] = None
    ):
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 9))
        df = results.pivot_table(index='day', columns='minute', values='action', dropna=False)
        df.columns = [f'{int(i // 60):02d}:{int(i % 60):02d}' for i in df.columns]
        cmap = LinearSegmentedColormap.from_list('RedWhiteGreen', ['crimson', 'white', 'green'])
        g = sns.heatmap(df, mask=df.isnull(), cmap=cmap, vmin=-1, vmax=1, xticklabels=60, ax=ax)
        g.set_facecolor('black')
        cbar = g.collections[0].colorbar
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(['Short', 'Flat', 'Long'])
        ax.set_title(f"{phase} set (iteration {iteration})", )
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        ax.yaxis.label.set_visible(False)
        if save_path is not None:
            assert fig is not None, "Cannot save path if figure is None."
            fig.savefig(os.path.join(save_path, f'Allocation_iter{iteration}_{phase}.png'), dpi=200)
            plt.close(fig)

    @classmethod
    def plot_action_diff_blocks(
            cls,
            results_a: pd.DataFrame,
            results_b: pd.DataFrame,
            minutes_a: typing.List[int],
            minutes_b: typing.List[int],
            phase: str,
            iteration: int,
            save_path: str,
            vmin: float = -5,
            vmax: float = 5,
    ) -> None:
        """Create one subplot per *(trajectory_a, trajectory_b)* pair, visualising
        the timestamps where *action_a != action_b*.
        """
        # --------------------------------------------------------------
        # figure scaffold – *wider* than tall
        # --------------------------------------------------------------
        n_rows, n_cols = cls._find_n_rows_n_cols(len(minutes_a))
        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6 * n_cols, 11 * n_rows),  # wider, shorter
        )
        axs = np.array(axs).flatten()

        cmap = LinearSegmentedColormap.from_list(
            "r_g", [(0, "red"), (0.5, "grey"), (1, "green")], N=256
        )
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        for i_ax, (t_a, t_b) in enumerate(zip(minutes_a, minutes_b)):
            ax = axs[i_ax]

            # ----------------------------------------------------------
            # prepare data
            # ----------------------------------------------------------
            df_a = results_a[results_a["trajectory"] == t_a].copy()
            df_b = results_b[results_b["trajectory"] == t_b].copy()

            df_a["minute_index"] = df_a["minute"] - t_a
            df_b["minute_index"] = df_b["minute"] - t_b

            df_a["action"] = pd.to_numeric(df_a["action"], errors="coerce")
            df_b["action"] = pd.to_numeric(df_b["action"], errors="coerce")

            merged = df_a.merge(df_b, on=["day", "minute_index"], suffixes=("_a", "_b"))
            differing = merged[merged["action_a"] != merged["action_b"]].copy()

            if differing.empty:
                ax.axis("off")
                ax.set_title(f"Traj {t_a} vs {t_b} – no differing actions")
                continue

            pivot_a = differing.pivot(index="day", columns="minute_index", values="action_a")
            pivot_b = differing.pivot(index="day", columns="minute_index", values="action_b")

            # interleave rows so each day gets two half‑rows
            stacked_rows, stacked_index = [], []
            for day in pivot_a.index:
                stacked_rows.append(pivot_a.loc[day])
                stacked_index.append(f"{day} A")
                stacked_rows.append(pivot_b.loc[day])
                stacked_index.append(f"{day} B")

            heat_data = pd.DataFrame(stacked_rows, index=stacked_index)

            # ----------------------------------------------------------
            # heatmap – aspect "auto" (rectangular cells)
            # ----------------------------------------------------------
            sns.heatmap(
                heat_data,
                cmap=cmap,
                norm=norm,
                ax=ax,
                cbar=False,
                linewidths=0,
                square=False,  # ← allow rectangular cells
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_aspect("auto")

            # ----------------------------------------------------------
            # axis formatting
            # ----------------------------------------------------------
            num_cols = heat_data.shape[1]
            tick_positions = np.arange(num_cols)[::2] + 0.5
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(
                [f"{int(c // 60):02d}:{int(c % 60):02d}" for c in heat_data.columns][::2],
                rotation=45,
                ha="right",
            )

            n_days = len(pivot_a.index)
            label_positions = [2 * d + 1 for d in range(n_days) if d % 3 == 0]
            label_strings = [str(pivot_a.index[d]) for d in range(n_days) if d % 3 == 0]
            ax.set_yticks(label_positions)
            ax.set_yticklabels(label_strings, rotation=0, font={'size': 7})
            ax.tick_params(axis="y", length=0)

            # ----------------------------------------------------------
            # separator lines between days
            # ----------------------------------------------------------
            n_days = len(pivot_a.index)
            xmin, xmax = ax.get_xlim()
            for k in range(1, n_days):
                y = 2 * k
                ax.hlines(
                    y=y,
                    xmin=xmin,
                    xmax=xmax,
                    colors="black",
                    linewidth=0.6,
                    clip_on=False,
                )

            ax.set_title(f"Policy actions where they differ")

        # hide unused axes
        for j in range(i_ax + 1, len(axs)):
            axs[j].set_visible(False)

        fig.tight_layout()
        os.makedirs(save_path, exist_ok=True)
        outfile = os.path.join(save_path, f"ActionDiff_iter{iteration}_{phase}.png")
        fig.savefig(outfile, dpi=150)
        plt.close(fig)

        print(f"Figure saved → {outfile}")

    @classmethod
    def plot_action_diff_rewards(
            cls,
            results_a: pd.DataFrame,
            results_b: pd.DataFrame,
            minutes_a: typing.List[int],
            minutes_b: typing.List[int],
            phase: str,
            iteration: int,
            save_path: str
    ):
        n_rows, n_cols = cls._find_n_rows_n_cols(len(minutes_a))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 12 * n_rows))
        axs = np.array(axs).flatten()

        # Colormap for reward difference
        #cmap = sns.diverging_palette(10, 135, center='dark', as_cmap=True)  # Blue-white-red
        cmap = sns.color_palette(palette='RdBu', as_cmap=True)

        for i, (trajectory_a, trajectory_b) in enumerate(zip(minutes_a, minutes_b)):
            df_a = results_a[results_a['trajectory'] == trajectory_a]
            df_b = results_b[results_b['trajectory'] == trajectory_b]

            df_a['minute_index'] = df_a['minute'] - trajectory_a
            df_b['minute_index'] = df_b['minute'] - trajectory_b

            merged = df_a.merge(
                df_b,
                on=['day', 'minute_index'],
                suffixes=('_a', '_b')
            )
            # Only keep where actions differ
            differing = merged[merged['action_a'] != merged['action_b']].copy()

            if differing.empty:
                axs[i].axis('off')
                axs[i].set_title(f"Minutes {trajectory_a} and {trajectory_b} (no diffs)")
                continue

            # Compute reward difference
            differing['reward_diff'] = differing['reward_nostd_a'] - differing['reward_nostd_b']

            # Pivot table to plot
            pivot = differing.pivot(index='day', columns='minute_index', values='reward_diff')
            pivot.columns = [f'{int(j // 60):02d}:{int(j % 60):02d}' for j in pivot.columns]

            g = sns.heatmap(
                pivot,
                cmap=cmap,
                center=0,
                ax=axs[i],
                cbar=True,
                #linewidths=0.5,
                #linecolor='black'
            )

            axs[i].set_title(f"Difference in p&l where action is different\nMinutes {trajectory_a} and {trajectory_b}")
            axs[i].tick_params(axis='x', rotation=45)
            axs[i].tick_params(axis='y', rotation=0)
            axs[i].yaxis.label.set_visible(False)

        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f'ActionRewardDiff_iter{iteration}_{phase}.png'))
        plt.close(fig)

    @classmethod
    def plot_actions(
            cls,
            results: pd.DataFrame,
            minutes: typing.List[int],
            phase: str,
            iteration: int,
            save_path: str
    ):
        n_rows, n_cols = cls._find_n_rows_n_cols(len(minutes))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 8*n_rows))
        axs = np.array(axs).flatten()
        cmap = LinearSegmentedColormap.from_list('RedWhiteGreen', ['crimson', 'white', 'green'])
        for i, trajectory in enumerate(minutes):
            _df = results[results['trajectory'] == trajectory]
            _df = _df.pivot_table(index='day', columns='minute', values='action', dropna=False)
            _df.columns = [f'{int(j // 60):02d}:{int(j % 60):02d}' for j in _df.columns]
            g = sns.heatmap(_df, mask=_df.isnull(), cmap=cmap, vmin=-1, vmax=1, ax=axs[i], cbar=False)
            g.set_facecolor('black')
            axs[i].set_title(f"Minute {trajectory}")
            axs[i].tick_params(axis='x', rotation=45)
            axs[i].tick_params(axis='y', rotation=0)
            axs[i].yaxis.label.set_visible(False)
        #fig.suptitle(f"{phase} set (iteration {iteration})", weight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f'Actions_iter{iteration}_{phase}.png'))
        plt.close(fig)
    @classmethod
    def plot_actions_weekly(
            cls,
            results: pd.DataFrame,
            minutes: typing.List[int],
            phase: str,
            iteration: int,
            save_path: str
    ):
        n_rows, n_cols = cls._find_n_rows_n_cols(len(minutes))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(12*n_cols, 8*n_rows))
        axs = np.array(axs).flatten()
        cmap = LinearSegmentedColormap.from_list('RedWhiteGreen', ['crimson', 'white', 'green'])
        for i, trajectory in enumerate(minutes):
            _df = results[results['trajectory'] == trajectory]
            _df = _df.copy() #copy to remove warning
            _df['Date'] = pd.to_datetime(_df['day'].astype(str), format='%Y%m%d')
            _df['week'] = _df['Date'].dt.isocalendar().week
            _df['day_of_week'] = _df['Date'].dt.dayofweek
            _df = _df.drop(columns = ['Date', 'day'])
            _df = _df.pivot_table(index='week', columns=['day_of_week', 'minute'], values='action', dropna=False)
            _df.columns = [f'{int(j[1] // 60):02d}:{int(j[1] % 60):02d}' for j in _df.columns]
            g = sns.heatmap(_df, mask=_df.isnull(), cmap=cmap, vmin=-1, vmax=1, ax=axs[i], cbar=False)
            g.set_facecolor('black')
            axs[i].set_title(f"Minute {trajectory}")
            axs[i].tick_params(axis='x', rotation=45)
            axs[i].tick_params(axis='y', rotation=0)
            axs[i].yaxis.label.set_visible(False)
        #fig.suptitle(f"{phase} set (iteration {iteration})", weight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f'Actions_iter{iteration}_{phase}_weekly.png'))
        plt.close(fig)
    @staticmethod
    def _find_n_rows_n_cols(n_plots: int) -> typing.Tuple[int, int]:
        n_rows = int(n_plots ** 0.5)
        while n_plots % n_rows != 0:
            n_rows -= 1
        return n_rows, n_plots // n_rows

    @staticmethod
    def plot_feature_importances(
            regressors: typing.List[typing.Dict[str, xgboost.XGBRegressor]],
            features: typing.List[str],
            iteration: int,
            save_path: str,
    ):
        n_rows = len(regressors)
        n_cols = len(regressors[0])
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(16*n_cols, 16*n_rows))
        axs = axs.flatten()
        for i, reg in enumerate(regressors):
            title_suffix = ''
            if n_rows > 1:  # Double Q
                title_suffix = f' (Regressor {i})'
            for j, (action, regressor) in enumerate(reg.items()):
                axs[j + i * n_cols].bar(features, regressor.feature_importances_)
                axs[j + i * n_cols].set_title(f'Action: {action}{title_suffix}')
                axs[j + i * n_cols].margins(x=0)
                axs[j + i * n_cols].tick_params(axis='x', rotation=90)
                axs[j + i * n_cols].grid(alpha=0.5, linestyle=':')
        fig.suptitle(f"Iteration {iteration}", weight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f'FI_iter{iteration}.png'))#, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def plot_loss(loss, dates, iteration, save_path):
        n_cols = len(loss)
        v_max = max(loss[-1].max(), loss[0].max(), loss[1].max())
        v_min = min(loss[-1].min(), loss[0].min(), loss[1].min())
        fig, axs = plt.subplots(1, n_cols, figsize=(8 * n_cols, 8))
        fig.suptitle(f"Loss")
        for i in loss.keys():
            axs[i].set_title(f"Loss for Q function estimator of action = {[-1, 0, 1][i]}")
            axs[i].plot(range(len(loss[i])), loss[i])
            axs[i].set_ylim(ymin=v_min, ymax=v_max)
            axs[i].xaxis.set_major_locator(mticker.IndexLocator(dates.shape[0] // 5, 0))
            axs[i].xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, pos: dates.iloc[int(x)].strftime('%Y-%m-%d')))
        fig.savefig(os.path.join(save_path, f'Loss_{iteration}.png'))  # , bbox_inches='tight')

    @classmethod
    def plot_costs_pnl(
            cls,
            phase: str,
            iterations: int,
            seeds: typing.List[int],
            save_path: str
	):
        #costs
        fig, ax = plt.subplots(figsize=(16, 9))
        # Get a colormap
        cmap = cm.get_cmap('tab10', iterations)
        # Plot a line with its confidence interval for each FQI iteration
        for i in range(iterations):
            iteration = i + 1
            df = pd.DataFrame()
            for seed in seeds:
                _df = cls._read_csv_results(os.path.join(save_path, f'seed{seed}/Results_iter{iteration}_{phase}.csv'))
                _df['costs'] = _df.groupby(by='trajectory')['cost'].cumsum()
                index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp']
                data = np.mean(_df.groupby(by='trajectory')['costs'].apply(list).to_list(), axis=0)
                df = pd.concat([df, pd.Series(data=data, index=index)], axis=1)
            n = df.shape[0]
            for c in df.columns:
                ax.plot(range(n), df[c], color=cmap(i))

        ax.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
        ax.set_title("Validation cost")
        ax.set_ylabel("Cost")
        ax.legend()
        ax.margins(x=0)
        ax.xaxis.set_major_locator(mticker.IndexLocator(n // 10, 0))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].strftime('%Y-%m-%d')))
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
        fig.savefig(os.path.join(save_path, f'Costs_CI_{phase}.png'), dpi=200)
        plt.close(fig)

        #pnl
        fig, ax = plt.subplots(figsize=(16, 9))
        # Get a colormap
        cmap = cm.get_cmap('tab10', iterations)
        # Plot a line with its confidence interval for each FQI iteration
        for i in range(iterations):
            iteration = i + 1
            df = pd.DataFrame()
            for seed in seeds:
                _df = cls._read_csv_results(
                    os.path.join(save_path, f'seed{seed}/Results_iter{iteration}_Validation.csv'))
                _df['pnl'] = _df.groupby(by='trajectory')['pnl'].cumsum()
                index = _df[_df['trajectory'] == _df.loc[0, 'trajectory']]['timestamp']
                data = np.mean(_df.groupby(by='trajectory')['pnl'].apply(list).to_list(), axis=0)
                df = pd.concat([df, pd.Series(data=data, index=index)], axis=1)
            n = df.shape[0]
            for c in df.columns:
                ax.plot(range(n), df[c], color=cmap(i))

        ax.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
        ax.set_title("Validation P&L without costs")
        ax.set_ylabel("P&L")
        ax.legend()
        ax.margins(x=0)
        ax.xaxis.set_major_locator(mticker.IndexLocator(n // 10, 0))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: index.iloc[int(x)].strftime('%Y-%m-%d')))
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
        fig.savefig(os.path.join(save_path, f'PNL_CI_{phase}.png'), dpi=200)
        plt.close(fig)


if __name__ == '__main__':


    Plotter.plot_percentage_returns(phase='Train', iterations=2, seeds = [2192,26500,34766,35080,47976,56517,69233,79126,92477,97511], save_path = "C:/Users/Riccardo/OneDrive - Politecnico di Milano/Webeep/Thesis/RL_Trading/results/with_halfopt_30min_5_std_dev_0.005")