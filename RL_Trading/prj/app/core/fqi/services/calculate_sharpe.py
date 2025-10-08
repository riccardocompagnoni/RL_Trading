import json
import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


def calculate_sharpe(path, seeds, phase, data_path, confidence_level = 0.95):
    sharpes = []
    volatilies = []
    vars = []
    cvars = []

    df_data = pd.read_csv(data_path)

    df_data = df_data[['timestamp', 'ask_0', 'bid_0']]
    df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])
    df_data['mid'] = (df_data.ask_0 + df_data.bid_0)/2
    df_data['date'] = df_data.timestamp.dt.date
    df_data = df_data.drop(columns=['ask_0', 'bid_0', 'timestamp'])
    df_data = df_data.groupby(['date']).mean().reset_index()


    for seed in seeds:
        iterations_sharpes = []
        iterations_volatilies = []
        iterations_var = []
        iterations_cvar = []
        folder = os.path.join(path, f'seed{seed}')
        files = [
            fname
            for fname in os.listdir(folder)
            if fname.startswith('Results_iter') and fname.endswith(f'{phase}.csv')
        ]
        files.sort(key=lambda f: int(re.search(r'Results_iter(\d+)_', f).group(1)))

        for fname in files:
            full_path = os.path.join(folder, fname)
            df = pd.read_csv(full_path)
            df = df[['day', 'reward_nostd']]
            n_days_sqrt = np.sqrt(len(df.day.unique()))
            df = df.groupby('day').sum().reset_index()
            sharpe = n_days_sqrt*df['reward_nostd'].mean() / df['reward_nostd'].std()
            vol = n_days_sqrt*df['reward_nostd'].std()

            df['date'] = pd.to_datetime(df['day'], format='%Y%m%d').dt.date
            df = df.merge(df_data, on='date')
            df['norm_daily_reward'] = df['reward_nostd']/df['mid']
            sorted_returns = np.sort(df['norm_daily_reward'].to_numpy())

            index = int((1 - confidence_level) * len(sorted_returns))

            print(len(sorted_returns))
            var = sorted_returns[index]
            cvar = np.mean(sorted_returns[:index])


            iterations_sharpes.append(sharpe)
            iterations_volatilies.append(vol)
            iterations_var.append(var)
            iterations_cvar.append(cvar)

        sharpes.append(iterations_sharpes)
        volatilies.append(iterations_volatilies)
        vars.append(iterations_var)
        cvars.append(iterations_cvar)


    sharpes = np.array(sharpes)
    volatilities = np.array(volatilies)
    vars = np.array(vars)
    cvars = np.array(cvars)

    quantiles_25 = np.quantile(sharpes, 0.25, axis=0)
    quantiles_75 = np.quantile(sharpes, 0.75, axis=0)
    medians = np.median(sharpes, axis=0)
    print(f'Sharpes for {phase.lower()}:\n')
    for i in range(sharpes.shape[1]):
        print(f'It{i+1}: {medians[i]:.2f} (IQR: {quantiles_25[i]:.2f}  {quantiles_75[i]:.2f})')

    print('\n\n\n')



    return sharpes, volatilities, vars, cvars


def plot_all_metrics(sharpes, volatilities, vars, cvars, path, seeds, alpha_label='5%', figsize=(6,3), dpi=150):


    n_seeds = sharpes.shape[0]
    for arr in (volatilities, vars, cvars):
        if arr.shape[0] != n_seeds:
            raise ValueError("All metric arrays must have the same length.")
    if len(seeds) != n_seeds:
        raise ValueError("`seeds` length must match metrics length.")

    os.makedirs(path, exist_ok=True)


    x = np.arange(n_seeds)

    def _barplot(values, title, ylabel, filename, y_pct=False):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.bar(x, values)
        ax.set_xticks(x)
        ax.set_xticklabels(seeds, rotation=0)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Seed")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        if y_pct:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        fig.tight_layout()
        out = os.path.join(path, filename)
        fig.savefig(out)
        plt.close(fig)


    _barplot(
        sharpes,
        title="Sharpe by seed",
        ylabel="Sharpe",
        filename="sharpe.png",
    )

    _barplot(
        volatilities,
        title="Volatility by seed",
        ylabel="Volatility",
        filename="volatility.png",
    )

    _barplot(
        vars,
        title=f"VaR {alpha_label} by seed",
        ylabel=f"VaR {alpha_label}",
        filename=f"var_{alpha_label.replace('%', 'pct')}.png",
        y_pct=True
    )

    _barplot(
        cvars,
        title=f"CVaR {alpha_label} by seed",
        ylabel=f"CVaR {alpha_label}",
        filename=f"cvar_{alpha_label.replace('%', 'pct')}.png",
        y_pct=True
    )





def plot_sharpes(sharpes, path, seeds):
    #sharpes = sharpes[:, :3]
    n_seeds, n_iters = sharpes.shape

    plt.figure(figsize = (7, 5))

    for i in range(n_seeds):
        plt.scatter(
            np.arange(1, n_iters+1),
            sharpes[i],
            label=f"seed {seeds[i]}",
            alpha=0.7
        )

    plt.xlabel("Iteration")
    plt.ylabel("Sharpe ratio")
    plt.title("Sharpe ratios by seed and iteration")
    plt.legend(title="Seeds", ncol=2, fontsize=8)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'sharpe_ratios.png'))
    plt.show()



def calculate_sharpe_oamp(path, seeds, iteration, phase):
    sharpes = []
    paths = [os.path.join(path, f'seed{seed}', f'Results_iter{iteration}_{phase}.csv') for seed in seeds]
    paths.append(os.path.join(path, 'oamp_optuna', 'oamp_log.csv'))
    for full_path in paths:

        df = pd.read_csv(full_path)
        n_days_sqrt = np.sqrt(len(df.day.unique()))
        df = df.groupby('day').sum()
        sharpe = n_days_sqrt*df['reward_nostd'].mean() / df['reward_nostd'].std()

        sharpes.append(sharpe)

    sharpes = np.array(sharpes)

    quantile_25 = np.quantile(sharpes, 0.25, axis=0)
    quantile_75 = np.quantile(sharpes, 0.75, axis=0)
    median = np.median(sharpes, axis=0)
    print(f'Sharpes for {phase.lower()}:\n')
    print(f'{median:.2f} (IQR: {quantile_25:.2f}  {quantile_75:.2f})')

    print('\n\n\n')

    return sharpes


def plot_sharpes_oamp(sharpes, path, seeds):
    #sharpes = sharpes[:, :3]
    labels = [str(seed) for seed in seeds]
    labels.append('oamp')

    sharpes, labels = (list(t) for t in zip(*sorted(zip(sharpes, labels))))

    plt.figure(figsize = (9, 5))
    plt.scatter(
        np.arange(1, len(labels)+1),
        sharpes,
        alpha=0.7
    )
    plt.xlabel("Seed")
    plt.ylabel("Sharpe ratio")
    plt.title("Sharpe ratios by seed")
    plt.xticks(np.arange(1, len(labels)+1), labels)
    plt.grid()
    #plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'sharpe_ratios_oamp.png'))
    plt.show()


def boh():
    paths = ['/home/a2a/a2a/RL_Trading/results/asym_reward0.025_overtrue',
             '/home/a2a/a2a/RL_Trading/results/asym_reward_cost0.05',
             '/home/a2a/a2a/RL_Trading/results/asym_reward_cost0.1',
             '/home/a2a/a2a/RL_Trading/results/asym_reward_cost0.5',
             '/home/a2a/a2a/RL_Trading/results/asym_reward_cost1',
             '/home/a2a/a2a/RL_Trading/results/asym_reward_cost2']

    paths = ['/home/a2a/a2a/RL_Trading/results/sym_reward_quant0.2',
             '/home/a2a/a2a/RL_Trading/results/sym_reward_quant0.25',
             '/home/a2a/a2a/RL_Trading/results/sym_reward_quant0.35',
             '/home/a2a/a2a/RL_Trading/results/sym_reward_quant0.4_overtrue'
             ]

    labels = ['0.025', '0.05', '0.1', '0.5', '1.0', '2.0']
    labels = ['0.2', '0.25', '0.35', '0.4']


    data_path = '/home/a2a/a2a/RL_Trading/prj/app/core/data/M1_ICE.csv'
    save_path = '/home/a2a/a2a/RL_Trading/results/quant_stats'


    data = {'sharpes': [],
            'volatilities': [],
            'vars': [],
            'cvars': []
            }

    alpha_label = '5%'
    figsize = (6, 3)
    dpi = 150

    for path in paths:
        seeds = []
        for elem in os.listdir(path):
            if elem.startswith('seed'):
                seeds.append(int(elem.replace('seed', '')))

        sharpes, volatilities, vars, cvars = calculate_sharpe(path, seeds, 'Validation', data_path)
        data['sharpes'].append(np.mean(sharpes[:, -1]))
        data['volatilities'].append(np.mean(volatilities[:, -1]))
        data['vars'].append(np.mean(vars[:, -1]))
        data['cvars'].append(np.mean(cvars[:, -1]))

    x = np.arange(len(paths))

    def _barplot(values, title, ylabel, filename, y_pct=False):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.bar(x, values)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Experiment")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        if y_pct:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        fig.tight_layout()
        out = os.path.join(save_path, filename)
        fig.savefig(out)
        plt.close(fig)

    _barplot(
        data['sharpes'],
        title="Sharpe by experiment",
        ylabel="Sharpe",
        filename="sharpe.png",
    )

    _barplot(
        data['volatilities'],
        title="Volatility by experiment",
        ylabel="Volatility",
        filename="volatility.png",
    )

    _barplot(
        data['vars'],
        title=f"VaR {alpha_label} by experiment",
        ylabel=f"VaR {alpha_label}",
        filename=f"var_{alpha_label.replace('%', 'pct')}.png",
        y_pct=True
    )

    _barplot(
        data['cvars'],
        title=f"CVaR {alpha_label} by experiment",
        ylabel=f"CVaR {alpha_label}",
        filename=f"cvar_{alpha_label.replace('%', 'pct')}.png",
        y_pct=True
    )


if __name__ == '__main__':
    #boh()

    path = '/home/a2a/a2a/RL_Trading/results/experts_overnight/1y23_7y23'
    data_path = '/home/a2a/a2a/RL_Trading/prj/app/core/data/M1_ICE.csv'

    train = False
    validation = True
    test = False
    oamp = False
    iteration = 13

    seeds = []
    for elem in os.listdir(path):
        if elem.startswith('seed'):
            seeds.append(int(elem.replace('seed', '')))

    if validation:
        if oamp:
            sharpes = calculate_sharpe_oamp(path, seeds, iteration,'Test')
            plot_sharpes_oamp(sharpes, path, seeds)
        else:
            sharpes, volatilities, vars, cvars = calculate_sharpe(path, seeds, 'Validation', data_path)

            #plot_sharpes(sharpes, path, seeds)
            plot_all_metrics(sharpes[:, -1], volatilities[:, -1], vars[:, -1], cvars[:, -1], path, seeds)



    if train:
        if oamp:
            sharpes = calculate_sharpe_oamp(path, seeds, iteration,'Train')
            plot_sharpes_oamp(sharpes, path, seeds)
        else:
            calculate_sharpe(path, 'Train')

    if test:
        if oamp:
            path = os.path.join(path, 'test_next_year')
            sharpes = calculate_sharpe_oamp(path, seeds, iteration,'Test')
            plot_sharpes_oamp(sharpes, path, seeds)
        else:
            with open(os.path.join(path, 'parameters_opt.json')) as f:
                iterations = json.load(f)['iterations']
            path = os.path.join(path, 'test_next_year_no_retrain/2022')
            sharpes, volatilities, vars, cvars = calculate_sharpe(path, 'Test', data_path)
            plot_all_metrics(sharpes[:, iterations], volatilities[:, iterations], vars[:, iterations], cvars[:, iterations], path, seeds)
            #plot_sharpes(sharpes, path, seeds)





