import json
import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def calculate_volatility(path, phase):
    sharpes = []
    for seed in seeds:
        iterations = []
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
            n_days_sqrt = np.sqrt(len(df.day.unique()))
            df = df.groupby('day').sum()
            vol = n_days_sqrt*df['reward_nostd'].std()

            iterations.append(vol)

        sharpes.append(iterations)
    sharpes = np.array(sharpes)

    quantiles_25 = np.quantile(sharpes, 0.25, axis=0)
    quantiles_75 = np.quantile(sharpes, 0.75, axis=0)
    medians = np.median(sharpes, axis=0)
    print(f'Sharpes for {phase.lower()}:\n')
    for i in range(sharpes.shape[1]):
        print(f'It{i+1}: {medians[i]:.2f} (IQR: {quantiles_25[i]:.2f}  {quantiles_75[i]:.2f})')

    print('\n\n\n')

    return sharpes


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


if __name__ == '__main__':
    path = 'C:/Users/Riccardo/OneDrive - Politecnico di Milano/Webeep/Thesis/RL_Trading/results/delta_std_pers10_optnostd_extended_long/test_next_year'

    train = False
    validation = True
    test = False
    oamp = True
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
            sharpes = calculate_volatility(path, 'Validation')
            plot_sharpes(sharpes, path, seeds)


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
            path = os.path.join(path, 'test_next_year')
            sharpes = calculate_sharpe(path, 'Test')[:, :iterations]
            plot_sharpes(sharpes, path, seeds)




