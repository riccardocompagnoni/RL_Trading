import json

from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib import cm, ticker as mticker
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.normpath(os.path.join(current_dir, '..', '..', '..', '..', '..','..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..', 'RL_Trading', 'prj', 'app')))

from RL_Trading.prj.app.core.fqi.services.plotter import Plotter



def method_1():

    results_path = '/data2/bonetti_a2a/a2a_riccardo/RL_Trading/results/delta_std_pers30_optnostd_extended/seed95111/Results_iter3_Validation.csv'
    data_path = '/data2/bonetti_a2a/a2a_riccardo/RL_Trading/prj/app/core/data/M1_ICE.csv'
    save_path = ''

    results_df = pd.read_csv(results_path)
    data_df = pd.read_csv(data_path)
    data_df = data_df[['timestamp', 'ask_0', 'bid_0']]

    results_df['date'] = pd.to_datetime(results_df['day'], format='%Y%m%d')
    delta = pd.to_timedelta(results_df['minute'], unit='m')
    results_df['timestamp'] = results_df['date'] + delta
    results_df = results_df[['timestamp', 'action', 'reward_nostd']]

    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df['mid'] = (data_df['ask_0'] + data_df['bid_0'])/2
    data_df = data_df.drop(columns = ['ask_0', 'bid_0'])

    df = results_df.merge(data_df, on='timestamp', how='left')
    df['percentage_return'] = df['reward_nostd']/df['mid']
    df['cum_percentage_return'] = np.cumsum(df['percentage_return'])
    df = df.set_index('timestamp')



    ax = df[['cum_percentage_return']].plot(figsize = (16, 8))
    locator = mdates.AutoDateLocator(minticks=15, maxticks=20)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    yloc = mticker.MaxNLocator(nbins=12, min_n_ticks=8)
    ax.yaxis.set_major_locator(yloc)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))

    ax.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
    plt.title('Cumulative percentage returns')
    plt.ylabel('Percentage return')
    plt.xlabel('')
    plt.savefig(
        '/data2/bonetti_a2a/a2a_riccardo/RL_Trading/results/delta_std_pers30_optnostd_extended/seed95111/cum_plot.png')


def method_2():
    results_path = '/data2/bonetti_a2a/a2a_riccardo/RL_Trading/results/delta_std_pers30_optnostd_extended/seed95111/Results_iter3_Validation.csv'
    data_path = '/data2/bonetti_a2a/a2a_riccardo/RL_Trading/prj/app/core/data/M1_ICE.csv'
    save_path = ''

    results_df = pd.read_csv(results_path)
    data_df = pd.read_csv(data_path)
    data_df = data_df[['timestamp', 'ask_0', 'bid_0']]

    results_df['date'] = pd.to_datetime(results_df['day'], format='%Y%m%d')
    delta = pd.to_timedelta(results_df['minute'], unit='m')
    results_df['timestamp'] = results_df['date'] + delta
    results_df = results_df[['timestamp', 'action', 'reward_nostd']]

    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df['mid'] = (data_df['ask_0'] + data_df['bid_0'])/2
    #data_df = data_df.drop(columns = ['ask_0', 'bid_0'])

    df = results_df.merge(data_df, on='timestamp', how='left')
    df = df.set_index('timestamp')

    percentages = []
    prev_action = 0

    for index, row in df.iterrows():
        action = row['action']
        percentage = 0

        if action != prev_action:
            if prev_action == 5:
                percentage = ((row['ask_0'] - 0.00625) / starting_price) - 1
            elif prev_action == -5:
                percentage = ((row['bid_0'] + 0.00625) / starting_price) - 1

            if action == 5:
                starting_price = row['bid_0'] + 0.00625
            elif action == -5:
                starting_price = row['ask_0'] - 0.00625
        prev_action = action
        percentages.append(percentage)

    df['percentage_return'] = percentages
    df['cum_percentage_return'] = np.cumsum(df['percentage_return'])

    ax = df[['cum_percentage_return']].plot(figsize = (16, 8))
    locator = mdates.AutoDateLocator(minticks=15, maxticks=20)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    yloc = mticker.MaxNLocator(nbins=12, min_n_ticks=8)
    ax.yaxis.set_major_locator(yloc)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))

    ax.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
    plt.title('Cumulative percentage returns')
    plt.ylabel('Percentage return')
    plt.xlabel('')
    plt.savefig(
        '/data2/bonetti_a2a/a2a_riccardo/RL_Trading/results/delta_std_pers30_optnostd_extended/seed95111/cum_plot2.png')



def method_3():
    results_path = '/data2/bonetti_a2a/a2a_riccardo/RL_Trading/results/xgb_delta_both_0.0005_pers10_nest250_span1_featstd/test_next_year/seed88674/Results_iter3_Test.csv'
    data_path = '/data2/bonetti_a2a/a2a_riccardo/RL_Trading/prj/app/core/data/M1_ICE.csv'
    save_path = ''

    results_df = pd.read_csv(results_path)
    data_df = pd.read_csv(data_path)
    data_df = data_df[['timestamp', 'ask_0', 'bid_0']]

    results_df['date'] = pd.to_datetime(results_df['day'], format='%Y%m%d')
    delta = pd.to_timedelta(results_df['minute'], unit='m')
    results_df['timestamp'] = results_df['date'] + delta
    results_df = results_df[['timestamp', 'action', 'reward_nostd']]

    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df['mid'] = (data_df['ask_0'] + data_df['bid_0'])/2
    #data_df = data_df.drop(columns = ['ask_0', 'bid_0'])

    df = results_df.merge(data_df, on='timestamp', how='left')
    df = df.set_index('timestamp')

    percentages = []
    prev_action = 0

    for index, row in df.iterrows():
        action = row['action']
        percentage = 0

        if action != prev_action:
            if prev_action == 5:
                percentage = ((row['ask_0'] - 0.00625) / starting_price) - 1
            elif prev_action == -5:
                percentage = ((row['bid_0'] + 0.00625) / starting_price) - 1

            if action == 5:
                starting_price = row['bid_0'] + 0.00625
            elif action == -5:
                starting_price = row['ask_0'] - 0.00625
        prev_action = action
        percentages.append(percentage)

    df['percentage_return'] = percentages

    df['cum_percentage_return'] = np.cumsum(df['percentage_return'])

    ax = df[['cum_percentage_return']].plot(figsize = (16, 8))
    locator = mdates.AutoDateLocator(minticks=15, maxticks=20)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    yloc = mticker.MaxNLocator(nbins=12, min_n_ticks=8)
    ax.yaxis.set_major_locator(yloc)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))

    ax.grid(color='gray', linestyle=(0, (5, 5)), linewidth=0.5)
    plt.title('Cumulative percentage returns')
    plt.ylabel('Percentage return')
    plt.xlabel('')
    plt.savefig(
        '/data2/bonetti_a2a/a2a_riccardo/RL_Trading/results/xgb_delta_both_0.0005_pers10_nest250_span1_featstd/test_next_year/seed88674/cum_plot3.png')






if __name__ == '__main__':

    path = '/home/a2a/a2a/RL_Trading/results/experts_overnight/1y23_7y23'
    print(path)
    persistence = 30
    train = False
    validation = True
    test = False

    remove_costs = False

    seeds = []
    for elem in os.listdir(path):
        if elem.startswith('seed'):
            seeds.append(int(elem.replace('seed', '')))

    if validation:
        iterations = []
        for elem in os.listdir(os.path.join(path, 'seed'+str(seeds[0]))):
            if elem.startswith('FI_iter'):
                iterations.append(int(elem.replace('FI_iter', '').replace('.png', '')))

        #Plotter.plot_seeds('Validation', np.max(iterations), seeds, path)
        #Plotter.plot_seeds_percentage('Validation', np.max(iterations), seeds, path, persistence, remove_costs)
        Plotter.plot_seeds_percentage('Validation', 1, seeds, path, persistence, remove_costs, labels=[np.max(iterations)])

        #Plotter.plot_percentage_pl('Validation', np.max(iterations), seeds, path, persistence, True, remove_costs)
        #Plotter.plot_bootstrapped('Validation', np.max(iterations), seeds, path)

    if train:
        iterations = []
        for elem in os.listdir(os.path.join(path, 'seed'+str(seeds[0]))):
            if elem.startswith('Actions_iter') and elem.endswith('_Train.png'):
                iterations.append(int(elem.replace('Actions_iter', '').replace('_Train.png', '')))

        Plotter.plot_seeds('Train', np.max(iterations), seeds, path)
        Plotter.plot_seeds_percentage('Train', np.max(iterations), seeds, path, persistence, remove_costs)
        #Plotter.plot_percentage_pl('Train', np.max(iterations), seeds, path, persistence, True, remove_costs)

    if test:
        with open(os.path.join(path, 'parameters_opt.json')) as f:
            iterations = json.load(f)['iterations']
        path = os.path.join(path, 'test_next_year_no_retrain/2022')
        #iterations = []
        #for elem in os.listdir(os.path.join(path, 'seed' + str(seeds[0]))):
        #    if elem.startswith('Actions_iter') and elem.endswith('_Test.png'):
        #        iterations.append(int(elem.replace('Actions_iter', '').replace('_Test.png', '')))
        #        iterations = np.max(iterations)

        #Plotter.plot_seeds('Test', np.max(iterations), seeds, path)
        Plotter.plot_seeds_percentage('Test', iterations, seeds, path, persistence, remove_costs)
        #Plotter.plot_percentage_pl('Test', iterations, seeds, path, persistence, True, remove_costs)
        #Plotter.plot_bootstrapped('Test', np.max(iterations), seeds, path)
