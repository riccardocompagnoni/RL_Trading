import pickle
import argparse
import json
import shutil
from RL_Trading.prj.app.core.fqi.entities.fqi_dataset_builder_factory import FQIDatasetBuilderFactory
from RL_Trading.prj.app.core.fqi.entities.trading_env import TradingEnv
from RL_Trading.prj.app.core.fqi.entities.trainer import  Trainer
from RL_Trading.prj.app.core.fqi.services.plotter import Plotter
from RL_Trading.prj.app.core.fqi.trlib.algorithms.reinforcement.fqi import FQI
import os




parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument("--optuna_output_path", type=str)
parser.add_argument("--output_path", type=str)
parser.add_argument('--test_year', type=int, default=2020)
parser.add_argument('--quantile', type=float, default=None)
#parser.add_argument('--save_root', type=str)
parser.add_argument("--skip_covid", type=bool, default=False)
parser.add_argument('--filter_method',  type=str, default="None")

args = parser.parse_args()
test_years = [args.test_year]
features_parameters = json.load(open(os.path.join(args.optuna_output_path, "features_params.json")))
parameters_opt = json.load(open(os.path.join(args.optuna_output_path, "parameters_opt.json")))
trajectory_window = None
trajectory_number = parameters_opt['trajectory_number']
save_root = args.output_path
print(args.quantile)
print(args.optuna_output_path)
if not os.path.exists(os.path.join(args.output_path, args.filter_method)):
    name = args.optuna_output_path.split("/")[-2]
    if args.quantile == None:
        os.makedirs(os.path.join(args.output_path, f"{args.filter_method}_{name}"))
    else:
        os.makedirs(os.path.join(args.output_path, f"{args.filter_method}_{args.quantile}_{name}"))

if args.quantile == None:
    dir_output = os.path.join(args.output_path, f"{args.filter_method}_{name}")
else:
    dir_output = os.path.join(args.output_path, f"{args.filter_method}_{args.quantile}_{name}")
seeds = [int(f.split("d")[1]) for f in os.listdir(args.model_path) if
                 os.path.isdir(os.path.join(args.model_path, f)) and "seed" in f]
print(seeds)
features_parameters['skip_covid'] = args.skip_covid
testing_dataset_builder = FQIDatasetBuilderFactory.create("IK", test_years, **features_parameters)
test_env = TradingEnv(testing_dataset_builder)
if args.quantile == None:
    q_threshold_per_seed = None
else:
    q_threshold_per_seed = Trainer.get_percentile_threshold(seeds, args.model_path, parameters_opt['iterations'], args.quantile) #dictionary (seed, q_value)
    print(q_threshold_per_seed)

if args.filter_method == "None":
    args.filter_method = None

for seed in seeds:
    os.makedirs(os.path.join(dir_output, f"seed{seed}"))
    for iteration in range(parameters_opt['iterations']):
        policy = pickle.load(open(os.path.join(args.model_path, f'seed{seed}', f'Policy_iter{iteration+1}.pkl'), "rb"))  # read policy
        if args.filter_method == None:
            gain = test_env.test(
                policy=policy,
                save_csv=True,
                save_plots=True,
                save_root=os.path.join(dir_output, f"seed{seed}"),
                phase="Test",
                iteration=iteration + 1,
                trajectory_number=trajectory_number,
                save_iteration_data=True,
                trajectory_window=trajectory_window,
            )
        else:
            gain = test_env.test(
                            policy=policy,
                            save_csv=True,
                            save_plots=True,
                            save_root=os.path.join(dir_output, f"seed{seed}"),
                            phase="Test",
                            iteration=iteration+1,
                            trajectory_number=trajectory_number,
                            save_iteration_data=True,
                            trajectory_window=trajectory_window,
                            filter_method=args.filter_method,
                            q_threshold=q_threshold_per_seed[seed][iteration+1]
                        )
        print(f"ITERATION {iteration}, seed {seed}: Algorithm tested on Test set (gain = {gain:,.2f}%).")

Plotter.plot_percentage_returns("Test", parameters_opt['iterations'], seeds, dir_output)
#Plotter.plot_percentage_pl("Test", parameters_opt['iterations'], seeds, dir_output, True)

#for d in os.listdir(dir_output):
#    if os.path.isdir(os.path.join(dir_output, d)):
#        shutil.rmtree(os.path.join(dir_output, d))