import os

from RL_Trading.prj.app.core.fqi.services.plotter import Plotter

save_path = "/Users/giovannidispoto/Desktop/bkp/Train 2018-2019, Validation 2020"
seeds = [int(f.split("d")[1]) for f in os.listdir(save_path) if
                 os.path.isdir(os.path.join(save_path, f)) and f != "__pycache__"]


Plotter.plot_percentage_returns_percentiles(percentiles=[1, 0.99, 0.95, 0.9], phase="Validation", iterations=1, seeds=seeds, save_path=save_path)