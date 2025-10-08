import os
from RL_Trading.prj.app.core.fqi.services.plotter import Plotter

#no_skip_conte_covid = "/Users/giovannidispoto/Desktop/PhD/new_version/influence/RL_Trading/results/test/FQI_1718_19_multiseed_dfqi_optimized_trajectory_ohe_no_skip"
#skip_conte_covid = "/Users/giovannidispoto/Desktop/PhD/new_version/influence/RL_Trading/results/test/FQI_1718_19_multiseed_dfqi_optimized_trajectory_ohe_skip_conte_i_covid_test"
#no_skip_conte_skip_covid ="/Users/giovannidispoto/Desktop/PhD/new_version/influence/RL_Trading/results/test/FQI_1718_19_multiseed_dfqi_optimized_trajectory_ohe_no_skip_skip_covid_on_test"
#skip_conte_skip_covid = "/Users/giovannidispoto/Desktop/PhD/new_version/influence/RL_Trading/results/test/FQI_1718_19_multiseed_dfqi_optimized_trajectory_ohe_skip_conte_I_skip_covid_test"

#directories = {
#"no_skip_conte_covid": "/Users/giovannidispoto/Desktop/PhD/new_version/influence/RL_Trading/results/test/FQI_1718_19_multiseed_dfqi_optimized_trajectory_ohe_no_skip",
#"skip_conte_covid": "/Users/giovannidispoto/Desktop/PhD/new_version/influence/RL_Trading/results/test/FQI_1718_19_multiseed_dfqi_optimized_trajectory_ohe_skip_conte_i_covid_test",
#"no_skip_conte_skip_covid" :"/Users/giovannidispoto/Desktop/PhD/new_version/influence/RL_Trading/results/test/FQI_1718_19_multiseed_dfqi_optimized_trajectory_ohe_no_skip_skip_covid_on_test",
#"skip_conte_skip_covid": "/Users/giovannidispoto/Desktop/PhD/new_version/influence/RL_Trading/results/test/FQI_1718_19_multiseed_dfqi_optimized_trajectory_ohe_skip_conte_I_skip_covid_test",
#}

directories = {
"no_skip": "/Users/giovannidispoto/Desktop/PhD/new_version/influence/RL_Trading/results/test/FQI_1920_21_multiseed_dfqi_optimized_trajectory_ohe_no_skip_test_observed",
"skip": "/Users/giovannidispoto/Desktop/PhD/new_version/influence/RL_Trading/results/test/FQI_1920_21_multiseed_dfqi_optimized_trajectory_ohe_skip_covid_test_observed",
}



#label_dictionary = {
#"no_skip_conte_covid": "Anomaly in Training, Anomaly in Test",
#"skip_conte_covid": "Manual removal Anomaly in Training, Anomaly in Test",
#"no_skip_conte_skip_covid" :"Anomaly in Training, Manual removal anomaly in Test",
#"skip_conte_skip_covid": "Manual removal anomaly in training, Manual removal anomaly in Test",
#}

label_dictionary = {
"no_skip": "Anomaly in Train (Covid)",
"skip": "No Anomaly in Train (Covid)",
}

seeds = {"seeds_no_skip": [int(f.split("d")[1]) for f in os.listdir(directories['no_skip']) if
                 os.path.isdir(os.path.join(directories['no_skip'], f)) and f != "__pycache__"],
         "seeds_skip": [int(f.split("d")[1]) for f in os.listdir(directories['skip']) if
                 os.path.isdir(os.path.join(directories['skip'], f)) and f != "__pycache__"]
         }

#seeds = {"seeds_no_skip_conte_covid": [int(f.split("d")[1]) for f in os.listdir(directories['no_skip_conte_covid']) if
#                 os.path.isdir(os.path.join(directories['no_skip_conte_covid'], f)) and f != "__pycache__"],
#         "seeds_skip_conte_covid": [int(f.split("d")[1]) for f in os.listdir(skip_conte_covid) if
#                 os.path.isdir(os.path.join(skip_conte_covid, f)) and f != "__pycache__"],
#         "seeds_no_skip_conte_skip_covid":[int(f.split("d")[1]) for f in os.listdir(no_skip_conte_skip_covid) if
#                 os.path.isdir(os.path.join(no_skip_conte_skip_covid, f)) and f != "__pycache__"],
#         "seeds_skip_conte_skip_covid": [int(f.split("d")[1]) for f in os.listdir(skip_conte_skip_covid) if
#                 os.path.isdir(os.path.join(skip_conte_skip_covid, f)) and f != "__pycache__"]

#         }



Plotter.plot_percentage_returns_more_cases_without_Q(phase = "Test", iterations = 1, directories = directories, seeds = seeds, save_path="/Users/giovannidispoto/Desktop/PhD/new_version/influence/RL_Trading/plots", label_dictionary=label_dictionary)