import optuna

if __name__ == '__main__':

    # Replace with your actual storage URL and study name
    #FQI_1718_19_multiseed_dfqi_optimized_trajectory_ohe_no_skip
    #
    #study_name = "FQI_1920_21_multiseed_dfqi_optimized_trajectory_ohe_skip_covid"
    study_name = "1y22_7y22"
    storage_url = f"sqlite://////home/a2a/a2a/RL_Trading/results/1y22_7y22/optuna_study.db"
    # Load the study
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    # Retrieve trials
    trials = study.trials

    # Print hyperparameters for each trial
    for trial in trials:
       # if trial.value != None and (float(trial.value) >= 10 and float(trial.value) < 11):
       if trial.value != None:
            print(f"Trial number: {trial.number}")
            print(f"Trial state: {trial.state}")
            print(f"Hyperparameters: {trial.params}")
            print(f"Objective value: {trial.value}")
            print("--------------")
