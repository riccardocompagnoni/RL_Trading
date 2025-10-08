experiments=("FQI_1718_19_multiseed_fqi_optimize_trajectory_skip_conte_I_ohe")
#"FQI_1718_19_multiseed_dfqi_optimize_trajectory_test_observed_persistence_10_volumes"
# Iterate over the array
for experiment in "${experiments[@]}"
do
    echo "Running $experiment"
    # Add commands to run each experiment here
    python3 prj/app/core/fqi/services/train_up_to_year.py --study_name $experiment

done