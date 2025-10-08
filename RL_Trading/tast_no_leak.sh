for quantile  in 0.99 0.95 0.90 0.85;
    do
      for filter_method in first_step every_step_flat every_step_propagate;
        do
            python3 prj/app/core/fqi/services/test_no_leak.py --model_path /Users/giovannidispoto/Desktop/PhD/new_version/influence/RL_Trading/results/test/FQI_1718_19_multiseed_dfqi_optimized_trajectory_ohe_no_skip --quantile $quantile --filter_method $filter_method --optuna_output_path /Users/giovannidispoto/Desktop/PhD/new_version/influence/RL_Trading/results/FQI_1718_19_multiseed_dfqi_optimized_trajectory_ohe_no_skip/ --test_year 2020 --output_path /Users/giovannidispoto/Desktop/PhD/new_version/influence/RL_Trading/test_no_leak_output/no_skip
        done
    done


#parser.add_argument('--model_path', type=str)
#parser.add_argument("--optuna_output_path", type=str)
#parser.add_argument("--output_path", type=str)
#parser.add_argument('--test_year', type=int, default=2020)
#parser.add_argument('--quantile', type=float, default=0.90)