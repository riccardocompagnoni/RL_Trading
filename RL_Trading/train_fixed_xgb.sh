for start_hour in 8 9 ;
    do
      for horizon in 5 10 20 30 45 60;
        do
          for number_of_deltas in 20 30 40 60;
          do
            DIR_NAME="/data/intesa/giovanni/fixed_xgb_experiments_1516_17/start_hour_${start_hour}_horizon_${horizon}_deltas_${number_of_deltas}"
            mkdir "${DIR_NAME}"
            python3 prj/app/core/fqi/services/train_grid.py --save_path "${DIR_NAME}" --start_hour $start_hour --horizon $horizon --number_of_deltas $number_of_deltas --n_jobs 5 --train_years 2015 2016 --test_years 2017
          done
        done
    done