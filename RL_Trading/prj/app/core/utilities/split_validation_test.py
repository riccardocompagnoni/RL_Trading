import pandas as pd
from datetime import datetime

if __name__ == '__main__':
    data_root = '/home/a2a/a2a/RL_Trading/prj/app/core/data'
    dataset = 'M1_ICE'
    

    df = pd.read_csv(f'{data_root}/{dataset}.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])


    #print(df.timestamp.dt.date.unique())


    start = datetime(2020, 3, 1)
    end = datetime(2023, 7, 1)

    low_val = datetime(2023,1, 1)
    up_val = datetime(2023,7, 1)

    test_start = datetime(2024, 1, 1)
    test_limit = datetime(2024, 10, 1)


    df[(df.timestamp >= low_val) & (df.timestamp < up_val)].to_csv(f'{data_root}/{dataset}_2021.csv', index=False)
    df[((df.timestamp >= start) & (df.timestamp < low_val))].to_csv(f'{data_root}/{dataset}_2019.csv', index=False)
    df[((df.timestamp >= up_val) & (df.timestamp < end))].to_csv(f'{data_root}/{dataset}_2020.csv', index=False)
    df[(df.timestamp>=test_start) & (df.timestamp<test_limit)].to_csv(f'{data_root}/{dataset}_2022.csv', index=False)




    """
    train_val = datetime(2020, 9, 1)
    val_test = datetime(2024, 4, 1)
    test_limit = datetime(2024, 10, 1)
    

    df[df.timestamp<train_val].to_csv(f'{data_root}/{dataset}_2021.csv', index=False)
    df[(df.timestamp>=train_val) & (df.timestamp<val_test)].to_csv(f'{data_root}/{dataset}_2020.csv', index=False)
    #df[(df.timestamp>=val_test) & (df.timestamp<test_limit)].to_csv(f'{data_root}/{dataset}_2022.csv', index=False)
    """