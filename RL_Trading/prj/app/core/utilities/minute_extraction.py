import os
import pandas as pd


if __name__ == '__main__':

    paths =  ["C:/Users/Riccardo/Documents/TTF_all/TTF_M2/"]

    for path in paths:
        df_all = pd.DataFrame()
        k = 0
        for file in os.listdir(path):
            if file.endswith('.csv'):
                df = pd.read_csv(path + file)
            else:
                df = pd.read_json(path+file)

            df['timestamp'] = pd.to_datetime(df['timestamp'])

            print('starting grouping')
            df = df.dropna(subset=['ask_0', 'bid_0'], how='any').groupby(pd.Grouper(key='timestamp', freq='1min')).nth(0).reset_index(drop=True)
            #df = df.groupby(pd.Grouper(key='timestamp', freq='1min')).first().dropna(subset=['ask_0', 'bid_0'], how='any').reset_index()
            print('finished grouping')

            df_all = pd.concat([df_all, df])
            k += 1
            print(k)
            print(df_all.shape)

        df_all['timestamp'] = df_all['timestamp'].apply(pd.to_datetime)
        df_all.to_csv(path + "minute_aggregation_order.csv")


