import itertools
import pandas as pd
from RL_Trading.prj.app.core.models.entities.dataset_builder import DatasetBuilder
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationDatasetBuilder(DatasetBuilder):
    """A class to create the dataset used for classification task.

    :param target_offset: time difference between next and current mid in the target delta mid calculation expressed
     as the number of records. The parameter steps_tolerance of the parent class sets the tolerance for calculating the
     target delta mid (i.e., if the final mid is NaN, the last available mid in a lookback window of steps_tolerance
     size is used).
    """
    target_offset: int

    @staticmethod
    def _clean_df(df: pd.DataFrame, keep_volumes: int, depth: int = 0) -> pd.DataFrame:
        # Remove rows for which mid price is NaN.
        df = df.dropna(subset=['mid'])
        # Remove useless columns.
        if keep_volumes:
            columns_to_drop = \
                ['mid', 'next_mid', 'L1-BidPrice', 'L1-AskPrice', 'buy', 'sell', 'unknown'] + \
                [f'L{level}-{side}Price' for (level, side) in itertools.product(range(1, depth), ['Bid', 'Ask'])]
        else:
            columns_to_drop = \
                ['mid', 'next_mid', 'L1-BidPrice', 'L1-AskPrice', 'buy', 'sell', 'unknown'] + \
                [f'L{level}-{side}Size' for (level, side) in itertools.product(range(1, depth + 1), ['Bid', 'Ask'])] + \
                [f'L{level}-{side}Size_0' for (level, side) in itertools.product(range(1, depth + 1), ['Bid', 'Ask'])] + \
                [f'L{level}-{side}Price' for (level, side) in itertools.product(range(1, depth), ['Bid', 'Ask'])]
            
        columns_to_keep = [c for c in df.columns if c not in columns_to_drop]
        df = df[columns_to_keep]
        # Remove columns with unique values
        unique_counts = df.nunique()
        columns_to_keep = unique_counts[unique_counts > 1].index
        df = df[columns_to_keep]
        # Return
        return df
    
    @staticmethod
    def _add_LOB_VOL_features(df: pd.DataFrame,number_of_levels: int, mids_offset: int) -> pd.DataFrame:
        """
        References:
        -----------
        - Modeling high-frequency limit order book dynamics with support vector machines
        - Enhancing Trading Strategies with Order Book Signals
        
        """
        new_features = {}

        new_features['Mean_AskVolumes'] = df[[f"L{i}-AskSize_0" for i in range(1,number_of_levels + 1)]].mean(axis = 1)
        new_features['Mean_BidVolumes'] = df[[f"L{i}-BidSize_0" for i in range(1,number_of_levels + 1)]].mean(axis = 1)
    
        for i in range(1, number_of_levels + 1):
            new_features[f'L{i}_AskBid_Diff'] = df[f'L{i}-AskSize_0'] - df[f'L{i}-BidSize_0']
    
        time_diff = (df.index.to_series().diff().dt.total_seconds())

        for i in range(1, number_of_levels + 1):
            new_features[f'L{i}_AskSize_dt'] = df[f'L{i}-AskSize_0'].diff() / time_diff
            new_features[f'L{i}_BidSize_dt'] = df[f'L{i}-BidSize_0'].diff() / time_diff

        new_features['Volume_Imbalance_L1'] = (df["L1-BidSize_0"] - df["L1-AskSize_0"]) / (df["L1-BidSize_0"] + df["L1-AskSize_0"])
        new_features[f'Volume_Imbalance_L1_window_{mids_offset}_Sum'] = new_features['Volume_Imbalance_L1'].fillna(0).rolling(window=mids_offset, min_periods=1).sum()
        new_features[f'Volume_Imbalance_L1_window_{mids_offset}_Mean'] = new_features['Volume_Imbalance_L1'].fillna(0).rolling(window=mids_offset, min_periods=1).mean()
        new_features[f'Volume_Imbalance_L1_window_{mids_offset}_Std'] = new_features['Volume_Imbalance_L1'].fillna(0).rolling(window=mids_offset, min_periods=1).std()

        new_features['Total_BidVolume'] = df[[f"L{i}-BidSize_0" for i in range(1,number_of_levels + 1)]].sum(axis=1)
        new_features['Total_AskVolume'] = df[[f"L{i}-AskSize_0" for i in range(1,number_of_levels + 1)]].sum(axis=1)
        new_features['Total_Volume_Imbalance'] = (new_features['Total_BidVolume'] - new_features['Total_AskVolume']) / (new_features['Total_BidVolume'] + new_features['Total_AskVolume'])
        new_features[f'Total_Volume_Imbalance_window_{mids_offset}_Sum'] = new_features['Total_Volume_Imbalance'].fillna(0).rolling(window=mids_offset, min_periods=1).sum()
        new_features[f'Total_Volume_Imbalance_window_{mids_offset}_Mean'] = new_features['Total_Volume_Imbalance'].fillna(0).rolling(window=mids_offset, min_periods=1).mean()
        new_features[f'Total_Volume_Imbalance_window_{mids_offset}_Std'] = new_features['Total_Volume_Imbalance'].fillna(0).rolling(window=mids_offset, min_periods=1).std()
        
        # for win_size in [mids_offset, mids_offset*2]:
        #     for i in range(1,number_of_levels + 1):
        #         new_features[f'L{i}-BidSize_window_{win_size}_sum'] = df[f'L{i}-BidSize_0'].fillna(0).rolling(window=win_size, min_periods=1).sum()
        #         new_features[f'L{i}-AskSize_window_{win_size}_sum'] = df[f'L{i}-AskSize_0'].fillna(0).rolling(window=win_size, min_periods=1).sum()
        #         #new_features[f'L{i}-BidSize_window_{win_size}_min'] = df[f'L{i}-BidSize_0'].fillna(0).rolling(window=win_size,min_periods=1).min()                                                                              
        #         #new_features[f'L{i}-AskSize_window_{win_size}_min'] = df[f'L{i}-AskSize_0'].fillna(0).rolling(window=win_size,min_periods=1).min()                                                                            
        #         #new_features[f'L{i}-BidSize_window_{win_size}_max'] = df[f'L{i}-BidSize_0'].fillna(0).rolling(window=win_size,min_periods=1).max()                                                                                      
        #         #new_features[f'L{i}-AskSize_window_{win_size}_max'] = df[f'L{i}-AskSize_0'].fillna(0).rolling(window=win_size,min_periods=1).max()  
        #         new_features[f'L{i}-BidSize_window_{win_size}_mean'] = df[f'L{i}-BidSize_0'].fillna(0).rolling(window=win_size,min_periods=1).mean()                                                                                     
        #         new_features[f'L{i}-AskSize_window_{win_size}_mean'] = df[f'L{i}-AskSize_0'].fillna(0).rolling(window=win_size,min_periods=1).mean()                                                                                    
            
        df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
        return df

    @staticmethod
    def _add_LOB_Price_features(df: pd.DataFrame,number_of_levels: int, mids_offset: int) -> pd.DataFrame:
        """
        References:
        -----------
        - Modeling high-frequency limit order book dynamics with support vector machines

        """
        new_features = {}

        for i in range(1,number_of_levels+1):
            new_features[f'L{i}-AskBid_Spread'] = df[f'L{i}-AskPrice'] - df[f'L{i}-BidPrice']
            new_features[f'L{i}-AskBid_MidPrice'] = (df[f'L{i}-AskPrice'] + df[f'L{i}-BidPrice']) / 2
            
            if i >= 2:
                new_features[f'L{i}-Ask_Spread_L1'] = df[f'L{i}-AskPrice'] - df[f'L1-AskPrice']
                new_features[f'L1-Bid_Spread_L{i}'] = df[f'L1-BidPrice'] - df[f'L{1}-BidPrice']
            
        new_features['Mean_AskPrices'] = df[[f"L{i}-AskPrice" for i in range(1,number_of_levels + 1)]].mean(axis = 1)
        new_features['Mean_BidPrices'] = df[[f"L{i}-BidPrice" for i in range(1,number_of_levels + 1)]].mean(axis = 1)
        new_features_df = pd.DataFrame(new_features, index=df.index)
        new_features_df['Spread_accumulated_difference'] = new_features_df[[f"L{i}-AskBid_Spread" for i in range(1,number_of_levels + 1)]].sum(axis=1)
        
        df = pd.concat([df, new_features_df], axis=1)

        return df