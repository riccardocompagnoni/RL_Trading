import sys
sys.path.append("../../../prj/app")
import time
import logging
import pandas as pd
from tree_influence.explainers import BoostIn, SubSample
from lightgbm import LGBMRegressor
from RL_Trading.prj.app.core.fqi.entities.fqi_dataset_builder_factory import FQIDatasetBuilderFactory
from  RL_Trading.prj.app.core.fqi.entities.trading_env import TradingEnv


start = time.time()
features_parameters = {
            'roll_date_offset': 1,
            'mids_window': 1,
            'mids_offset': 60,
            'steps_tolerance': 5,
            'number_of_deltas': 20,
            'opening_hour': 8,
            'closing_hour': 18,
            'ohe_temporal_features': True,
            'persistence': 60,
            'actions': [-1, 0, 1],
            'add_trade_imbalance': False,
            'volume_features': False,
        }
training_dataset_builder = FQIDatasetBuilderFactory.create("IK", [2018], **features_parameters)
test_dataset_builder = FQIDatasetBuilderFactory.create("IK", [2018], **features_parameters)
env_train = TradingEnv(training_dataset_builder)
env_test = TradingEnv(test_dataset_builder)
df_train = env_train.df.dropna(subset='reward')
df_test = env_test.df.dropna(subset='reward')
df_test = df_test[(df_train.time.dt.month == 2) & (df_train.time.dt.day == 20)]
#df_test = df_test[(df_test.time.dt.month == 5) & (df_test.time.dt.day == 25)]
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger("Logger")
print(f'Elapsed time: {time.time()-start:,.2f} seconds')
X_train = df_train[env_train.current_state_features + ['action']].ffill().bfill()
y_train = df_train['reward']

X_test = df_test[env_test.current_state_features + ['action']].ffill().bfill()
y_test = df_test['reward']

model = LGBMRegressor(n_jobs=-1).fit(X_train.values, y_train.values)
explainer = BoostIn(logger=log).fit(model, X_train.values, y_train.values)
print("Start explaining")
self_influence = explainer.get_self_influence(X_test.values, y_test.values, batch_size=100)
self_influence_df = pd.concat([pd.DataFrame(df_test.time.values, columns=["Time"]), pd.DataFrame(self_influence, columns=["Influence"])], axis = 1)
self_influence_df.to_csv("self_influence_training.csv", index=False)
print(self_influence_df.shape)
#print("self-influence: ", self_influence_df)
#print("Get Influence on Test")
#influence = explainer.get_local_influence(X_test.values, y_test.values)
#influence_df = pd.DataFrame(influence)
#print(influence_df.shape)
#influence_df.to_csv("influence_training_on_test.csv", index=False)

#print(influence_df)


