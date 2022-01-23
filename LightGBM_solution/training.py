from pathlib import Path
import pandas as pd
import gc
from lightgbm import LGBMClassifier

from preprocessing import * # load the data_preprocess function
from feature_engineering import * # load the feature_generating function

DATA_PATH = '<my data path>'
PARAMS = {'colsample_bytree': 0.8776176426571266,
          'learning_rate': 0.13584383574944736,
          'max_bin': 22,
          'max_depth': 20,
          'min_child_samples': 2137,
          'num_leaves': 93,
          'reg_alpha': 0.32607195296165126,
          'reg_lambda': 74.72987624093412,
          'subsample': 0.8157147194111745}
N_ESTIMATORS = 500

### read & split training data
df = data_preprocess(DATA_PATH, use_test=True)
df = feature_generating(df)
df = df.iloc[:, 3:]
train_ind = df['target'][~pd.isnull(df['target'])].index
test_ind = df['target'][pd.isnull(df['target'])].index
X_train, y_train = df.drop('target', axis=1).iloc[train_ind, :], df['target'][train_ind]
X_test = df.drop('target', axis=1).iloc[test_ind, :].reset_index(drop=True)
del df
gc.collect()

### training
model = LGBMClassifier(n_estimators=N_ESTIMATORS, random_state=0, **PARAMS)
model.fit(X_train, y_train)
df_pred = pd.DataFrame({'id': range(len(X_test)), 'target': model.predict_proba(X_test)[:,1]})
df_pred.to_csv(Path(DATA_PATH)/'submit_lgb_tune.csv', index=False)