import numpy as np
import gc
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from hyperopt import hp, fmin, STATUS_OK, tpe, Trials
from hyperopt.pyll.base import scope

from preprocessing import * # load the data_preprocess function
from feature_engineering import * # load the feature_generating function

DATA_PATH = '<my data path>'
VALID_SIZE = 0.2
N_ESTIMATORS = 500
NUM_EVAL = 50

### read & split training data
df = data_preprocess(DATA_PATH, use_test=False)
df = feature_generating(df)
df = df.iloc[:, 3:]
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], 
                                                    test_size=VALID_SIZE, random_state=0, shuffle=False)
del df
gc.collect()

### tuning by hyperopt
def objective_function(params):
    model = LGBMClassifier(n_estimators=N_ESTIMATORS, random_state=0, **params)
    model.fit(X_train, y_train)
    metric = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    return {'loss': -metric, 'status': STATUS_OK}
    
space = {'min_child_samples': scope.int(hp.quniform('min_child_samples', 100, 5000, 1)),
          'max_depth': scope.int(hp.quniform('max_depth', 4, 20, 1)),
          'max_bin': scope.int(hp.quniform('max_bin', 5, 50, 1)),
          'num_leaves': scope.int(hp.quniform('num_leaves', 10, 100, 1)),
          'learning_rate': hp.uniform('learning_rate', 0, 1),
          'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
          'subsample': hp.uniform('subsample', 0.5, 1),
          'reg_alpha': hp.loguniform('reg_alpha', np.log(0.1), np.log(100)),
          'reg_lambda': hp.loguniform('reg_lambda', np.log(0.1), np.log(100))}

trials = Trials()
best_param = fmin(objective_function, space, algo=tpe.suggest, max_evals=NUM_EVAL, 
                  trials=trials, rstate=np.random.default_rng(42))