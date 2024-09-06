import sys
sys.path.append('..')

import time
import pickle
import optuna
import numpy as np
import pandas as pd


from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from src.utils import OffsetScaler, get_fps_offset
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import seaborn as sns


mae = 'neg_mean_absolute_error'
mse = 'neg_mean_squared_error'
rmse = 'neg_root_mean_squared_error'
roc_auc = 'neg_roc_auc_score'

N_JOBS = 24
RANDOM_SEED = 42


def get_objectives(X, y):
    objectives = {}

    # Logistic Regression
    def logistic_regression_objective(trial):
        params = {
            'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
            'max_iter': trial.suggest_int('max_iter', 100, 500)
        }
        clf = LogisticRegression(**params)
        kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        score = cross_val_score(clf, X, y, n_jobs=-1, cv=kfold, scoring='roc_auc')
        return score.mean() - score.std()


    # KNN
    def knn_objective(trial):
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 1, 50),
            'leaf_size': trial.suggest_int('leaf_size', 10, 50),
            'p': trial.suggest_categorical('p', [1, 2]),
            'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        }
        clf = KNeighborsClassifier(n_jobs=N_JOBS, **params)
        kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        score = cross_val_score(clf, X, y, n_jobs=-1, cv=kfold, scoring='roc_auc')
        return score.mean() - score.std()

    # SVC
    def svc_objective(trial):
        params = {
            'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
        }
        clf = SVC(**params)
        kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        score = cross_val_score(clf, X, y, n_jobs=-1, cv=kfold, scoring='roc_auc')
        return score.mean() - score.std()

    # Random Forest
    def random_forest_objective(trial):
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 400, 500, 800, 1400, 2000]),
            'max_depth': trial.suggest_int('max_depth', 2, 40),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
        clf = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=N_JOBS, **params)
        kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        score = cross_val_score(clf, X, y, n_jobs=1, cv=kfold, scoring='roc_auc')
        return score.mean() - score.std()

    # XGBoost
    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', [200, 400, 1000, 2000, 5000]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.05, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.05, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 0.1, 1.0, log=True),
        }
        clf = xgb.XGBClassifier(random_state=RANDOM_SEED, n_jobs=N_JOBS, verbosity=0, **params)
        kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        score = cross_val_score(clf, X, y, n_jobs=1, cv=kfold, scoring='roc_auc')
        return score.mean() - score.std()

    # CatBoost
    def catboost_objective(trial):
        params = {
            'iterations': trial.suggest_categorical('n_estimators', [50, 100, 200, 400, 800, 1000, 2000, 5000]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1.0, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'depth': trial.suggest_int('depth', 3, 10)
        }
        clf = cb.CatBoostClassifier(random_seed=RANDOM_SEED, thread_count=N_JOBS, verbose=False, **params)
        kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        score = cross_val_score(clf, X, y, n_jobs=1, cv=kfold, scoring='roc_auc')
        return score.mean() - score.std()

    # LightGBM
    def lgbm_objective(trial):
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 200, 400, 800, 1000, 2000, 5000]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150)
        }
        clf = lgb.LGBMClassifier(random_state=RANDOM_SEED, n_jobs=N_JOBS, verbose=0, **params)
        kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        score = cross_val_score(clf, X, y, n_jobs=1, cv=kfold, scoring='roc_auc')
        return score.mean() - score.std()

    # Adding each objective to the dict
    objectives['LR'] = logistic_regression_objective
    objectives['KNN'] = knn_objective
    objectives['SVC'] = svc_objective
    objectives['RF'] = random_forest_objective
    objectives['XGB'] = xgb_objective
    objectives['CB'] = catboost_objective
    objectives['LGB'] = lgbm_objective

    return objectives


if __name__ == '__main__':
    X = pd.read_pickle('../data/processed/X_train_2.pkl')
    y = pd.read_pickle('../data/processed/y_train_2.pkl')

    FPS_OFFSET = get_fps_offset(X.columns)
    scaler = OffsetScaler(FPS_OFFSET)
    X_scaled = scaler.fit_transform(X.values)

    objectives = get_objectives(X_scaled, y)
    for name in ['SVC', 'RF', 'XGB', 'CB', 'LGB']:
    # for name in ['SVC', 'RF', 'XGB', 'LGB']:
    # for name in ['RF']:
        obj = objectives[name]
        study = optuna.create_study(direction='maximize')
        study.optimize(obj, n_trials=2)
        with open('../data/tuning/' + name + '_dataset_2_.pkl', 'wb') as f:
            pickle.dump(study, f)

