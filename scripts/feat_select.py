import pickle
import optuna
from optuna.samplers import TPESampler

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb

RANDOM_SEED = 42


class FeatureSelectionOptuna:
    """
    This class implements feature selection using Optuna optimization framework.

    Parameters:

    - model (object): The predictive model to evaluate; this should be any object that implements fit() and predict() methods.
    - loss_fn (function): The loss function to use for evaluating the model performance. This function should take the true labels and the
                          predictions as inputs and return a loss value.
    - features (list of str): A list containing the names of all possible features that can be selected for the model.
    - X (DataFrame): The complete set of feature data (pandas DataFrame) from which subsets will be selected for training the model.
    - y (Series): The target variable associated with the X data (pandas Series).
    - splits (list of tuples): A list of tuples where each tuple contains two elements, the train indices and the validation indices.
    - feat_count_penalty (float, optional): A factor used to penalize the objective function based on the number of features used.
    """

    def __init__(self,
                 model,
                 X,
                 y,
                 cv=None,
                 features=None,
                 feat_ids=None,
                 scoring=None,
                 ):
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv
        self.features = features
        self.feat_ids = feat_ids
        self.scoring = scoring

    def __call__(self,
                 trial: optuna.trial.Trial):
        list_of_lists = [v for k, v in self.feat_ids.items() if trial.suggest_categorical(k, [True, False])]
        ids = pd.Series(list_of_lists).explode().astype(int).values

        # feature = {name: trial.suggest_categorical(name, [True, False]) for name in self.features}
        # feats_selected = [k for k, v in feature.items() if v]

        # print(group_feats_selected)

        X_selected = np.concatenate([self.X.iloc[:, ids], self.X.iloc[:, -3:]], axis=-1)
        # X_selected = self.X[feats_selected + group_feats_selected].copy()

        cv_res = cross_val_score(
            self.model,
            X_selected, self.y,
            cv=self.cv,
            scoring=self.scoring)
        score = (cv_res.mean() - cv_res.std())
        return score


def main():
    X_train = pd.read_pickle('../data/processed/X_train_all.pkl.zip')
    y_train = pd.read_pickle('../data/processed/y_train.pkl')
    with open('../data/processed/feat_ids_all.pkl', 'rb') as f:
        feat_ids = pickle.load(f)

    # clf = HistGradientBoostingClassifier(random_state=RANDOM_SEED)
    clf = xgb.XGBClassifier(random_state=RANDOM_SEED, verbosity=0,
                            tree_method='gpu_hist', predictor='gpu_predictor')
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_SEED)

    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(
        study_name='xgb_gpu_all_molfeats_groupped',
        storage=f"sqlite:///../data/tuning/optuna.db",
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    # Set all groups and features to True
    default_params = {k: True for k in feat_ids.keys()}
    study.enqueue_trial(default_params)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study.optimize(
        FeatureSelectionOptuna(
            model=clf,
            cv=cv,
            features=[],
            feat_ids=feat_ids,
            scoring='roc_auc',
            X=X_train,
            y=y_train,
        ),
        n_trials=300,
        n_jobs=2,
        show_progress_bar=True)


if __name__ == '__main__':
    main()