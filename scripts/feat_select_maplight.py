import optuna
from optuna.samplers import TPESampler

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold
import lightgbm as lgb
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
                 features=None,
                 group_feats=None,
                 loss_fn=roc_auc_score,
                 ):
        self.model = model
        self.X = X
        self.y = y
        self.features = features
        self.group_feats = group_feats
        self.loss_fn = loss_fn

    def __call__(self,
                 trial: optuna.trial.Trial):
        """
        self.group_feats = {
            'group_1': ['feat_1', 'feat_2', ... ],
            ...
        }

        """

        groups = [k for k in self.group_feats if trial.suggest_categorical(k, [True, False])]
        group_feats_selected = []
        [group_feats_selected.extend(self.group_feats[k]) for k in groups]

        feature = {name: trial.suggest_categorical(name, [True, False]) for name in self.features}
        feats_selected = [k for k, v in feature.items() if v]

        X_selected = self.X[feats_selected + group_feats_selected].copy()

        kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_SEED)
        # rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_SEED)

        cv_res = cross_val_score(
            self.model,
            X_selected, self.y,
            cv=kfold,
            scoring='roc_auc')
        score = (cv_res.mean() - cv_res.std())
        return score


def main():
    X_train = pd.read_pickle('../data/processed/X_train_mpl_mord.pkl.zip')
    y_train = pd.read_pickle('../data/processed/y_train.pkl')

    groups = {
        'morgan': [],
        'avalon': [],
        'erg': [],
        'gin': [],
    }

    features = []

    for c in X_train.columns:
        if 'morgan_' in c:
            groups['morgan'].append(c)
            continue
        if 'avalon_' in c:
            groups['avalon'].append(c)
            continue
        if 'erg_' in c:
            groups['erg'].append(c)
            continue
        if 'gin_' in c:
            groups['gin'].append(c)
            continue
        features.append(c)

    # clf = lgb.LGBMClassifier(random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
    # clf = xgb.XGBClassifier(random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
    #                               tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=1)
    clf = HistGradientBoostingClassifier(random_state=RANDOM_SEED)

    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(
        study_name='hgb_cpu_fps_groups_rd',
        storage=f"sqlite:///../data/tuning/optuna_cpu.db",
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    # Set all groups and features to True
    opt_params = [k for k in groups] + features
    default_params = {k: True for k in opt_params}
    study.enqueue_trial(default_params)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study.optimize(
        FeatureSelectionOptuna(
            model=clf,
            features=features,
            group_feats=groups,
            X=X_train,
            y=y_train,
        ),
        n_trials=10,
        n_jobs=1,
        show_progress_bar=True)


if __name__ == '__main__':
    main()
