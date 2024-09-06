import argparse
import pickle
import optuna
import numpy as np
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import root_mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold

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
                 loss_fn=roc_auc_score,
                 cv=None,
                 feat_count_penalty=0,
                 ):

        self.model = model
        self.X = X
        self.y = y

        if features is None:
            self.features = list(X.columns)
        else:
            self.features = features

        self.loss_fn = loss_fn

        if cv is None:
            kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            self.splits = list(kfold.split(X))
        else:
            self.splits = list(cv.split(X))

        self.feat_count_penalty = feat_count_penalty

    def __call__(self,
                 trial: optuna.trial.Trial):

        # Select True / False for each feature
        selected_features = [trial.suggest_categorical(name, [True, False]) for name in self.features]

        # List with names of selected features
        selected_feature_names = [name for name, selected in zip(self.features, selected_features) if selected]

        # Optional: adds a feat_count_penalty for the amount of features used
        n_used = len(selected_feature_names)
        total_penalty = n_used * self.feat_count_penalty

        X_selected = self.X[selected_feature_names].copy()

        from sklearn.model_selection import cross_val_score

        kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_res = cross_val_score(self.model, X_selected, self.y, cv=kfold, scoring='roc_auc')
        score = (cv_res.mean() - cv_res.std()) - total_penalty
        return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_jobs', type=int, default=12, help='CPU cores used')
    args = parser.parse_args()


    X_train = pd.read_pickle('../data/processed/X_train_2.pkl.zip')
    y_train = pd.read_pickle('../data/processed/y_train_2.pkl')
    feature_list = [s for s in X_train.columns if '_' in s]

    cls_params = {
        "random_state": RANDOM_SEED,
        "n_jobs": args.n_jobs,
        "verbose": False
    }

    model = RandomForestClassifier(**cls_params)
    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # We first try the model using all features
    default_features = {ft: True for ft in feature_list}
    study.enqueue_trial(default_features)

    optuna.logging.set_verbosity(optuna.logging.WARN)

    study.optimize(FeatureSelectionOptuna(
        model=model,
        features=feature_list,
        X=X_train,
        y=y_train,
        # feat_count_penalty = 1e-4,
    ), n_trials=4096, show_progress_bar=True)

    with open('../data/tuning/feat_sel_study_rf_ds01.pkl', 'wb') as f:
        pickle.dump(study, f)

if __name__ == '__main__':
    main()
