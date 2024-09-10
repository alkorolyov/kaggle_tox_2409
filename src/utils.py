import time
import multiprocessing as mp
from codecs import ignore_errors

import numpy as np
import pandas as pd
import datamol as dm
from rdkit import Chem

from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from joblib import Parallel, delayed

from Auto3D import auto3D


class OffsetScaler(BaseEstimator, TransformerMixin):
    """
    Applies StandardScaler the part of the input vector. Only values after offset are transformed [offset:],
    while keeping intact the initial part of vector [:offset].
    """

    def __init__(self, offset=0):
        self.offset = offset
        self.scaler = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.scaler = StandardScaler().fit(X[:, self.offset:])
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        x_fix = X[:, :self.offset]
        x_scale = self.scaler.transform(X[:, self.offset:])
        x_trans = np.hstack([x_fix, x_scale])
        return x_trans

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            x_trans = self._transform(X.values)
            return pd.DataFrame(x_trans, index=X.index, columns=X.columns)
        elif isinstance(X, np.ndarray):
            return self._transform(X)


class BlendingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = models
        self.weights = weights if weights is not None else [1] * len(models)

    def fit(self, X, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        for name, model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.array([model.predict_proba(X) for name, model in self.models])
        weighted_avg_predictions = np.average(predictions, axis=0, weights=self.weights)
        pred_idx = np.argmax(weighted_avg_predictions, axis=1)
        return self.classes_[pred_idx]

    def predict_proba(self, X):
        predictions = []
        for name, model in self.models:
            proba = model.predict_proba(X)
            predictions.append(proba)

        predictions = np.array(predictions)
        weighted_avg_predictions = np.average(predictions, axis=0, weights=self.weights)
        return weighted_avg_predictions

def get_fps_cols(cols):
    res = []
    for c in cols:
        try:
            res.append(str(int(c)))
        except:
            pass
    return res


def get_fps_offset(cols):
    res = []
    for c in cols:
        try:
            res.append(int(c))
        except:
            pass
    return len(res)


def scale(X: pd.DataFrame) -> pd.DataFrame:
    offset = get_fps_offset(X.columns)
    scaler = OffsetScaler(offset)
    X_scaled_vals = scaler.fit_transform(X.values)
    return pd.DataFrame(X_scaled_vals, columns=X.columns, index=X.index)


def _process_chunk(s: pd.Series, func, *args):
    return s.apply(func, args=args)


def apply_mp(s: pd.Series, func, *args, n_jobs: int = mp.cpu_count()):
    num_splits = min(len(s), n_jobs * 2)
    chunks = np.array_split(s, num_splits)
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.starmap(_process_chunk, [(chunk, func, *args) for chunk in chunks])
    return pd.concat(results)


def mol2smi(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol) if pd.notna(mol) else None


def smi2mol(smiles: str) -> Chem.Mol:
    return Chem.MolFromSmiles(smiles) if pd.notna(smiles) else None


def embed3d(smiles: pd.Series, n_jobs=-1, n_confs=1):
    mols = smiles.apply(Chem.MolFromSmiles)
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    return Parallel(n_jobs=n_jobs)(delayed(dm.conformers.generate)(mol, n_confs=n_confs, ignore_failure=True) for mol in mols)


def embed_auto3d(smiles: pd.Series, use_gpu=True, verbose=False):
    args = auto3D.options(k=1, use_gpu=use_gpu, verbose=verbose)
    return auto3D.smiles2mols(smiles, args)


def eval_model(name, model, X, y, random_seed=42, cv=None, scoring='roc_auc'):
    tic = time.time()

    if isinstance(X, pd.DataFrame):
        X = X.values

    if cv is None:
        cv = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    cv_res = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    score = cv_res.mean() - cv_res.std()

    toc = time.time()
    print("%-15s %3.4f    (%3.3f ± %3.3f)    %.1fs" % (name, score, cv_res.mean(), cv_res.std(), toc - tic))
    return cv_res


def to_submit(probs):
    return pd.Series(probs[:, 1], index=pd.RangeIndex(len(probs), name='id'), name='Y')


def drop_nans_non_unique(df):
    df.dropna(axis=1, inplace=True)
    df = df.loc[:, df.nunique() > 1].copy()
    return df



