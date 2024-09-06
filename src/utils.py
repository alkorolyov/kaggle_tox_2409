import time
import multiprocessing as mp
import numpy as np
import pandas as pd
from rdkit import Chem

from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import KFold, cross_val_score

class OffsetScaler(BaseEstimator, TransformerMixin):
    """
    Applies StandardScaler the part of the input vector. Only values after offset are transformed [offset:],
    while keeping intact the initial part of vector [:offset].
    """

    def __init__(self, offset=0):
        self.offset = offset
        self.scaler = None

    def fit(self, X, y=None):
        self.scaler = StandardScaler().fit(X[:, self.offset:])
        return self

    def transform(self, X, y=None):
        x_fix = X[:, :self.offset]
        x_scale = self.scaler.transform(X[:, self.offset:])
        x_trans = np.hstack([x_fix, x_scale])
        return x_trans


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


def eval_model(name, model, X, y, random_seed=42):
    tic = time.time()

    kfold = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    cv_res = cross_val_score(model, scale(X), y, cv=kfold, scoring='roc_auc')

    score = cv_res.mean() - cv_res.std()

    toc = time.time()
    print("%7s: %3.3f    (%3.3f ± %3.3f)    %.1fs" % (name, score, cv_res.mean(), cv_res.std(), toc - tic))
    return cv_res