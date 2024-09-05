import multiprocessing as mp
import numpy as np
import pandas as pd
from rdkit import Chem

from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator


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