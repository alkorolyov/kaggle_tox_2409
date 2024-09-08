import multiprocessing

import pandas as pd
import datamol as dm
import numpy as np
import seaborn as sns

from molfeat.calc import RDKitDescriptors2D, FPCalculator, MordredDescriptors
from molfeat.trans import MoleculeTransformer
from sklearn.preprocessing import OneHotEncoder

import collections.abc as collections
from molfeat.trans.concat import FeatConcat

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from src.config import mem
from src.utils import embed3d, eval_model
from src.descriptors import get_dgl_predictions, get_hft_predictions

from rdkit import RDLogger, Chem

RDLogger.DisableLog('rdApp.*')

if __name__ == '__main__':
    train = dm.read_csv("../data/processed/train.csv", smiles_column="smi", index_col=0)
    test = dm.read_csv("../data/processed/test.csv", smiles_column="smi", index_col=0)
    y_train = pd.read_pickle('../data/processed/y_train.pkl')
    ohe = OneHotEncoder(sparse_output=False)


    def get_x_train(feats):
        return np.concatenate([feats, ohe.fit_transform(train[["prop"]])], axis=1)


    def get_x_test(feats):
        return np.concatenate([feats, ohe.transform(train[["prop"]])], axis=1)


    from molfeat.trans.fp import FPVecTransformer
    from molfeat.trans.pretrained import PretrainedDGLTransformer
    from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer

    dgl_params = [
        {'kind': 'gin_supervised_contextpred'},
        {'kind': 'gin_supervised_infomax'},
        {'kind': 'gin_supervised_edgepred'},
        {'kind': 'gin_supervised_masking'},
    ]

    hft_params = [
        {'kind': 'MolT5', 'notation': 'smiles', 'random_seed': 42},
        {'kind': 'GPT2-Zinc480M-87M', 'notation': 'smiles', 'random_seed': 42},
        {'kind': 'Roberta-Zinc480M-102M', 'notation': 'smiles', 'random_seed': 42},
    ]

    transformers = [
        FPVecTransformer("ecfp:4", length=1024, n_jobs=1, dtype=np.float32),
        FPVecTransformer("maccs", length=1024, n_jobs=1, dtype=np.float32),
        FPVecTransformer("topological", length=1024, n_jobs=1, dtype=np.float32),
        FPVecTransformer("avalon", length=1024, n_jobs=1, dtype=np.float32),
        FPVecTransformer('erg', length=315, dtype=np.float32),
        FPVecTransformer("layered", n_jobs=1, length=1024, dtype=np.float32),
        FPVecTransformer("secfp", length=1024, n_jobs=1, dtype=np.float32),
        FPVecTransformer("estate", n_jobs=1, dtype=np.float32),
        FPVecTransformer('pattern', length=1024, dtype=float),

        FPVecTransformer("mordred", n_jobs=-1, dtype=np.float32),
        FPVecTransformer("desc2D", n_jobs=-1, dtype=np.float32, replace_nan=True),
        FPVecTransformer("cats2D", n_jobs=-1, dtype=np.float32, replace_nan=True),
        FPVecTransformer("pharm2D", n_jobs=-1, length=1024, dtype=np.float32),
        FPVecTransformer("scaffoldkeys", n_jobs=-1, dtype=np.float32, replace_nan=True),
        FPVecTransformer("skeys", n_jobs=-1, dtype=np.float32, replace_nan=True),
    ]

    transformers3d = [
        FPVecTransformer("desc3D", length=639, dtype=np.float32, replace_nan=True),
        FPVecTransformer("cats3D", length=126, dtype=np.float32, replace_nan=True),
        FPVecTransformer("pharm3D", length=1024, dtype=np.float32),
        FPVecTransformer("electroshape", length=15, dtype=np.float32, replace_nan=True),
        FPVecTransformer("usr", length=12, dtype=np.float32),
        FPVecTransformer("usrcat", length=60, dtype=np.float32),
    ]

    # featurizer = FeatConcat(transformers, dtype=np.float32)

    # calcucalte feats and cache them
    for trans in transformers:
        feats = mem.cache(trans.transform)(train.smi)

    for params in dgl_params:
        feats = mem.cache(get_dgl_predictions, ignore=['n_jobs', 'dtype'])(train.smi, params)

    for params in hft_params:
        feats = mem.cache(get_hft_predictions, ignore=['n_jobs', 'dtype', 'device'])(train.smi, params, device='cuda',
                                                                                     n_jobs=-1)

    # for kind, params in hft_params.items():
    #     print('Initializing', end=' ')
    #     trans = PretrainedHFTransformer(kind, **params, n_jobs=-1)
    #     print(kind, end=' ')
    #     feats = mem.cache(trans.transform)(train.smi)
    #     print(feats.shape[1])