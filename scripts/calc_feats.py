import sys
sys.path.append('..')

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
from src.utils import embed3d, eval_model, embed_auto3d
from src.descriptors import calc_dgl_feats, calc_hft_feats, calc_2d_feats, calc_3d_feats

from rdkit import RDLogger, Chem

RDLogger.DisableLog('rdApp.*')

if __name__ == '__main__':
    train_df = dm.read_csv("../data/processed/train.csv", smiles_column="smi", index_col=0)[:2]
    test_df = dm.read_csv("../data/processed/test.csv", smiles_column="smi", index_col=0)[:2]
    y_train = pd.read_pickle('../data/processed/y_train.pkl')
    ohe = OneHotEncoder(sparse_output=False)


    def get_x_train(feats):
        return np.concatenate([feats, ohe.fit_transform(train_df[["prop"]])], axis=1)


    def get_x_test(feats):
        return np.concatenate([feats, ohe.transform(train_df[["prop"]])], axis=1)


    from molfeat.trans.fp import FPVecTransformer

    from molfeat.trans.pretrained import PretrainedDGLTransformer
    from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer

    ADD_3D_FEATS = True

    feats2D_params = [
        # fps
        {'kind': 'ecfp:4', 'length': 1024},
        {'kind': 'maccs', 'length': 167},
        {'kind': 'topological', 'length': 1024},
        {'kind': 'avalon', 'length': 1024},
        {'kind': 'erg', 'length': 315},
        {'kind': 'layered', 'length': 1024},
        {'kind': 'secfp', 'length': 1024},
        {'kind': 'pattern', 'length': 1024},
        {'kind': 'pharm2D', 'n_jobs': -1, 'length': 1024},

        # normalize
        {'kind': 'estate', 'length': 79},
        {'kind': 'desc2D', 'n_jobs': -1, 'length': 216, 'replace_nan': True},
        {'kind': 'mordred', 'n_jobs': -1, 'length': 1613},
        {'kind': 'cats2D', 'n_jobs': -1, 'length': 189, 'replace_nan': True},
        {'kind': 'scaffoldkeys', 'n_jobs': -1, 'length': 42, 'replace_nan': True},
        {'kind': 'skeys', 'n_jobs': -1, 'length': 42, 'replace_nan': True},
    ]

    feat3D_params = [
        {'kind': 'desc3D', 'length': 639, 'replace_nan': True},
        {'kind': 'cats3D', 'length': 126, 'replace_nan': True},
        {'kind': 'pharm3D', 'length': 1024, },
        {'kind': 'electroshape', 'length': 15, 'replace_nan': True},
        {'kind': 'usr', 'length': 12},
        {'kind': 'usrcat', 'length': 60},
    ]

    dgl_params = [
        {'kind': 'gin_supervised_contextpred', 'n_jobs': -1},
        {'kind': 'gin_supervised_infomax', 'n_jobs': -1},
        {'kind': 'gin_supervised_edgepred', 'n_jobs': -1},
        {'kind': 'gin_supervised_masking', 'n_jobs': -1},
    ]

    hft_params = [
        {'kind': 'MolT5', 'notation': 'smiles', 'random_seed': 42, 'device': 'cuda'},
        {'kind': 'GPT2-Zinc480M-87M', 'notation': 'smiles', 'random_seed': 42, 'device': 'cuda'},
        {'kind': 'Roberta-Zinc480M-102M', 'notation': 'smiles', 'random_seed': 42, 'device': 'cuda'},
    ]

    train_feats = {}
    test_feats = {}

    for params in feats2D_params:
        print(params)
        train_feats[params['kind']] = mem.cache(calc_2d_feats, ignore=['n_jobs', 'dtype'])(train_df.smi, **params)
        test_feats[params['kind']] = mem.cache(calc_2d_feats, ignore=['n_jobs', 'dtype'])(test_df.smi, **params)

    for params in dgl_params:
        print(params)
        train_feats[params['kind']] = mem.cache(calc_dgl_feats, ignore=['n_jobs', 'dtype'])(train_df.smi, **params)
        test_feats[params['kind']] = mem.cache(calc_dgl_feats, ignore=['n_jobs', 'dtype'])(test_df.smi, **params)

    if ADD_3D_FEATS:
        mem.cache(embed_auto3d, ignore=['use_gpu', 'verbose'])(train_df.smi)
        mem.cache(embed_auto3d, ignore=['use_gpu', 'verbose'])(test_df.smi)

        for params in feat3D_params:
            print(params)
            train_feats[params['kind']] = mem.cache(calc_3d_feats, ignore=['n_jobs', 'dtype'])(train_df.smi, **params)
            test_feats[params['kind']] = mem.cache(calc_3d_feats, ignore=['n_jobs', 'dtype'])(test_df.smi, **params)

    # for params in hft_params:
    #     feats = mem.cache(get_hft_predictions, ignore=['n_jobs', 'dtype', 'device'])(train.smi, params, device='cpu', n_jobs=-1)

    # for kind, params in hft_params.items():
    #     print('Initializing', end=' ')
    #     trans = PretrainedHFTransformer(kind, **params, n_jobs=-1)
    #     print(kind, end=' ')
    #     feats = mem.cache(trans.transform)(train.smi)
    #     print(feats.shape[1])

    # for kind, params in hft_params.items():
    #     print('Initializing', end=' ')
    #     trans = PretrainedHFTransformer(kind, **params, n_jobs=-1)
    #     print(kind, end=' ')
    #     feats = mem.cache(trans.transform)(train.smi)
    #     print(feats.shape[1])