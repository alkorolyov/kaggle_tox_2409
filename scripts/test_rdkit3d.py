import multiprocessing

import pandas as pd
import datamol as dm
import numpy as np
import seaborn as sns

from molfeat.calc import RDKitDescriptors2D, FPCalculator, MordredDescriptors
from molfeat.trans import MoleculeTransformer, FPVecTransformer
from sklearn.preprocessing import OneHotEncoder

import collections.abc as collections
from molfeat.trans.concat import FeatConcat

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from src.config import mem
from src.utils import embed3d, eval_model, OffsetScaler
from src.descriptors import calc_dgl_feats, calc_hft_feats, calc_3d_feats, calc_2d_feats

from rdkit import RDLogger, Chem
RDLogger.DisableLog('rdApp.*')

if __name__ == '__main__':
    from rdkit import RDConfig
    from rdkit.Chem.SaltRemover import SaltRemover; SaltRemover()
    FPVecTransformer('morgan')


    # train = dm.read_csv('../data/processed/train.csv', smiles_column='smi', index_col=0)[:5]
    # test = dm.read_csv('../data/processed/test.csv', smiles_column='smi', index_col=0)[:5]
    #
    # n_confs = 3
    # mols = mem.cache(embed3d, ignore=['n_jobs'])(train.smi[0:1], n_confs=n_confs, n_jobs=1)
    # trans = FPVecTransformer('desc3D', length=639, n_jobs=1)
    #
    # print('trans.transform(mols, ...)')
    # # print(trans.transform(mols, conformer_id=0, ignore_errors=False)[0, :5].round(3))
    # print(trans.transform(mols, conformer_id=1, ignore_errors=False)[0, :5].round(3))
    #
    # print('trans(mols, ... ')
    # # print(trans(mols, conformer_id=0, ignore_errors=False)[0, :5].round(3))
    # print(trans(mols, conformer_id=1, ignore_errors=False)[0, :5].round(3))
    #
    # from molfeat.calc.descriptors import RDKitDescriptors3D
    #
    # rd3d = RDKitDescriptors3D()
    # print('rd3d = RDKitDescriptors3D()')
    # # print(rd3d(mols[0], conformer_id=0)[:5].round(3))
    # print(rd3d(mols[0], conformer_id=1)[:5].round(3))
