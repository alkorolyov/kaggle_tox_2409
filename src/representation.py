from tqdm import tqdm
import pandas as pd
import numpy as np

from src.config import feats2D_params, feats3D_params, dgl_params, mem
from src.utils import smi2mol, embed3d
from src.fingerprints import get_fingerprints
from src.descriptors import get_rd_descriptors, get_md_descriptors, calc_2d_feats, calc_3d_feats, calc_dgl_feats

from joblib import Parallel, delayed
from rdkit import RDLogger


def get_representation(smiles: str, config=None) -> pd.Series | None:
    """
    This function computes the vector representation for a single molecule
    Representation consists of stacked vectors of fingerprints and rdkit descriptors in the following order:
     - morgan fingerprints, size MORGAN_FP_SIZE, radius MORGAN_RADIUS
     - avalon fingerprints, size AVALON_FP_SIZE
     - erg fingerprints, size ERG_FP_SIZE
     - rdkit descriptors, size RD_DESCS_SIZE
    """

    RDLogger.DisableLog('rdApp.*')

    mol = smi2mol(smiles)
    if mol is None:
        return

    # default config
    if config is None:
        config = {
            'fps_config': None,
            'rd_descriptors': None,
            'md_descriptors': None
        }

    results = []

    for k, v in config.items():
        if k == 'fps_config':
            results.append(get_fingerprints(mol, v))
        elif k == 'rd_descriptors':
            results.append(get_rd_descriptors(mol, v))
        elif k == 'md_descriptors':
            results.append(get_md_descriptors(mol, v))

    return pd.concat(results)


def get_representation_from_series(smiles: pd.Series, config=None, n_jobs=1) -> pd.DataFrame:
    if n_jobs == 1:
        return smiles.apply(get_representation, config)
    elif n_jobs > 1:
        res = Parallel(n_jobs=n_jobs)(delayed(get_representation)(smi, config) for smi in smiles)
        return pd.DataFrame(res, index=smiles.index)
    else:
        raise ValueError('n_jobs must be 1 or greater.')


def calc_all_features(smiles, inclue_3D=False):
    def get_length(arr):
        for a in arr:
            if a is not None:
                return a.shape[0]

    def feats_np_to_df(arr, params):
        length = get_length(arr)
        nan_arr = np.array([np.nan] * length)
        arr = [nan_arr if a is None else a for a in arr]  # fill None with numpy nan array
        arr = np.stack(arr)
        cols = [params['kind'] + '_' + str(i) for i in range(length)]
        return pd.DataFrame(arr, columns=cols, index=smiles.index)

    df = pd.DataFrame(index=smiles.index)

    for params in tqdm(feats2D_params, desc='feats2D'):
        arr = mem.cache(calc_2d_feats, ignore=['n_jobs'], verbose=False)(smiles, **params)
        # print(params['kind'], arr[1].shape)
        feats_df = feats_np_to_df(arr, params)
        df = pd.concat([df, feats_df], axis=1)

    if inclue_3D:
        mem.cache(embed3d)(smiles)
        for params in tqdm(feats3D_params, desc='feats3D'):
            arr = mem.cache(calc_3d_feats, verbose=False)(smiles, **params)
            # print(params['kind'], arr[1].shape)
            feats_df = feats_np_to_df(arr, params)
            df = pd.concat([df, feats_df], axis=1)

    for params in tqdm(dgl_params, desc='dgl'):
        try:
            arr = mem.cache(calc_dgl_feats, verbose=False)(smiles, **params)
            feats_df = feats_np_to_df(arr, params)
            df = pd.concat([df, feats_df], axis=1)
        except:
            pass
    return df