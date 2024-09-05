import pandas as pd

from src.utils import smi2mol
from src.fingerprints import get_fingerprints
from src.descriptors import get_rd_descriptors, get_md_descriptors

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
