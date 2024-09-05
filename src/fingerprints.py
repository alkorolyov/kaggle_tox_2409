import numpy as np
import pandas as pd

from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import rdReducedGraphs, rdFingerprintGenerator
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import ExplicitBitVect

MORGAN_FP_SIZE = 1024
MORGAN_RADIUS = 2
AVALON_FP_SIZE = 1024
ERG_FP_SIZE = 315  # Constant value, cannot be changed
FPS_SIZE = MORGAN_FP_SIZE + MORGAN_FP_SIZE + ERG_FP_SIZE

FP_DEFAULT_CONFIG = {
    'morgan': {
        'n_bits': 1024,
        'radius': 2,
    },
    'avalon': {
        'n_bits': 1024,
    },
    'erg': {},
}


def fp2numpy(x: ExplicitBitVect):
    return np.frombuffer(x.ToBitString().encode(), 'u1') - ord('0')


def get_morgan_fp(mol, radius=MORGAN_RADIUS, n_bits=MORGAN_FP_SIZE) -> np.ndarray:
    bitvect = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return fp2numpy(bitvect)


def get_avalon_fp(mol, n_bits=AVALON_FP_SIZE) -> np.ndarray:
    bitvect = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
    return fp2numpy(bitvect)


def get_erg_fp(mol) -> np.ndarray:
    return rdReducedGraphs.GetErGFingerprint(mol)


def get_fingerprints(mol, config=None) -> pd.Series:

    fps = []

    if config is None:
        config = FP_DEFAULT_CONFIG

    for k, v in config.items():
        if k == 'morgan':
            fps.append(get_morgan_fp(mol, **v))
        elif k == 'avalon':
            fps.append(get_avalon_fp(mol, **v))
        elif k == 'erg':
            fps.append(get_erg_fp(mol))
    res = np.concatenate(fps)
    idx = pd.RangeIndex(start=0, stop=len(res)).astype(str)
    return pd.Series(res, index=idx)
