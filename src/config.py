import os
import warnings
from joblib import Memory

os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
warnings.filterwarnings("ignore")

mem = Memory(location='../data/.cache')

RANDOM_SEED = 42

feats2D_params = [
    # fps
    {'kind': 'ecfp:4', 'length': 1024},
    {'kind': 'topological', 'length': 1024, 'n_jobs': -1},
    {'kind': 'avalon', 'length': 1024, 'n_jobs': -1},
    {'kind': 'layered', 'length': 1024, 'n_jobs': -1},
    {'kind': 'secfp', 'length': 1024, 'n_jobs': -1},
    {'kind': 'pattern', 'length': 1024},
    {'kind': 'pharm2D', 'length': 1024, 'n_jobs': -1},

    # fixed length fps
    {'kind': 'erg', 'n_jobs': -1},
    {'kind': 'maccs', 'n_jobs': -1},

    # fixed length descriptors - to normalize
    {'kind': 'estate', 'n_jobs': -1},
    {'kind': 'desc2D', 'n_jobs': -1},
    {'kind': 'mordred', 'n_jobs': -1},
    {'kind': 'cats2D', 'n_jobs': -1},
    {'kind': 'scaffoldkeys', 'n_jobs': -1},
    {'kind': 'skeys', 'n_jobs': -1},
]

feats3D_params = [
    {'kind': 'pharm3D', 'length': 1024, 'n_jobs': -1},
    {'kind': 'desc3D', 'n_jobs': -1},
    {'kind': 'cats3D', 'n_jobs': -1},
    {'kind': 'electroshape', 'n_jobs': -1},
    {'kind': 'usr', },
    {'kind': 'usrcat', },
]

dgl_params = [
    {'kind': 'gin_supervised_contextpred'},
    {'kind': 'gin_supervised_infomax'},
    {'kind': 'gin_supervised_edgepred'},
    {'kind': 'gin_supervised_masking'},
]

hf_params = [
    # {'kind': 'ChemBERTa-77M-MLM', 'random_seed': 42},
    {'kind': 'ChemBERTa-77M-MTR', 'random_seed': 42},
    # {'kind': 'ChemGPT-1.2B', 'random_seed': 42},
    {'kind': 'ChemGPT-19M', 'random_seed': 42},
    {'kind': 'ChemGPT-4.7M', 'random_seed': 42},
    {'kind': 'GPT2-Zinc480M-87M', 'random_seed': 42},
    # {'kind': 'MolT5', 'random_seed': 42},
    {'kind': 'Roberta-Zinc480M-102M', 'random_seed': 42},
]
