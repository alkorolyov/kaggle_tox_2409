import os
import warnings
from joblib import Memory

os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
warnings.filterwarnings("ignore")

mem = Memory(location='../data/.cache')

RANDOM_SEED = 42
