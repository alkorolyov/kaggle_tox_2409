from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerParent
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


remover = SaltRemover(defnData='[Cl,Na,K,N,O]')
uncharger = Uncharger()


def preprocess(mol):
    m = remover.StripMol(mol)
    m = uncharger.uncharge(m)
    # m = TautomerParent(m, skipStandardize=True)
    m = TautomerParent(m)
    return m