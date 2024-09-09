from rdkit import Chem
from Auto3D.auto3D import options, smiles2mols

if __name__ == '__main__':
    # smiles = ['CCNCC', ]
    smiles = ['[H]c1c2c(c3c(C([H])([H])[H])c([H])c(=O)oc3c1[H])C1(C([H])([H])[H])OOC1(C([H])([H])[H])O2']
    args = options(k=20, use_gpu=True, verbose=True)
    mols = smiles2mols(smiles, args)

    # get the energy and atomic positions out of the mol objects
    for mol in mols:
       print(mol.GetProp('_Name'))
       print('Energy: ', mol.GetProp('E_tot'))  # unit Hartree
       conf = mol.GetConformer()
       for i in range(conf.GetNumAtoms()):
          atom = mol.GetAtomWithIdx(i)
          pos = conf.GetAtomPosition(i)
          print(f'{atom.GetSymbol()} {pos.x:.3f} {pos.y:.3f} {pos.z:.3f}')