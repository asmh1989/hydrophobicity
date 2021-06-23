# -*- encoding: utf-8 -*-
"""
@Description:       :
search conformer
@Date     :2021/06/23 10:58:30
@Author      :likun.yang
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

"""
do this, from sdf to smiles then to 3d, to remove the 3d 
memory of the mol
"""
path = "/home/yanglikun/git/protein/conformation/dataset/platinum_diverse_dataset_2017_01.sdf"
sdfMol = Chem.SDMolSupplier(path)[0]
# molSmiles = Chem.MolToSmiles(sdfMol)
# mol = Chem.MolFromSmiles(molSmiles)
# mol_H = Chem.AddHs(mol)
mol_H = Chem.AddHs(sdfMol)
# Generate conformers (stored in side the mol object)
num_of_conformer = 50
cids = AllChem.EmbedMultipleConfs(
    mol_H,
    numConfs=num_of_conformer,
    randomSeed=1,
    pruneRmsThresh=0.1,
    numThreads=0,
)  # randomSeed 为了复现结果 #numThreads=0 means use all theads aviliable

#############################################################
# mmff optimazation
# AllChem.MMFFOptimizeMoleculeConfs(
#     mol_H, maxIters=400, mmffVariant="MMFF94s", numThreads=0
# )  # default maxIters is 200


################################################
"""
begin to compare conformers and the target
"""
mol = Chem.RemoveHs(mol_H)  # remove H to calc RMSD
RMSD = []
num_of_conformer = mol.GetNumConformers()
print("Number of conformers:{}".format(num_of_conformer))
for idx in range(num_of_conformer):
    rmsd = rdMolAlign.GetBestRMS(mol, sdfMol, idx)
    RMSD.append(rmsd)
res = np.array(RMSD)
per_under2 = np.count_nonzero(res <= 2) / num_of_conformer
print("{:.0%} within RMSD 2".format(per_under2))
per_under1 = np.count_nonzero(res <= 1) / num_of_conformer
print("{:.0%} within RMSD 1".format(per_under1))


# ids = list(cids)  # You can reach conformers by ids


# results_UFF = AllChem.UFFOptimizeMoleculeConfs(mol_h_UFF, maxIters=max_iter)
# # results_MMFF = AllChem.MMFFOptimizeMoleculeConfs(mol_h_MMFF,maxIters=max_iter)


# # Search for the min energy conformer from results(tuple(is_converged,energy))
# print("Searching conformers by UFF ")
# for index, result in enumerate(results_UFF):
#     if min_energy_UFF > result[1]:
#         min_energy_UFF = result[1]
#         min_energy_index_UFF = index
#         print(min_energy_index_UFF, ":", min_energy_UFF)

# # print("\nSearching conformers by MMFF ")
# # for index, result in enumerate(results_MMFF):
# #    if(min_energy_MMFF>result[1]):
# #        min_energy_MMFF=result[1]
# #        min_energy_index_MMFF=index
# #        print(min_energy_index_MMFF,":",min_energy_MMFF)


# # Write minimum energy conformers into a SDF file
# w = Chem.SDWriter("minimum-energy-conformer-UFF.sdf")
# w.write(Chem.Mol(mol_h_UFF, False, min_energy_index_UFF))
# w.flush()
# w.close()

from rdkit.Chem import rdMolAlign

rdMolAlign
