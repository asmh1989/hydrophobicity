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

platinum_diverse_dataset_path = "/home/yanglikun/git/protein/conformation/dataset/platinum_diverse_dataset_2017_01.sdf"
bostrom_path = "/home/yanglikun/git/protein/conformation/dataset/bostrom.sdf"
platinum_diverse_dataset = Chem.SDMolSupplier(platinum_diverse_dataset_path)
bostrom_dataset = Chem.SDMolSupplier(bostrom_path)


def process_mol(sdfMol):
    """
    do this, from sdf to smiles then to 3d, to remove the 3d 
    memory of the mol
    """
    molSmiles = Chem.MolToSmiles(sdfMol)
    mol = Chem.MolFromSmiles(molSmiles)
    mol_H = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_H, randomSeed=1)
    return mol_H


# AllChem.MMFFOptimizeMolecule(mol_H, maxIters=200, mmffVariant="MMFF94s")
# Chem.AssignAtomChiralTagsFromStructure(mol_H, replaceExistingTags=True)

###################################################################################


def gen_conf(
    mol_H, num_of_conformer=50, RmsThresh=0.1, randomSeed=1, numThreads=0
):
    cids = AllChem.EmbedMultipleConfs(
        mol_H,
        numConfs=num_of_conformer,
        maxAttempts=1000,
        randomSeed=randomSeed,
        pruneRmsThresh=RmsThresh,
        numThreads=numThreads,
    )  # randomSeed 为了复现结果 #numThreads=0 means use all theads aviliable
    return cids


def calc_RMSD(mol_H, ref_mol):
    RMSD = []
    num_of_conformer = mol_H.GetNumConformers()
    mol_No_H = Chem.RemoveHs(mol_H)  # remove H to calc RMSD
    for idx in range(num_of_conformer):
        _rmsd = rdMolAlign.GetBestRMS(mol_No_H, ref_mol, prbId=idx)
        RMSD.append(_rmsd)
    return np.array(RMSD)


def run_Conf_Search(sdf_dataset):
    # num_mols = len(sdf_dataset)
    counter_1 = 0
    counter_2 = 0
    random_ll = np.random.randint(1000, size=(1000))
    for id in random_ll:
        mol = sdf_dataset[int(id)]
        mol_H = process_mol(mol)
        gen_conf(
            mol_H,
            num_of_conformer=250,
            RmsThresh=0.1,
            randomSeed=1,
            numThreads=0,
        )  # confs stored in side the mol object
        rmsd = calc_RMSD(mol_H, mol)
        if np.any(rmsd <= 1):
            counter_1 += 1
        if np.any(rmsd <= 2):
            counter_2 += 1
    reprocuce_rate_within_1 = counter_1 / 1000
    reprocuce_rate_within_2 = counter_2 / 1000
    print("{:.0%} reprocuce rate within RMSD 1".format(reprocuce_rate_within_1))
    print("{:.0%} reprocuce rate within RMSD 2".format(reprocuce_rate_within_2))


# def ECleaing():


# def RmsCleaning():


#############################################################
# mmff optimazation
# AllChem.MMFFOptimizeMoleculeConfs(
#     mol_H, maxIters=400, mmffVariant="MMFF94s", numThreads=0
# )  # default maxIters is 200
######################################################
# change dielectricConstant
# for id in range(200):
#     prop = AllChem.MMFFGetMoleculeProperties(mol_H, mmffVariant="MMFF94s")
#     prop.SetMMFFDielectricConstant(10)
#     ff = AllChem.MMFFGetMoleculeForceField(mol_H, prop, confId=id)
#     ff.Minimize()
##############################################################
# uff optimazation
# AllChem.UFFOptimizeMoleculeConfs(mol_H, maxIters=400, numThreads=0)
################################################

######################################################################
"""
begin to compare conformers and the target
"""
# mol_No_H = Chem.RemoveHs(mol_H)  # remove H to calc RMSD


# RMSD = []
# num_of_conformer = mol_No_H.GetNumConformers()
# print("Number of conformers:{}".format(num_of_conformer))

# for idx in range(num_of_conformer):
#     _rmsd = rdMolAlign.GetBestRMS(mol_No_H, sdfMol, prbId=idx)
#     RMSD.append(_rmsd)
# res = np.array(RMSD)


# per_under2 = np.count_nonzero(res <= 2.0) / num_of_conformer
# print("{:.0%} within RMSD 2".format(per_under2))

# per_under1 = np.count_nonzero(res <= 1.0) / num_of_conformer
# print("{:.0%} within RMSD 1".format(per_under1))
##########################################################

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
