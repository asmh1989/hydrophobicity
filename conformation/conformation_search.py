# -*- encoding: utf-8 -*-
"""
@Description:       :
search conformer
@Date     :2021/06/23 10:58:30
@Author      :likun.yang
"""

import copy

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.ML.Cluster import Butina

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
    return mol


# AllChem.MMFFOptimizeMolecule(mol_H, maxIters=200, mmffVariant="MMFF94s")
# Chem.AssignAtomChiralTagsFromStructure(mol_H, replaceExistingTags=True)

###################################################################################
def change_epsilon(mol_H, epilon=1, maxIter=200):

    prop = AllChem.MMFFGetMoleculeProperties(
        mol_H, mmffVariant="MMFF94s"
    )  # get MMFF prop
    prop.SetMMFFDielectricConstant(
        epilon
    )  # change dielectric constant, default value is 1
    for id in range(mol_H.GetNumConformers()):
        ff = AllChem.MMFFGetMoleculeForceField(
            mol_H, prop, confId=id
        )  # load force filed
        ff.Minimize(maxIter)  # minimize the confs


def gen_conf(mol, RmsThresh=0.1, randomSeed=1, numThreads=0):
    nr = int(AllChem.CalcNumRotatableBonds(mol))
    Chem.AssignAtomChiralTagsFromStructure(mol, replaceExistingTags=True)
    if nr <= 3:
        nc = 50
    elif nr > 6:
        nc = 300
    else:
        nc = nr ** 3
    mol_H = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_H, randomSeed=1)
    AllChem.EmbedMultipleConfs(
        mol_H,
        numConfs=nc,
        maxAttempts=1000,
        randomSeed=randomSeed,
        pruneRmsThresh=RmsThresh,
        numThreads=numThreads,
    )  # randomSeed 为了复现结果 #numThreads=0 means use all theads aviliable
    return mol_H


def calcConfRMSD(mol_H, ref_mol):
    RMSD = []
    mol_No_H = Chem.RemoveHs(mol_H)  # remove H to calc RMSD
    for indx, conformer in enumerate(mol_No_H.GetConformers()):
        conformer.SetId(indx)
        _rmsd = rdMolAlign.GetBestRMS(mol_No_H, ref_mol, prbId=indx)
        RMSD.append(_rmsd)
    return np.array(RMSD)


def calc_energy(mol_H, minimizeIts=200):
    results = {}
    num_conformers = mol_H.GetNumConformers()
    for conformerId in range(num_conformers):
        prop = AllChem.MMFFGetMoleculeProperties(mol_H, mmffVariant="MMFF94s")
        ff = AllChem.MMFFGetMoleculeForceField(mol_H, prop, confId=conformerId)
        ff.Initialize()
        if minimizeIts > 0:
            ff.Minimize(maxIts=minimizeIts)
        results[conformerId] = ff.CalcEnergy()
    return results


def calcConfEnergy(mol_H, maxIters=0):
    res = AllChem.MMFFOptimizeMoleculeConfs(
        mol_H, mmffVariant="MMFF94s", maxIters=maxIters, numThreads=0
    )
    return np.array(res)[:, 1]


def energyCleaning(mol_H, confEnergies, cutEnergy=5):
    res = confEnergies
    res = res - res.min()  # get relative energies
    removeIds = np.where(res > cutEnergy)[0]
    for id in removeIds:
        mol_H.RemoveConformer(int(id))
    for idx, conformer in enumerate(mol_H.GetConformers()):
        conformer.SetId(idx)
    return mol_H


def groupEnergyCleaing(mol_H, ConfEnergies, ButinaClusters):
    keepIdx = []
    molCopy = copy.deepcopy(mol_H)
    mol_H.RemoveAllConformers()
    for cluster in ButinaClusters:
        energyGroup = ConfEnergies[list(cluster)]
        df = pd.DataFrame([energyGroup, cluster]).T
        df = df.sort_values(0)
        idx = df[1].values[:1]
        keepIdx.extend(idx)
    for idx in keepIdx:
        mol_H.AddConformer(molCopy.GetConformer(int(idx)))
    return mol_H


def groupCleaing(mol_H, ButinaClusters):
    molCopy = copy.deepcopy(mol_H)
    mol_H.RemoveAllConformers()
    keepId = []
    for cluster in ButinaClusters:
        id = cluster[:2]
        keepId.extend(id)
    for id in keepId:
        mol_H.AddConformer(molCopy.GetConformer(int(id)))
    return mol_H


def getButinaClusters(mol_H, RmstThresh=0.5):
    molNoH = Chem.RemoveHs(mol_H)
    numConfs = molNoH.GetNumConformers()
    rmsma = AllChem.GetConformerRMSMatrix(molNoH)
    ButinaClusters = Butina.ClusterData(
        rmsma, numConfs, distThresh=RmstThresh, isDistData=True, reordering=True
    )
    return ButinaClusters


def postRmsClening(mol_H, RmstThresh=0.5):
    molCopy = copy.deepcopy(mol_H)
    ButinaClusters = getButinaClusters(mol_H, RmstThresh=RmstThresh)
    mol_H.RemoveAllConformers()
    for cluster in ButinaClusters:
        idx = cluster[0]
        mol_H.AddConformer(molCopy.GetConformer(idx))
    return mol_H


def localMinCleaning(mol_H, ConfEn):
    localMin = findLocalMin(ConfEn)
    idx = np.where(localMin)[0]
    molCopy = copy.deepcopy(mol_H)
    mol_H.RemoveAllConformers()
    for i in idx:
        mol_H.AddConformer(molCopy.GetConformer(int(i)))
    return mol_H


def run_Conf_Search(sdf_dataset, sample_size=100, cutEnergy=10, RmstThresh=0.5):
    # num_mols = len(sdf_dataset)
    counter_1 = 0
    counter_2 = 0
    total_num_of_conformer = 0
    sample_size = sample_size
    np.random.seed(0)
    random_ll = np.random.randint(2859, size=(sample_size))
    for id in random_ll:
        sdf_mol = sdf_dataset[int(id)]
        mol = process_mol(sdf_mol)
        mol_H = gen_conf(
            mol, RmsThresh=0.5, randomSeed=1, numThreads=0
        )  # confs stored in side the mol object
        # confEnergies = calcConfEnergy(mol_H, maxIters=200)  #  get its energy
        # mol_H = energyCleaning(mol_H, confEnergies, cutEnergy=cutEnergy)
        # ButinaClusters = getButinaClusters(mol_H, RmstThresh=RmstThresh)
        # mol_H = groupEnergyCleaing(mol_H, confEnergies, ButinaClusters)
        # mol_H = groupCleaing(mol_H, ButinaClusters)
        # mol_H = localMinCleaning(mol_H, confEnergies)
        # mol_H = groupEnergyCleaing(mol_H, confEnergies, ButinaClusters)
        # mol_H = energyCleaning(mol_H, confEnergies, cutEnergy=cutEnergy)
        # mol_H = postRmsClening(mol_H, RmstThresh=RmstThresh)
        # change_epsilon(mol_H, epilon=100)
        num_of_conformer = mol_H.GetNumConformers()
        total_num_of_conformer += num_of_conformer
        rmsd = calcConfRMSD(mol_H, sdf_mol)
        if np.any(rmsd <= 1):
            counter_1 += 1
        if np.any(rmsd <= 2):
            counter_2 += 1
    mean_conformer = total_num_of_conformer / sample_size
    reprocuce_rate_within_1 = counter_1 / sample_size
    reprocuce_rate_within_2 = counter_2 / sample_size
    print("{} mean num of conformers".format(int(mean_conformer)))
    print("{:.0%} reprocuce rate within RMSD 1".format(reprocuce_rate_within_1))
    print("{:.0%} reprocuce rate within RMSD 2".format(reprocuce_rate_within_2))


import matplotlib.pyplot as plt


def plotEn(en):
    plt.plot(en)
    plt.scatter(range(len(en)), en)
    for i in range(len(en)):
        plt.annotate(i, xy=(i, en[i]))


def findLocalMin(data):
    res = np.r_[True, data[1:] < data[:-1]] & np.r_[data[:-1] < data[1:], True]
    return res


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
