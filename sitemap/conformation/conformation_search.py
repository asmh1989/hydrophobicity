# -*- encoding: utf-8 -*-
"""
@Description:       :
利用ETKDG算法生成conformations,然后利用能量和RMS在尽量不损失精度的情况下，减小conformation的
数量。
@Date     :2021/06/23 10:58:30
@Author      :likun.yang
"""

import concurrent.futures
import copy
from multiprocessing import cpu_count

# sys.path.append("/home/yanglikun/git/")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# from protein import mol_surface
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.ML.Cluster import Butina

from sitemap.hydrophobicity.mol_surface import sa_surface

# import sys


platinum_diverse_dataset_path = "protein/conformation/dataset/platinum_diverse_dataset_2017_01.sdf"
bostrom_path = "protein/conformation/dataset/bostrom.sdf"
platinum_diverse_dataset = Chem.SDMolSupplier(platinum_diverse_dataset_path)
bostrom_dataset = Chem.SDMolSupplier(bostrom_path)


def process_mol(sdf_mol):
    """
    do this, from sdf to smiles then to 3d, to remove the 3d 
    memory of the mol
    """
    mol_smiles = Chem.MolToSmiles(sdf_mol)
    mol = Chem.MolFromSmiles(mol_smiles)
    return mol


# AllChem.MMFFOptimizeMolecule(mol_h, max_iters=200, mmffVariant="MMFF94s")
# Chem.AssignAtomChiralTagsFromStructure(mol_h, replaceExistingTags=True)

###################################################################################
def change_epsilon2(mol_h, id, max_iter, epilon):
    prop = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")  # get MMFF prop
    prop.SetMMFFDielectricConstant(epilon)  # change dielectric constant, default value is 1
    ff = AllChem.MMFFGetMoleculeForceField(mol_h, prop, confId=id)  # load force filed
    ff.Initialize()
    ff.Minimize(max_iter)  # minimize the confs
    en = ff.CalcEnergy()
    return (id, en)


def change_epsilon_parallel(mol_h, epilon=1, max_iter=200):
    """并行化"""
    res = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [
            executor.sucmit(change_epsilon2, mol_h, id, max_iter, epilon) for id in range(mol_h.GetNumConformers())
        ]
        for future in concurrent.futures.as_completed(futures):
            # add result to total data
            res.append(future.result())
        # for id in range(mol_h.GetNumConformers()):
        #    future = executor.submit(task, mol_h, id, max_iter, epilon)
        # for future in concurrent.futures.as_completed(_futures):
        #    res.append(future.result())
    res.sort()  # in-place
    return np.array(res)[:, 1]


def change_epsilon(mol_h, epilon=1, max_iter=0):
    """可以并行化"""
    ens = []
    prop = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")  # get MMFF prop
    prop.SetMMFFDielectricConstant(epilon)  # change dielectric constant, default value is 1

    for id in range(mol_h.GetNumConformers()):
        ff = AllChem.MMFFGetMoleculeForceField(mol_h, prop, confId=id)  # load force filed
        # ff.Minimize(max_iter)  # minimize the confs
        en = ff.CalcEnergy()
        ens.append(en)
    return np.array(ens)


def gen_conf(mol, rms_thresh=0.1, random_seed=1, num_threads=0):
    nr = int(AllChem.CalcNumRotatableBonds(mol))
    Chem.AssignAtomChiralTagsFromStructure(mol, replaceExistingTags=True)
    if nr <= 3:
        nc = 50
    elif nr > 6:
        nc = 300
    else:
        nc = nr ** 3
    mol_h = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h, random_seed=1)
    AllChem.EmbedMultipleConfs(
        mol_h,
        num_confs=nc,
        maxAttempts=1000,
        random_seed=random_seed,
        pruneRmsThresh=rms_thresh,
        num_threads=num_threads,
    )  # random_seed 为了复现结果 #num_threads=0 means use all theads aviliable
    return mol_h


def calc_conf_rmsd(mol_h, ref_mol):
    rmsd = []
    mol_no_h = Chem.RemoveHs(mol_h)  # remove H to calc rmsd
    for indx, conformer in enumerate(mol_no_h.GetConformers()):
        conformer.SetId(indx)
        _rmsd = rdMolAlign.GetBestRMS(mol_no_h, ref_mol, prbId=indx)
        rmsd.append(_rmsd)
    return np.array(rmsd)


def calc_energy(mol_h, minimize_its=200):
    results = {}
    num_conformers = mol_h.GetNumConformers()
    for conformer_id in range(num_conformers):
        prop = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
        ff = AllChem.MMFFGetMoleculeForceField(mol_h, prop, confId=conformer_id)
        ff.Initialize()
        if minimize_its > 0:
            ff.Minimize(maxIts=minimize_its)
        results[conformer_id] = ff.CalcEnergy()
    return results


def calc_conf_energy(mol_h, max_iters=0):
    res = AllChem.MMFFOptimizeMoleculeConfs(mol_h, mmffVariant="MMFF94s", max_iters=max_iters, num_threads=0)
    return np.array(res)[:, 1]


def energy_cleaning(mol_h, conf_energies, cut_energy=5):
    res = conf_energies
    res = res - res.min()  # get relative energies
    # res = res - res.mean()
    remove_ids = np.where(res > cut_energy)[0]
    for id in remove_ids:
        mol_h.RemoveConformer(int(id))
    for idx, conformer in enumerate(mol_h.GetConformers()):
        conformer.SetId(idx)
    return mol_h


def group_energy_cleaing(mol_h, conf_energies, butina_clusters):
    keep_idx = []
    mol_copy = copy.deepcopy(mol_h)
    mol_h.RemoveAllConformers()
    for cluster in butina_clusters:
        energy_group = conf_energies[list(cluster)]
        df = pd.DataFrame([energy_group, cluster]).T
        df = df.sort_values(0)
        idx = df[1].values[:1]
        keep_idx.extend(idx)
    for idx in keep_idx:
        mol_h.AddConformer(mol_copy.GetConformer(int(idx)))
    return mol_h


def group_cleaing(mol_h, butina_clusters):
    mol_copy = copy.deepcopy(mol_h)
    mol_h.RemoveAllConformers()
    keep_id = []
    for cluster in butina_clusters:
        id = cluster[:2]
        keep_id.extend(id)
    for id in keep_id:
        mol_h.AddConformer(mol_copy.GetConformer(int(id)))
    return mol_h


def get_butina_clusters(mol_h, rmst_thresh=0.5):
    mol_noh = Chem.RemoveHs(mol_h)
    num_confs = mol_noh.GetNumConformers()
    rmsma = AllChem.GetConformerRMSMatrix(mol_noh)
    butina_clusters = Butina.ClusterData(rmsma, num_confs, distThresh=rmst_thresh, isDistData=True, reordering=True)
    return butina_clusters


def post_rms_clening(mol_h, rmst_thresh=0.5):
    mol_copy = copy.deepcopy(mol_h)
    butina_clusters = get_butina_clusters(mol_h, rmst_thresh=rmst_thresh)
    mol_h.RemoveAllConformers()
    for cluster in butina_clusters:
        idx = cluster[0]
        mol_h.AddConformer(mol_copy.GetConformer(idx))
    return mol_h


def local_min_cleaning(mol_h, conf_en):
    local_min = find_local_min(conf_en)
    idx = np.where(local_min)[0]
    mol_copy = copy.deepcopy(mol_h)
    mol_h.RemoveAllConformers()
    for i in idx:
        mol_h.AddConformer(mol_copy.GetConformer(int(i)))
    return mol_h


def find_local_min(data):
    res = np.r_[True, data[1:] < data[:-1]] & np.r_[data[:-1] < data[1:], True]
    return res


def count_csp_in_sas(mol_h, n=100):
    mol_noh = Chem.RemoveHs(mol_h)
    res = []
    atoms = np.array([i.GetSymbol() for i in mol_noh.GetAtoms()])
    num_confs = mol_noh.GetNumConformers()
    for i in range(num_confs):
        conf = mol_noh.GetConformer(i)
        pos = conf.GetPositions()
        sa = sa_surface(pos, atoms, n=n, enable_ext=False)
        carbon_idx = np.where((atoms == "C") | (atoms == "S") | (atoms == "P"))[0]
        tmp = np.isin(sa[:, -1], carbon_idx)
        num_carbon = np.count_nonzero(tmp)
        res.append(num_carbon / n)
    return np.array(res)


def no_use(mol_h, n=100):
    mol_noh = Chem.RemoveHs(mol_h)
    res = []
    atoms = np.array([i.GetSymbol() for i in mol_noh.GetAtoms()])
    num_confs = mol_noh.GetNumConformers()
    for i in range(num_confs):
        conf = mol_noh.GetConformer(i)
        pos = conf.GetPositions()
        sa = sa_surface(pos, atoms, n=n, enable_ext=False)
        carbon_idx = np.where((atoms == "C") | (atoms == "S") | (atoms == "P"))[0]
        tmp = np.isin(sa[:, -1], carbon_idx)
        num_carbon = np.count_nonzero(tmp)
        res.append(num_carbon / n)
    return np.array(res)


def normalize_data(data):
    """
    project grad to [-1,1]
    grad = grad - grad.mean(axis=0) / grad.max(axis=0) - grad.min(axis=0)
    """
    a = data - data.mean(axis=0)
    b = data.max(axis=0) - data.min(axis=0)
    c = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    return c


def change_energy_base_on_sas(en, normalized_num_c, coff=1):
    return en - coff * (en * normalized_num_c)


def change_energy_base_on_sas_no_use(en, sasa, coff=100):
    return en - coff * sasa


def run_conf_search(sdf_dataset, sample_size=100, cut_energy=25, rmst_thresh=1, coff=0.1, epilon=1):
    # num_mols = len(sdf_dataset)
    counter_1 = 0
    counter_2 = 0
    total_num_of_conformer = 0
    delta_sasa = []
    delta_e = []
    sample_size = sample_size
    np.random.seed(0)  # 为了结果的复现
    random_ll = np.random.randint(2859, size=(sample_size))
    for id in random_ll:
        sdf_mol = sdf_dataset[int(id)]
        mol = process_mol(sdf_mol)
        mol_h = gen_conf(mol, rms_thresh=0.5, random_seed=1, num_threads=0)  # confs stored in side the mol object
        conf_energies = change_epsilon(mol_h, epilon=epilon)
        # conf_energies = calc_conf_energy(mol_h, max_iters=0)  #  get its energy
        # conf_energies = change_epsilon_parallel(mol_h, epilon=epilon)
        # lowestEnId = conf_energies.argmin()
        # normalized_num_c = count_csp_in_sas(mol_h, n=100)
        # NumCSP = count_csp_in_sas(mol_h, n=100)

        # conf_energies = change_energy_base_on_sas(
        #    conf_energies, normalized_num_c, coff=coff
        # )
        # sasa = no_use(mol_h)
        # conf_energies = change_energy_base_on_sas_no_use(conf_energies, sasa, coff=coff)
        mol_h = energy_cleaning(mol_h, conf_energies, cut_energy=cut_energy)
        mol_h = post_rms_clening(mol_h, rmst_thresh=rmst_thresh)
        # butina_clusters = get_butina_clusters(mol_h, rmst_thresh=rmst_thresh)
        # mol_h = group_energy_cleaing(mol_h, conf_energies, butina_clusters)
        # mol_h = group_cleaing(mol_h, butina_clusters)
        # mol_h = local_min_cleaning(mol_h, conf_energies)
        # mol_h = energy_cleaning(mol_h, conf_energies, cut_energy=cut_energy)

        num_of_conformer = mol_h.GetNumConformers()
        total_num_of_conformer += num_of_conformer
        rmsd = calc_conf_rmsd(mol_h, sdf_mol)
        # lowestRMSId = rmsd.argmin()

        # delta_sasa.append(NumCSP[lowestRMSId] - NumCSP[lowestEnId])
        # delta_e.append(conf_energies[lowestRMSId] - conf_energies[lowestEnId])
        if np.any(rmsd <= 1):
            counter_1 += 1
        if np.any(rmsd <= 2):
            counter_2 += 1
    mean_conformer = total_num_of_conformer / sample_size
    reprocuce_rate_within_1 = counter_1 / sample_size
    reprocuce_rate_within_2 = counter_2 / sample_size
    print("{} mean num of conformers".format(int(mean_conformer)))
    print("{:.0%} reprocuce rate within rmsd 1".format(reprocuce_rate_within_1))
    print("{:.0%} reprocuce rate within rmsd 2".format(reprocuce_rate_within_2))
    # return (np.array(delta_sasa), np.array(delta_e), random_ll)


def plot_en(en, anno):
    plt.plot(en)
    plt.scatter(range(len(en)), en)
    for i, j in enumerate(anno):
        plt.annotate(round(j, 2), xy=(i, en[i]))


def align_conf(mol, ref):
    num_conf = mol.GetNumConformers()
    for id in range(num_conf):
        AllChem.AlignMol(mol, ref, prbCid=id, refCid=0)
    return 0


######################################################################
"""
begin to compare conformers and the target
"""
# mol_no_h = Chem.RemoveHs(mol_h)  # remove H to calc rmsd


# rmsd = []
# num_of_conformer = mol_no_h.GetNumConformers()
# print("Number of conformers:{}".format(num_of_conformer))

# for idx in range(num_of_conformer):
#     _rmsd = rdMolAlign.GetBestRMS(mol_no_h, sdf_mol, prbId=idx)
#     rmsd.append(_rmsd)
# res = np.array(rmsd)


# per_under2 = np.count_nonzero(res <= 2.0) / num_of_conformer
# print("{:.0%} within rmsd 2".format(per_under2))

# per_under1 = np.count_nonzero(res <= 1.0) / num_of_conformer
# print("{:.0%} within rmsd 1".format(per_under1))
##########################################################

# ids = list(cids)  # You can reach conformers by ids


# results_UFF = AllChem.UFFOptimizeMoleculeConfs(mol_h_UFF, max_iters=max_iter)
# # results_MMFF = AllChem.MMFFOptimizeMoleculeConfs(mol_h_MMFF,max_iters=max_iter)


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
