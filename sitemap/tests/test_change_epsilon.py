import concurrent.futures
import copy
import logging
from multiprocessing import cpu_count

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


def change_epsilon(mol_h, epilon=1, max_iter=200):
    """可以并行化"""
    prop = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")  # get MMFF prop
    prop.SetMMFFDielectricConstant(epilon)  # change dielectric constant, default value is 1

    for id in range(mol_h.GetNumConformers()):
        ff = AllChem.MMFFGetMoleculeForceField(mol_h, prop, confId=id)  # load force filed
        ff.Minimize(max_iter)  # minimize the confs


def task(mol_h, id, max_iter, epilon):
    mol_copy = copy.deepcopy(mol_h)
    mol_h.RemoveAllConformers()
    prop = AllChem.MMFFGetMoleculeProperties(mol_copy, mmffVariant="MMFF94s")  # get MMFF prop
    prop.SetMMFFDielectricConstant(epilon)  # change dielectric constant, default value is 1
    ff = AllChem.MMFFGetMoleculeForceField(mol_copy, prop, confId=id)  # load force filed
    ff.Initialize()
    ff.Minimize(max_iter)  # minimize the confs
    mol_h.AddConformer(mol_copy.GetConformer(int(id)))
    return mol_h

    # en = ff.CalcEnergy()
    # return (id, en)


def change_epsilon_fast(mol_h, epilon=1, max_iter=200):
    """并行化"""
    res = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [executor.submit(task, mol_h, id, max_iter, epilon) for id in range(mol_h.GetNumConformers())]
        for future in concurrent.futures.as_completed(futures):
            # add result to total data
            res.append(future.result())
        # for id in range(mol_h.GetNumConformers()):
        #    future = executor.submit(task, mol_h, id, max_iter, epilon)
        # for future in concurrent.futures.as_completed(_futures):
        #    res.append(future.result())
    # res.sort()
    return futures

    # for id in range(mol_h.GetNumConformers()):
    #     ff = AllChem.MMFFGetMoleculeForceField(
    #         mol_h, prop, confId=id
    #     )  # load force filed
    #     ff.Minimize(max_iter)  # minimize the confs


def test_change_epsilon():
    s = "CCCCN(CCCC)C(=O)n1cc(C(=O)NC(Cc2ccccc2)C(O)C[NH2+]Cc2cccc(OC)c2)\
c2ccccc21"
    logging.info("11111")
    mol = Chem.MolFromSmiles(s)
    logging.info("222")
    AllChem.EmbedMolecule(mol, randomSeed=1)  # in-place
    logging.info("333")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, 10, numThreads=0, randomSeed=1)  # in-place
    # logging.info("start change_epsilon ...")
    # change_epsilon(mol, epilon=0.001, max_iter=200)
    # logging.info("end change_epsilon ...")

    # logging.info("start change_epsilon_fast ...")
    res = change_epsilon_fast(mol, epilon=0.001, max_iter=200)
    # logging.info("end change_epsilon_fast ...")

    return res


"""'
ref
with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
         futures = [executor.submit(self.analyze, patent_number) for patent_number in self.patent_number_list]
         for future in concurrent.futures.as_completed(futures):
             # add result to total data
             total_data['data'].append(future.result())
"""
