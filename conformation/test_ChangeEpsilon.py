import concurrent.futures
import logging
from multiprocessing import cpu_count

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


def task(mol_H, id, maxIter, epilon):
    prop = AllChem.MMFFGetMoleculeProperties(
        mol_H, mmffVariant="MMFF94s"
    )  # get MMFF prop
    prop.SetMMFFDielectricConstant(
        epilon
    )  # change dielectric constant, default value is 1
    ff = AllChem.MMFFGetMoleculeForceField(
        mol_H, prop, confId=id
    )  # load force filed
    ff.Minimize(maxIter)  # minimize the confs
    return 0


def change_epsilon_fast(mol_H, epilon=1, maxIter=200):
    """并行化"""
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=cpu_count()
    ) as executor:
        for id in range(mol_H.GetNumConformers()):
            executor.submit(task, mol_H, id, maxIter, epilon)

    # for id in range(mol_H.GetNumConformers()):
    #     ff = AllChem.MMFFGetMoleculeForceField(
    #         mol_H, prop, confId=id
    #     )  # load force filed
    #     ff.Minimize(maxIter)  # minimize the confs


def change_epsilon(mol_H, epilon=1, maxIter=200):
    """可以并行化"""
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


def test_ChangeEpsilon():
    s = "CCCCN(CCCC)C(=O)n1cc(C(=O)NC(Cc2ccccc2)C(O)C[NH2+]Cc2cccc(OC)c2)\
c2ccccc21"
    logging.info("11111")
    mol = Chem.MolFromSmiles(s)
    logging.info("222")
    AllChem.EmbedMolecule(mol)  # in-place
    logging.info("333")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, 100, numThreads=0)  # in-place
    logging.info("start change_epsilon ...")
    change_epsilon(mol, epilon=10, maxIter=200)
    logging.info("end change_epsilon ...")

    logging.info("start change_epsilon_fast ...")
    change_epsilon_fast(mol, epilon=10, maxIter=200)
    logging.info("end change_epsilon_fast ...")

    return 0
