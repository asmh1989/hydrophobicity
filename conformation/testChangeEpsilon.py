from rdkit import Chem
from rdkit.Chem import AllChem


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


def testChangeEpsilon():
    s = "CCCCN(CCCC)C(=O)n1cc(C(=O)NC(Cc2ccccc2)C(O)C[NH2+]Cc2cccc(OC)c2)c2ccccc21"
    mol = Chem.MolFromSmiles(s)
    AllChem.EmbedMolecule(mol)  # in-place
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, 100)  # in-place
    change_epsilon(mol, epilon=10, maxIter=200)
    return 0
