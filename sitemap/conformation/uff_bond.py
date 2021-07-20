# -*- coding: utf-8 -*-
"""
cal uff bond term
return bond energy and bond force
Ref : 
1) paper, uff, a full periodic table force filed for molecular mechanics and molelular
dynamics simulations

2)open source, rdkit

Created on Wed May 12 09:01:49 2021

@author: likun.yang
"""
import os

import numpy as np
import pandas as pd

# from rdkit import Chem

uff_par_path = "protein/conformation/data/uff.par"

uff_par = pd.read_csv(uff_par_path, sep="\\s+", skiprows=148)  # load the uff par

"""
par meaning
took from rdkit

    r1;            //!<  valence bond radius
    theta0;        //!< valence angle
    x1;            //!< vdW characteristic length
    D1;            //!< vdW atomic energy
    zeta;          //!< vdW scaling term
    Z1;            //!< effective charge
    V1;            //!< sp3 torsional barrier parameter
    U1;            //!< torsional contribution for sp2-sp3 bonds
    GMP_Xi;        //!< GMP Electronegativity;
    GMP_Hardness;  //!< GMP Hardness
    GMP_Radius;    //!< GMP Radius value
"""

covalent_radii = {"H": 0.23, "C": 0.68, "O": 0.68, "N": 0.68}
"""
2021-5-31 Likun.Yang
# covalent radii
# ref: rdkit Chem.GetPeriodicTable().GetRcovalent()
very crude!!
Need Optimazation !!!
based on bond order !!!
one data source maybe in
"https://chem.libretexts.org/Ancillary_Materials/Reference/Reference_Tables/Atomic_and_Molecular_Properties/A3%3A_Covalent_Radii"
"""


def read_xyz(filename, dir="."):
    """
    xyz format
    3 #numer of atoms
      # blank line
    H 0 0 0
    O 0 0 2
    H 0 0 4

    return coor,elements #in numpy array
    """
    file_path = os.path.join(dir, filename)
    df = pd.read_csv(file_path, header=None, sep="\\s+", skiprows=2)
    coors = df.iloc[:, 1:].values
    eles = df.iloc[:, 0].values.astype("str")
    eles = np.char.upper(eles)  # convert to uppercase
    return (coors, eles)


def get_distance(a, b):
    """
    return distance between point a,b
    """
    return np.sqrt(np.sum(np.square(a - b)))


def get_distance_matrix(coors):
    """
    input: numpy array atoms coors xyz
    return: distance matrix
    """
    res = np.zeros((len(coors), len(coors)))
    for i in range(len(coors)):
        for j in range(i + 1, len(coors)):
            d = get_distance(coors[i], coors[j])
            res[i, j] = d
            res[j, i] = d
    return res


def get_AC(coors, eles, distance_matrix, covalent_factor=1.3):
    """

    Generate adjacent matrix from atoms and coordinates.

    AC is a (num_atoms, num_atoms) matrix with 1 being covalent bond and 0 is not

    covalent_factor - 1.3 is an arbitrary factor

    args:
        coors:mol's xyz coors
        eles: mol's elements

    optional
        covalent_factor - increase covalent bond length threshold with facto

    returns:
        AC - adjacent matrix

    """

    # Calculate distance matrix
    # dMat = get_distance_matrix(coors)
    num_atoms = coors.shape[0]
    AC = np.zeros((num_atoms, num_atoms), dtype=int)
    for i in range(num_atoms):
        a_i = eles[i]
        Rcov_i = covalent_radii[a_i] * covalent_factor
        for j in range(i + 1, num_atoms):
            a_j = eles[j]
            Rcov_j = covalent_radii[a_j] * covalent_factor
            if distance_matrix[i, j] < Rcov_i + Rcov_j:
                AC[i, j] = 1
                AC[j, i] = 1

    return AC


def get_bond_order():
    """
    set default value as 1, for simplicity

    Need Change after !!!

    (2021-5-28)Likun.Yang

    Note: 1.42 for aminoacid

    """
    return 1


def get_atom_type(atom):
    """
    Just for simplicity
    very crude
    Need Change after
    only support:  H C O N
    and set atom type as : “Generic“
    note: maybe rdkit atom.GetHybridization()??? Need test!
    2021-5-28
    likun.yang
    """
    atom = atom.upper()
    if atom == "H":
        return "H_"
    elif atom == "C":
        return "C_3"
    elif atom == "O":
        return "O_3"
    elif atom == "N":
        return "N_3"
    else:
        raise KeyError("Not Support Atom Type")


def get_uff_par(atom_type, term):
    return uff_par[term][uff_par["Atom"] == atom_type].values[0]


def cal_bond_energy_and_grad(bond, bond_order, coors, elemets, distance_matrix):
    """
    E_bond = 1/2 * force_cons * (r_curr - r_optima)^2
    """
    atom_i = bond[0]
    atom_j = bond[1]
    atom_type_i = get_atom_type(elemets[atom_i])
    atom_type_j = get_atom_type(elemets[atom_j])

    r_curr = distance_matrix[atom_i, atom_j]
    r_desired = cal_real_bond_length(bond_order, atom_type_i, atom_type_j)
    print("R_desired = {}".format(r_desired))
    force_cons = cal_bond_force_cons(atom_type_i, atom_type_j, r_desired)
    # print("Force Constant = {:6.2f}".format(force_cons))
    E_bond = 0.5 * force_cons * (r_curr - r_desired) ** 2
    # print("Ebond = {:6.2f}".format(E_bond))

    u_vec = (coors[atom_i] - coors[atom_j]) / r_curr
    G_bond = force_cons * (r_curr - r_desired) * u_vec
    return (E_bond, G_bond)


def cal_real_bond_length(bondorder, atom_type_i, atom_type_j):
    """
    return real ideal bond length
    r_ij = r_i + r_j + r_bo - r_en #
    """

    r1_i = get_uff_par(atom_type_i, "r1")
    r1_j = get_uff_par(atom_type_j, "r1")
    Xi_i = get_uff_par(atom_type_i, "Xi")
    Xi_j = get_uff_par(atom_type_j, "Xi")

    # this is the pauling correction
    r_bo = -0.1332 * (r1_i + r1_j) * np.log(bondorder)

    # O'Keefe and Breese electronegativity correction
    r_en = r1_i * r1_j * (np.sqrt(Xi_i) - np.sqrt(Xi_j)) ** 2 / (Xi_i * r1_i + Xi_j * r1_j)

    res = r1_i + r1_j + r_bo - r_en
    return res


def cal_bond_force_cons(atom_type_i, atom_type_j, r_ij_desired):
    Z1_i = get_uff_par(atom_type_i, "Z1")
    Z1_j = get_uff_par(atom_type_j, "Z1")

    G = 332.06  # bond force constant prefactor
    res = 2.0 * G * Z1_i * Z1_j / r_ij_desired ** 3
    return res


def get_bond_list(adjacent_matrix):
    bond_list = []
    shape = adjacent_matrix.shape[0]
    for i in range(shape):
        for j in range(i + 1, shape):
            if adjacent_matrix[i, j] == 1:
                bond_list.append((i, j))
    return bond_list


def get_bonds_energy_grad(coors, elemets):
    bond_order = get_bond_order()
    distance_matrix = get_distance_matrix(coors)
    ac = get_AC(coors, elemets, distance_matrix)
    bond_list = get_bond_list(ac)
    E = 0
    grad = np.zeros(coors.shape)
    for bond in bond_list:
        atom_i = bond[0]
        atom_j = bond[1]
        tmp = cal_bond_energy_and_grad(bond, bond_order, coors, elemets, distance_matrix)
        E += tmp[0]
        G = tmp[1]
        grad[atom_i] += G
        grad[atom_j] -= G
    return (E, grad)
