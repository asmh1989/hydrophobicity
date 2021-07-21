# -*- encoding: utf-8 -*-
"""
@Description:       :
the uff non-bonded term
return energy and gradient
@Date     :2021/06/04 15:58:46
@Author      :likun.yang
"""


import numpy as np
from uff_bond import get_atom_type, get_distance, get_uff_par


def calc_nonbonded_minimum(atom_type_i, atom_type_j):
    x1_i = get_uff_par(atom_type_i, "x1")
    x1_j = get_uff_par(atom_type_j, "x1")
    return np.sqrt(x1_i * x1_j)


def calc_nonbonded_depth(atom_type_i, atom_type_j):
    d1_i = get_uff_par(atom_type_i, "D1")
    d1_j = get_uff_par(atom_type_j, "D1")
    return np.sqrt(d1_i * d1_j)


def cal_energy(r, d_well_depth):
    r6 = np.power(r, 6)
    r12 = r6 * r6
    res = d_well_depth * (r12 - 2.0 * r6)
    return res


def cal_grad(r, d_well_depth, d_xij, pos_i, pos_j, dist):
    r7 = np.power(r, 7)
    r13 = np.power(r, 13)
    pre_factor = 12.0 * d_well_depth / d_xij * (r7 - r13)
    d_grad = pre_factor * (pos_i - pos_j) / dist
    return d_grad


def get_energy_grad(coors, eles):
    e = 0
    grad = np.zeros(coors.shape)
    n_atoms = len(coors)
    for i in range(n_atoms):
        atom_i = eles[i]
        pos_i = coors[i]
        atom_type_i = get_atom_type(atom_i)
        for j in range(i + 1, n_atoms):
            atom_j = eles[j]
            pos_j = coors[j]
            atom_type_j = get_atom_type(atom_j)
            d_xij = calc_nonbonded_minimum(atom_type_i, atom_type_j)
            d_well_depth = calc_nonbonded_depth(atom_type_i, atom_type_j)
            dist = get_distance(pos_i, pos_j)
            r = d_xij / dist
            e += cal_energy(r, d_well_depth)
            g = cal_grad(r, d_well_depth, d_xij, pos_i, pos_j, dist)
            grad[i] += g
            grad[j] -= g
    return (e, grad)
