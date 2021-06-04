# -*- encoding: utf-8 -*-
'''
@Description:       :
the uff non-bonded term
return energy and gradient
@Date     :2021/06/04 15:58:46
@Author      :likun.yang
'''


from conformation.uff_bond import get_atom_type, get_uff_par, get_distance

import numpy as np


def calcNonbondedMinimum(atom_type_i, atom_type_j):
    x1_i = get_uff_par(atom_type_i, 'x1')
    x1_j = get_uff_par(atom_type_j, 'x1')
    return np.sqrt(x1_i*x1_j)


def calcNonbondedDepth(atom_type_i, atom_type_j):

    D1_i = get_uff_par(atom_type_i, 'D1')
    D1_j = get_uff_par(atom_type_j, 'D1')
    return np.sqrt(D1_i * D1_j)


def calEnergy(r, d_wellDepth):
    r6 = np.power(r, 6)
    r12 = r6 * r6
    res = d_wellDepth * (r12 - 2.0 * r6)
    return res


def calGrad(r, d_wellDepth, d_xij, pos_i, pos_j, dist):
    r7 = np.power(r, 7)
    r13 = np.power(r, 13)
    preFactor = 12. * d_wellDepth / d_xij * (r7 - r13)
    dGrad = preFactor * (pos_i - pos_j) / dist
    return dGrad


def get_energy_grad(coors, eles):
    e = 0
    grad = np.zeros(coors.shape)
    n_atoms = len(coors)
    for i in range(n_atoms):
        atom_i = eles[i]
        pos_i = coors[i]
        atom_type_i = get_atom_type(atom_i)
        for j in range(i+1, n_atoms):
            atom_j = eles[j]
            pos_j = coors[j]
            atom_type_j = get_atom_type(atom_j)
            d_xij = calcNonbondedMinimum(atom_type_i, atom_type_j)
            d_wellDepth = calcNonbondedDepth(atom_type_i, atom_type_j)
            dist = get_distance(pos_i, pos_j)
            r = d_xij / dist
            e += calEnergy(r, d_wellDepth)
            g = calGrad(r, d_wellDepth, d_xij, pos_i, pos_j, dist)
            grad[i] += g
            grad[j] -= g
    return (e, grad)


def optimization_SD(coors, eles, maxIter=30):
    pos = coors
    for i in range(maxIter):
        E, G = get_energy_grad(pos, eles)
        pos = pos - (0.000001 * G)
        print('E = {:6.2f}\n'.format(E), 'G = {}'.format(G.round(2)))
    return pos
