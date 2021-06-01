# -*- coding: utf-8 -*-
"""
Created on Wed May 12 09:01:49 2021

@author: likun.yang
"""
import os
import numpy as np
import pandas as pd
# from rdkit import Chem

uff_par_path = '/home/yanglikun/git/protein/conformation/data/uff.par'

uff_par = pd.read_csv(uff_par_path, sep='\s+',
                      skiprows=148)  # load the uff par

'''
par meaning
took from rdkit

   double r1;            //!<  valence bond radius
   double theta0;        //!< valence angle
   double x1;            //!< vdW characteristic length
   double D1;            //!< vdW atomic energy
   double zeta;          //!< vdW scaling term
   double Z1;            //!< effective charge
   double V1;            //!< sp3 torsional barrier parameter
   double U1;            //!< torsional contribution for sp2-sp3 bonds
   double GMP_Xi;        //!< GMP Electronegativity;
   double GMP_Hardness;  //!< GMP Hardness
   double GMP_Radius;    //!< GMP Radius value
'''

covalent_radii = {'H': 0.23, 'C': 0.68, 'O': 0.68, 'N': 0.68}
'''
2021-5-31 Likun.Yang
# covalent radii
# ref: rdkit Chem.GetPeriodicTable().GetRcovalent()
very crude!!
Need Optimazation !!!
based on bond order !!!
one data source maybe in
"https://chem.libretexts.org/Ancillary_Materials/Reference/Reference_Tables/Atomic_and_Molecular_Properties/A3%3A_Covalent_Radii"
'''


def sorted_pair(a, b):
    if (b < a):
        tmp = b
        b = a
        a = tmp
    return [a, b]


def make_angles(bond):
    newconn = []
    N = len(bond)
    print("There are %d bonds" % (N))
    for i in range(N):
        bond[i] = sorted_pair(bond[i][0], bond[i][1])

    for i in range(N):
        c = bond[i]
        i1 = i+1
        for j in range(i1, N):
            d = bond[j]
            if (c[0] == d[0] and not c[1] == d[1]):
                newconn.append(sorted_pair(c[1], d[1]))
            elif (c[1] == d[0] and not c[0] == d[1]):
                newconn.append(sorted_pair(c[0], d[1]))
            elif (c[0] == d[1] and not c[1] == d[0]):
                newconn.append(sorted_pair(c[1], d[0]))
            elif (c[1] == d[1] and not c[0] == d[0]):
                newconn.append(sorted_pair(c[0], d[0]))
    print("There are %d angles" % (len(newconn)))
    for n in newconn:
        bond.append(n)
    sorted_con = sorted(bond)
    print("There are %d conect" % (len(sorted_con)))
    return sorted_con

    # def calcljE(d):
    #     return 1/np.power(d, 12) - 1/np.power(d, 6)

    # def get_total_E(dis_matrix):
    #     energy = 0
    #     for i in range(dis_matrix.shape[0]):
    #         for j in range(i+1, dis_matrix.shape[0]):
    #             energy += calcljE(dis_matrix[i, j])
    #     return energy

    # def get_grad(atom_coors):
    #     grad = np.zeros((atom_coors.shape[0], 3))
    #     for i in range(atom_coors.shape[0] - 1):
    #         for j in range(i+1, atom_coors.shape[0]):
    #             d = np.linalg.norm(atom_coors[i] - atom_coors[j])
    #             norma_vec = (atom_coors[i] - atom_coors[j]) / d
    #             force = (1/np.power(d, 12) - 1/np.power(d, 6)) * norma_vec
    #             grad[i] += force
    #             grad[j] -= force
    #     return grad

    # def optimization_SD(points):
    #     pos = points
    #     for i in range(100):
    #         dis_ma = gen_distance_matrix(pos)
    #         energy = get_total_E(dis_ma)
    #         grad = get_grad(pos)
    #         pos = pos - (10 * grad)
    #         print(energy)
    #     return pos
