# -*- coding: utf-8 -*-
"""
Created on 2021-6-1

uff angle term
return energy and gradient

@author: likun.yang
"""
import numpy as np

from uff_bond import *


def get_bonds_v2(coors, eles):
    distance_ma = get_distance_matrix(coors)
    ac = get_AC(coors, eles, distance_ma)
    return get_bond_list(ac)


def sorted_pair(a, b):
    if b < a:
        tmp = b
        b = a
        a = tmp
    return [a, b]


def get_angles_list(bonds):
    """
    bonds: all bonds in a mol
    return :angles in a molecule
    angle format: [start,node,end], NOTE: node always in the mid
    """

    angles = []
    N = len(bonds)
    # print("There are %d bonds" % (N))
    # for i in range(N):
    #     bonds[i] = sorted_pair(bonds[i][0], bonds[i][1])

    for i in range(N):
        c = bonds[i]
        i1 = i + 1
        for j in range(i1, N):
            d = bonds[j]
            if c[0] == d[0] and not c[1] == d[1]:
                node = c[0]
                angle = (c[1], node, d[1])
                angles.append(angle)
            elif c[1] == d[0] and not c[0] == d[1]:
                node = c[1]
                angle = (c[0], node, d[1])
                angles.append(angle)
            elif c[0] == d[1] and not c[1] == d[0]:
                node = c[0]
                angle = (c[1], node, d[0])
                angles.append(angle)
            elif c[1] == d[1] and not c[0] == d[0]:
                node = c[1]
                angle = (c[0], node, d[0])
                angles.append(angle)
    # print("There are %d angles" % (len(angles)))
    return angles


def cal_angle_force_constant(atom_type_i, atom_type_j, atom_type_k, theta0):
    G = 332.06
    bondorder = get_bond_order()  # fake function now(2021-6-2)
    Z1_1 = get_uff_par(atom_type_i, "Z1")
    Z1_3 = get_uff_par(atom_type_k, "Z1")
    cosTheta0 = np.cos(theta0)
    r12 = cal_real_bond_length(bondorder, atom_type_i, atom_type_j)
    r23 = cal_real_bond_length(bondorder, atom_type_i, atom_type_j)
    r13 = np.sqrt(r12 ** 2 + r23 ** 2 - 2 * r12 * r23 * cosTheta0)
    beta = (2 * G) / (r12 * r23)

    preFactor = beta * Z1_1 * Z1_3 / int(np.power(r13, 5))
    rTerm = r12 * r23
    innerBit = (
        3.0 * rTerm * (1.0 - cosTheta0 * cosTheta0) - r13 * r13 * cosTheta0
    )
    res = preFactor * rTerm * innerBit
    return res


def cal_cosTheta(a, b, c):
    """
    a,b,c: np.array, (xyz)
    return : cos theta between ab, bc vector
    """
    dis_ab = get_distance(a, b)
    dis_bc = get_distance(b, c)
    norm_ab = (a - b) / dis_ab
    norm_bc = (c - b) / dis_bc
    return norm_ab.dot(norm_bc)


def cal_energy_expansion_coeff(theta0):
    """
    Note: can ONLY apply to general case, Need optimatzation later
    can ref to rdkit uff
    (Likun)
    """
    sinTheta0 = np.sin(theta0)
    cosTheta0 = np.cos(theta0)
    d_C2 = 1.0 / (4.0 * max((sinTheta0 * sinTheta0), 1e-8))
    d_C1 = -4.0 * d_C2 * cosTheta0
    d_C0 = d_C2 * (2.0 * cosTheta0 * cosTheta0 + 1.0)
    return (d_C0, d_C1, d_C2)


def cal_dE_dtheta(forceConstant, d_C1, d_C2, sinTheta, sin2Theta):
    dE_dTheta = (
        -1.0 * forceConstant * (d_C1 * sinTheta + 2.0 * d_C2 * sin2Theta)
    )
    return dE_dTheta


def cal_angle_energy_grad(angle, coors, eles):
    """
    Note : Can only apply to general case. Need Optimazation
    ref : rdkit uff
    2021-6-2 Likun.Yang
    """
    atom_i = angle[0]
    atom_j = angle[1]
    atom_k = angle[2]

    atom_type_i = get_atom_type(eles[atom_i])
    atom_type_j = get_atom_type(eles[atom_j])
    atom_type_k = get_atom_type(eles[atom_k])

    theta0 = get_uff_par(atom_type_j, "theta0")  # the node atom
    theta0 = theta0 / 180.0 * np.pi  # convert to rad

    pos_a = coors[atom_i]
    pos_b = coors[atom_j]
    pos_c = coors[atom_k]

    force_cons = cal_angle_force_constant(
        atom_type_i, atom_type_j, atom_type_k, theta0
    )

    d_C0, d_C1, d_C2 = cal_energy_expansion_coeff(theta0)
    # angle between 2 bonds that construct the angle

    cosTheta = cal_cosTheta(pos_a, pos_b, pos_c)
    # print(cosTheta)

    sinThetaSq = 1.0 - cosTheta * cosTheta
    cos2Theta = cosTheta * cosTheta - sinThetaSq  # cos2x = cos^2x - sin^2x

    angle_term = d_C0 + d_C1 * cosTheta + d_C2 * cos2Theta
    Energy = force_cons * angle_term
    #####
    # start to cal force
    dist_ba = get_distance(pos_b, pos_a)
    dist_bc = get_distance(pos_b, pos_c)

    norm_ab_vec = (pos_a - pos_b) / dist_ba
    norm_cb_vec = (pos_c - pos_b) / dist_bc
    # print(ba_vec, bc_vec)
    # print(cosTheta*ba_vec)

    sinTheta = max(np.sqrt(sinThetaSq), 1e-8)
    # print(sinTheta)
    sin2Theta = 2.0 * sinTheta * cosTheta

    dE_dTheta = cal_dE_dtheta(force_cons, d_C1, d_C2, sinTheta, sin2Theta)
    dTheta_dPos_a = (
        1 / dist_ba * (norm_cb_vec - cosTheta * norm_ab_vec)
    ) / -sinTheta
    dTheta_dPos_c = (
        1 / dist_bc * (norm_ab_vec - cosTheta * norm_cb_vec)
    ) / -sinTheta
    grad_a = dE_dTheta * dTheta_dPos_a
    grad_c = dE_dTheta * dTheta_dPos_c
    grad_b = -(grad_a + grad_c)

    return (Energy, grad_a, grad_b, grad_c)


"""
getting  gradient is Not that straightforward, a good ref is :

Determination of Forces from a Potential in Molecular Dynamics (google it)

this is how rdkit determine grad:


dCos_dS[6] =
                      {1.0 / dist[0] * (r[1].x - cosTheta * r[0].x),
                       1.0 / dist[0] * (r[1].y - cosTheta * r[0].y),
                       1.0 / dist[0] * (r[1].z - cosTheta * r[0].z),
                       1.0 / dist[1] * (r[0].x - cosTheta * r[1].x),
                       1.0 / dist[1] * (r[0].y - cosTheta * r[1].y),
                       1.0 / dist[1] * (r[0].z - cosTheta * r[1].z)};

  g[0][0] += dE_dTheta * dCos_dS[0] / (-sinTheta);
  g[0][1] += dE_dTheta * dCos_dS[1] / (-sinTheta);
  g[0][2] += dE_dTheta * dCos_dS[2] / (-sinTheta);

  g[1][0] += dE_dTheta * (-dCos_dS[0] - dCos_dS[3]) / (-sinTheta);
  g[1][1] += dE_dTheta * (-dCos_dS[1] - dCos_dS[4]) / (-sinTheta);
  g[1][2] += dE_dTheta * (-dCos_dS[2] - dCos_dS[5]) / (-sinTheta);

  g[2][0] += dE_dTheta * dCos_dS[3] / (-sinTheta);
  g[2][1] += dE_dTheta * dCos_dS[4] / (-sinTheta);
  g[2][2] += dE_dTheta * dCos_dS[5] / (-sinTheta);
"""


def get_angles_energy_grad(coors, eles):
    E = 0
    grad = np.zeros(coors.shape)

    bonds = get_bonds_v2(coors, eles)
    angles = get_angles_list(bonds)
    for angle in angles:
        # print(angle)
        atom_i = angle[0]
        atom_j = angle[1]
        atom_k = angle[2]
        tmp = cal_angle_energy_grad(angle, coors, eles)
        E += tmp[0]
        grad[atom_i] += tmp[1]
        grad[atom_j] += tmp[2]
        grad[atom_k] += tmp[3]
    return (E, grad)

