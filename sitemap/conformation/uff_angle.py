# -*- coding: utf-8 -*-
"""
Created on 2021-6-1

uff angle term
return energy and gradient

@author: likun.yang
"""
import numpy as np

from sitemap.conformation.uff_bond import (
    cal_real_bond_length,
    get_AC,
    get_atom_type,
    get_bond_list,
    get_bond_order,
    get_distance,
    get_distance_matrix,
    get_uff_par,
)


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
    n = len(bonds)
    # print("There are %d bonds" % (n))
    # for i in range(n):
    #     bonds[i] = sorted_pair(bonds[i][0], bonds[i][1])

    for i in range(n):
        c = bonds[i]
        i1 = i + 1
        for j in range(i1, n):
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
    g = 332.06
    bondorder = get_bond_order()  # fake function now(2021-6-2)
    z1_1 = get_uff_par(atom_type_i, "Z1")
    z1_3 = get_uff_par(atom_type_k, "Z1")
    cos_theta0 = np.cos(theta0)
    r12 = cal_real_bond_length(bondorder, atom_type_i, atom_type_j)
    r23 = cal_real_bond_length(bondorder, atom_type_i, atom_type_j)
    r13 = np.sqrt(r12 ** 2 + r23 ** 2 - 2 * r12 * r23 * cos_theta0)
    beta = (2 * g) / (r12 * r23)

    pre_factor = beta * z1_1 * z1_3 / int(np.power(r13, 5))
    r_term = r12 * r23
    inner_bit = 3.0 * r_term * (1.0 - cos_theta0 * cos_theta0) - r13 * r13 * cos_theta0
    res = pre_factor * r_term * inner_bit
    return res


def cal_cos_theta(a, b, c):
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
    sin_theta0 = np.sin(theta0)
    cos_theta0 = np.cos(theta0)
    d_c2 = 1.0 / (4.0 * max((sin_theta0 * sin_theta0), 1e-8))
    d_c1 = -4.0 * d_c2 * cos_theta0
    d_c0 = d_c2 * (2.0 * cos_theta0 * cos_theta0 + 1.0)
    return (d_c0, d_c1, d_c2)


def cal_de_dtheta(force_constant, d_c1, d_c2, sin_theta, sin2_theta):
    de_dtheta = -1.0 * force_constant * (d_c1 * sin_theta + 2.0 * d_c2 * sin2_theta)
    return de_dtheta


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

    force_cons = cal_angle_force_constant(atom_type_i, atom_type_j, atom_type_k, theta0)

    d_c0, d_c1, d_c2 = cal_energy_expansion_coeff(theta0)
    # angle between 2 bonds that construct the angle

    cos_theta = cal_cos_theta(pos_a, pos_b, pos_c)
    # print(cos_theta)

    sin_theta_sq = 1.0 - cos_theta * cos_theta
    cos2_theta = cos_theta * cos_theta - sin_theta_sq  # cos2x = cos^2x - sin^2x

    angle_term = d_c0 + d_c1 * cos_theta + d_c2 * cos2_theta
    energy = force_cons * angle_term
    #####
    # start to cal force
    dist_ba = get_distance(pos_b, pos_a)
    dist_bc = get_distance(pos_b, pos_c)

    norm_ab_vec = (pos_a - pos_b) / dist_ba
    norm_cb_vec = (pos_c - pos_b) / dist_bc
    # print(ba_vec, bc_vec)
    # print(cos_theta*ba_vec)

    sin_theta = max(np.sqrt(sin_theta_sq), 1e-8)
    # print(sin_theta)
    sin2_theta = 2.0 * sin_theta * cos_theta

    de_dtheta = cal_de_dtheta(force_cons, d_c1, d_c2, sin_theta, sin2_theta)
    dtheta_dpos_a = (1 / dist_ba * (norm_cb_vec - cos_theta * norm_ab_vec)) / -sin_theta
    dtheta_dpos_c = (1 / dist_bc * (norm_ab_vec - cos_theta * norm_cb_vec)) / -sin_theta
    grad_a = de_dtheta * dtheta_dpos_a
    grad_c = de_dtheta * dtheta_dpos_c
    grad_b = -(grad_a + grad_c)

    return (energy, grad_a, grad_b, grad_c)


"""
getting  gradient is Not that straightforward, a good ref is :

Determination of Forces from a Potential in Molecular Dynamics (google it)

this is how rdkit determine grad:


dCos_dS[6] =
                      {1.0 / dist[0] * (r[1].x - cos_theta * r[0].x),
                       1.0 / dist[0] * (r[1].y - cos_theta * r[0].y),
                       1.0 / dist[0] * (r[1].z - cos_theta * r[0].z),
                       1.0 / dist[1] * (r[0].x - cos_theta * r[1].x),
                       1.0 / dist[1] * (r[0].y - cos_theta * r[1].y),
                       1.0 / dist[1] * (r[0].z - cos_theta * r[1].z)};

  g[0][0] += de_dtheta * dCos_dS[0] / (-sin_theta);
  g[0][1] += de_dtheta * dCos_dS[1] / (-sin_theta);
  g[0][2] += de_dtheta * dCos_dS[2] / (-sin_theta);

  g[1][0] += de_dtheta * (-dCos_dS[0] - dCos_dS[3]) / (-sin_theta);
  g[1][1] += de_dtheta * (-dCos_dS[1] - dCos_dS[4]) / (-sin_theta);
  g[1][2] += de_dtheta * (-dCos_dS[2] - dCos_dS[5]) / (-sin_theta);

  g[2][0] += de_dtheta * dCos_dS[3] / (-sin_theta);
  g[2][1] += de_dtheta * dCos_dS[4] / (-sin_theta);
  g[2][2] += de_dtheta * dCos_dS[5] / (-sin_theta);
"""


def get_angles_energy_grad(coors, eles):
    e = 0
    grad = np.zeros(coors.shape)

    bonds = get_bonds_v2(coors, eles)
    angles = get_angles_list(bonds)
    for angle in angles:
        # print(angle)
        atom_i = angle[0]
        atom_j = angle[1]
        atom_k = angle[2]
        tmp = cal_angle_energy_grad(angle, coors, eles)
        e += tmp[0]
        grad[atom_i] += tmp[1]
        grad[atom_j] += tmp[2]
        grad[atom_k] += tmp[3]
    return (e, grad)
