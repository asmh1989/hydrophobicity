# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:37:11 2021

@author: yangl

计算pocket网格的疏水性
"""
import numpy as np
import pandas as pd

import pdb_io
from common import mkdir_by_file, vdw_radii
from find_pocket import layer_grids
from mol_surface import sa_surface_vec


atomic_hydrophobicity_file_path = 'data/atomic_hydrophobicity.csv'
atomic_hydrophobicity = pd.read_csv(
    atomic_hydrophobicity_file_path).iloc[:, :3].values


def get_atomic_sovation_para(resn, atom):
    ''' find atomic_sovation_para in the atomic_hydrophobicity table
    '''
    if atom == 'OXT':
        atomic_sovation_para = -6  # termeial O,与残基无关
    elif atom == 'NA':
        atomic_sovation_para = -12  # 因为pandas读取时会错误的将‘NA’认为 na（miss value)
    else:
        index = np.where(((atomic_hydrophobicity[:, 0] == resn) & (
            atomic_hydrophobicity[:, 1] == atom)))
        atomic_sovation_para = atomic_hydrophobicity[index, 2]
    return atomic_sovation_para


def get_accessible_solvent_area(sa, vdw_r, index, n=40):
    '''
    sa : solvent_accessible_points
    n为产生单位圆时取得点数，要与sa_surface用的点数相同
    index:原子在体系中的index'''
    # len(solvent_accessible_points[index]) : 该原子贡献的SASpoints
    return (4*np.pi*np.square(vdw_r + 1.4) / n * sa[sa[:, -1] == index].shape[0])


def find_within_radii_atoms(grid, atom_coors, elements, resns, sa, n=40):
    '''
    找到以grid为球心，半径=radii之内的所有原子,
    并返回 其坐标 , atomic_sovation_para, assessable_solvent_area, 
    以及 （格点与原子距离) 减去 （原子的VDW半径 + 1.4）

    grid:格点
    atom_coors:体系的原子坐标
    elements: 体系的元素
    resns:残基
    radii：半径，默认为9
    '''
    d = np.sum(np.square(grid[:3] - atom_coors), axis=1)
    felt_atoms = atom_coors[d < 81.01]  # 9^2
    indexes = np.where(d < 81.01)[0]
    d = np.sqrt(d[d < 81.01])
    atomic_sovation_para = np.zeros(len(indexes))
    area = np.zeros(len(indexes))
    #eles = np.array(['X']*len(indexes))
    for i, index in enumerate(indexes):
        element = elements[index]
        resn = resns[index]
        vdw_r = vdw_radii[element]
        atomic_sovation_para[i] = get_atomic_sovation_para(resn, element)
        area[i] = get_accessible_solvent_area(sa, vdw_r, index, n=n)
        d[i] = d[i] - vdw_r - 1.4

    # insert atomic_sovation_para
    felt_atoms = np.insert(felt_atoms, 3, atomic_sovation_para, axis=1)
    felt_atoms = np.insert(felt_atoms, 4, area, axis=1)  # aera
    felt_atoms = np.insert(felt_atoms, 5, d, axis=1)  # insert distance
    #felt_atoms = np.insert(felt_atoms,6,eles,axis=1)
    return (felt_atoms)


def get_grid_layer_value(grid):
    return grid[-1]


def find_within_radii_grids(grid, grids):
    '''
    grids: shape(n,4)
    '''
    d = np.sum(np.square(grid[:3] - grids[:, :3]), axis=1)
    felt_grids = grids[d < 81.01]  # 9^2 + .1
    d = np.sqrt(d[d < 81.01])
    layer_velues = felt_grids[:, -1]
    new_water_hydro = layer_velues.dot(
        np.exp(-0.7*d)) / np.sum(np.exp(-0.7*d))
    return np.insert(grid, 4, new_water_hydro)
    # return(d, felt_grids)


def cal_hydro_atoms(felt_atoms):
    hydro_atom = 0
    for atom in felt_atoms:
        atomic_sovation_para = atom[3]
        area = atom[4]
        distance = atom[5]
        hydro_atom += atomic_sovation_para * area * np.exp(-0.7*(distance))
    return hydro_atom


def cal_grids_hydro(layerd_grids, atom_coors, elements, resns, solvent_accessible_points, n=40):
    '''
    for all grid points
    pas_r: probe radii
    n:生成单位球时取点的个数，要与sas保持一致
    radii： 格点的寻找半径，在此范围内的atoms对格点的疏水性有影响
    '''
    grids_hydro = []
    for grid in layerd_grids:
        felt_atoms = find_within_radii_atoms(
            grid, atom_coors, elements, resns, solvent_accessible_points, n=n)
        new_layerd_grid = find_within_radii_grids(grid, layerd_grids)
        _hydro = cal_hydro_atoms(felt_atoms) + \
            get_grid_layer_value(new_layerd_grid)
        grid = np.insert(grid[:3], 3, _hydro)
        grids_hydro.append(grid)
    grids_hydro = np.vstack(grids_hydro)
    grids_hydro[:, -1] = grids_hydro[:, -1] * 0.1  # 为了显示，疏水性 * 0.1
    return grids_hydro


def run_hydro_vec(filename, n=40, pas_r=20, dir='.'):
    mkdir_by_file(dir, isDir=True)
    atom_coors, eles, resns = pdb_io.read_pdb(filename)
    layered_grids = layer_grids(
        atom_coors, eles, n=n, pr=pas_r)
    pdb_io.to_pdb(
        layered_grids, '{}/{}_pocket_grids_layered.pdb'.format(dir, filename[:-4]))
    sa = sa_surface_vec(atom_coors, eles, n=n, pr=1.4)
    pdb_io.to_xyz(sa, '{}/{}_SAS.xyz'.format(dir, filename[:-4]))
    hyo = cal_grids_hydro(layered_grids, atom_coors, eles, resns, sa, n=n)
    pdb_io.to_pdb(hyo, '{}/{}_hyo.pdb'.format(dir, filename[:-4]))
    print('Done')
