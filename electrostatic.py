# -*- coding: utf-8 -*-
"""
Created on Fri May  7 14:38:41 2021

@author: yangl
使用简单的库伦定律来计算蛋白质的静电势能
"""

import numpy as np
from sz_py_ext import cal_electro as cal_electro_rust

charged_dict = {'ASP_OD1': -0.5, 'ASP_OD2': -0.5,
                'GLU_OE1': -0.5, 'GLU_OE2': -0.5,
                'LYS_NZ': 1, 'ARG_NH1': 0.5, 'ARG_NH2': 0.5,
                'MN_MN': 2, 'CA_CA': 2, 'MG_MG': 2, 'ZN_ZN': 2}


def get_grids_elec(grids, charged_atoms, n=4):
    elecs = np.zeros(len(grids))
    for index, grid in enumerate(grids):
        #grid = grid.reshape((1,3))
        elec = cal_electro(grid, charged_atoms, n=n)
        elecs[index] = elec
    return elecs


def cal_electro(grid, charged_atoms, n=4):
    '''
    n:介质的介电常数

    调用rust写的扩展, 实现以下代码:
    tmp = np.sqrt(np.sum(np.square(grid[:3] - charged_atoms[:, :3]), axis=1))
            .dot(charged_atoms[:, -1])
    (1/(4*np.pi*n)) * tmp
    '''
    return cal_electro_rust(grid, charged_atoms, n)


def join(r, e):
    r_e = []
    for i in zip(r, e):
        r_e.append('_'.join(i))
    return np.array(r_e)


def get_charge(coors, r_e):
    '''
    返回在charged_dict 中元素的电荷和其坐标
    coors: 蛋白质的坐标
    r_e: 将residue_name and atom name 用‘_'合并
    '''

    res = []
    for key in charged_dict:
        tmp = coors[r_e == key]
        tmp = np.insert(tmp, 3, charged_dict[key], axis=1)
        res.append(tmp)
    return np.vstack(res)


def run_electrosatatic(grids, coors, eles, residue_names):
    r_e = join(residue_names, eles)
    charged = get_charge(coors, r_e)
    return get_grids_elec(grids, charged, n=4)
