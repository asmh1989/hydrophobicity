# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:03:24 2021
找到分子的pocket
1）建立格点，都标记为1
2）以分子的atom为圆心，r=vdw + 1.4做圆，圆内所有格点标记为0
4）以pas（protein accessible surface） 为圆心，r=10 or 20? ,圆内所有格点标记为0
@author: yangl
"""

# 打格点

import numpy as np
import pandas as pd

from common import vdw_radii


def gen_grid(coors, n=1):
    """ 
    coor: 分子的xyz
    n: 一埃内取点的个数
    """
    n = 1/n  # np.arange中取倒数
    x_min = min(coors[:, 0])
    x_max = max(coors[:, 0])
    x_range = np.arange(int(x_min), int(x_max), n)
    y_min = min(coors[:, 1])
    y_max = max(coors[:, 1])
    y_range = np.arange(int(y_min), int(y_max), n)
    z_min = min(coors[:, 2])
    z_max = max(coors[:, 2])
    z_range = np.arange(int(z_min), int(z_max), n)
    xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
    # 将其用ravel展开成一维，放入dataframe中，并都标记为 1
    res = pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel(), 'z': zz.ravel()})
    return res.values


def sas_search_del(coors, elements, grids, pr=1.4):
    '''
    找到所有以 atom 为圆心，半径=vdw + pr 圆内所有的 格点
    并将其标记为 0
    coors: 分子的xyz坐标
    elements: 分子中元素
    grids: 为该分子生成的格点，np.array
    pr: 伸长的半径，一般为水分子的半径 1.4
    '''
    for index, coor in enumerate(coors):  # 循环格点
        r = vdw_radii[elements[index]] + pr
        d_ma = np.sum(np.square(coor - grids), axis=1)
        grids = grids[d_ma > np.square(r)]
    return grids


def pas_search(coors, elements, grids, n=40, pr=13):
    '''
    pas:protein accessible surface
    找到所有以 pas 为圆心，半径=pr 圆内所有的 格点
    并将其标记为 0
    coors:分子的xyz坐标
    elements:分子的元素
    grids:经过sa_search_* 处理之后的格点
    '''
    from mol_surface import mol_surface
    pas = mol_surface.sa_surface(coors, elements, n=n, pr=pr)
    for coor in np.vstack(pas):  # 循环pas中的每个点
        d_ma = np.sum(np.square(coor - grids), axis=1)
        grids = grids[d_ma > np.square(pr)]
    return grids


def find_pockets(atoms_coors, elements, n=40, pas_r=13):
    #data = pd.read_csv(filename,header=None,sep='\s+')
    #coor = data.iloc[:,1:].values
    #ele = data.iloc[:,0].values
    grids = gen_grid(atoms_coors)
    grids = sas_search_del(atoms_coors, elements, grids, pr=1.4)
    grids = pas_search(atoms_coors, elements, grids, n=n, pr=pas_r)
    return grids


# if __name__ ==  '__main__':
#    test('6FS6-mono_noe4z.xyz')
