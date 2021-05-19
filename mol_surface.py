# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 14:55:08 2021
生成分子的connolly surface即分子平面
ref: A fast algorithm for generating smooth molecular dot surface representations 


先生成分子的 extended-radius（er) surface，然后再在er surface上做probe，半径为1.4，水分子的半径
最后去除probe之间的overlap

潜在问题：当取点太少时，比如 50左右，会因为probe 与probe之间的overlap不够，导致部分点没法消除

@author: likun yang
"""

import numpy as np

from common import vdw_radii


goldenRatio = (1 + 5**0.5)/2


def dotsphere(n=100):
    """ use Fibonacci Lattice to even distribute points on a unit sphere
    n: number of points on sphere
    先用Fibonacci Lattice在单位圆上均匀取点
    下一步根据原子的坐标平移单位圆"""

    i = np.arange(0, n)
    theta = 2 * np.pi * i / goldenRatio
    phi = np.arccos(1 - 2*(i+0.5)/n)
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * \
        np.sin(phi), np.cos(phi)  # 球坐标
    return(np.array([x, y, z]).T)


def sa_surface_vec(coors, elements, n=40, pr=1.4):
    """ 生solvent accessible ,返回list，list的index为原子的序号
    coors: 体系的xyz坐标，shape：(m * 3)
    elements: 元素，shape：（m * 1))
    n:生成的圆上格点的数目 
    pr:probe radaii"""

    dots = np.zeros((len(coors) * n, 4))
    for i in range(coors.shape[0]):
        _b = dotsphere(n=n)  # 生成圆上的点
        r = vdw_radii[elements[i]] + pr  # 半径
        _b = _b * r  # 根据 vdw半径对点进行放缩
        _c = _b + coors[i]  # 平移生成的圆
        _c = np.insert(_c, 3, i, axis=1)  # 加入原子序号
        dots[i*n:(i+1)*n] = _c
    for index, coor in enumerate(coors):
        r = vdw_radii[elements[index]] + pr

        # 计算所有球上点与当前原子的距离
        d_ma = np.sum(np.square(coor - dots[:, :-1]), axis=1)

        # 过滤在r^2长度内的点,也就是index 球内的点, 重叠的部分
        dots = dots[d_ma > np.square(r-0.001)]

    return(dots)


def sa_surface_no_ele(coors, n=40, pr=1.4):
    dots = np.zeros((len(coors) * n, 4))
    for i in range(coors.shape[0]):
        _b = dotsphere(n=n)  # 生成圆上的点
        _b = _b * pr  # 根据 vdw半径对点进行放缩
        _c = _b + coors[i]  # 平移生成的圆
        _c = np.insert(_c, 3, i, axis=1)  # 加入原子序号
        dots[i*n:(i+1)*n] = _c
    for coor in coors:
        d_ma = np.sum(np.square(coor - dots[:, :-1]), axis=1)
        dots = dots[d_ma > np.square(pr-0.001)]

    return(dots)


def connolly_surface(coors, elements, n=40, pr=1.4):
    '''   
    结果 不如 mol_surface.py 好！
    coors: 体系的xyz坐标，shape：(m * 3)
    elements: 元素，shape：（m * 1))
    r:比vdw半径伸长的半径
    '''
    sas_points = sa_surface_vec(coors, elements, n=int(n/2), pr=pr)  # 生成sas
    dots = sa_surface_no_ele(sas_points[:, :-1], n=n, pr=pr)
    # return dots


#    for index,sa_point in enumerate(sas_points):
#        points = dots[dots[:,3] == index] #该sas point 产生的点
#        coor = coors[int(sa_point[3])]
#        d_ma = np.sum(np.square(coor - points[:,:-1]),axis=1)
#        points = points[d_ma < np.square(2.8)]
#        res.append(points)
#    return np.vstack(res)
#
#        vec_sas_atom = sa_point[:-1] - coors[int(sa_point[3])] #sas点与原子的vector
#        vec_sas_probe = sa_point[:-1] - points[:,:-1] #sas点与probe上点的vector
#        v1 = vec_sas_atom / np.linalg.norm(vec_sas_atom)
#        v2 = vec_sas_probe / np.linalg.norm(vec_sas_probe)
#        cos_angle = v1.dot(v2.T) # cos values of angle bwtween vec_sas_atom and vec_sas_probe
#        points = points[cos_angle > 0.95] #只要角度小于15°的
    result = []
    for point in dots:
        for atom in coors:
            d = np.sum(np.square(point[:-1] - atom))
            if d < 10:
                result.append(point)
                break
    return np.vstack(result)
