# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 14:55:08 2021
生成分子的connolly surface即分子平面
ref: A fast algorithm for generating smooth molecular dot surface
representations


先生成分子的 extended-radius（er) surface，然后再在er surface上做probe，半径为1.4，
水分子的半径最后去除probe之间的overlap

潜在问题：当取点太少时，比如 30左右，会因为probe 与probe之间的overlap不够，导致部分点没法消除

@author: likun yang
"""

import numpy as np
from sz_py_ext import sa_surface as sa_surface_rust
from sz_py_ext import sa_surface_no_ele as sa_surface_no_ele_rust

from sitemap.core import vdw_radii

GoldenRatio = (1 + 5 ** 0.5) / 2


def dotsphere(n=100):
    """ use Fibonacci Lattice to even distribute points on a unit sphere
    n: number of points on sphere
    先用Fibonacci Lattice在单位圆上均匀取点
    下一步根据原子的坐标平移单位圆"""

    i = np.arange(0, n)
    theta = 2 * np.pi * i / GoldenRatio
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    x, y, z = (
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi),
    )  # 球坐标
    return np.array([x, y, z]).T


def sa_surface(coors, elements, n=40, pr=1.4, enable_ext=True, index=True):
    """ 生solvent accessible ,返回list，list的index为原子的序号
    coors: 体系的xyz坐标，shape：(m * 3)
    elements: 元素，shape：（m * 1))
    n:生成的圆上格点的数目
    pr:probe radaii"""

    if enable_ext:
        return sa_surface_rust(coors, elements, n, pr, index)

    dots = np.zeros((len(coors) * n, 4))

    for i in range(coors.shape[0]):
        _b = dotsphere(n=n)  # 生成圆上的点
        r = vdw_radii[elements[i]] + pr  # 半径
        _b = _b * r  # 根据 vdw半径对点进行放缩
        _c = _b + coors[i]  # 平移生成的圆
        _c = np.insert(_c, 3, i, axis=1)  # 加入原子序号
        dots[i * n : (i + 1) * n] = _c
    for index, coor in enumerate(coors):
        r = vdw_radii[elements[index]] + pr

        # 计算所有球上点与当前原子的距离
        d_ma = np.sum(np.square(coor - dots[:, :-1]), axis=1)

        # 过滤在r^2长度内的点,也就是index 球内的点, 重叠的部分
        dots = dots[d_ma > np.square(r - 0.001)]

    return dots


def sa_surface_no_ele(coors, n=40, pr=1.4, enable_ext=True, index=True):
    if enable_ext:
        return sa_surface_no_ele_rust(coors, n, pr, index)
    dots = np.zeros((len(coors) * n, 4))

    for i in range(coors.shape[0]):
        _b = dotsphere(n=n)  # 生成圆上的点

        _b = _b * pr  # 根据 vdw半径对点进行放缩
        _c = _b + coors[i]  # 平移生成的圆
        _c = np.insert(_c, 3, i, axis=1)  # 加入原子序号
        dots[i * n : (i + 1) * n] = _c
    for coor in coors:
        d_ma = np.sum(np.square(coor - dots[:, :-1]), axis=1)
        dots = dots[d_ma > np.square(pr - 0.001)]

    return dots


def connolly_surface(coors, elements, n=50, pr=1.4, enable_ext=True):
    """
    coors: 体系的xyz坐标，shape：(m * 3)
    elements: 元素，shape：（m * 1))
    r:比vdw半径伸长的半径
    """
    sas_points = sa_surface(coors, elements, n=n, pr=pr, enable_ext=enable_ext)  # 生成sas
    dots = sa_surface_no_ele(sas_points[:, :-1], n=n, pr=pr, enable_ext=enable_ext)  # 以sas为球心，pr为半径做球
    # return dots

    # 开始去除探测小球最外面的点, 认为离原子距离的平方小于10.01都是内层的点
    dots[:, 3] = 0  # label as 0
    for atom in coors:
        d = np.sum(np.square(atom - dots[:, :3]), axis=1)
        indexes = np.where((d < 10.01) & (dots[:, 3] == 0))
        dots[indexes, 3] = 1
    return dots[dots[:, 3] == 1]

    # result = []
    # for point in dots:
    #     for atom in coors:
    #         d = np.sum(np.square(point[:-1] - atom))
    #         if d < 10:
    #             result.append(point)
    #             break
    # return np.vstack(result)
