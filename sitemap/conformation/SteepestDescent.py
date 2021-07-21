# -*- encoding: utf-8 -*-
"""
@Description:       :
SteepestDescent optimzer 

@Date     :2021/06/07 14:23:13
@Author      :likun.yang
"""
# import copy

import numpy as np

# from sitemap.conformation.uff_nonBonded import *


def test_f(x):
    return np.square(x - 3.0)


def test_grad_f(x):
    return 2 * (x - 3.0)


def test2(a):
    e = a[0] ** 2 + 2 * a[1] ** 2
    gx = 2 * a[0]
    gy = 4 * a[1]
    return (e, (gx, gy))


def is_near(a, b, epsilon):
    """
    returns abs(a - b) < epsilon
    """
    return np.abs(a - b) < epsilon


def normalize_grad(grad):
    """
    project grad to [-1,1]
    grad = grad - grad.mean(axis=0) / grad.max(axis=0) - grad.min(axis=0)
    """
    a = grad - grad.mean(axis=0)
    b = grad.max(axis=0) - grad.min(axis=0)
    c = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    return c


def steepest_descent(func, coors, eles, max_iter=100, torerance=1e-05):
    currrent_pos = coors.astype("float64")
    step = 0.2
    trust_radius = 0.75
    counter = 0
    trj = []
    energy = []
    e1 = 1e09
    for _ in range(max_iter):
        e1 = e1
        counter += 1
        e2, grad = func(currrent_pos, eles)
        energy.append(e2)
        grad = normalize_grad(grad)
        # print("energy= {:6.2f}".format(Energy))
        print("grandient= \n{}".format(grad.round(2)))
        temp_step = -grad * step
        new_step = np.where(temp_step > trust_radius, trust_radius, temp_step)  # positive big step
        new_step = np.where(new_step < -trust_radius, -trust_radius, new_step)  # negative big step4
        # if np.all(np.abs(new_step) < torerance):
        #    break
        currrent_pos += new_step  # update positon
        # print(currrent_pos)
        trj.append(np.array(currrent_pos))  # deep copy
        if is_near(e2, e1, 1.0e-6):
            break
        if e2 > e1:  # // decrease stepsize
            step *= 0.1
        elif e2 < e1:  # // increase stepsize
            e1 = e2
            step *= 1.15
            if step > 1.0:
                step = 1.0
    print(counter)
    return (currrent_pos, trj, np.array(energy))
    # # print(new_step)
    #

    """
    # e_n2 = func(currrent_pos, eles)[0]
    # if temp_step > trust_radius:  # positive big step # can use np.where
    #     currrent_pos += trust_radius
    # elif temp_step < -trust_radius:  # negative big step
    #     currrent_pos -= trust_radius
    # else:
    #     currrent_pos += temp_step

    # print("e_n1 = {:6.2f}\n".format(e_n1))
    # print(currrent_pos)

    # if is_near(e_n2, e_n1, 1.0e-7):
    #    break
    # if e_n2 > e_n1:  # // decrease stepsize
    #     step *= 0.1
    # elif e_n2 < e_n1:  # // increase stepsize
    #     # e_n1 = e_n2
    #     step *= 2.15
    #     if step > 1.0:
    #         step = 1.0
    """
