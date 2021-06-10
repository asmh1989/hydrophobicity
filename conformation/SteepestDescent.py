# -*- encoding: utf-8 -*-
"""
@Description:       :
SteepestDescent optimzer 

@Date     :2021/06/07 14:23:13
@Author      :likun.yang
"""
import copy

import numpy as np

from uff_nonBonded import *


def test_f(x):
    return np.square(x - 3.0)


def test_grad_f(x):
    return 2 * (x - 3.0)


def test2(a):
    e = a[0] ** 2 + 2 * a[1] ** 2
    gx = 2 * a[0]
    gy = 4 * a[1]
    return (e, (gx, gy))


def IsNear(a, b, epsilon):
    """
    returns abs(a - b) < epsilon
    """
    return np.abs(a - b) < epsilon


def normalizeGrad(grad):
    """
    project grad to [-1,1]

    grad = grad - grad.mean(axis=0) / grad.max(axis=0) - grad.min(axis=0)
   
    """
    a = grad - grad.mean(axis=0)
    b = grad.max(axis=0) - grad.min(axis=0)
    c = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    return c


def steepest_descent(func, coors, eles, maxIter=100, torerance=1e-05):
    currrent_pos = coors.astype("float64")
    step = 0.2
    trustRadius = 0.75
    counter = 0
    trj = []
    energy = []
    for _ in range(maxIter):
        counter += 1
        Energy, grad = func(currrent_pos, eles)
        energy.append(Energy)
        grad = normalizeGrad(grad)
        # print("energy= {:6.2f}".format(Energy))
        print("grandient= \n{}".format(grad.round(2)))
        tempStep = -grad * step
        newStep = np.where(
            tempStep > trustRadius, trustRadius, tempStep
        )  # positive big step
        newStep = np.where(
            newStep < -trustRadius, -trustRadius, newStep
        )  # negative big step4
        if np.all(np.abs(newStep) < torerance):
            break
        currrent_pos += newStep
        # print(currrent_pos)
        trj.append(np.array(currrent_pos))  # deep copy
        print(counter)
    return (currrent_pos, trj, np.array(energy))
    # # print(newStep)
    #

    """
    # e_n2 = func(currrent_pos, eles)[0]
    # if tempStep > trustRadius:  # positive big step # can use np.where
    #     currrent_pos += trustRadius
    # elif tempStep < -trustRadius:  # negative big step
    #     currrent_pos -= trustRadius
    # else:
    #     currrent_pos += tempStep

    # print("e_n1 = {:6.2f}\n".format(e_n1))
    # print(currrent_pos)

    # if IsNear(e_n2, e_n1, 1.0e-7):
    #    break
    # if e_n2 > e_n1:  # // decrease stepsize
    #     step *= 0.1
    # elif e_n2 < e_n1:  # // increase stepsize
    #     # e_n1 = e_n2
    #     step *= 2.15
    #     if step > 1.0:
    #         step = 1.0
    """
