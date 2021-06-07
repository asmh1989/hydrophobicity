# -*- coding: utf-8 -*-

"""
normalize data into [-1,1]
equation: x_i = x_i - x_mean / x_max - x_min
"""


def normalize(x):
    """x: 1D numpy array
    return normalized value
    """
    x_max = x.max()
    x_min = x.min()
    x_mean = x.mean()
    return (x - x_mean) / (x_max - x_min)
