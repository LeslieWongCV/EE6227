# -*- coding: utf-8 -*-
# @Time    : 2021/3/1 10:27 上午
# @Author  : Yushuo Wang
# @FileName: __init__.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/

import numpy as np
from math import pi as pi

def init(ps, d, xmax, xmin):
    popu_ = np.random.random((ps, d)) * (xmax - xmin) - xmax
    # popu_ = np.ones((ps, d)) * 2
    return popu_


def ft_score(x, ps, d, index):
    tmp = np.zeros(ps)
    f = np.zeros((ps))
    if index == '?':

        if len(x.shape) == 1:
            for i in range(d):
                f += (10 ** 6) ** ((i - 1) / (d - 1)) * x[i] ** 2
        else:
            for i in range(d):
                f += (10 ** 6) ** ((i - 1) / (d - 1)) * x[:, i] ** 2

    elif index == 2:
        if len(x.shape) == 1:  # rosenbrock
            for i in range(d - 1):
                f += 100 * ((x[:, i]) ** 2 - x[:, i + 1]) ** 2 + (x[:, i] - 1) ** 2
        else:
            for i in range(d - 1):
                f += 100 * ((x[:, i]) ** 2 - x[:, i + 1]) ** 2 + (x[:, i] - 1) ** 2

    elif index == 1:
        if len(x.shape) == 1:

            for i in range(d):
                f += x[i]**2
        else:
            for i in range(d):
                f += x[:, i]**2

    elif index == 3:
        tmp = np.zeros(ps)
        tmp2 = np.zeros(ps)

        for i in range(d):
            tmp += x[:, i] ** 2
            tmp2 += np.cos(2 * np.pi * x[:, i])

        f = -20 * np.exp(-0.2 * np.sqrt(1/d * tmp)) \
                 - np.exp(1/d * tmp2) + 20 + np.e

    elif index == 4:
        tmp = np.zeros(ps)
        tmp2 = np.zeros(ps)
        for i in range(d):
            tmp += (x[:, i] ** 2) / 4000
            tmp2 *= np.cos(x[:, i] / np.sqrt(i+1))

        f = tmp - tmp2

    elif index == 5:
        kmax = 21
        a = 0.5
        b = 3
        tmp = np.zeros(ps)
        tmp2 = np.zeros(ps)
        for i in range(d):
            for k in range(kmax):
                tmp += a ** k * np.cos(2 * np.pi * b ** k * (x[:, i] + 0.5))
        for k in range(kmax):
            tmp2 += a ** k * np.cos(2 * np.pi * b ** k * 0.5)
        f = tmp - d * tmp2

        #################################
        # x = x - 80 * np.random.random((NP, D))
        # X = x * np.random.random((NP, D))
    elif index == 6:
        if len(x.shape) == 1:
            for i in range(d):
                f += x[i] ** 2 - 10 * np.cos(2 * pi * x[i]) + 10
        else:
            for i in range(d):
                f += x[:, i] ** 2 - 10 * np.cos(2 * pi * x[:, i]) + 10
    ################################

    elif index == 7:
        y = np.copy(x)
        y_orin = np.copy(x)
        y_mask = np.where(abs(x) < 1/2)
        y = np.round(2 * y) / 2
        y[y_mask] = y_orin[y_mask]
        for i in range(d):
            f += y[:, i] ** 2 - 10*np.cos(2 * np.pi * y[:, i]) + 10

    elif index == 8:
        tmp = np.zeros(ps)
        for i in range(d):
            tmp +=  x[:, i] * np.sin(np.sqrt(np.abs(x[:, i])))
        f = 418.9829 * d - tmp

    elif index == 9:
        tmp = np.zeros(ps)
        tmp2 = np.zeros(ps)
        for i in range(d):
            tmp += (x[:, i] ** 2) / 4000
            tmp2 *= np.cos(x[:, i] / np.sqrt(i+1))

        f = tmp - tmp2

    elif index == 10:
        tmp = np.zeros(ps)
        tmp2 = np.zeros(ps)

        for i in range(d):
            tmp += x[:, i] ** 2
            tmp2 += np.cos(2 * np.pi * x[:, i])

        f = -20 * np.exp(-0.2 * np.sqrt(1/d * tmp)) \
                 - np.exp(1/d * tmp2) + 20 + np.e

    return f