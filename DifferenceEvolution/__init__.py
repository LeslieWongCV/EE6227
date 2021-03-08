# -*- coding: utf-8 -*-
# @Time    : 2021/2/10 11:58 上午
# @Author  : Yushuo Wang
# @FileName: __init__.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import GenericAlgorithm as GA
from math import *
def fitness_score(input, NP, D):

    Z = np.zeros(NP)
    for i in range(D):
        Z += input[:, i]**2
    #Z = 1/(input-5) * np.sin(10 * pi * input) + 2
    #Z = input[:, 0]**2 + input[:, 1]**2 + (input[:, 2] + 500)**2 + input[:, 3]**2 + input[:, 4]**2 - 200
    return Z


def init_popu(NP, D):
    X = np.random.random([NP, D]) * 200 - 100
    X_fitness = fitness_score(X, NP, D)
    return X, X_fitness
