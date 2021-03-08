# -*- coding: utf-8 -*-
# @Time    : 2021/2/8 4:11 下午
# @Author  : Yushuo Wang
# @FileName: DifferenceEvolution.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import GenericAlgorithm as GA
#from math import *

NP = 20
D = 1
F = 0.1  # (0, 0.15) in traditional DifferenceEvolution
Cr = 0.5

X = np.arange(-5, 5, 0.1)
Y = X**2
ITER = 10


def fitness_score(X, NP, D):
    Z = np.zeros(NP)
    for i in range(D):
        Z += X[:, i]**2
    # Z = X * np.sin(10 * pi * X) + 2
    # Z = input[:, 0]**2 + input[:, 1]**2 + input[:, 2]**2 + input[:, 3]**2 + input[:, 4]**2
    return Z


def init_popu(NP, D):
    X = np.random.random([NP, D]) * 10 + -5
    X_fitness = fitness_score(X, NP, D)
    return X, X_fitness


def mutation(popu):
    offspringSet = np.zeros((NP, D))
    for i in range(NP):
        sub_popu = np.delete(popu, i, axis=0)
        rand_sub = np.arange(sub_popu.shape[0])
        np.random.shuffle(rand_sub)
        offspringSet[i] = F * (sub_popu[rand_sub[0]] - sub_popu[rand_sub[1]]) + sub_popu[rand_sub[2]]  # no random is better? -NO
    _ = 1 + 1
    return offspringSet


def crossover(popu, off_v, NP, D, Cr):
    off_u = np.copy(popu)
    Cr_rand = np.random.rand(NP, D)
    Cr_r_index = np.where(Cr_rand <= Cr)
    off_u[Cr_r_index] = off_v[Cr_r_index]
    if D != 1:
        for i in range(NP):
            j_rand = np.random.randint(0, D, size=NP)
            off_u[i, j_rand[i]] = popu[i, j_rand[i]]
    _ = 1 + 1
    return off_u


def selection(off_u, popu, popu_fitness):

    mask = np.where(fitness_score(off_u, NP, D) <= popu_fitness)
    off_final = np.copy(popu)
    off_final[mask] = off_u[mask]
    return off_final

#if __name__ == '__main__':
popu, popu_fitness = init_popu(NP, D)
init_ft = popu_fitness

GA.go_plot(X, Y, popu, popu_fitness, 1, 0, 'DifferenceEvolution')

for i in range(ITER):

    iter = i

    off_v = mutation(popu)
    _ = 1 + 1
    off_u = crossover(popu, off_v, NP, D, Cr)

    off_final = selection(off_u, popu, popu_fitness)
    off_ft = fitness_score(off_final, NP, D)
    GA.go_plot(X, Y, off_final, off_ft, iter, 0, 'DifferenceEvolution')

    popu = off_final
    popu_fitness = fitness_score(popu, NP, D)

_ = 1 + 1

