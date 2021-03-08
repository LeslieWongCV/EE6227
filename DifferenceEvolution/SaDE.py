# -*- coding: utf-8 -*-
# @Time    : 2021/2/10 11:24 上午
# @Author  : Yushuo Wang
# @fileName: SaDE.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
import DifferenceEvolution as DE
import GenericAlgorithm as GA

NP = 50
D = 10
CRm = np.random.normal(0.5, 0.1)

X = np.arange(-5, 5, 0.1)
Y = X ** 2
ITER = 100
rep_set = np.zeros(4)
LP = 2


# ITER = 3


def rand_1_mutation(popu_):
    offspringSet = np.zeros((NP, D))
    for j in range(NP):
        f = np.random.normal(0.5, 0.3)
        sub_popu = np.delete(popu_, j, axis=0)
        rand_sub = np.arange(sub_popu.shape[0])
        np.random.shuffle(rand_sub)
        offspringSet[j] = sub_popu[rand_sub[4]] + f * (sub_popu[rand_sub[0]] - sub_popu[rand_sub[1]])
    return offspringSet


def rand2best_2_mutation(popu_):
    offspringSet = np.zeros((NP, D))
    for k in range(NP):
        f = np.random.normal(0.5, 0.3)
        sub_popu = np.delete(popu_, k, axis=0)
        rand_sub = np.arange(sub_popu.shape[0])
        np.random.shuffle(rand_sub)
        offspringSet[k] = sub_popu[rand_sub[4]] + f * (sub_popu[rand_sub[0]] - sub_popu[rand_sub[1]]) \
                          + f * (sub_popu[rand_sub[2]] - sub_popu[rand_sub[3]])
    return offspringSet


def cur2rand_1_mutation(popu_):
    offspringSet = np.zeros((NP, D))
    for l in range(NP):
        f = np.random.normal(0.5, 0.3)
        cur = popu_[l]
        sub_popu = np.delete(popu_, l, axis=0)
        rand_sub = np.arange(sub_popu.shape[0])
        np.random.shuffle(rand_sub)
        offspringSet[l] = cur + f * (sub_popu[rand_sub[0]] - cur) \
                          + f * (sub_popu[rand_sub[1]] - sub_popu[rand_sub[2]])
    return offspringSet


def rand_2_mutation(popu_):
    offspringSet = np.zeros((NP, D))
    for i in range(NP):
        f = np.random.normal(0.5, 0.3)
        sub_popu = np.delete(popu_, i, axis=0)
        rand_sub = np.arange(sub_popu.shape[0])
        np.random.shuffle(rand_sub)
        offspringSet[i] = sub_popu[rand_sub[0]] + f * (sub_popu[rand_sub[1]] - sub_popu[rand_sub[2]]) \
                          + f * (sub_popu[rand_sub[3]] - sub_popu[rand_sub[4]])
    return offspringSet


def crossover(popu, off_v, NP, D, CRm):
    off_u = np.copy(popu)
    Cr_rand = np.random.rand(NP * 4, D)
    Cr = np.random.normal(CRm, 0.1, (4 * NP, D))  # CRi = N(CRm, Std)
    Cr_r_index = np.where(Cr_rand <= Cr)  # the result of np.where is a tuple including 2 array
    off_u[Cr_r_index] = off_v[Cr_r_index]
    rand_j = np.random.randint(0, D, size=NP)
    for i in range(NP):
        off_u[i, rand_j] = popu[i, rand_j]
    _ = 1 + 1
    return off_u, Cr


def selection(off_u, popu):
    popu_fitness = DE.fitness_score(popu, 4 * NP, D)
    mask = np.where(DE.fitness_score(off_u, 4 * NP, D) <= popu_fitness)

    off_final = np.copy(popu)
    off_final[mask] = off_u[mask]
    replace_num = len(mask[0])
    rep = np.zeros(4)
    rep[0] = sum(mask[0] <= 50)
    rep[1] = sum(mask[0] <= 100) - rep[0]
    rep[2] = sum(mask[0] <= 150) - rep[1] - rep[0]
    rep[3] = len(mask[0]) - rep[2] - rep[1] - rep[0]
    _ = 1 + 1
    return off_final, rep, replace_num, mask


# f = np.random.normal(0.5, 0.3)  # randomly sampled from normal distribution and applied to EACH TARGET VECTOR
p = np.random.uniform(0, 1)
p1 = 0.25
p2 = 0.5
p3 = 0.75
popu, popu_ft = DE.init_popu(4 * NP, D)
CRtmp_mean = 0

for i in range(ITER):

    for i in range(LP):

        off_v1 = rand_1_mutation(popu[:50])
        off_v2 = rand2best_2_mutation(popu[50:100])
        off_v3 = cur2rand_1_mutation(popu[100:150])
        off_v4 = rand_2_mutation(popu[150:])

        off_v = np.concatenate((off_v1, off_v2, off_v3, off_v4))

        off_u, CRi = crossover(popu, off_v, NP, D, CRm)

        off_final, replace, replace_num, replace_CR = selection(off_u, popu)
        rep_set += replace
        CRtmp_mean += np.mean(CRi[replace_CR])
        _ = 2 + 2
        p1 = (replace[0]) / replace_num
        p2 = (replace[0] + replace[1]) / replace_num
        p3 = (replace[0] + replace[1] + replace[2]) / replace_num
        p = np.random.uniform(0, 1)
        if p < p2:
            if p > p1:
                l = 1
            else:
                l = 0
        else:
            if p > p3:
                l = 3
            else:
                l = 2

        popu = off_final
        popu = np.concatenate((popu[l * 50:50 + l * 50], popu[l * 50:50 + l * 50], popu[l * 50:50 + l * 50],
                               popu[l * 50:50 + l * 50]))  # strategy picking
        # off_ft = fitness_score(off_final, 4*NP, D)
        off_ft = DE.fitness_score(popu, 4 * NP, D)
        _ = 1 + 1

        if i % 2 == 0:
            # GA.go_plot(X, Y, popu[:50], off_ft[:50], 1, 0, 'SADE')
            print(np.min(off_ft))

    # popu_ft = fitness_score(popu, NP, D)


    _ = 1 + 1
    CRm = CRtmp_mean / LP

_ = 1 + 1
