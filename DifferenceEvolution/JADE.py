# -*- coding: utf-8 -*-
# @Time    : 2021/2/14 2:42 下午
# @Author  : Yushuo Wang
# @FileName: JADE.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import DifferenceEvolution as DE
import GenericAlgorithm as GA

NP = 50
D = 5
CR_u = np.random.normal(0.5, 0.1)

X = np.arange(-5, 5, 0.1)
Y = X**2
rep_set = np.zeros(4)
LP = 50
ITER = 500
PBEST_RATE = 0.2
mask = np.arange(NP)
c = 0.5
F_u = 0.1


def current2Pbest_1_mutation(popu_, ft, mask, F_u):
    F_set = np.zeros(NP)
    ext_arc = np.ma.array(popu_, mask = False)
    _ = 1 + 1
    ext_arc.mask[mask] = True
    ext_arc = ext_arc.compressed()
    ext_arc = np.reshape(ext_arc, [-1, D])
    ext_arc = np.concatenate((ext_arc, popu_))

    offspringSet = np.zeros((NP, D))
    for i in range(NP):
        #f = np.random.normal(0.5, 0.3)
        f = np.random.standard_cauchy() + F_u
        cur = popu_[i]
        sub_popu = popu_[np.argpartition(ft, -round(PBEST_RATE*NP))[-round(PBEST_RATE*NP):]]

        #sub_popu = np.delete(popu_, i, axis=0)
        rand_sub = np.arange(sub_popu.shape[0])  # will avoid choosing same one 2 times
        np.random.shuffle(rand_sub)  # !!

        rand_popu = np.arange(popu_.shape[0])
        np.random.shuffle(rand_popu)

        rand_ext = np.arange(ext_arc.shape[0])
        np.random.shuffle(rand_ext)

        offspringSet[i] = cur + f*(sub_popu[rand_sub[0]] - cur) \
                          + f*(popu_[rand_popu[0]] - ext_arc[rand_ext[0]])
        F_set[i] = f
    _ = 1 + 1
    return offspringSet, F_set


def crossover(popu, off_v, NP, D, CR_u):
    off_u = np.copy(popu)
    Cr_rand = np.random.rand(NP, D)
    Cr = np.random.normal(CR_u, 0.1, (NP, D))  # CRi = N(CRu, 0.1)
    Cr_r_index = np.where(Cr_rand <= Cr)  # the result of np.where is a tuple including 2 array
    off_u[Cr_r_index] = off_v[Cr_r_index]
    _ = 1 + 1
    # rand_j = np.random.randint(0, D, size=NP)
    # for i in range(NP):
    #     off_u[i, rand_j] = popu[i, rand_j]

    return off_u, Cr



def selection(off_u, popu):
    popu_fitness = DE.fitness_score(popu, NP, D)
    mask = np.where(DE.fitness_score(off_u, NP, D) <= popu_fitness)
    off_final = np.copy(popu)
    off_final[mask] = off_u[mask]

    replace_num = len(mask[0])
    rep = sum(mask[0])

    _ = 1 + 1
    return off_final, rep, replace_num, mask


popu, popu_ft = DE.init_popu(NP, D)
popu_ft_init = popu_ft
#GA.go_plot(X, Y, popu, popu_ft, 1, 0, 'JADE')

_ = 1 + 1
for i in range(ITER):
    off_v, F_set = current2Pbest_1_mutation(popu, popu_ft, mask, F_u)
    off_u, CR = crossover(popu, off_v, NP, D, CR_u)
    _ = 1 + 1
    off_final, rep, replace_num, mask = selection(off_u, popu)

    CR_u = (1 - c) * CR_u + c*np.mean(CR[mask])
    F_u = (1 - c) * F_u + c*np.mean(F_set[mask])  # lehman? truncated ?

    popu = off_final
    popu_ft = DE.fitness_score(popu, NP, D)
    if i % 100 == 0:
        #GA.go_plot(X, Y, popu, popu_ft, 1, 0, 'JADE')
        print(np.min(popu_ft))

    _ = 1 + 1




_ = 1 + 1