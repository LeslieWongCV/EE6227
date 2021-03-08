# -*- coding: utf-8 -*-
# @Time    : 2021/2/23 12:14 下午
# @Author  : Yushuo Wang
# @FileName: CLPSO_correct.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/
import numpy as np
import sys
sys.path.append('../')
import DifferenceEvolution as DE
import GenericAlgorithm as GA
import copy
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from matplotlib import cm
import Problems

P = 4
D = 5
NP = 1000
W = 0.729
ITER = 10000
V = np.zeros((NP, D))
V_LOW = -5.12 + 3.12
V_UP = 5.12
xmin1 = [-100, -2.048, -32.768, -600, -5.12, -5.12, -500, -0.5, -2.048, -100, -1, -5, -5, -5, -5];
xmax1 = [-x for x in xmin1]
xmin1 = np.array(xmin1)
xmax1 = np.array(xmax1)
h = []
ft = []
std =[]
v_plot = []
p_best_plot = []


def CLPSO(popu, v, W, C1, C2):

    p_best_fd = p_best.copy()
    np.random.shuffle(p_best_fd)  # BUG 1

    rand_Pc = np.random.uniform(0, Pc[-1] + Pc[-1]/10, [NP, D])  #BUG 2
    mask_Pc = np.where(rand_Pc > Pc)
    p_best_fd[mask_Pc] = p_best[mask_Pc]

    g_best = np.repeat(p_best[np.argmin(p_best_ft)][np.newaxis, :], NP, 0)
    v += W * C1 * np.random.random((NP, D)) * C2 * (p_best_fd - popu) + C2 * np.random.random((NP, D)) * (g_best - popu)
    v = np.clip(v, V_LOW, V_UP)
    popu += v

    return popu, v


popu, popu_ft = Problems.init_popu(NP, D, P, V_UP, V_LOW)
#GA.go_plot(X, Y, popu, popu_ft, 1, 0, 'CLPSO_Diffi')
p_best = popu.copy()
v_set = np.zeros((NP, D))
v = np.zeros((NP, D))
p_best_ft = popu_ft.copy()

t = 5 * np.arange(0, 1, 1/NP)
Pc = 0.5 * (np.exp(t) - np.exp(t[0])) / np.exp(t[-1] - np.exp(t[0]))  # Sefl-Adaptive Pc
Pc = np.tile((Pc[np.newaxis, :]).T, D)  # Trans from D*NP to NP*D

for i in range(ITER):
    W = -(0.8 / ITER) * i + 1
    C1 = -(1.5 / ITER) * i + 3
    C2 = (1.5 / ITER) * i

    #popu_ft = Problems.fitness_score(popu, NP, D, P)

    popu, V = CLPSO(popu, V, W, C1, C2)

    popu_ft = Problems.fitness_score(popu, NP, D, P)

    h.append(np.mean(p_best))
    ft.append(np.min(p_best_ft))
    std.append(np.std(popu))
    v_plot.append(np.mean(V))

    p_best_mask = np.where(popu_ft <= p_best_ft)
    tmp = np.zeros((NP,D)) + p_best
    p_best[p_best_mask] = popu[p_best_mask]
    tmp = tmp - p_best
    #p_best_ft[p_best_mask] = popu_ft[p_best_mask]  # BUG fixed
    p_best_ft = Problems.fitness_score(p_best, NP, D, P)

    p_best_plot.append(np.mean(tmp))
    #p_best_fd = pp__best

    if i % 50000 == 0:
        #GA.go_plot(X, Y, popu, p_best_ft, 1, 0, 'CLPSO_Diffi')
        print('############ Generation {} ############'.format(str(i + 1)))
        print('Best Position：{}'.format(p_best[np.argmin(p_best_ft)]))
        print('Best Fitness Score：{}'.format(np.min(ft)))
        print('Mean Fitness Score：{}'.format(np.mean(ft)))
        print('Best 【popu】：{}'.format(np.min(popu_ft)))
        print('Mean 【popu】：{}'.format(np.mean(popu_ft)))
        _ = 1 + 1

print("---- End of (successful) Searching ----")
print('Best Position：{}'.format(p_best[np.argmin(p_best_ft)]))
print('Best Fitness Score：{}'.format(np.min(ft)))
print('Mean Fitness Score：{}'.format(np.mean(ft)))

ft = np.array(ft)
h = np.array(h)
std = np.array(std)
x_ = np.arange(0, ITER, 1)
plt.figure()
plt.title('-Function{}-pbest fitness score'.format(P))
plt.plot(x_, np.log(ft))
plt.show()
plt.figure()
plt.title('-Function{}-popu std'.format(P))
plt.plot(x_, std)
plt.show()
plt.figure()
plt.title('-Function{}-pbest mean'.format(P))
plt.plot(x_, h)
plt.show()
plt.figure()
plt.title('-Function{}-v mean'.format(P))
plt.plot(x_, v_plot)
plt.show()

plt.figure()
plt.title('-Function{}-p_best_change mean'.format(P))
plt.plot(x_, p_best_plot)
plt.show()

_ = 1 + 1


