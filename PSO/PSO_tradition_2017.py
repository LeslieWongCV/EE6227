# -*- coding: utf-8 -*-
# @Time    : 2021/2/15 3:27 下午
# @Author  : Yushuo Wang
# @FileName: PSO_tradition.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/

import numpy as np
import sys
sys.path.append('../')
#from DE.DE import init_popu, fitness_score, mutation, crossover
import DifferenceEvolution as DE
import Problems
import GenericAlgorithm as GA
import copy
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from __init__ import init, ft_score

P = 1
index = P
NP = 1000
D = 10
ITER = 5000
C1 = 1.0
C2 = 1.0
W = 0.729
Dis = 5
best_all = []
Name = ['Sphere function', 'Rosenbrock’s function', 'Ackley’s function', 'Griewanks’s function', 'Weierstrass function',
        'Rastrigin’s function', 'Noncontinuous Rastrigin’s function', 'Schwefel’s function',
        'Griewanks’s function at exteme', 'Ackley’s function at extreme']
VRmin = [-100, -2.048, -32.768, -600, -0.5, -5.12, -5.12, -500, -5, -5]
VRmax = [-x for x in VRmin]


def PSO(popu_, v_set_):
    g_best = np.repeat(popu_best[np.argmin(popu_best_ft)][np.newaxis, :], NP, 0)  # gbest update
    v_tmp = C1 * np.random.uniform(0, 1) * (popu_best - popu_) \
             + C2 * np.random.uniform(0, 1) * (g_best - popu_)  # v update
    v_set_ = W * v_set_ + v_tmp
    popu_ += v_set_

    return popu_, v_set_


for index in range(9, 11):
    ph = []
    pft = []
    pstd = []
    pv_plot = []
    res = []
    for param_index in range(1, 6):
        W = 0.75 + param_index * 0.25 / 5
        W = round(100 * (W - 0.05)) / 100

        V_UP = VRmax[index - 1]
        V_LOW = VRmin[index - 1]
        popu = init(NP, D, V_UP, V_LOW)
        popu_best = copy.deepcopy(popu)
        popu_ft = ft_score(popu, NP, D, index)
        v_set = np.zeros((NP, D))
        popu_best_ft = copy.deepcopy(popu_ft)
        h = []
        ft = []
        std = []
        v_plot = []

        for i in range(ITER):
            # W = -(0.8 / ITER) * i + 1
            # C1 = -(1.5 / ITER) * i + 3
            # C2 = (1.5 / ITER) * i
            popu, v_set = PSO(popu, v_set)
            popu_ft = ft_score(popu, NP, D, index)
            mask = np.where(popu_ft < popu_best_ft)

            popu_best_ft[mask] = popu_ft[mask]  # popu remains
            popu_best[mask] = popu[mask]  # popu remains

            h.append(np.mean(popu))
            ft.append(np.min(popu_best_ft))
            std.append(np.std(popu))
            v_plot.append(np.mean(v_set))

            if i % 1000 == 0:
                #GA.go_plot(X, Y, popu, popu_ft, 1, 0, 'PSO{}'.format(str(i+1)))
                print('############ Generation {} ############'.format(str(i + 1)))
                print('Best Position：{}'.format(popu_best[np.argmin(popu_best_ft)]))
                print('Best Fitness Score：{}'.format(np.min(popu_best_ft)))
                print('Mean Fitness Score：{}'.format(np.mean(popu_ft)))
                _ = 1 + 1

        print("---- End of (successful) Searching No.{} ----".format(i))
        print('Best Fitness Score：{}'.format(np.min(popu_best_ft)))
        print('Mean Fitness Score：{}'.format(np.mean(popu_ft)))
        pft.append(np.log10(ft))
        ph.append(h)
        pstd.append(std)
        pv_plot.append(v_plot)

        x_ = np.arange(0, ITER, 1)
        x__ = np.arange(0, ITER, 1)
        plt.figure()
        plt.title('-Function{}-pbest fitness score'.format(P))
        plt.plot(x_, np.log10(ft))
        plt.show()
        plt.figure()
        plt.title('-Function{}-popu std'.format(P))
        plt.plot(x_, std)
        plt.show()
        plt.figure()
        plt.title('-Function{}-popu mean'.format(P))
        plt.plot(x_, h)
        plt.show()
        plt.figure()
        plt.title('-Function{}-v mean'.format(P))
        plt.plot(x_, v_plot)
        plt.show()
        res.append(np.min(ft))
    # title_ = str(index) + ' [' + Name[index - 1] + ']' + 'W:' + str(W)
    # x = np.arange(0, ITER * 10, 10)
    # x_ = np.arange(0, (ITER + 1) * 10, 10)
    # plt.subplot(2, 2, 1)
    #
    # lgbest, = plt.plot(x, np.log10(pft[]), label='log10(1Gbest)', color='g')
    # plt.legend(handles=[lgbest], labels=['gbest_ft'], loc='best')
    #
    # plt.subplot(2, 2, 2)
    # lpopustd, = plt.plot(x, std, label='popu std')
    # plt.legend(handles=[lpopustd], labels=['popu std'], loc='best')
    #
    # plt.subplot(2, 2, 3)
    # lownbestmean, = plt.plot(x, h, label='pbest_ft mean')
    # plt.legend(handles=[lownbestmean], labels=['pbest_ft mean'], loc='best')
    #
    # plt.subplot(2, 2, 4)
    # lownbestmean, = plt.plot(x, v_plot, label='v mean')
    # #plt.bar(x, exc, label='No. of re-generated')
    # plt.legend(handles=[lownbestmean], labels=['v mean'],loc='best')
    # #title_ = str(index) + ' [' + Name[index - 1] + ']'
    # plt.legend()
    # plt.suptitle("Func_{}".format(title_))
    # plt.savefig(f'/Users/leslie/Downloads/CLPSO_matlab/imgs/res_PSO/res{index}_{Name[index - 1]}_W={W}.png')
    # plt.show()
    x = np.arange(0, ITER * 10, 10)
    title_ = str(index) + ' [' + Name[index - 1] + ']'
    plt.title("Func_{}".format(title_))
    plt.plot(x, pft[0], color='green', label='w = 0.75')
    plt.plot(x, pft[1], color='lightsalmon', label='w = 0.80')
    plt.plot(x, pft[2], color='skyblue', label='w = 0.85')
    plt.plot(x, pft[3], color='slategrey', label='w = 0.90')
    plt.plot(x, pft[4], color='orchid', label='w = 0.95')
    plt.legend()  # 显示图例

    plt.xlabel('no. of function evaluations')
    plt.ylabel('log(g_best)')
    plt.savefig(f'/Users/leslie/Downloads/CLPSO_matlab/imgs/res_PSO/res{index}_{Name[index - 1]}.png')
    plt.show()

    best_all.append(res)

# f = open("/Users/leslie/Downloads/CLPSO_matlab/imgs/res_CLPSO/ model_Weight.txt",'a')
# for y in range(2):
#
#     f.write(best_all[y])
#     f.write("\n")

