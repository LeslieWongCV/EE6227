# -*- coding: utf-8 -*-
# @Time    : 2021/2/27 6:29 下午
# @Author  : Yushuo Wang
# @FileName: CLPSO_final.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/
import sys
import matplotlib.pyplot as plt
from __init__ import init, ft_score
sys.path.append('../')
import numpy as np

index = 1
NP = 1000
D = 10
ITER = 5000
W = 0.9
C = 2.1
M = 7
VRmin = [-100, -2.048, -32.768, -600, -0.5, -5.12, -5.12, -500, -5, -5]
VRmax = [-x for x in VRmin]
V_UP = VRmax[index - 1]
V_LOW = VRmin[index - 1]
Name = ['Sphere function', 'Rosenbrock’s function', 'Ackley’s function', 'Griewanks’s function', 'Weierstrass function',
        'Rastrigin’s function', 'Noncontinuous Rastrigin’s function', 'Schwefel’s function',
        'Griewanks’s function at exteme', 'Ackley’s function at extreme']
best_all = []


def gen_tournament(popu_, id):
    # p_best_fd = own_p_best.copy()
    # np.random.shuffle(p_best_fd)  # BUG 1
    # mask_tourmnt = np.where(ft_score(own_p_best, NP, D, index) < ft_score(p_best_fd, NP, D, index))
    # p_best_fd[mask_tourmnt] = own_p_best[mask_tourmnt]

    tournament = np.zeros((NP, D))
    for j in id:
        for k in range(D):
            tmp_t = np.copy(popu_)

            tmp_ = tmp_t[np.argpartition(-ft_score(tmp_t, NP, D, index), -round(0.2 * NP))[-round(0.2 * NP):]]
            np.random.shuffle(tmp_)
            try:
                tournament[j][k] = tmp_[0][k]
            except:
                pass

            # tmp_2 = tmp_[:2]
            # if ft_score(tmp_2[0], 1, D, index) < ft_score(tmp_2[1], 1, D, index):
            #     tournament[j][k] = tmp_2[0][k]
            # else:
            #     tournament[j][k] = tmp_2[1][k]  # tournament 生成
    return tournament  # p_best_fd


for index in range(1, 11):
    res = []
    for param_index in range(1, 6):
        W = 0.75 + param_index * 0.25 / 5
        W = round(100 * (W - 0.05)) / 100  # W 0.75 - 0.95
        # print(W)

        V_UP = VRmax[index - 1]
        V_LOW = VRmin[index - 1]
        NP_index = np.arange(NP)
        g_best_ft = []
        g_best = []
        h = []
        ft = []
        std = []
        v_plot = []
        exc = []
        flag_i = np.zeros(NP)
        t = 5 * np.arange(0, 1, 1 / NP)
        Pc = 0.5 * (np.exp(t) - np.exp(t[0])) / np.exp(t[-1] - np.exp(t[0]))  # Sefl-Adaptive Pc
        Pc = np.tile((Pc[np.newaxis, :]).T, D)  # Trans from D*NP to NP*D
        v = np.random.random((NP, D))  # * (V_UP - V_LOW) + 0.5*V_UP

        popu = init(NP, D, V_UP, -V_UP)
        popu_ft = ft_score(popu, NP, D, index)
        own_p_best = np.copy(popu)
        p_best = np.copy(popu)
        g_best_ft.append(popu_ft[0])
        examplar = gen_tournament(popu, np.arange(NP))
        examplar_update = np.copy(examplar)

        for i in range(ITER):
            # W = -(0.2 / ITER) * i + 1
            # W = 0.9 - (i+1)*0.7/(i+1)
            # #C = -(1.5 / ITER) * i + 3
            # C2 = (1.5 / ITER) * i
            # V_LOW = W * V_LOW
            # V_UP = W * V_UP
            mask_flag_exc = np.where(flag_i > M)
            flag_i[mask_flag_exc] = np.zeros(NP)[mask_flag_exc]
            if len(mask_flag_exc) != 1:
                examplar_update = gen_tournament(own_p_best, mask_flag_exc)
            # for j in mask_flag_exc:
            examplar[mask_flag_exc] = examplar_update[mask_flag_exc]
            # examplar = np.copy(examplar_update)
            rand_Pc = np.random.uniform(0, Pc[-1] + Pc[-1] / 10, [NP, D])
            mask_pc = np.where(rand_Pc < Pc)
            # mask_pc = np.arange(NP)
            # own_ft = ft_score(own_p_best, NP, D)
            p_best = np.copy(own_p_best)
            p_best[mask_pc] = examplar[mask_pc]  # own_p_best or others

            v = W * C * np.random.random((NP, D)) * (p_best - popu)
            v = np.clip(v, V_LOW, V_UP)
            popu += v

            mask_up_exc = np.where(popu > V_UP)  # calculate the exceed
            mask_low_exc = np.where(popu < V_LOW)
            mask_exc = np.unique(np.concatenate((mask_up_exc[0], mask_low_exc[0])))

            tmp = np.copy(own_p_best)
            popu_ft = ft_score(popu, NP, D, index)
            own_ft = ft_score(own_p_best, NP, D, index)
            mask_ownb_update = np.where(popu_ft < own_ft)

            own_p_best[mask_ownb_update] = popu[mask_ownb_update]
            own_p_best[mask_exc] = tmp[mask_exc]  # exceed

            own_ft = ft_score(own_p_best, NP, D, index)
            # if np.min(own_ft) < g_best_ft[-1]:  # record the gbest
            g_best_ft.append(np.min(own_ft))
            g_best.append(own_p_best[np.argmin(own_ft)])

            union_1 = np.setdiff1d(NP_index, mask_ownb_update)
            union_2 = np.intersect1d(mask_exc, mask_ownb_update)
            mask_flag_i = np.union1d(union_1, union_2)
            flag_i[mask_flag_i] += np.ones(NP)[mask_flag_i]  # m

            h.append(popu)
            ft.append(np.mean(own_ft))
            std.append(np.std(popu))
            v_plot.append(np.mean(v))
            exc.append(round(len(mask_exc) / 10))


            # if i % 1000 == 0:
            #     #GA.go_plot(X, Y, popu, ft_score(popu, NP, D, index), 1, 0, 'CLPSO_Diffi')
            #     print('############ Generation {} ############'.format(round(i/10)))
            #     print('Best Position：{}'.format(own_p_best[np.argmin(own_ft)]))
            #     print('Best Fitness Score：{}'.format(np.min(own_ft)))
            #     print('Mean Fitness Score：{}'.format(np.mean(own_ft)))
            #     #print("*" * 15)
            #     # print('Best popu Fitness Score：{}'.format(np.min(popu_ft)))
            #     # print('Mean popu Fitness Score：{}'.format(np.mean(popu_ft)))
            #     _ = 1 + 1
        print("Func{}".format(index))
        print('############ Generation {} ############'.format(500))
        print('Best Fitness Score：{}'.format(np.min(own_ft)))
        print('Mean Fitness Score：{}'.format(np.mean(own_ft)))
        print('############ End of Iteration ############')

        # x_ = np.arange(0, 5000, 1)
        # plt.figure()
        # plt.plot(x_, ft)
        # plt.show()

        x = np.arange(0, ITER * 10, 10)
        x_ = np.arange(0, (ITER + 1) * 10, 10)
        plt.subplot(2, 2, 1)
        lgbest, = plt.plot(x_, np.log10(g_best_ft), label='log10(1Gbest)', color='g')
        plt.legend(handles=[lgbest], labels=['gbest_ft'], loc='best')

        plt.subplot(2, 2, 2)
        lpopustd, = plt.plot(x, std, label='popu std')
        plt.legend(handles=[lpopustd], labels=['popu std'], loc='best')

        plt.subplot(2, 2, 3)
        lownbestmean, = plt.plot(x, ft, label='pbest_ft mean')
        plt.legend(handles=[lownbestmean], labels=['pbest_ft mean'], loc='best')

        plt.subplot(2, 2, 4)
        # lownbestmean, = plt.bar(x, exc, label='v mean')
        plt.bar(x, exc, label='No. of re-generated')
        # plt.legend(handles=[lownbestmean], labels=['v mean'],loc='best')
        title_ = str(index) + ' [' + Name[index - 1] + ']' + 'W:' + str(W)
        plt.legend()
        plt.suptitle("Func_{}".format(title_))
        plt.savefig(f'/Users/leslie/Downloads/CLPSO_matlab/imgs/res_CLPSO/Fuc{index}_{Name[index - 1]}_W={W}.png')
        # plt.savefig(f'res{index}_{Name[index-1]}')
        plt.show()

        res.append(np.min(own_ft))

    best_all.append(res)

# f = open("/Users/leslie/Downloads/CLPSO_matlab/imgs/res_CLPSO/ model_Weight.txt", 'a')
# for y in range(2):
#     f.write(str(best_all[y]))
#     f.write("\n")
#
# f.close()
# dataframe = pd.DataFrame({'Function_1': best_all[0], 'Function_2': best_all[1], 'Function_3': best_all[2],
#                           'Function_4': best_all[3], 'Function_5': best_all[4], 'Function_6': best_all[5],
#                           'Function_7': best_all[6], 'Function_8': best_all[7]})
# dataframe.to_csv("res_CLPSO.csv", index=False, sep=',')

