# -*- coding: utf-8 -*-
# @Time    : 2021/1/9 5:23 下午
# @Author  : Yushuo Wang
# @FileName: 01.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/

import matplotlib.pyplot as plt
import numpy as np
from math import *
import random

KILL_PARAM = 1.0
INIT_POPU = [-0.5, 0, 0.5, 1.0, 1.5, 1.7]
POPU = []
SCORE = []
ITERATION = 8
iter = 0

X = np.arange(-2, 2, 0.01)
Y = X**2 #np.sin(10 * pi * X) + 2


'''
Genetic Algorithm: Kill the Weak
'''

def init_pop(init_x, num):
    popul = []
    for i in range(num):
        tmp = random.uniform(-0.15, 0.15)
        tmp = round(tmp, 4)
        popul.append(init_x + tmp)
    return popul

# def one_point_cross_over(inputx):
#     sum = sum(inputx)
#
#     for i in inputx:


def gen_pop(pop_in):
    pop_out = pop_in + pop_in
    return pop_out
    #pop_out = []    # cross over
    # for i in range(len(pop_in)):
    #     try:
    #         tmp = pop_in[i + 1]
    #         father = pop_in[i]
    #         mother = pop_in[i + 1]
    #         pop_out.append(round(random.uniform(father, mother), 4))
    #         i += 1
    #     except:
    #         pass
    #
    # pop_out += pop_in


def mutation(x_in):
    tmp = []
    for i in x_in:
        tmp.append(i + round(random.uniform(-0.01, 0.01), 4))
    return tmp


def enco_f(a_x):
    a_y = []
    for i in a_x:
        res = i**2# * np.sin(10 * pi * i) + 2
        a_y.append(res)
    return a_y


def kill_indiv(y_in):
    global KILL_PARAM
    new_pop = []

    for i in range(len(y_in)):
        if y_in[i] > KILL_PARAM:
            new_pop.append(POPU[i])
    KILL_PARAM += 0.3
    return new_pop


def go_plot():
    plt.figure()
    plt.title("Generic_Algorithm")
    plt.plot(X, Y)
    # plt.plot(POPU, SCORE, 'ro')
    plt.scatter(POPU, SCORE, marker='v', color='green')
    plt.savefig(f'img{iter}')
    plt.show()


'''
Iteration
'''

for i in INIT_POPU:
    POPU = POPU + init_pop(i, 5)

SCORE = enco_f(POPU)

go_plot()

_ = 1 + 1

for i in range(ITERATION):
    iter = i + 1
    _ = 1 + 1

    POPU = kill_indiv(SCORE)
    POPU = gen_pop(POPU)
    POPU = mutation(POPU)

    SCORE = enco_f(POPU)

    go_plot()
_ = 1 + 1
POPU = kill_indiv(SCORE)
SCORE = enco_f(POPU)
go_plot()

_ = 1 + 1



'''
def gen_genetic():
    population = []
    for i in range(0, 100):
        unit = random.randint(0, 255)
        unit = bin(unit)[2:]
        if len(unit) < 8:
            unit = unit + '0'*(8-len(unit))

        population.append(unit)
    return population



# 1 2 3 4 5 6 7 8     feature 3 6 7 do not influence the score 



def fitness_score(x_set):
    score_set = []
    for i in x_set:
        fitScore = 5 * int(i[0]) + int(i[1]) + 10 * int(i[3]) + 3 * int(i[4]) * int(i[7])
        score_set.append(fitScore)
    return score_set


def score_select(s_set):




popu = gen_genetic()
print(gen_genetic())
pop_set = fitness_score(popu)
print(pop_set)
_ = 1 + 1
'''
