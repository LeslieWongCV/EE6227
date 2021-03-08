# -*- coding: utf-8 -*-
# @Time    : 2021/2/10 10:37 上午
# @Author  : Yushuo Wang
# @FileName: __init__.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/
import matplotlib.pyplot as plt


def go_plot(X, Y, POPU, SCORE, iter, save, name):
    '''

    :param X:
    :param Y:
    :param POPU:
    :param SCORE:
    :param iter:
    :return:
    '''

    plt.figure()
    plt.title(name)
    plt.plot(X, Y)
    # plt.plot(POPU, SCORE, 'ro')
    plt.scatter(POPU, SCORE, marker='v', color='green')

    if save:
        plt.savefig(f'img{iter}')
    plt.show()