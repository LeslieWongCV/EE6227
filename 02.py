# -*- coding: utf-8 -*-
# @Time    : 2021/1/18 9:25 下午
# @Author  : Yushuo Wang
# @FileName: 02.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
X1 = np.arange(-5.12, 5.12, 0.1)
X2 = np.arange(-5.12, 5.12, 0.1)
#X3 = np.arange(-5.12, 5.12, 0.25)

X1, X2 = np.meshgrid(X1, X2)
#R = np.sqrt(X**2 + Y**2)
R = - X1**2 - X2**2 #+ np.sqrt((1 - X))
#Z = np.sin(R)

# 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
ax.plot_surface(X1, X2, R, rstride=1, cstride=1, cmap='rainbow')

_ = 1 + 1
plt.show()