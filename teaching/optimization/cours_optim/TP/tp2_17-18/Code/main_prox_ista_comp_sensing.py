#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 09:15:34 2018

@author: alain_durmus
"""

import numpy as np
import matplotlib.pyplot as plt
from tp2_tools import *
import warnings
warnings.filterwarnings('ignore')
from ista import ista

lambda_l1 = 1
y, A = noisy_observations()
#


def proxL1(x, gamma):
    eps = np.finfo(float).eps
    return np.fmax(0, 1 - gamma / np.fmax(eps, np.abs(x))) * x


def grad_f_1(x):
    Q = A.T.dot(A)
    Ay = A.T.dot(y)
    grad_f_x = Q.dot(x) - Ay
    Lip_grad_f = np.linalg.eigvalsh(Q).max()
    return grad_f_x, Lip_grad_f


def fun_total_1(x):
    return lambda_l1 * np.linalg.norm(x, ord=1) + (1. / 2) * np.linalg.norm(A.dot(x) - y)**2


dim = A.shape[1]

x_final_ista, fun_iterate_ista = ista(
    dim, proxL1, grad_f_1, fun_total_1, lambda_l1, n_it=500)

plt.figure()
plt.plot(fun_iterate_ista)

plot_image(x_final_ista)

print("||Ax-y||=", np.linalg.norm(A.dot(x_final_ista) - y))
