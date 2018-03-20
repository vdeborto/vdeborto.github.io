#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 14:37:58 2018

@author: alain_durmus
"""


import numpy as np
import matplotlib.pyplot as plt
from tp2_tools import *
import warnings
warnings.filterwarnings('ignore')
from ista import ista

lambda_max = 1
y, A = noisy_observation_inf()


def prox_max(v, gamma):
    tol = 1e-8
    max_iter = 1e2
    rho = 1 / gamma

    n = len(v)
    t_low = np.min(v)
    t_up = np.max(v)

    def g(t):
        return (np.sum(np.fmax(0, rho * (v - t))) - 1)

    iteration = 0

    while (t_up - t_low > tol) & (iteration <= max_iter):
        t0 = (t_up + t_low) / 2
        if np.sign(g(t0)) == np.sign(g(t_low)):
            t_low = t0
        else:
            t_up = t0
        iteration = iteration + 1
    return np.fmin(t0, v)


def grad_f_1(x):
    Q = A.T.dot(A)
    Ay = A.T.dot(y)
    grad_f_x = Q.dot(x) - Ay
    Lip_grad_f = np.linalg.eigvalsh(Q).max()
    return grad_f_x, Lip_grad_f


def fun_total_1(x):
    return lambda_max * np.max(x) + (1. / 2) * np.linalg.norm(A.dot(x) - y)**2


dim = A.shape[1]

x_final_ista, fun_iterate_ista = ista(
    dim, prox_max, grad_f_1, fun_total_1, lambda_max, n_it=500)

plt.figure()
plt.plot(fun_iterate_ista)
plt.title(r'$\mathrm{Energy \ evolution}$')
plt.xlabel('iterates')
plt.ylabel('energy value')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.show()
