#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 16:19:57 2018

@author: alain_durmus
"""


import numpy as np
import matplotlib.pyplot as plt
from tp2_tools import *
import warnings
warnings.filterwarnings('ignore')
from ista import ista_mat

lambda_nuclear = 1
Y, A = noisy_observation_nuclear()


def proxL1(x, gamma):
    eps = np.finfo(float).eps
    return np.fmax(0, 1 - gamma / np.fmax(eps, np.abs(x))) * x

def prox_nuclear(X,gamma):
    u,s,vh = np.linalg.svd(X)
    s_prox = proxL1(s,gamma)
    return u.dot(np.diag(s_prox).dot(vh))

def grad_f_1(X):
    grad_f_x = A*X-A*Y
    Lip_grad_f = 1
    return grad_f_x, Lip_grad_f

def fun_total_1(X):
    return lambda_max*np.linalg.norm(X,ord = 'nuc') + (1./2)*np.linalg.norm(Y-A*X)**2

dim_1 = A.shape[0]
dim_2 = A.shape[1]

x_final_ista, fun_iterate_ista = ista_mat(dim_1,dim_2, prox_nuclear, grad_f_1, fun_total_1,lambda_nuclear, n_it=500)

plt.figure()
plt.plot(fun_iterate_ista)