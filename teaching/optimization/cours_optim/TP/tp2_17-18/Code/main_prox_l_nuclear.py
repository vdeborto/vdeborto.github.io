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


def prox_nuclear(X, gamma):
    u, s, vh = np.linalg.svd(X)
    s_prox = proxL1(s, gamma)
    return u.dot(np.diag(s_prox).dot(vh))


def grad_f_1(X):
    grad_f_x = A * X - A * Y
    Lip_grad_f = 1
    return grad_f_x, Lip_grad_f


def fun_total_1(X):
    return lambda_nuclear * np.linalg.norm(X, ord='nuc') + (1. / 2) * np.linalg.norm(Y - A * X)**2


dim_1 = A.shape[0]
dim_2 = A.shape[1]

x_final_ista, fun_iterate_ista = ista_mat(
    dim_1, dim_2, prox_nuclear, grad_f_1, fun_total_1, lambda_nuclear, n_it=500)

energy_plt = plt.figure()
plt.plot(fun_iterate_ista)
plt.title(r'$\mathrm{Energy \ evolution}$')
plt.xlabel('iterates')
plt.ylabel('energy value')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
energy_plt.show()

matrix_plt = plt.figure()
matrix_plt.add_subplot(1, 3, 1)
plt.axis('off')
plt.imshow(Y)
plt.title(r'$\mathrm{Measurement}$')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
matrix_plt.add_subplot(1, 3, 2)
plt.axis('off')
plt.imshow(x_final_ista)
plt.title(r'$\mathrm{Estimated}$')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
matrix_plt.add_subplot(1, 3, 3)
plt.axis('off')
plt.imshow(A * x_final_ista)
plt.title(r'$\mathrm{Uncomplete estimated}$')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
matrix_plt.show()
