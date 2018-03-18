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

lambda_l1 = 10 ** -5
n_it = 20
n = 32
r_sparse = 1
r_info = 1
y, A = noisy_observations(n, r_sparse, r_info)


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
    dim, proxL1, grad_f_1, fun_total_1, lambda_l1, n_it)

energy_plt = plt.figure(1)
plt.plot(fun_iterate_ista)

images_plt = plt.figure('images')
images_plt.add_subplot(1, 2, 1)
plt.axis('off')
plt.imshow(back_to_image(np.linalg.inv(A).dot(y)), cmap='gray')

# fig.add_subplot(1, 2, 2)
# plot_image(x_final_ista)

print("||Ax-y||=", np.linalg.norm(A.dot(x_final_ista) - y))
plt.show()
