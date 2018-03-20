"""
Created on Tue Mar 13 09:15:34 2018

@author: alain_durmus
"""

import numpy as np
import matplotlib.pyplot as plt
from tp2_tools import *
import warnings
warnings.filterwarnings('ignore')
import time
from ista import ista

lambda_l1 = 10 ** -2
n_it = 500
n = 32
r_sparse = 0.2
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

start_time = time.time()
x_final_ista, fun_iterate_ista = ista(
    dim, proxL1, grad_f_1, fun_total_1, lambda_l1, n_it)
elapsed_time = time.time() - start_time

energy_plt = plt.figure(1)
plt.plot(fun_iterate_ista)
plt.title(r'$\mathrm{Energy \ evolution}$')
plt.xlabel('iterates')
plt.ylabel('energy value')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
energy_plt.show()

images_plt = plt.figure('images')
images_plt.add_subplot(1, 3, 1)
plt.axis('off')
plt.imshow(load_image('barb.bmp', n), cmap='gray')
plt.title(r'$\mathrm{Original \ image}$')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
images_plt.add_subplot(1, 3, 2)
plt.axis('off')
plt.imshow(back_to_image(np.linalg.inv(A).dot(y)), cmap='gray')
plt.title(r'$\mathrm{Measurement}$')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
images_plt.add_subplot(1, 3, 3)
plt.axis('off')
plt.imshow(back_to_image(x_final_ista), cmap='gray')
plt.title(r'$\mathrm{Recontructed}$')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
images_plt.show()

# fig.add_subplot(1, 2, 2)
# plot_image(x_final_ista)

print("||Ax-y||=", np.linalg.norm(A.dot(x_final_ista) - y))
print("Elapsed time=", elapsed_time, 's')
