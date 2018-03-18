#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 12:01:41 2018

@author: alain_durmus
"""

import numpy as np
import matplotlib.pyplot as plt
from tp2_tools import *
import warnings
warnings.filterwarnings('ignore')

y, A = noisy_observations()

def proxL1(x, gamma):
    return np.fmax(0, 1 - gamma / np.fmax(1e-15, np.abs(x))) * x

def ista(y, A, mu=1., n_it=100):
    Q = A.T.dot(A)
    Ay = A.T.dot(y)
    
    gamma = 1 / (mu * np.linalg.eigvalsh(Q).max())
    x = np.zeros(A.shape[1])
    
    obj = []
    for it in range(n_it):
        x = proxL1(x - gamma*mu*(Q.dot(x) - Ay), gamma)
        obj.append( np.linalg.norm(x, ord=1) + mu/2*np.linalg.norm(A.dot(x)-y)**2 )

    return x, obj

mu = 2.
xista, objista = ista(y, A, mu=mu, n_it=50)

plt.figure()
plt.plot(objista)

plot_image(xista)

print("||Ax-y||=", np.linalg.norm(A.dot(xista)-y))
