#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 20:42:44 2018

@author: alain_durmus
"""


def prox_max(v,gamma):
    tol = 1e-8
    max_iter = 1e2
    rho = 1/gamma
    
    n = len(v)
    t_low = np.min(v)
    t_up = np.max(v)
    
    g = lambda(t): (np.sum(np.fmax(0,rho*(v-t)))-1)
    
    iteration = 0
    
    while (t_up-t_low > tol) & (iteration <= max_iter):
        t0 = (t_up+t_low)/2
        if np.sign(g(t0)) == np.sign(g(t_low)):
            t_low = t0
        else:
            t_up = t0
        iteration = iteration +1
    return np.fmin(t0,v)
