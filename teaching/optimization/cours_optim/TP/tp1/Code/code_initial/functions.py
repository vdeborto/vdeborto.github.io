#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:04:54 2018

@author: alain
"""
import numpy as np




def mk_quad(m,M,ndim=2):
    def f(x):
       x = np.asarray(x)
       y = x.copy()
       scaling = np.ones(ndim)
       scaling[0] = m
       scaling[1] = M
       y *= scaling
       return np.sum(y**2)

    def f_prime(x):
       x = np.asarray(x)
       y = x.copy()
       scaling = np.ones(ndim)
       scaling[0] = m
       scaling[1] = M
       y *= scaling
       return 2*scaling*y

    def hessian(x):       
       scaling = np.ones(ndim)
       scaling[0] = m
       scaling[1] = M
       return 2*np.diag(scaling)

    return f, f_prime, hessian


def rosenbrock(x):
    y=x
    return np.sum((1 - y[:-1])**2 + 100*(y[1:] - y[:-1]**2)**2)


def rosenbrock_prime(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der


def rosenbrock_hessian_(x):
    x, y = x
    return np.array((
                    (1 - 4*100*y + 12*100*x**2, -4*x*100),
                    (             -4*100*x,    2*100),
                   ))


def rosenbrock_hessian(x):
    x = np.asarray(x)
    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)
    return H