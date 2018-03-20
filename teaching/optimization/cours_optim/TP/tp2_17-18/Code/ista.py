#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 14:37:17 2018

@author: alain_durmus
"""


import numpy as np
import matplotlib.pyplot as plt


def ista(dim, prox_op_g, grad_f, fun_total, lambda_l, n_it=10):
    #    Variables d'entrée :
    #    dim : dimension du problème
    #    prox_op_g : opérateur proximal de g qui prend en entrée x et gamma
    #    grad_f : gradient de f
    #    fun_total : fonction F = f+lambda*g
    #    lambda_l : paramètre lambda dans F
    #    n_it : nombre d'itérations
    #    Variables de sortie : x, fun_iterate
    #    x : itéré final de Ista
    #    fun_iterate : suite (f(x_k))

    x = np.zeros(dim)
    grad_f_x, Lip_grad_f = grad_f(x)
    gamma = 1. / (Lip_grad_f)
    fun_iterate = []
    for it in range(n_it):
        x = prox_op_g(x - gamma * grad_f_x, lambda_l * gamma)
        fun_iterate.append(fun_total(x))
        grad_f_x, _ = grad_f(x)
        if np.mod(it, 10) == 0:
            print('iteration number: ', it, end="\r", flush=True)
    return x, fun_iterate


def ista_mat(dim_1, dim_2, prox_op_g, grad_f, fun_total, lambda_l, n_it=500):
    x = np.zeros((dim_1, dim_2))
    grad_f_x, Lip_grad_f = grad_f(x)
    gamma = 1. / (Lip_grad_f)
    fun_iterate = []
    for it in range(n_it):
        x = prox_op_g(x - gamma * grad_f_x, lambda_l * gamma)
        fun_iterate.append(fun_total(x))
        grad_f_x, _ = grad_f(x)
        if np.mod(it, 10) == 0:
            print('iteration number: ', it, end="\r", flush=True)
    return x, fun_iterate
