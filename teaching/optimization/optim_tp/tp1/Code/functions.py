#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:04:54 2018

@author: alain
"""
import numpy as np


def mk_quad(m, M, ndim=2):
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
        return 2 * scaling * y

    def hessian(x):
        scaling = np.ones(ndim)
        scaling[0] = m
        scaling[1] = M
        return 2 * np.diag(scaling)

    return f, f_prime, hessian


def rosenbrock(x):
    y = x
    return np.sum((1 - y[:-1])**2 + 100 * (y[1:] - y[:-1]**2)**2)


def rosenbrock_prime(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200 * (xm - xm_m1**2) - 400 * \
        (xm_p1 - xm**2) * xm - 2 * (1 - xm)
    der[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2]**2)
    return der


def rosenbrock_hessian_(x):
    x, y = x
    return np.array((
                    (1 - 4 * 100 * y + 12 * 100 * x**2, -4 * x * 100),
                    (-4 * 100 * x,    2 * 100),
                    ))


def rosenbrock_hessian(x):
    x = np.asarray(x)
    H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200 * x[0]**2 - 400 * x[1] + 2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200 * x[1:-1]**2 - 400 * x[2:]
    H = H + np.diag(diagonal)
    return H


def mk_wave(A, sigma, omega):

    def wave_fun(x):
        return A * np.exp(-sigma * x) * np.sin(omega * x)

    def wave_grad(x):
        dim = np.max(np.shape(x))
        grad = np.zeros([3, dim])
        grad[0, :] = np.sin(omega * x)
        grad[1, :] = -A * x * np.sin(omega * x)
        grad[2, :] = A * x * np.cos(omega * x)
        grad = np.exp(-sigma * x) * grad
        return grad

    def wave_hessian(x):
        dim = np.max(np.shape(x))
        s = np.sin(omega * x) * x * np.exp(-sigma * x)
        c = np.cos(omega * x) * x * np.exp(-sigma * x)
        scal = A * x ** 2 * np.exp(-sigma * x)
        hessian = np.zeros([3, 3, dim])
        hessian[0, 1, :] = -x * s
        hessian[1, 0, :] = -x * s
        hessian[0, 2, :] = hessian[2, 0, :] = x * c
        hessian[1, 1, :] = scal * s
        hessian[2, 2, :] = -scal * s
        hessian[1, 2, :] = hessian[2, 1, :] = -scal * c
        return hessian
    return wave_fun, wave_grad, wave_hessian


def generate_data(x, A, sigma, omega, noise=0, n_outliers=0, random_state=0):
    y = A * np.exp(-sigma * x) * np.sin(omega * x)
    rnd = np.random.RandomState(random_state)
    error = noise * rnd.randn(x.size)
    outliers = rnd.randint(0, x.size, n_outliers)
    error[outliers] *= 35
    return y + error


def mk_nonlinreg(x_train, y_train):
    def nonlinreg_fun(param):
        wave_fun = mk_wave(param[0], param[1], param[2])[0]
        return np.sum((wave_fun(x_train) - y_train) ** 2)

    def nonlinreg_grad(param):
        wave_fun, wave_grad = mk_wave(param[0], param[1], param[2])[:2]
        grad = 2 * wave_grad(x_train) * (wave_fun(x_train) - y_train)
        return np.sum(grad, 1)

    def nonlinreg_hessian(param, method='newton', mu=0.1):
        wave_fun, wave_grad, wave_hessian = mk_wave(
            param[0], param[1], param[2])
        grad = wave_grad(x_train)
        if method == 'newton':
            hess1 = 2 * wave_hessian(x_train) * (wave_fun(x_train) - y_train)
            hess1 = np.sum(hess1, 2)
        elif method == 'lm':
            hess1 = mu * np.identity(3)
        hess2 = 2 * np.dot(grad, grad.T)
        return hess1 + hess2

    return nonlinreg_fun, nonlinreg_grad, nonlinreg_hessian


def mk_lin(a, b):

    def lin_fun(x):
        return a * x + b

    def lin_grad(x):
        grad = np.ones([2, x.size])
        grad[0, :] = x
        return grad

    def lin_hessian(x, method='newton', mu=0.1):
        return np.zeros([2, 2, x.size])
    return lin_fun, lin_grad, lin_hessian


def mk_linreg(x_train, y_train):
    def linreg_fun(param):
        lin_fun = mk_lin(param[0], param[1])[0]
        return np.sum((lin_fun(x_train) - y_train) ** 2)

    def linreg_grad(param):
        lin_fun, lin_grad = mk_lin(param[0], param[1])[:2]
        grad = 2 * lin_grad(x_train) * (lin_fun(x_train) - y_train)
        return np.sum(grad, 1)

    def linreg_hessian(param, method='newton', mu=0.1):
        lin_fun, lin_grad, lin_hessian = mk_lin(
            param[0], param[1])
        grad = lin_grad(x_train)
        if method == 'newton':
            hess1 = 2 * lin_hessian(x_train) * (lin_fun(x_train) - y_train)
            hess1 = np.sum(hess1, 2)
        elif method == 'lm':
            hess1 = mu * np.identity(2)
        hess2 = 2 * np.dot(grad, grad.T)
        return hess1 + hess2

    return linreg_fun, linreg_grad, linreg_hessian


def V(x):
    return x ** -12 - 2 * x ** - 6


def V2(x):
    return x ** -6 - 2 * x ** - 3


def V2der(x):
    return -6 * x ** -7 + 6 * x ** -4


def J(u):
    N = len(u) / 3
    u_v = np.reshape(u, (N, 3))

    M = np.zeros([N, N, 3])
    M -= u_v
    M = M - np.transpose(M, (1, 0, 2))

    M = np.sum(M ** 2, 2)
    np.fill_diagonal(M, 1)
    M = V2(M)
    np.fill_diagonal(M, 0)

    return .5 * np.sum(M)


def grad_J(u):
    N = len(u) / 3
    u_v = np.reshape(u, (N, 3))

    M = np.zeros([N, N, 3])
    M -= u_v
    M = M - np.transpose(M, (1, 0, 2))
    # print(np.reshape(M,(N ** 2,3)).shape)
    # print(np.reshape(M,(N ** 2,3)))
    # print(M[:,:,1])

    Mnorm = np.sum(M ** 2, 2)
    np.fill_diagonal(Mnorm, 1)
    Mnorm = V2der(Mnorm)
    np.fill_diagonal(Mnorm, 0)
    # print('norm')
    # print(np.reshape(Mnorm,N**2).shape)
    # print(np.reshape(Mnorm,N**2))
    # print(Mnorm)

    grad = np.reshape(Mnorm, (N**2, 1)) * np.reshape(M, (N ** 2, 3))
    grad = np.reshape(grad, (N, N, 3))
    grad = np.sum(grad, 1)
    return np.ravel(grad)
