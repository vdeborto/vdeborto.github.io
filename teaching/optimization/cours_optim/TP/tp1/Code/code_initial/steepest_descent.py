#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:30:23 2018

@author: alain
"""


import numpy as np

def steepest_descent_constant_step(f, grad, x0, iterations, error_point, error_grad, h):
    dim = np.max(np.shape(x0))
    x_list = np.zeros([dim,iterations])  
    f_list = np.zeros(iterations)  
    x = x0
    x_old = x
    grad_x = grad(x)
    for i in xrange(iterations):
        x = x - h*grad(x)
        grad_x = grad(x)
        f_x = f(x)
        x_list[:,i] = x
        f_list[i] = f_x
        if i % 10 == 0:
            # print "  iter={}, grad={}, alpha={}, x={}, f(x)={}".format(i, pk, alpha, x, f(x))
            print "  iter={}, x={}, f(x)={}".format(i, x, f(x))

        if (np.linalg.norm(x - x_old) < error_point)|(np.linalg.norm(grad_x) < error_grad):
            break
        x_old = x
    return dict([('x_list',x_list[:,0:i]), ('f_list',f_list[0:i])])

def armijo_rule(alpha_0,x,f,f_x,grad_x,d_x,c,beta): #d_x est la direction de descente d_x . grad_x <= 0
    # test f(x_new) \leq f(x_0) + c alpha ps{d_x}{grad_x}
    test = 1
    alpha = alpha_0
    while test: 
        x_new = x+alpha*d_x;
        if (f(x_new)<=f_x+c*alpha*np.dot(grad_x,d_x)):
            test = 0
        else:
            alpha = alpha*beta
    return alpha
            

def steepest_descent_armijo(f, grad, x0, iterations, error_point, error_grad, c=0.1,L=100,beta=0.5):
    dim = np.max(np.shape(x0))
    x_list = np.zeros([dim,iterations])  
    f_list = np.zeros(iterations)  
    x = x0
    x_old = x
    grad_x = grad(x)
    d_x = -grad_x
    f_x = f(x)
    alpha_0 = -(1./L)*np.dot(d_x,grad_x)/np.power(np.linalg.norm(d_x),2)
    h = armijo_rule(alpha_0,x,f,f_x,grad_x,d_x,c,beta)
    for i in xrange(iterations):
        x = x + h*d_x
        grad_x = grad(x)
        f_x = f(x)
        d_x = -grad_x
        alpha_0 = -(1./L)*np.dot(d_x,grad_x)/np.power(np.linalg.norm(d_x),2)
        h = armijo_rule(alpha_0,x,f,f_x,grad_x,d_x,c,beta)
        x_list[:,i] = x
        f_list[i] = f_x
        if i % 10 == 0:
            # print "  iter={}, grad={}, alpha={}, x={}, f(x)={}".format(i, pk, alpha, x, f(x))
            print "  iter={}, x={}, f(x)={}".format(i, x, f(x))

        if (np.linalg.norm(x - x_old) < error_point)|(np.linalg.norm(grad_x) < error_grad):
            break
        x_old = x
    return dict([('x_list',x_list[:,0:i]), ('f_list',f_list[0:i])])


def conjugate_gradient_armijo(f, grad, x0, iterations, error_point, error_grad, c=0.1,L=100,beta=0.5):
    dim = np.max(np.shape(x0))
    x_list = np.zeros([dim,iterations])  
    f_list = np.zeros(iterations)  
    x = x0
    x_old = x
    grad_x = grad(x)
    d_x = -grad_x
    f_x = f(x)
    alpha_0 = -(1./L)*np.dot(d_x,grad_x)/np.power(np.linalg.norm(d_x),2)
    h = armijo_rule(alpha_0,x,f,f_x,grad_x,d_x,c,beta)
    for i in xrange(iterations):
        x = x + h*d_x
        grad_x_old = grad_x
        grad_x = grad(x)
        f_x = f(x)
        kappa = np.dot(grad_x - grad_x_old, grad_x)/np.power(np.linalg.norm(grad_x),2)
        d_x = kappa*d_x -grad_x
        alpha_0 = -(1./L)*np.dot(d_x,grad_x)/np.power(np.linalg.norm(d_x),2)
        h = armijo_rule(alpha_0,x,f,f_x,grad_x,d_x,c,beta)
        x_list[:,i] = x
        f_list[i] = f_x
        if i % 10 == 0:
            # print "  iter={}, grad={}, alpha={}, x={}, f(x)={}".format(i, pk, alpha, x, f(x))
            print "  iter={}, x={}, f(x)={}".format(i, x, f(x))

        if (np.linalg.norm(x - x_old) < error_point)|(np.linalg.norm(grad_x) < error_grad):
            break
        x_old = x
    return dict([('x_list',x_list[:,0:i]), ('f_list',f_list[0:i])])



def wolf_rule(alpha_0,x,f,grad,f_x,grad_x,d_x,c_1,c_2): #d_x est la direction de descente  d_x . grad_x <= 0
    # test f(x_new) \leq f(x_0) + c_1 alpha ps{d_x}{grad_x} et \ps{x_new}{d_x} \geq c_2 \ps{x_0}{d_x}
    # sinon alpha <- alpha * beta
    # On cherche au fur et mesure un opt dans [minorant, majorant]
    test = 1
    iteration = 0
    alpha = alpha_0
    minorant = 0
    majorant = 1000
    while (test)&(iteration<=1000): 
        x_new = x+alpha*d_x;
        if (f(x_new)<=f_x+c_1*alpha*np.dot(grad_x,d_x))&(np.dot(grad(x_new),d_x) >= c_2*np.dot(grad_x,d_x) ):
            test = 0
        elif (f(x_new)>=f_x+c_1*alpha*np.dot(grad_x,d_x)):
            majorant = alpha
            alpha = (majorant +minorant)/2
            iteration = iteration +1
        else:
            minorant = alpha
            alpha = (majorant +minorant)/2
            iteration = iteration +1
    return alpha




def steepest_descent_wolf(f, grad, x0, iterations, error_point, error_grad, c_1=0.1,c_2=0.9,L=100):
    dim = np.max(np.shape(x0))
    x_list = np.zeros([dim,iterations])  
    f_list = np.zeros(iterations)  
    x = x0
    x_old = x
    grad_x = grad(x)
    d_x = -grad_x
    f_x = f(x)
    alpha_0 = -(1./L)*np.dot(d_x,grad_x)/np.power(np.linalg.norm(d_x),2)
    h = wolf_rule(alpha_0,x,f,grad,f_x,grad_x,d_x,c_1,c_2)
    for i in xrange(iterations):
        x = x + h*d_x
        grad_x = grad(x)
        f_x = f(x)
        d_x = -grad_x
        alpha_0 = -(1./L)*np.dot(d_x,grad_x)/np.power(np.linalg.norm(d_x),2)
        h = wolf_rule(alpha_0,x,f,grad,f_x,grad_x,d_x,c_1,c_2)
        x_list[:,i] = x
        f_list[i] = f_x
        if i % 10 == 0:
            # print "  iter={}, grad={}, alpha={}, x={}, f(x)={}".format(i, pk, alpha, x, f(x))
            print "  iter={}, x={}, f(x)={}".format(i, x, f(x))

        if (np.linalg.norm(x - x_old) < error_point)|(np.linalg.norm(grad_x) < error_grad):
            break
        x_old = x
    return dict([('x_list',x_list[:,0:i]), ('f_list',f_list[0:i])])




def conjugate_gradient_wolf(f, grad, x0, iterations, error_point, error_grad, c_1=0.1,c_2=0.9,L=100):
    dim = np.max(np.shape(x0))
    x_list = np.zeros([dim,iterations])  
    f_list = np.zeros(iterations)  
    x = x0
    x_old = x
    grad_x = grad(x)
    d_x = -grad_x
    f_x = f(x)
    alpha_0 = -(1./L)*np.dot(d_x,grad_x)/np.power(np.linalg.norm(d_x),2)
    h = wolf_rule(alpha_0,x,f,grad,f_x,grad_x,d_x,c_1,c_2)
    for i in xrange(iterations):
        x = x + h*d_x
        grad_x_old = grad_x
        grad_x = grad(x)
        f_x = f(x)
        kappa = np.dot(grad_x - grad_x_old, grad_x)/np.power(np.linalg.norm(grad_x),2)
        d_x = kappa*d_x -grad_x
        alpha_0 = -(1./L)*np.dot(d_x,grad_x)/np.power(np.linalg.norm(d_x),2)
        h = wolf_rule(alpha_0,x,f,grad,f_x,grad_x,d_x,c_1,c_2)
        x_list[:,i] = x
        f_list[i] = f_x
        if i % 10 == 0:
            # print "  iter={}, grad={}, alpha={}, x={}, f(x)={}".format(i, pk, alpha, x, f(x))
            print "  iter={}, x={}, f(x)={}".format(i, x, f(x))

        if (np.linalg.norm(x - x_old) < error_point)|(np.linalg.norm(grad_x) < error_grad):
            break
        x_old = x
    return dict([('x_list',x_list[:,0:i]), ('f_list',f_list[0:i])])
