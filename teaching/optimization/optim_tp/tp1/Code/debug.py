#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:09:48 2018

@author: alain
"""

import numpy as np
import matplotlib.pyplot as plt
from functions import *
from steepest_descent import * 

#Plot of Rosenbrock's banana function: f(x,y)=(1-x)^2+100(y-x^2)^2

rosenbrockfunction = lambda x,y: (1-x)**2+100*(y-x**2)**2

n = 1000 # number of discretization points along the x-axis
m = 1000 # number of discretization points along the x-axis
a=-0.5; b=2. # extreme points in the x-axis
c=-1.5; d=4. # extreme points in the y-axis

X,Y = np.meshgrid(np.linspace(a,b,n), np.linspace(c,d,m))

Z = rosenbrockfunction(X,Y)

CS= plt.contour(X,Y,Z,np.logspace(-0.5,3.5,20,base=10),cmap='gray')
plt.clabel(CS,inline=1, fontsize=10)
plt.title(r'$\mathrm{Rosenbrock\, function: } f(x,y)=(1-x)^2+100(y-x^2)^2$')
plt.xlabel('x')
plt.ylabel('y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.plot([1], [1], 'rx', markersize=12)

plt.show()

ndim = 2
f = rosenbrock
grad = rosenbrock_prime
x0 = np.array([1.1,2.1])
error_point = 10**(-7)
error_grad = 10**(-7)
h=0.001
iterations = 1000

#
#result = steepest_descent_constant_step(f, grad, x0, iterations, error_point, error_grad, h)
#x_list = result['x_list']
#all_x_i = x_list[0,:]
#all_y_i = x_list[1,:]
#plt.plot(all_x_i, all_y_i, linewidth=2)
#plt.plot(all_x_i, all_y_i)

#pl.plot(logging_f.all_x_i, logging_f.all_y_i, 'k.', markersize=2)


result = steepest_descent_armijo(f, grad, x0, iterations, error_point, error_grad)
x_list = result['x_list']
all_x_i = x_list[0,:]
all_y_i = x_list[1,:]
plt.plot(all_x_i, all_y_i, linewidth=2)
plt.plot(all_x_i, all_y_i)


result = steepest_descent_wolf(f, grad, x0, iterations, error_point, error_grad)
x_list = result['x_list']
all_x_i = x_list[0,:]
all_y_i = x_list[1,:]
plt.plot(all_x_i, all_y_i, 'b-', linewidth=2)
plt.plot(all_x_i, all_y_i, 'k+')

result = conjugate_gradient_wolf(f, grad, x0, iterations, error_point, error_grad)
x_list = result['x_list']
all_x_i = x_list[0,:]
all_y_i = x_list[1,:]
plt.plot(all_x_i, all_y_i, 'b-', linewidth=2)
plt.plot(all_x_i, all_y_i, 'k+')
