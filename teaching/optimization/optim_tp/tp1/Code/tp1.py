#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:14:44 2018

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

plt.show()

ndim = 2
f = rosenbrock
grad = rosenbrock_prime
x0 = np.array([1.1,2.1])
error_point = 10**(-7)
error_grad = 10**(-7)
h=0.0001
iterations = 100

x_list, f_list = steepest_descent_constant_step(f, grad, x0, iterations, error_point, error_grad, h)

all_x_i = x_list[0,:]
all_y_i = x_list[1,:]
plt.plot(all_x_i, all_y_i, 'b-', linewidth=2)
plt.plot(all_x_i, all_y_i, 'k+')
#pl.plot(logging_f.all_x_i, logging_f.all_y_i, 'k.', markersize=2)
plt.plot([1], [1], 'rx', markersize=12)




pl.figure(index, figsize=(3, 2.5))
    pl.clf()
    pl.axes([0, 0, 1, 1])

    X = np.concatenate((x[np.newaxis, ...], y[np.newaxis, ...]), axis=0)
    z = np.apply_along_axis(f, 0, X)
    log_z = np.log(z + .01)
    pl.imshow(log_z,
            extent=[x_min, x_max, y_min, y_max],
            cmap=pl.cm.gray_r, origin='lower',
            vmax=log_z.min() + 1.5*log_z.ptp())
    contours = pl.contour(log_z,
                        levels=levels.get(f, None),
                        extent=[x_min, x_max, y_min, y_max],
                        cmap=pl.cm.gnuplot, origin='lower')
    levels[f] = contours.levels
    pl.clabel(contours, inline=1,
                fmt=super_fmt, fontsize=14)

    pl.plot(all_x_i, all_y_i, 'b-', linewidth=2)
    pl.plot(all_x_i, all_y_i, 'k+')

    pl.plot(logging_f.all_x_i, logging_f.all_y_i, 'k.', markersize=2)

    pl.plot([0], [0], 'rx', markersize=12)


    pl.xticks(())
    pl.yticks(())
    pl.xlim(x_min, x_max)
    pl.ylim(y_min, y_max)
    pl.draw()

    pl.figure(index + 100, figsize=(4, 3))
    pl.clf()
    pl.semilogy(np.maximum(np.abs(all_f_i), 1e-30), linewidth=2,
                label='# iterations')
    pl.ylabel('Error on f(x)')
    pl.semilogy(logging_f.counts,
                np.maximum(np.abs(logging_f.all_f_i), 1e-30),
                linewidth=2, color='g', label='# function calls')
    pl.legend(loc='upper right', frameon=True, prop=dict(size=11),
              borderaxespad=0, handlelength=1.5, handletextpad=.5)
    pl.tight_layout()
    pl.draw()