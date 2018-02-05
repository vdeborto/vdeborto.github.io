#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 19:19:42 2018

@author: alain
"""

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
    return x_list[:,0:i]; f_list[0:i]


def newton(f, g, H, x0, iterations, error):
  x = x0
  x_old = x
  c2 = 0.9
  for i in xrange(iterations):
    pk = -np.linalg.solve(H(x), g(x))
    alpha = step_length(f, g, x, 1.0, pk, c2)
    x = x + alpha * pk
    if i % 50 == 0:
      # print "  iter={}, grad={}, alpha={}, x={}, f(x)={}".format(i, pk, alpha, x, f(x))
      print "  iter={}, x={}, f(x)={}".format(i, x, f(x))

    if np.linalg.norm(x - x_old) < error:
      break
    x_old = x
  return x, i + 1



def bfgs(f, g, x0, iterations, error):
  xk = x0
  c2 = 0.9
  I = np.identity(xk.size)
  Hk = I

  for i in xrange(iterations):
    # compute search direction
    gk = g(xk)
    pk = -Hk.dot(gk)

    # obtain step length by line search
    alpha = step_length(f, g, xk, 1.0, pk, c2)

    # update x
    xk1 = xk + alpha * pk
    gk1 = g(xk1)

    # define sk and yk for convenience
    sk = xk1 - xk
    yk = gk1 - gk

    # compute H_{k+1} by BFGS update
    rho_k = float(1.0 / yk.dot(sk))

    Hk1 = (I - rho_k * np.outer(sk, yk)).dot(Hk).dot(I - \
           rho_k * np.outer(yk, sk)) + rho_k * np.outer(sk, sk)

    if i % 10 == 0:
      # print "  iter={}, grad={}, alpha={}, x={}, f(x)={}".format(i, pk, alpha, xk, f(xk))
      print "  iter={}, x={}, f(x)={}".format(i, xk, f(xk))

    if np.linalg.norm(xk1 - xk) < error:
      xk = xk1
      break

    Hk = Hk1
    xk = xk1

  return xk, i + 1


def l_bfgs(f, g, x0, iterations, error, m=10):
  xk = x0
  c2 = 0.9
  I = np.identity(xk.size)
  Hk = I

  sks = []
  yks = []

  def Hp(H0, p):
    m_t = len(sks)
    q = g(xk)
    a = np.zeros(m_t)
    b = np.zeros(m_t)
    for i in reversed(xrange(m_t)):
      s = sks[i]
      y = yks[i]
      rho_i = float(1.0 / y.T.dot(s))
      a[i] = rho_i * s.dot(q)
      q = q - a[i] * y

    r = H0.dot(q)

    for i in xrange(m_t):
      s = sks[i]
      y = yks[i]
      rho_i = float(1.0 / y.T.dot(s))
      b[i] = rho_i * y.dot(r)
      r = r + s * (a[i] - b[i])

    return r

  for i in xrange(iterations):
    # compute search direction
    gk = g(xk)
    pk = -Hp(I, gk)

    # obtain step length by line search
    alpha = step_length(f, g, xk, 1.0, pk, c2)

    # update x
    xk1 = xk + alpha * pk
    gk1 = g(xk1)

    # define sk and yk for convenience
    sk = xk1 - xk
    yk = gk1 - gk

    sks.append(sk)
    yks.append(yk)
    if len(sks) > m:
      sks = sks[1:]
      yks = yks[1:]

    # compute H_{k+1} by BFGS update
    rho_k = float(1.0 / yk.dot(sk))

    if i % 10 == 0:
      print "  iter={}, grad={}, alpha={}, x={}, f(x)={}".format(i, pk, \
        alpha, xk, f(xk))

    if np.linalg.norm(xk1 - xk) < error:
      xk = xk1
      break

    xk = xk1

return xk, i + 1