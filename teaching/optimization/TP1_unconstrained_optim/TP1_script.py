import numpy as np
import numpy.random as rnd
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from functions import *
from steepest_descent import *

# EXERCISE 1

# show Rosenbrock


def rosenbrock_display(x, y): return (1 - x)**2 + 100 * (x**2 - y)**2


nx = 1000  # number of discretization points along the x-axis
ny = 1000  # number of discretization points along the y-axis
a = -0.5
b = 2.  # extreme points in the x-axis
c = -1.5
d = 4  # extreme points in the y-axis

X, Y = np.meshgrid(np.linspace(a, b, nx), np.linspace(c, d, ny))

Z = rosenbrock_display(X, Y)

CS = plt.contour(X, Y, Z, np.logspace(-0.5, 3.5, 20, base=10))
plt.title(r'$\mathrm{Rosenbrock \ function: } f(x,y)=(x-1)^2+100(x^2 - y)^2$')
plt.xlabel('x')
plt.ylabel('y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.show()

# show quadratic


def quad_display(x, y, M, m): return M * x**2 + m * y**2


M = 10
m = 1

nx = 1000  # number of discretization points along the x-axis
ny = 1000  # number of discretization points along the y-axis
a = -2
b = 2.  # extreme points in the x-axis
c = -2
d = 2  # extreme points in the y-axis

X, Y = np.meshgrid(np.linspace(a, b, nx), np.linspace(c, d, ny))

Z = quad_display(X, Y, M, m)

CS = plt.contour(X, Y, Z, np.linspace(0, 10, 15))
plt.title(r'$\mathrm{Quadratic \ function: } f(x,y)=Mx^2 + my^2$')
plt.xlabel('x')
plt.ylabel('y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.show()

# Rosenbrock + fixed step size gradient

f = rosenbrock
grad = rosenbrock_grad
x0 = np.array([1.1, 2.1])
error_point = 10**-10
error_grad = 10**-10
h = 10**-3
iterations = 10000
result = steepest_descent_constant_step(
    f, grad, x0, iterations, error_point, error_grad, h)

x_list = result['x_list']

all_x_i = np.append(x0[0], x_list[0, :])
all_y_i = np.append(x0[1], x_list[1, :])
plt.plot(all_x_i, all_y_i, 'k+-')
plt.plot(x0[0], x0[1], 'r+')
plt.plot([1], [1], 'g+')
plt.title(
    r'$\mathrm{Rosenbrock \ minimization: fixed \ step \ size \ gradient}$')
plt.xlabel('x')
plt.ylabel('y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.show()

# Rosenbrock + normalized step size gradient

f = rosenbrock
grad = rosenbrock_grad
x0 = np.array([1.1, 2.1])
error_point = 10**-10
error_grad = 10**-10
h = 10**-2
iterations = 10000
result = steepest_descent_normalized_step(
    f, grad, x0, iterations, error_point, error_grad, h)

x_list = result['x_list']

all_x_i = np.append(x0[0], x_list[0, :])
all_y_i = np.append(x0[1], x_list[1, :])
plt.plot(all_x_i, all_y_i, 'k+-')
plt.plot(x0[0], x0[1], 'r+')
plt.plot([1], [1], 'g+')
plt.title(
    r'$\mathrm{Rosenbrock \ minimization: normalized \ step \ size \ gradient}$')
plt.xlabel('x')
plt.ylabel('y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.show()

# quadratic + fixed size + normalized size

M = 5.
m = 1.
n = 3
(f, grad, hess) = mk_quad(M, m, n)
x0 = np.array([5., 5., 5.])
error_point = 10**-10
error_grad = 10**-10
h = 10**-3
iterations = 10000

result = steepest_descent_constant_step(
    f, grad, x0, iterations, error_point, error_grad, h)

x_list = result['x_list']

all_x_i = np.append(x0[0], x_list[0, :])
all_y_i = np.append(x0[1], x_list[1, :])
plt.plot(all_x_i, all_y_i, 'k+-')
plt.plot(x0[0], x0[1], 'r+')
plt.plot([0], [0], 'g+')
plt.title(
    r'$\mathrm{Quadratic \ minimization: constant \ step \ size \ gradient}$')
plt.xlabel('x')
plt.ylabel('y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.show()

result = steepest_descent_normalized_step(
    f, grad, x0, iterations, error_point, error_grad, h)

x_list = result['x_list']

all_x_i = np.append(x0[0], x_list[0, :])
all_y_i = np.append(x0[1], x_list[1, :])
plt.plot(all_x_i, all_y_i, 'k+-')
plt.plot(x0[0], x0[1], 'r+')
plt.plot([0], [0], 'g+')
plt.title(
    r'$\mathrm{Quadratic \ minimization: normalized \ step \ size \ gradient}$')
plt.xlabel('x')
plt.ylabel('y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.show()

# EXERCISE 2

# Rosenbrock + Armijo + Wolfe

f = rosenbrock
grad = rosenbrock_grad
x0 = np.array([1.1, 2.1])
error_point = 10**-10
error_grad = 10**-10
iterations = 10000
result = steepest_descent_armijo(
    f, grad, x0, iterations, error_point, error_grad)

x_list = result['x_list']

all_x_i = np.append(x0[0], x_list[0, :])
all_y_i = np.append(x0[1], x_list[1, :])
plt.plot(all_x_i, all_y_i, 'k+-')
plt.plot(x0[0], x0[1], 'r+')
plt.plot([1], [1], 'g+')
plt.title(
    r'$\mathrm{Rosenbrock \ minimization: steepest \ + \ Armijo \ rule}$')
plt.xlabel('x')
plt.ylabel('y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.show()

f = rosenbrock
grad = rosenbrock_grad
x0 = np.array([1.1, 2.1])
error_point = 10**-10
error_grad = 10**-10
iterations = 10000
result = steepest_descent_wolfe(
    f, grad, x0, iterations, error_point, error_grad)

x_list = result['x_list']

all_x_i = np.append(x0[0], x_list[0, :])
all_y_i = np.append(x0[1], x_list[1, :])
plt.plot(all_x_i, all_y_i, 'k+-')
plt.plot(x0[0], x0[1], 'r+')
plt.plot([1], [1], 'g+')
plt.title(r'$\mathrm{Rosenbrock \ minimization: steepest \ + Wolfe \ rule}$')
plt.xlabel('x')
plt.ylabel('y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.show()

# Quadratic + Armijo + Wolfe

M = 5.
m = 1.
n = 3
(f, grad, hess) = mk_quad(M, m, n)
x0 = np.array([5., 5., 5.])
error_point = 10**-10
error_grad = 10**-10
iterations = 10000

result = steepest_descent_armijo(
    f, grad, x0, iterations, error_point, error_grad)

x_list = result['x_list']

all_x_i = np.append(x0[0], x_list[0, :])
all_y_i = np.append(x0[1], x_list[1, :])
plt.plot(all_x_i, all_y_i, 'k+-')
plt.plot(x0[0], x0[1], 'r+')
plt.plot([0], [0], 'g+')
plt.title(
    r'$\mathrm{Quadratic \ minimization: steepest \ descent \ + \ Armijo \ rule}$')
plt.xlabel('x')
plt.ylabel('y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.show()

result = steepest_descent_wolfe(
    f, grad, x0, iterations, error_point, error_grad)

x_list = result['x_list']

all_x_i = np.append(x0[0], x_list[0, :])
all_y_i = np.append(x0[1], x_list[1, :])
plt.plot(all_x_i, all_y_i, 'k+-')
plt.plot(x0[0], x0[1], 'r+')
plt.plot([0], [0], 'g+')
plt.title(
    r'$\mathrm{Quadratic \ minimization: steepest \ descent \ + \ Wolfe \ rule}$')
plt.xlabel('x')
plt.ylabel('y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.show()

# EXERCISE 3

# Rosenbrock + conjugate gradient

f = rosenbrock
grad = rosenbrock_grad
x0 = np.array([1.1, 2.1])
error_point = 10**-10
error_grad = 10**-10
iterations = 10000
result = conjugate_gradient_armijo(
    f, grad, x0, iterations, error_point, error_grad)

x_list = result['x_list']

all_x_i = np.append(x0[0], x_list[0, :])
all_y_i = np.append(x0[1], x_list[1, :])
plt.plot(all_x_i, all_y_i, 'k+-')
plt.plot(x0[0], x0[1], 'r+')
plt.plot([1], [1], 'g+')
plt.title(
    r'$\mathrm{Rosenbrock \ minimization: conjugate \ gradient \ + \ Armijo \ rule}$')
plt.xlabel('x')
plt.ylabel('y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.show()

f = rosenbrock
grad = rosenbrock_grad
x0 = np.array([1.1, 2.1])
error_point = 10**-10
error_grad = 10**-10
iterations = 10000
result = conjugate_gradient_wolfe(
    f, grad, x0, iterations, error_point, error_grad)

x_list = result['x_list']

all_x_i = np.append(x0[0], x_list[0, :])
all_y_i = np.append(x0[1], x_list[1, :])
plt.plot(all_x_i, all_y_i, 'k+-')
plt.plot(x0[0], x0[1], 'r+')
plt.plot([1], [1], 'g+')
plt.title(
    r'$\mathrm{Rosenbrock \ minimization: conjugate \ gradient \ + \ Wolfe \ rule}$')
plt.xlabel('x')
plt.ylabel('y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.show()

# Quadratic + conjugate gradient

M = 5.
m = 1.
n = 3
(f, grad, hess) = mk_quad(M, m, n)
x0 = np.array([5., 5., 5.])
error_point = 10**-10
error_grad = 10**-10
iterations = 10000

result = conjugate_gradient_armijo(
    f, grad, x0, iterations, error_point, error_grad)

x_list = result['x_list']

all_x_i = np.append(x0[0], x_list[0, :])
all_y_i = np.append(x0[1], x_list[1, :])
plt.plot(all_x_i, all_y_i, 'k+-')
plt.plot(x0[0], x0[1], 'r+')
plt.plot([0], [0], 'g+')
plt.title(
    r'$\mathrm{Quadratic \ minimization: conjugate \ gradient \ and \ Armijo \ rule}$')
plt.xlabel('x')
plt.ylabel('y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.show()

result = conjugate_gradient_wolfe(
    f, grad, x0, iterations, error_point, error_grad)

x_list = result['x_list']

all_x_i = np.append(x0[0], x_list[0, :])
all_y_i = np.append(x0[1], x_list[1, :])
plt.plot(all_x_i, all_y_i, 'k+-')
plt.plot(x0[0], x0[1], 'r+')
plt.plot([0], [0], 'g+')
plt.title(
    r'$\mathrm{Quadratic \ minimization: conjugate \ gradient \ and \ Wolfe \ rule}$')
plt.xlabel('x')
plt.ylabel('y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.show()

# EXERCISE 4

# Lennard-Jones show

r = np.linspace(0.85, 1.5, 1000)

Vr = V(r)
plt.plot(r, Vr)
plt.xlabel('rayon')
plt.ylabel('Force')
plt.title(r'$\mathrm{Potentiel \ de \ van \ der \ Walls}$')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.show()

# N = 4

N = 4
error_point = 10 ** -10
error_grad = 10 ** -10
x0 = rnd.random(3 * N)
x0[0:2] = 0
iterations = 10000

result = steepest_descent_armijo(
    J, grad_J, x0, iterations, error_point, error_grad)

init_pos = np.reshape(x0, (N, 3))
final_pos = np.reshape(result['x_list'][:, -1], (N, 3))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(init_pos[:, 0], init_pos[:, 1], init_pos[:, 2], '^')
ax.scatter(final_pos[:, 0], final_pos[:, 1], final_pos[:, 2], '^')
for i in range(4):
    ax.plot([init_pos[i, 0], final_pos[i, 0]], [init_pos[i, 1],
                                                final_pos[i, 1]], [init_pos[i, 2], final_pos[i, 2]], 'g')
    for j in range(4):
        ax.plot([final_pos[i, 0], final_pos[j, 0]], [final_pos[i, 1],
                                                     final_pos[j, 1]], [final_pos[i, 2], final_pos[j, 2]], 'r')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

print(J(np.ravel(init_pos)))
print(J(np.ravel(final_pos)))

# N = 13

N = 13
error_point = 10 ** -10
error_grad = 10 ** -10
x0 = rnd.random(3 * N)
x0[0:2] = 0
iterations = 100000

result = steepest_descent_armijo(
    J, grad_J, x0, iterations, error_point, error_grad)

init_pos = np.reshape(x0, (N, 3))
final_pos = np.reshape(result['x_list'][:, -1], (N, 3))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(final_pos[:, 0], final_pos[:, 1], final_pos[:, 2], '^')
for i in range(13):
    ax.plot([init_pos[i, 0], final_pos[i, 0]], [init_pos[i, 1],
                                                final_pos[i, 1]], [init_pos[i, 2], final_pos[i, 2]], 'g')
moy = np.mean(final_pos, 0)
d = np.linalg.norm(final_pos - moy, axis=1)
center = np.argmin(d)
vertices = final_pos[np.arange(len(final_pos)) != center, :]
vertdiff = np.zeros([N - 1, N - 1, 3])
vertdiff -= vertices
vertdiff = vertdiff - np.transpose(vertdiff, (1, 0, 2))
vertnorm = np.sum(vertdiff ** 2, 2)
for i in range(12):
    indsort = np.argsort(vertnorm[:, i])
    for j in range(5):
        js = indsort[j]
        ax.plot([vertices[i, 0], vertices[js, 0]], [vertices[i, 1],
                                                    vertices[js, 1]], [vertices[i, 2], vertices[js, 2]], 'r')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

print J(np.ravel(init_pos))
print J(np.ravel(final_pos))

# EXERCISE 5

# generate data

A = 5
sigma = 0.1
omega = 0.1 * 2 * np.pi
param_true = np.array([A, sigma, omega])

wave_fun, wave_grad, wave_hessian = mk_wave(A, sigma, omega)

noise = 0.1

x_min = 0
x_max = 20

x_train = np.linspace(x_min, x_max, 30)
y_train = generate_data(x_train, A, sigma, omega, noise=noise, n_outliers=5)

plt.plot(x_train, y_train, 'o', label='data')
plt.plot(x_train, wave_fun(x_train), 'r-', label='signal')
plt.title(r'$\mathrm{Data \ with \ outliers}$')
plt.xlabel('x')
plt.ylabel('y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.legend()
plt.show()

# loss functions

r = np.linspace(0, 5, 100)

linear = r**2
huber = r**2
huber[huber > 1] = 2 * r[huber > 1] - 1
soft_l1 = 2 * (np.sqrt(1 + r**2) - 1)
cauchy = np.log1p(r**2)
arctan = np.arctan(r**2)

plt.plot(r, linear, label='linear')
plt.plot(r, huber, label='Huber')
plt.plot(r, soft_l1, label='soft_l1')
plt.plot(r, cauchy, label='Cauchy')
plt.plot(r, arctan, label='arctan')
plt.xlabel("$r$")
plt.ylabel(r"$\rho(r^2)$")
plt.title(r'$\mathrm{Different \ loss \ functions}$')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.legend()
plt.show()

# regression with Levenberg/Newton/Wolfe

f, grad, hess = mk_nonlinreg(x_train, y_train)

x0 = rnd.random(3)
true_hessian = hess(x0)
lm_hessian = hess(x0, 'lm', 0.1)
print "true cond = {}, lm cond = {}".format(np.linalg.cond(true_hessian), np.linalg.cond(lm_hessian))

error_point = 10 ** -10
error_grad = 10 ** -10
h = 10 ** -3
iterations = 10000

result_newton = newton_descent(
    f, grad, hess, x0, iterations, error_point, error_grad, h)
result_lm = newton_descent(f, grad, hess, x0, iterations,
                           error_point, error_grad, h, method='lm', mu=10 ** -1)
result_wolfe = steepest_descent_wolfe(
    f, grad, x0, iterations, error_point, error_grad)

(param_newton, param_lm, param_wolfe) = (
    result['x_list'][:, -1] for result in (result_newton, result_lm, result_wolfe))
print "Newton parameters = {}, Levenberg-Marquardt parameters = {}, Wolfe parameters = {}".format(param_newton, param_lm, param_wolfe)
newton_wave_fun = mk_wave(param_newton[0], param_newton[1], param_newton[2])[0]
lm_wave_fun = mk_wave(param_lm[0], param_lm[1], param_lm[2])[0]
wolfe_wave_fun = mk_wave(param_wolfe[0], param_wolfe[1], param_wolfe[2])[0]

plt.plot(x_train, y_train, 'o', label='data')
plt.plot(x_train, wave_fun(x_train), 'r-', label='signal')
plt.plot(x_train, newton_wave_fun(x_train), 'c-', label='Newton')
plt.plot(x_train, lm_wave_fun(x_train), 'g-', label='Levenberg')
plt.plot(x_train, wolfe_wave_fun(x_train), 'b-', label='Wolfe')
plt.title(r'$\mathrm{Data \ with \ outliers}$')
plt.xlabel('x')
plt.ylabel('y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.legend()
plt.show()

# same study with linear regression

a = 2
b = 3
lin_fun, lin_grad, lin_hessian = mk_lin(a, b)

noise = 5

x_min = 0
x_max = 20
n_data = 50

x_train = np.linspace(x_min, x_max, n_data)
y_train = a * x_train + b + noise * rnd.randn(x_train.size)

plt.plot(x_train, y_train, 'o', label='data')
plt.plot(x_train, lin_fun(x_train), 'r-', label='signal')
plt.title(r'$\mathrm{Data \ with \ outliers}$')
plt.xlabel('x')
plt.ylabel('y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.legend()
plt.show()

f, grad, hess = mk_linreg(x_train, y_train)

error_point = 10 ** -10
error_grad = 10 ** -10
h = 10 ** -2
iterations = 1000
x0 = rnd.random(2)

result_newton = newton_descent(
    f, grad, hess, x0, iterations, error_point, error_grad, h)

result_lm = newton_descent(f, grad, hess, x0, iterations,
                           error_point, error_grad, h, method='lm', mu=0.01)

result_wolfe = steepest_descent_wolfe(
    f, grad, x0, iterations, error_point, error_grad)

(param_newton, param_lm, param_wolfe) = (
    result['x_list'][:, -1] for result in (result_newton, result_lm, result_wolfe))
print "Newton parameters = {}, Levenberg-Marquardt parameters = {}, Wolfe parameters = {}".format(param_newton, param_lm, param_wolfe)
newton_lin_fun, newton_lin_grad, newton_lin_hessian = mk_lin(
    param_newton[0], param_newton[1])
lm_lin_fun, lm_lin_grad, lm_lin_hessian = mk_lin(param_lm[0], param_lm[1])
wolfe_lin_fun, wolfe_lin_grad, wolfe_lin_hessian = mk_lin(
    param_wolfe[0], param_wolfe[1])

plt.plot(x_train, y_train, 'o', label='data')
plt.plot(x_train, lin_fun(x_train), 'r-', label='signal')
plt.plot(x_train, newton_lin_fun(x_train), 'c-', label='Newton')
plt.plot(x_train, lm_lin_fun(x_train), 'g-', label='Levenberg')
plt.plot(x_train, wolfe_lin_fun(x_train), 'b-', label='Wolfe')
plt.title(r'$\mathrm{Data \ with \ outliers}$')
plt.xlabel('x')
plt.ylabel('y')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.legend()
plt.show()
