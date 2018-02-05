import numpy as np

# quadratic function and Rosenbrock function


def mk_quad(m, M, ndim=2):
    def quad(x):
        y = np.copy(np.asarray(x))
        scal = np.ones(ndim)
        scal[0] = m
        scal[1] = M
        y *= scal
        return np.sum(y**2)

    def quad_grad(x):
        y = np.asarray(x)
        scal = np.ones(ndim)
        scal[0] = m
        scal[1] = M
        return 2 * scal * y

    def quad_hessian(x):
        scaling = np.ones(ndim)
        scaling[0] = m
        scaling[1] = M
        return 2 * np.diag(scaling)

    return quad, quad_grad, quad_hessian


def rosenbrock(x):
    y = np.asarray(x)
    return np.sum((y[0] - 1)**2 + 100 * (y[1] - y[0]**2)**2)


def rosenbrock_grad(x):
    y = np.asarray(x)
    grad = np.zeros_like(y)
    grad[0] = 400 * y[0] * (y[0]**2 - y[1]) + 2 * (y[0] - 1)
    grad[1] = 200 * (y[1] - y[0]**2)
    return grad


def rosenbrock_hessian_(x):
    y = np.asarray(x)
    return np.array((
                    (1 - 4 * 100 * y[1] + 12 * 100 * y[0]**2, -4 * y[0] * 100),
                    (-4 * 100 * y[0],    2 * 100),
                    ))

# Lennard-Jones functions


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

    Mnorm = np.sum(M ** 2, 2)
    np.fill_diagonal(Mnorm, 1)
    Mnorm = V2der(Mnorm)
    np.fill_diagonal(Mnorm, 0)

    grad = np.reshape(Mnorm, (N**2, 1)) * np.reshape(M, (N ** 2, 3))
    grad = np.reshape(grad, (N, N, 3))
    grad = np.sum(grad, 1)
    return 2 * np.ravel(grad)

# wave functions


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

# non linear regression functions


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

# linear regression functions


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
