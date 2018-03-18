import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from compute_wavelet_filter import compute_wavelet_filter


def upsampling(x, d):
    """
        up-sampling along dimension d by factor p=2
    """
    p = 2
    s = x.shape
    if d == 1:
        y = np.zeros((p * s[0], s[1]))
        y[::p, :] = x
    elif d == 2:
        y = np.zeros((s[0], p * s[1]))
        y[:, ::p] = x
    else:
        raise Exception('Not implemented')
    return y


def subsampling(x, d):
    # subsampling along dimension d by factor p=2
    p = 2
    if d == 1:
        y = x[::p, :]
    elif d == 2:
        y = x[:, ::p]
    else:
        raise Exception('Not implemented')
    return y


def rescale(f, a=0, b=1):
    """
        Rescale linearly the dynamic of a vector to fit within a range [a,b]
    """
    v = f.max() - f.min()
    g = (f - f.min()).copy()
    if v > 0:
        g = g * 1. / v
    return a * 1. + g * (b - a)


def load_image(name, n=-1, flatten=1, resc=1, grayscale=1):
    """
        Load an image from a file, rescale its dynamic to [0,1], turn it into a grayscale image
        and resize it to size n x n.
    """
    f = plt.imread(name)
    # turn into normalized grayscale image
    if grayscale == 1:
        if (flatten == 1) and np.ndim(f) > 2:
            f = np.sum(f, axis=2)
    if resc == 1:
        f = rescale(f)
    # change the size of the image
    if n > 0:
        if np.ndim(f) == 2:
            f = transform.resize(f, [n, n], 1)
        elif np.ndim(f) == 3:
            f = transform.resize(f, [n, n, f.shape[2]], 1)
    return f


def circshift1d(x, k):
    """ 
        Circularly shift a 1D vector
    """
    return np.roll(x, -k, axis=0)


def cconv(x, h, d):
    """
        Circular convolution along dimension d.
        h should be small and with odd size
    """
    if d == 2:
        # apply to transposed matrix
        return np.transpose(cconv(np.transpose(x), h, 1))
    y = np.zeros(x.shape)
    p = len(h)
    pc = int(round(float((p - 1) / 2)))
    for i in range(0, p):
        y = y + h[i] * circshift1d(x, i - pc)
    return y


def reverse(x):
    """
        Reverse a vector. 
    """
    return x[::-1]


def perform_wavortho_transf(f, Jmin, dir, h):
    """
        perform_wavortho_transf - compute orthogonal wavelet transform

        fw = perform_wavortho_transf(f,Jmin,dir,options);

        You can give the filter in options.h.

        Works in 2D only.

        Copyright (c) 2014 Gabriel Peyre
    """

    n = f.shape[1]
    Jmax = int(np.log2(n) - 1)
    # compute g filter
    u = np.power(-np.ones(len(h) - 1), range(1, len(h)))
    # alternate +1/-1
    g = np.concatenate(([0], h[-1:0:-1] * u))

    if dir == 1:
        ### FORWARD ###
        fW = f.copy()
        for j in np.arange(Jmax, Jmin - 1, -1):
            A = fW[:2 ** (j + 1):, :2 ** (j + 1):]
            for d in np.arange(1, 3):
                Coarse = subsampling(cconv(A, h, d), d)
                Detail = subsampling(cconv(A, g, d), d)
                A = np.concatenate((Coarse, Detail), axis=d - 1)
            fW[:2 ** (j + 1):, :2 ** (j + 1):] = A
        return fW
    else:
        ### BACKWARD ###
        fW = f.copy()
        f1 = fW.copy()
        for j in np.arange(Jmin, Jmax + 1):
            A = f1[:2 ** (j + 1):, :2 ** (j + 1):]
            for d in np.arange(1, 3):
                if d == 1:
                    Coarse = A[:2**j:, :]
                    Detail = A[2**j: 2**(j + 1):, :]
                else:
                    Coarse = A[:, :2 ** j:]
                    Detail = A[:, 2 ** j:2 ** (j + 1):]
                Coarse = cconv(upsampling(Coarse, d), reverse(h), d)
                Detail = cconv(upsampling(Detail, d), reverse(g), d)
                A = Coarse + Detail
            f1[:2 ** (j + 1):, :2 ** (j + 1):] = A
        return f1


def noisy_observations(n=32, r_sparse=0.2, r_info=0.5):
    """
    Measurement function.

    Parameters:
    - n is the image size (n x n);
    - r_sparse is the ratio of non-zero coefficients (wavelet domain) of the
    signal x to recover;
    - r_info is the ratio between the size of y and the size of x.

    Return y, A,  where:
    - y is the vector of measurements;
    - A is the sensing matrix (we look for x such that y = Ax);
    """

    im = rescale(load_image("barb.bmp", n))

    h = compute_wavelet_filter("Daubechies", 4)

    # Compute the matrix of wavelet transform
    mask = np.zeros((n, n))
    A0 = []
    for i in range(n):
        for j in range(n):
            mask[i, j] = 1
            wt = perform_wavortho_transf(mask, 0, +1, h)
            A0.append(wt.ravel())
            mask[i, j] = 0
    A0 = np.asarray(A0)

    # Gaussian matrix x Wavelet transform (keep ratio r_info)
    G = np.random.randn(int(np.floor(n**2 * r_info)), n**2) / n
    A = G.dot(A0)

    # Threshold the image (keep ratio r_sparse) and generate the measurements y
    # Same as x_true = A0.T.dot(im.flatten())
    x_true = perform_wavortho_transf(im, 0, +1, h).ravel()
    thshol = np.sort(np.abs(x_true.ravel()))[int((1 - r_sparse) * n**2)]
    x_true[np.abs(x_true) <= thshol] = 0
    y = A.dot(x_true)  # Vector of measurements

    return y, A


def back_to_image(x):
    n = int(np.sqrt(x.size))
    h = compute_wavelet_filter("Daubechies", 4)
    wt = x.reshape((n, n))
    im = perform_wavortho_transf(wt, 0, -1, h)
    return im


def plot_image(x):
    plt.figure(figsize=(1, 1))
    im = back_to_image(x)
    plt.imshow(im, cmap='gray')
    plt.axis('off')


def total_variation_op(n=32):
    """
    Measurement function.

    Parameters:
    - n is the image size (n x n);

    Return T a total variation operator.
    """
    h = compute_wavelet_filter("Daubechies", 4)

    # Compute the matrix of wavelet transform
    mask = np.zeros((n, n))
    A0 = []
    for i in range(n):
        for j in range(n):
            mask[i, j] = 1
            wt = perform_wavortho_transf(mask, 0, +1, h)
            A0.append(wt.ravel())
            mask[i, j] = 0
    A0 = np.asarray(A0)

    # Total variation operator
    dx = np.eye(n**2)
    dx -= np.roll(dx, 1, axis=1)
    dx = np.delete(dx, np.s_[n - 1::n], axis=0)

    dy = np.eye(n**2)
    dy -= np.roll(dy, n, axis=1)
    dy = np.delete(dy, np.s_[-n:], axis=0)

    T = np.r_[dx, dy].dot(A0)  # TV in the image domain

    T = np.r_[np.eye(n**2), T]  # For usual L1 norm, add identity

    return T


def noisy_observation_inf(n=2**4):
    A = np.zeros((n, n))
    while np.linalg.det(A) == 0:
        A = np.random.randn(n, n)
    y = A.dot(np.ones(n)) + np.random.randn(n)
    return y, A


def noisy_observation_nuclear(n=2**4):
    A = np.random.binomial(1, 0.1, size=(n, n))
    while np.sum(A) <= n / 5:
        A = np.random.binomial(1, 0.1, size=(n, n))
    X = np.ones((n, n))
    Y = A * X
    return Y, A
