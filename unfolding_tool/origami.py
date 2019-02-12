import numpy as np
from numpy.linalg import norm
from scipy.optimize import basinhopping


def MAXED(N, sigma2, R, f_def, params):
    """The MAXED unfolding algorithm."""

    # pull out algorithm-specific parameters
    Omega = params['Omega']

    # create the function that we will maximize, Z
    def Z(lam, N, sigma2, R, f_def, Omega):
        """A function, the maximization of which is equivalent to the
        maximization of """

        A = - np.sum(f_def * np.exp(- np.sum((lam * R.T).T, axis=0)))
        B = - (Omega * np.sum(lam**2 * sigma2))**(0.5)
        C = - np.sum(N * lam)

        # negate because it's a minimization
        return - (A + B + C)

    # create a lambda
    lam = np.ones(len(N))

    # apply the simulated annealing to the Z
    mk = {'args': (N, sigma2, R, f_def, Omega)}
    lam = basinhopping(Z, lam, minimizer_kwargs=mk).x

    # back out the spectrum values from the lam
    return f_def * np.exp(-np.sum((lam * R.T).T, axis=0))


def Gravel(N, sigma2, R, f_def, params):
    """The modified SandII algorithm used in the Gravel code."""

    # pull out algorithm-specific parameters
    max_iter = params['max_iter']
    tol = params['tol']

    # initalize
    iteration = 0
    f = f_def
    N0 = np.sum(R * f, axis=1)

    # begin iteration
    while iteration < max_iter and norm(N0 - N, ord=2) > tol:

        # break down equations into simpler terms
        a = (R * f)
        b = np.sum(R * f, axis=1)
        c = (N**2 / sigma2)
        log_term = np.log(N / b)

        # compute the uper and lower portion of the exponential
        top = np.sum((((a.T / b) * c) * log_term).T, axis=0)
        bot = np.sum(((a.T / b) * c).T, axis=0)

        # compute the coefficient array
        coef = np.exp(top / bot)

        # update the new f
        f = f * coef

        # update f
        N0 = np.sum(R * f, axis=1)
        iteration += 1

    return f


def STAYSL(N, sigma2, R, f_def, params):
    """The least-squares algorithm implemented in the STAY'SL unfolding code."""

    # pull out algorithm-specific parameters and values
    f_def_err = params['f_def_err']
    R_err = params['R_err']

    # if not calculated, the relative covariance matrices are computed
    M_No = np.diag(sigma2 / N)
    M_f_def = np.diag(f_def_err / f_def)
    M_R = np.diag(R_err / R)

    # compute covariance matrix N_Ao
    N_No = N.dot(M_No.dot(N))

    # compute 'c' matrix
    c = f_def * R

    # compute u matrix
    u = c * 0
    for i in range(len(N)):
        for j in range(len(f_def)):
            u[i, j] = np.sum(M_f_def[j] * c[i])

    # compute A vector
    A = np.sum(c, axis=1)

    # compute covariance matrix N_f_A
    N_f_A = c * 0
    for i in range(len(N)):
        for j in range(len(f_def)):
            N_f_A[i, j] = np.sum(c[i] * u[j])

    # compute covariance matrix N_R_A
    pass


def unfold(N, sigma2, R, f_def, method='MAXED', params={}):
    """A utility that deconvolutes (unfolds) neutron spectral data given
    typical inputs and a selection of unfolding algorithm."""

    # check input
    available_methods = ('MAXED', 'Gravel')
    assert method in available_methods, 'method must by literal in {}'.format(available_methods)
    assert len(N) == len(sigma2), 'N and sigma2 must be the same length.'
    assert R.shape == (len(N), len(f_def)), 'Shape of R must be consistent with other inputs.'

    # unfold with MAXED
    if method == 'MAXED':
        return MAXED(N, sigma2, R, f_def, params)

    # unfold with Gravel
    elif method == 'Gravel':
        return Gravel(N, sigma2, R, f_def, params)

    # unfold with STAY'SL
    elif method == 'STAYSL':
        return STAYSL(N, sigma2, R, f_def, params)

    return
