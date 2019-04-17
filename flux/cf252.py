import numpy as np
import scipy as sp
from scipy.constants import N_A
import sys
sys.path.insert(0, '../')
import paths
from group_structures import energy_groups


def watt_distribution(e, a, b):
    """Docstring."""

    # constant used in the distribution
    C = np.sqrt(np.pi * (b / (4 * a))) * (np.exp(b / (4 * a)) / a)

    # the function
    return C * np.exp(-a * e) * np.sinh(np.sqrt(b * e))


def cf252_source():
    """Docstring."""

    # californium data
    nu_bar = 3.757
    a = 0.847458
    b = 1.03419

    # source info
    # mass in grams
    m0 = 9.24E-6

    # time from assay date (s)
    t = 799286329

    # molar mass of californium 252
    M = 252.081626

    # halflife (s)
    halflife = 9.61E2 * 24 * 3600

    # calc decay constant
    decay_constant = np.log(2) / halflife

    # calc number of sample atoms
    N = (m0 * N_A) / M

    # calc initial activity
    A0 = N * decay_constant

    # decay to present day
    activity = A0 * np.exp(-decay_constant * t)

    # read in bin structure
    eb = energy_groups('scale252')

    # discretize the spectrum
    data = np.zeros(len(eb))

    #
    for i in range(len(eb[1:])):

        #
        data[i + 1] = sp.integrate.quad(watt_distribution, eb[i], eb[i + 1], args=(a, b))[0]

    # make sure scaled to one
    data = data / np.sum(data)

    # scale by number of neutrons per fisison and by activity
    data *= nu_bar * activity

    return data
