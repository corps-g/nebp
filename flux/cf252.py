import numpy as np
import scipy as sp
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

    # calculate activity
    activity = 1

    # read in bin structure
    eb = energy_groups('scale252')

    # discretize the spectrum
    data = np.zeros(len(eb))

    #
    for i in range(len(eb[1:])):

        #
        data[i + 1] = sp.integrate.quad(watt_distribution, eb[i], eb[i + 1], args=(a, b))[0] / (eb[i + 1] - eb[i])

    # make sure scaled to one
    data = data / np.sum(data)

    # scale by number of neutrons per fisison
    data *= nu_bar

    # scale by activity
    data *= activity

    return data
