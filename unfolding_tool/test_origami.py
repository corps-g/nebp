import numpy as np
import matplotlib.pyplot as plt
import origami as ori
import sys
sys.path.insert(0, '../')
import paths
from spectrum import Spectrum


def test_values():
    """Produces a test values for unfolding."""

    # produce a true spectrum
    f_true = np.array([5, 6, 4, 3, 3, 2, 1])

    # produce response functions
    R = np.zeros((3, len(f_true)))
    R[0] = np.array([9, 8, 6, 4, 2, 2, 1])
    R[1] = np.array([1, 2, 3, 5, 3, 2, 1])
    R[2] = np.array([0, 0, 0, 0, 3, 4, 3])

    # produce detector responses
    N = np.sum(R * f_true, axis=1)

    # produce errors
    sigma2 = np.ones(len(N)) * 0.1

    # produce a default spectrum
    f_def = np.array([4.5, 6.5, 3.5, 3, 2.5, 1.5, 0.5])

    # produces some trivial bin edges
    bin_edges = np.linspace(0, 7, 8)

    # return the appropriate values
    return f_true, f_def, N, sigma2, R, bin_edges


def test_origami():
    """Uses the test values to test different methods in origami."""

    # create a container for solutions
    solutions = {}

    # load test values
    f_true, f_def, N, sigma2, R, edges = test_values()
    solutions['True'] = Spectrum(edges, f_true, 0)
    solutions['Default'] = Spectrum(edges, f_def, 0)

    # test MAXED
    f_MAXED = ori.unfold(N, sigma2, R, f_def, method='MAXED', params={'Omega': 3})
    solutions['MAXED'] = Spectrum(edges, f_MAXED, 0)

    # test Gravel
    f_Gravel = ori.unfold(N, sigma2, R, f_def, method='Gravel', params={'max_iter': 1000, 'tol': 1E-4})
    solutions['Gravel'] = Spectrum(edges, f_Gravel, 0)

    # compare results visually
    fig = plt.figure(0)
    ax = fig.add_subplot(111)

    for name, spec in solutions.items():
        ax.plot(*spec.plot('plot', 'int'), label=name)

    # produce legend
    ax.legend()

    return

if __name__ == '__main__':
    test_origami()
