import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from unfold_nebp import Unfold_NEBP
from spectrum import Spectrum


def plot_unfolded_spectra():
    """Docstring."""

    # get data
    unfolded_data = Unfold_NEBP()

    # convert to spectrum object
    unfolded_spectrum_gravel = Spectrum(unfolded_data.eb, unfolded_data.sol_gravel, 0)
    unfolded_spectrum_maxed = Spectrum(unfolded_data.eb, unfolded_data.sol_maxed, 0)

    #
    fig = plt.figure(0, figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energy $MeV$')
    ax.set_ylabel(r'$\Phi$ $\frac{1}{cm^2 s MeV}$')

    #
    ax.plot(*unfolded_spectrum_gravel.plot('plot', 'diff'), color='grey', label='Gravel')
    ax.plot(*unfolded_spectrum_maxed.plot('plot', 'diff'), color='indigo', label='MAXED')

    #
    ax.legend()
    fig.savefig('plot/nebp.png', dpi=300)

    # -------------------------------------------------------------------------
    # evolution plot

    # setup plotting environment
    fig = plt.figure(1, figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energy $MeV$')
    ax.set_ylabel(r'$\Phi$ $\frac{1}{cm^2 s MeV}$')

    # create some colors
    colors = cm.magma(np.linspace(0, 1, len(unfolded_data.evolution)))[::-1]

    for i, iteration in enumerate(unfolded_data.evolution):

        # convert to spectrum object
        unfolded_spectrum = Spectrum(unfolded_data.eb, iteration, 0)

        # plot the data
        ax.plot(*unfolded_spectrum.plot('plot', 'diff'), color=colors[i], lw=0.5)

    # save it
    fig.savefig('plot/nebp_gravel_evolution.png', dpi=300)

    return


if __name__ == '__main__':
    plot_unfolded_spectra()
