import matplotlib.pyplot as plt
from unfold_nebp import Unfold_NEBP
from spectrum import Spectrum


def plot_unfolded_spectra():
    """Docstring."""

    # get data
    unfolded_data = Unfold_NEBP()

    # convert to spectrum object
    unfolded_spectrum = Spectrum(unfolded_data.eb, unfolded_data.sol, 0)

    #
    fig = plt.figure(0, figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energy $MeV$')
    ax.set_ylabel(r'$\Phi$ $\frac{1}{cm^2 s MeV}$')

    #
    ax.plot(*unfolded_spectrum.plot('plot', 'diff'), color='indigo')

    #
    fig.savefig('plot/nebp_gravel.png', dpi=300)

    return


if __name__ == '__main__':
    plot_unfolded_spectra()
