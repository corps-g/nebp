import sys
sys.path.insert(0, '../')
import paths
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import cm
from process_activities import Au_Foil_Data
from theoretical_activities import Au_Foil_Theoretical
from bss_calibration import BSS_Calibration
from bss_in_beam import BSS_Data
from unfold_nebp import Unfold_NEBP
from response import response_data
from spectrum import Spectrum
import seaborn


class Plotting_Tool():

    """An object that contains all of the plotting parameters and values needed
    for the nebp data."""

    def __init__(self):
        """Initializes the object with some default parameters set."""
        self.set_au_parameters()

    def set_au_parameters(self):
        """This includes all of the gold foils used in the experiment and the
        color scheme of those foils."""

        # list of rfs uses
        self.au_rf_names = []
        self.au_fancy_names = []
        for i in range(9):
            self.au_rf_names.append('ft_au{}'.format(i))
            self.au_fancy_names.append('Au {}'.format(i))

        # set the rf colors to be used
        self.au_rf_colors = seaborn.color_palette('hls', 9)

        return

    def set_poster_defaults(self):
        """Utility that sets parameters ideal for readability on a poster."""

        rc('font', **{'family': 'serif'})
        rcParams['xtick.direction'] = 'out'
        rcParams['ytick.direction'] = 'out'
        rcParams['xtick.labelsize'] = 24
        rcParams['ytick.labelsize'] = 24
        rcParams['lines.linewidth'] = 1.85
        rcParams['axes.labelsize'] = 24
        rcParams['legend.fontsize'] = 12
        rcParams.update({'figure.autolayout': True})
        return


def plotting_environment(fig_num, xlabel, ylabel, xscale='linear', yscale='linear',
                         xticks=None, xticklabels=None, figsize=(10, 6)):
    """Sets up a plotting environment using matplotlib."""

    # create the figure
    fig = plt.figure(fig_num, figsize=figsize)

    # add a single subplot
    ax = fig.add_subplot(111)

    # label the axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # set the scale of the axes
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    # set the xticks and their labels
    if xticks:
        ax.set_xticks(xticks)
    if xticklabels:
        ax.set_xticklabels(xticklabels)

    # return both the figure and the axes
    return fig, ax


def plot_activities():
    """Plots a comparison of the activies found """

    # plotting parameters
    tool = Plotting_Tool()

    # set values to poster defaults
    tool.set_poster_defaults()

    # load in the two datasets
    experimental = Au_Foil_Data()
    theoretical = Au_Foil_Theoretical(experimental)

    # setup plotting environment
    fig, ax = plotting_environment(14, 'Foil Position $In$', r'Activity per Atom $\frac{Bq}{atom}$', yscale='log',
                                   xticks=range(experimental.n), xticklabels=range(experimental.n))

    # plot the theoretical data
    style = {'color': 'blue', 'marker': 'o', 'markerfacecolor': 'None',
             'markeredgecolor': 'blue', 'linestyle': 'None', 'label': 'Theoretical',
             'mew': 1.2, 'ms': 10}
    ax.plot(range(experimental.n), theoretical.a_sat_atom, **style)

    # plot the experimental data
    style = {'color': 'red', 'marker': 'x', 'markerfacecolor': 'None',
             'markeredgecolor': 'red', 'linestyle': 'None', 'label': 'Experimental',
             'mew': 1.2, 'ms': 10}
    ax.plot(range(experimental.n), experimental.a_sat_atom, **style)

    # add legend and save
    ax.legend()
    plt.savefig('plot/compare_activities.png')

    # clear the figure
    fig.clear()

    return


def plot_au_rfs_and_unfolded():
    """This plot superimposes the spectra unfolded with the gold on top of the au response functions."""

    # plotting parameters
    tool = Plotting_Tool()

    # set values to poster defaults
    tool.set_poster_defaults()

    # get the data
    responses = response_data()

    # plotting environment
    fig, ax = plotting_environment(7, 'Energy $MeV$', 'Response Function $cm^2$', xscale='log', yscale='log', figsize=(12, 8))

    # set up the twin axis
    axt = ax.twinx()
    axt.set_xscale('log')
    axt.set_yscale('log')
    axt.set_ylabel(r'$\Phi$ ($cm^{-2}s^{-1}$)')
    axt.set_ylim(1E-3, 1E11)

    # set up axes lims
    ax.set_xlim(1E-11, 20)
    ax.set_ylim(5E-28, 1E-21)
    ax.spines['top'].set_visible(False)
    axt.spines['top'].set_visible(False)

    # loop through integral responses
    for i, name in enumerate(tool.au_rf_names):

        # parse out name and response
        response = responses[name]

        # plot the integral response
        ax.plot(*response.plot('plot', 'int'), label=tool.au_fancy_names[i], color=tool.au_rf_colors[i], lw=1.2)
        ax.errorbar(*response.plot('errorbar', 'int'), color=tool.au_rf_colors[i], ls='None', lw=1.2)

    # get nebp data
    unfolded_data = Unfold_NEBP()

    # convert to spectrum object
    unfolded_spectrum_maxed = Spectrum(unfolded_data.eb, unfolded_data.sol_maxed, 0)

    # plot the unfolded data
    axt.plot(*unfolded_spectrum_maxed.plot('plot', 'diff'), color='k', label='MAXED', lw=1.2)
    axt.text(1.1E-7, 1.1E9, r'$\Phi$', fontsize=24)

    #
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=9, fancybox=True,
                    framealpha=1.0, shadow=True, edgecolor='k', facecolor='white')
    fig.savefig('plot/rfs_and_unfolded.png', dpi=300, bbox_extra_artists=(leg,), bbox_inches='tight')
    fig.clear()
    return


def plot_all():
    """A utility that calls every plotting function in this file."""

    plot_activities()
    #plot_au_rfs_and_unfolded()

    return


if __name__ == '__main__':
    plot_all()
