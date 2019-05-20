import sys
sys.path.insert(0, '../')
import paths
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import cm
from group_structures import energy_groups
from fission import extract_fission_data
from process_activities import Au_Foil_Data
from theoretical_activities import Au_Foil_Theoretical
from bss_calibration import BSS_Calibration
from bss_in_beam import BSS_Data
from unfold_nebp import unfold_myriad
from response import response_data
from spectrum import Spectrum
import seaborn


class Plotting_Tool():

    """An object that contains all of the plotting parameters and values needed
    for the nebp data."""

    def __init__(self):
        """Initializes the object with some default parameters set."""
        self.set_cmap()
        self.set_au_parameters()

    def set_cmap(self):
        """This sets the colormaps that will be used throughout the thesis."""

        # set a colormap
        self.cmap = 'inferno'

        return

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
                         xticks=None, xticklabels=None, yticks=None, yticklabels=None,
                         figsize=(10, 6)):
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

    # set the ticks and their labels
    if xticks:
        ax.set_xticks(xticks)
    if xticklabels:
        ax.set_xticklabels(xticklabels)
    if yticks:
        ax.set_yticks(yticks)
    if yticklabels:
        ax.set_yticklabels(yticklabels)

    # return both the figure and the axes
    return fig, ax


def mirror_element(rr_density):
    """Given a fission rate density map, creates a symmetric mirror with an empty
    middle channel to more closely resemble an actual element."""

    # grab the shape of the matrix
    n_ax, n_rad = rr_density.shape

    # create an new, appropriately sized matrix
    new_map = np.zeros((n_ax, n_rad * 2 + 1))

    # flip the existing data onto the 'left' side of the new matrix
    new_map[:, 0:n_rad] = rr_density[:, ::-1]

    # fill the 'right' side of the new matrix with the existing data
    new_map[:, n_rad + 1:] = rr_density[:, ]

    # return this new matrix, noting the 'hole' of zeros in the center column
    return new_map


def plot_fission_rates():
    """A utility to visualize the fission data from the ksu-triga core."""

    # plotting parameters
    tool = Plotting_Tool()

    # set values to poster defaults
    tool.set_poster_defaults()

    # grab data
    core = extract_fission_data()

    # -------------------------------------------------------------------------
    #                                                            single element
    # plot a heatmap of the in-element fission rate densities
    # grab an element
    element = core.fuel['201']

    # initalize plotting environment
    fig, ax = plotting_environment(0, '$r$', '$z$', figsize=(3, 10),
                                   xticks=[10], xticklabels=[10], yticks=[20], yticklabels=[20])

    # plot the reaction rate density map
    ext = [-element.rad_dims[1], element.rad_dims[1], *element.ax_dims]
    ax.imshow(mirror_element(element.rr_density), vmin=core.min_rr_density,
              vmax=core.max_rr_density, extent=ext, cmap=tool.cmap)

    # save the elements plot
    fig.savefig('plot/rr_dist_B1.png', dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    #                                                           axial/elemental
    # loop for separate plot for each ring
    for i, ring in enumerate(core.rings):

        # set up plotting environment
        fig, ax = plotting_environment(i + 10, 'Fission Rate', '$z$ (cm)', figsize=(6, 10),
                                       xticks=[10], xticklabels=[10])

        # create a color object
        colors = cm.inferno(np.linspace(0, 1, len(core.ids[i])))

        # loop through each element in the ring
        for j, element_id in enumerate(core.ids[i]):

            # pull the particular element
            element = core.fuel[element_id]

            # plot the axial rr density from that element
            ax.plot(element.rr_density_ax, element.ax_mps, color=colors[j], label=core.names[i][j])

        # create a fancy legend
        leg = ax.legend(loc='center right', bbox_to_anchor=(1.4, 0.5), ncol=1,
                        fancybox=True, framealpha=1.0, shadow=True,
                        edgecolor='k', facecolor='white')

        # get rid of right and top lines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # save and close the figure
        fig.savefig('plot/axial_rr_density_{}.png'.format(ring), dpi=300,
                    bbox_extra_artists=(leg,), bbox_inches='tight')
        plt.close(fig)

    # -------------------------------------------------------------------------
    #                                                          radial/elemental
    # loop for separate plot for each ring
    for i, ring in enumerate(core.rings):

        # set up plotting environment
        fig, ax = plotting_environment(i + 20, 'r (cm)', 'Fission Rate', figsize=(8, 6), yticks=[10], yticklabels=[10])

        # create a color object
        colors = cm.inferno(np.linspace(0, 1, len(core.ids[i])))

        # loop through each element in the ring
        for j, element_id in enumerate(core.ids[i]):

            # pull the particular element
            element = core.fuel[element_id]

            # plot the axial rr density from that element
            ax.plot(element.rad_mps, element.rr_density_rad, color=colors[j], label=element_id)

        # create a fancy legend
        leg = ax.legend(loc='center right', bbox_to_anchor=(
                        1.2, 0.5), ncol=1, fancybox=True, framealpha=1.0,
                        shadow=True, edgecolor='k', facecolor='white')

        # get rid of right and top lines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # save and close the figure
        fig.savefig('plot/radial_rr_density_{}.png'.format(ring), dpi=300,
                    bbox_extra_artists=(leg,), bbox_inches='tight')
        plt.close(fig)

    # -------------------------------------------------------------------------
    #                                                       total fission rates
    # setup plotting environment
    fig, ax = plotting_environment(1, r'$\phi$ (deg)', 'Fission Rate', figsize=(10, 8),
                                   yticks=[10], yticklabels=[10])

    # set up a color pallete
    colors = seaborn.color_palette('hls', len(core.rings))

    # loop through data and plot
    for i in range(len(core.rr_totals)):

        # plot the ring's total reaction rate
        ax.plot(core.azis[i], core.rr_totals[i], marker='o', color=colors[i], label=core.rings[i])

    # create a legend
    leg = ax.legend(loc='upper center', bbox_to_anchor=(
                    0.5, 1.03), ncol=5, fancybox=True, framealpha=1.0,
                    shadow=True, edgecolor='k', facecolor='white')

    # save the figure and close
    fig.savefig('plot/totals_azi.png')
    plt.close(fig)

    return
    # -------------------------------------------------------------------------
    #                                                                    others
    # plot the axial and radial distributions for each ring
    # loop through each ring
    rr_dens = []
    rr_totals = []
    rr_densax = []
    rr_abstotals = []
    rings = ['A', 'B', 'C', 'D', 'E', 'F']
    for i in range(2, 7):

        # make a list of each element in that ring
        ring_elements = []
        for element_id in core.fuel.keys():
            if str(i) == element_id[0]:
                ring_elements.append(element_id)

        color = cm.rainbow(np.linspace(0, 1, len(ring_elements)))

        rr_den = 0
        rr_total = 0
        rr_denax = 0

        ring = []

        # now loop through the elements in a given ring
        for j, element_id in enumerate(ring_elements):

            rr_den += element.rr_density_rad / len(ring_elements)

            rr_denax += element.rr_density_ax / len(ring_elements)

            rr_total += element.total_fission_rate / len(ring_elements)

            ring.append(element.total_fission_rate)

            # grab the individual element
            element = core.fuel[element_id]


        rr_abstotals.append(ring)
        rr_totals.append(rr_total)
        rr_dens.append(rr_den)
        rr_densax.append(rr_denax)

    rr_abstotals[1].insert(6, 0)
    rr_abstotals[2].insert(3, 0)
    rr_abstotals[2].insert(15, 0)
    rr_abstotals[3].insert(0, 0)
    rr_abstotals[4].insert(9, 0)

    

    fig2 = plt.figure(100, figsize=(10, 10))
    ax2 = fig2.add_subplot(111)
    ax2.plot(np.linspace(0, 1, len(rr_totals)), rr_totals)
    fig2.savefig('plot/total_fr')
    plt.close(fig2)

    fig3 = plt.figure(100, figsize=(10, 10))
    ax3 = fig3.add_subplot(111)
    for i in range(len(rr_dens)):
        ax3.plot(element.rad_mps, rr_dens[i], label=rings[i+1])
    leg3 = ax3.legend(loc='center right', bbox_to_anchor=(1.4, 0.5), ncol=1,
                      fancybox=True, framealpha=1.0, shadow=True,
                      edgecolor='k', facecolor='white')
    fig3.savefig('plot/rr_dens', dpi=300,
                 bbox_extra_artists=(leg3,), bbox_inches='tight')
    plt.close(fig3)

    fig4 = plt.figure(100, figsize=(10, 10))
    ax4 = fig4.add_subplot(111)
    for i in range(len(rr_densax)):
        ax4.plot(rr_densax[i], element.ax_mps, label=rings[i+1])
    leg4 = ax4.legend(loc='center right', bbox_to_anchor=(1.4, 0.5), ncol=1,
                      fancybox=True, framealpha=1.0, shadow=True,
                      edgecolor='k', facecolor='white')
    fig4.savefig('plot/rr_densax', dpi=300,
                 bbox_extra_artists=(leg4,), bbox_inches='tight')
    plt.close(fig4)
    return


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


def plot_unfolded():
    """This plot superimposes the spectra unfolded with the gold on top of the au response functions."""

    # plotting parameters
    tool = Plotting_Tool()

    # set values to poster defaults
    tool.set_poster_defaults()

    # plotting environment
    fig, ax = plotting_environment(7, 'Energy $MeV$', r'$\Phi$ ($cm^{-2}s^{-1}$)', xscale='log', yscale='log', figsize=(12, 8))

    # set up axes lims
    ax.set_xlim(1E-11, 20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # get nebp data
    unfolded_data = unfold_myriad()

    # convert to spectrum object
    ds = Spectrum(unfolded_data['eb'], unfolded_data['ds'], 0)
    unfolded = Spectrum(unfolded_data['eb'], unfolded_data['all'], 0)

    # plot the unfolded data
    ax.plot(*ds.plot('plot', 'diff'), color='k', label='Default', lw=1.2)
    ax.plot(*unfolded.plot('plot', 'diff'), color='g', label='Gravel', lw=1.2)

    # create a legend and save
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=9, fancybox=True,
                    framealpha=1.0, shadow=True, edgecolor='k', facecolor='white')
    fig.savefig('plot/unfolded_{}.png'.format('all'), dpi=300, bbox_extra_artists=(leg,), bbox_inches='tight')
    fig.clear()
    return


def plot_all():
    """A utility that calls every plotting function in this file."""

    #plot_fission_rates()
    plot_activities()
    #plot_au_rfs_and_unfolded()
    plot_unfolded()

    return


if __name__ == '__main__':
    plot_all()
