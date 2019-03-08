import re
import numpy as np
import matplotlib.pyplot as plt
from energy_groups import energy_groups
import sys
sys.path.insert(0, '../')
import paths
from spectrum import Spectrum
from nebp_flux import nebp_flux



def grab_tally(filename):
    """Produces a dictionary of all of the tally data from one of the
    responses used in the analysis."""

    # open the file
    with open('mcnp/' + filename) as F:
        output = F.read()

    tallies_txt = output.split('1tally')[1:-3]

    tally = {}

    # loop through each tally section
    for tally_txt in tallies_txt:

        # grab tally number
        tally_number = int(tally_txt.split()[0])

        # create dict space
        tally[tally_number] = np.zeros((253, 2))

        # use regular expression to grab all data
        pattern = re.compile(r'\d.\d\d\d\d\dE[+-]\d\d \d.\d\d\d\d')
        results = re.findall(pattern, tally_txt)[:252]

        for i, result in enumerate(results):

            result = result.split()
            val, err = float(result[0]), float(result[0]) * float(result[1])

            tally[tally_number][i + 1] = np.array([val, err])

    return tally


def response_data():
    """This function consolidates ALL of the response data in this repo."""

    # create a dict that will house all of the data
    response_data = {}

    # get energy groups
    erg_struct = energy_groups('scale252')
    cos_struct = np.array([0.000000e+00, 1.736482e-01, 3.420201e-01, 5.000000e-01, 6.427876e-01,
                           7.660444e-01, 8.660254e-01, 9.396926e-01, 9.848078e-01, 9.961947e-01,
                           9.993908e-01, 9.998477e-01, 9.999619e-01, 1.000000e+00])

    # this is the energy dependent flux spectrum for the first distribution
    erg_flux_spectrum = nebp_flux('n', 'erg', erg_struct, cos_struct, 1)

    # the gold foil tube ------------------------------------------------------
    gold_tallys = grab_tally('ft_au.out')

    # loop through the gold tally
    for name, tally in gold_tallys.items():

        # grab integral responses
        if 134 <= name:

            # convert to name
            new_name = 'ft_au' + str(((name - 4) // 10) - 13)

            response_data[new_name] = Spectrum(erg_flux_spectrum.edges, tally[:, 0], tally[:, 1])

    # the indium foil tube ------------------------------------------------------
    indium_tallys = grab_tally('ft_in.out')

    # loop through the indium tally
    for name, tally in indium_tallys.items():

        # grab integral responses
        if 134 <= name:

            # convert to name
            new_name = 'ft_in' + str(((name - 4) // 10) - 13)

            response_data[new_name] = Spectrum(erg_flux_spectrum.edges, tally[:, 0], tally[:, 1])

    # the bonner spheres ------------------------------------------------------
    for sphere_size in [0, 2, 3, 5, 8, 10, 12]:
        bs_tallys = grab_tally('bs{}.out'.format(str(sphere_size)))

        # loop through the bs tally
        for name, tally in bs_tallys.items():

            # grab integral responses
            if name == 124:

                # convert to name
                new_name = 'bs{}'.format(str(sphere_size)) + str(((name - 4) // 10) - 13)

                response_data[new_name] = Spectrum(erg_flux_spectrum.edges, tally[:, 0], tally[:, 1])

    return response_data


def plot_response_data():
    """Pretty straight-forward."""

    # get the data
    responses = response_data()

    # plot response functions -------------------------------------------------
    fig = plt.figure(0, figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # establish colormap
    color = plt.cm.rainbow(np.linspace(0, 1, len(responses)))

    # loop through integral responses
    for i, item in enumerate(responses.items()):

        # parse out name and response
        name, response = item

        # plot the integral response
        ax.plot(*response.plot('plot', 'int'), label=name, color=color[i], lw=0.5)
        #ax.errorbar(*response.plot('errorbar', 'int'), color=color[i], ls='None', lw=0.5)

    # add legend and save
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True,
                    framealpha=1.0, shadow=True, edgecolor='k', facecolor='white')
    fig.savefig('plot/ft_au.png', dpi=300, bbox_extra_artists=(leg,), bbox_inches='tight')
    fig.clear()

    return


if __name__ == '__main__':
    rd = plot_response_data()
