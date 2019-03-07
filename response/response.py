import re
import numpy as np
import matplotlib.pyplot as plt
from energy_groups import energy_groups
import sys
sys.path.insert(0, '../')
import paths
from spectrum import Spectrum
from nebp_flux import nebp_flux


class Response(object):

    """This houses the data on a particular detector - both the
    response function, as well as the integral response."""

    def __init__(self, name, rf, ir):
        """Initializes the object with a response function and
        integral response."""
        self.name = name
        self.rf = rf
        self.ir = ir
        return


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
        results = re.findall(pattern, tally_txt)[:253]

        for i, result in enumerate(results):

            result = result.split()
            val, err = float(result[0]), float(result[0]) * float(result[1])

            tally[tally_number][i] = np.array([val, err])

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
    gold_tallys = grab_tally('ft_au.inpo')

    # loop through the gold tally
    for name, tally in gold_tallys.items():

        # skip the bonner stuff
        if name in (114, 124):
            continue

        # grab integral responses
        if 134 <= name <= 244:

            # convert to name
            new_name = 'ft_au' + str(((name - 4) // 10) - 13)

            response_data[new_name] = Spectrum(erg_flux_spectrum.edges, tally[:, 0], tally[:, 1])

        # grab integral responses
        elif 244 < name:

            # skip for now
            continue

    return response_data


def plot_response_data():
    """Pretty straight-forward."""

    # get the data
    responses = response_data()

    # plot integral responses -------------------------------------------------
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # loop through integral responses
    for name, response in responses.items():

        # plot the integral response
        ax.plot(*response.plot('plot', 'diff'), label=name)

    # add legend
    ax.legend()

    return


if __name__ == '__main__':
    rd = plot_response_data()
