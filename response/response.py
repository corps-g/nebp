import re
import numpy as np
from scipy.constants import N_A
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')
import paths
from spectrum import Spectrum
from group_structures import energy_groups, cosine_groups, radial_groups
from nebp_flux import extract_mcnp


def grab_tally(name, scaling_factor):
    """Produces a dictionary of all of the tally data from one of the
    responses used in the analysis."""

    # first, grab the data
    flux_data = extract_mcnp('n', 1)

    # split into values and error
    flux = flux_data[:, :, :, 0]

    #
    regional_pdf = np.sum(flux, axis=(1, 2))[:-1]
    regional_pdf /= np.sum(regional_pdf)

    #
    tally_regions = radial_groups('nebp')

    #
    tally = {}

    for i in range(len(tally_regions) - 1):

        # name
        filename = name + '{}.out'.format(i)

        # open the file
        with open(paths.main_path + '/response/mcnp/' + filename) as F:
            output = F.read()

        tallies_txt = output.split('1tally')[1:-3]

        # loop through each tally section
        for tally_txt in tallies_txt:

            # grab tally number
            tally_number = int(tally_txt.split()[0])

            if tally_number not in tally:
                # create dict space
                tally[tally_number] = np.zeros((253, 2))

            # use regular expression to grab all data
            pattern = re.compile(r'\d.\d\d\d\d\dE[+-]\d\d \d.\d\d\d\d')
            results = re.findall(pattern, tally_txt)[:252]

            for j, result in enumerate(results):

                result = result.split()
                val, err = float(result[0]), float(result[0]) * float(result[1])

                #
                val = val * regional_pdf[i] * scaling_factor
                err = (err * regional_pdf[i] * scaling_factor)**2

                tally[tally_number][j + 1] += np.array([val, err])

    # loop through each tally section
    for tally_txt in tallies_txt:

        # grab tally number
        tally_number = int(tally_txt.split()[0])
        tally[tally_number][:, 1] = np.sqrt(tally[tally_number][:, 1])

    return tally


def grab_pbs_tally(name, scaling_factor):
    """Docstring."""

    #
    tally = {}

    # name
    filename = name + '.out'

    # open the file
    with open(paths.main_path + '/response/mcnp/' + filename) as F:
        output = F.read()

    #
    #output = output.split('1tally')[1]

    tally[name] = np.zeros((253, 2))

    # use regular expression to grab all data
    pattern = re.compile(r'                 \d.\d\d\d\d\dE[+-]\d\d \d.\d\d\d\d')
    results = re.findall(pattern, output)[1:253]

    for j, result in enumerate(results):

        result = result.split()
        val, err = float(result[0]), float(result[0]) * float(result[1])

        val = val * scaling_factor
        err = err * scaling_factor

        tally[name][j + 1] += np.array([val, err])

    return tally


def response_data():
    """This function consolidates ALL of the response data in this repo."""

    # create a dict that will house all of the data
    response_data = {}

    # get energy groups
    erg_struct = energy_groups('scale252')

    # the gold foil tube ------------------------------------------------------
    scaling_factor = 1E-24 * 252
    gold_tallys = grab_tally('ft_au', scaling_factor)

    # loop through the gold tally
    for name, tally in gold_tallys.items():

        # grab integral responses
        if 134 <= name:

            # convert to name
            new_name = 'ft_au' + str(((name - 4) // 10) - 13)

            response_data[new_name] = Spectrum(erg_struct, tally[1:, 0], tally[1:, 1])

    # the indium foil tube ------------------------------------------------------
    scaling_factor = (7.31 * N_A * 1E-24 * 252) / (115 * 20)
    indium_tallys = grab_tally('ft_in', scaling_factor)

    # loop through the indium tally
    for name, tally in indium_tallys.items():

        # grab integral responses
        if 134 <= name:

            # convert to name
            new_name = 'ft_in' + str(((name - 4) // 10) - 13)

            response_data[new_name] = Spectrum(erg_struct, tally[1:, 0], tally[1:, 1])

    # the bonner spheres ------------------------------------------------------
    V = 5.02655E-02
    rho = 3.84
    M = 133.85
    scaling_factor = (rho * N_A * 1E-24 * V * 252) / M
    for sphere_size in [0, 2, 3, 5, 8, 10, 12]:
        bs_tallys = grab_tally('bs{}_'.format(str(sphere_size)), scaling_factor)

        # loop through the bs tally
        for name, tally in bs_tallys.items():

            # grab integral responses
            if name == 124:

                # convert to name
                new_name = 'bs{}'.format(str(sphere_size)) + str(((name - 4) // 10) - 13)

                response_data[new_name] = Spectrum(erg_struct, tally[1:, 0], tally[1:, 1])

    # the point bonner spheres ------------------------------------------------
    scaling_factor = (rho * N_A * 1E-24 * V * 252) / M
    for sphere_size in [0, 2, 3, 5, 8, 10, 12]:
        bs_tallys = grab_pbs_tally('pbs{}'.format(str(sphere_size)), scaling_factor)

        # loop through the bs tally
        for name, tally in bs_tallys.items():

            # convert to name
            new_name = 'pbs{}'.format(str(sphere_size))

            response_data[new_name] = Spectrum(erg_struct, tally[1:, 0], tally[1:, 1])

    return response_data
