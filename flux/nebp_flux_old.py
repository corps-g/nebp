import re
import numpy as np
import matplotlib.pyplot as plt
from energy_groups import energy_groups
import sys
sys.path.insert(0, '../')
import paths
from spectrum import Spectrum, Spectrum2D


def extract_mcnp(par):
    """Utility that grabs the flux data from an mcnp output file."""

    # check input
    assert par in ('n', 'g'), 'Incorrect particle type.'

    # for neutrons
    if par == 'n':

        # number of erg and cos groups
        erg_struct_hi = energy_groups('scale252')

        erg_struct_lo = energy_groups('hr6')

        cos_struct = np.array([0.000000e+00, 1.736482e-01, 3.420201e-01, 5.000000e-01, 6.427876e-01,
                               7.660444e-01, 8.660254e-01, 9.396926e-01, 9.848078e-01, 9.961947e-01,
                               9.993908e-01, 9.998477e-01, 9.999619e-01, 1.000000e+00])

        data_shape_hi = (len(cos_struct), len(erg_struct_hi))
        data_shape_lo = (len(cos_struct), len(erg_struct_lo))

        # open file w/ neutron data
        with open(paths.main_path + '/flux/mcnp/ksuna.out') as F:
            mcnp_file = F.read()

        # split hi res tally
        mcnp_tally = mcnp_file.split('1tally       11')[1].split('1tally       21')[0]

        # use regular expression to grab all data
        pattern = re.compile(r'    \d.\d\d\d\dE[+-]\d\d   \d.\d\d\d\d\dE[+-]\d\d \d.\d\d\d\d')
        results = re.findall(pattern, mcnp_tally)

        # create empty array to house data
        data_hi = np.empty((len(results), 2))

        # loop through
        for i, line in enumerate(results):

            # break line at spaces
            line = line.split()

            # put data into numpy array
            data_hi[i] = float(line[1]), float(line[2])

        # get rid of totals and all the zeros
        data_hi = data_hi[len(data_hi) // 2:]

        # convert error to absolute
        data_hi[:, 1] = data_hi[:, 0] * data_hi[:, 1]

        # now extract low res stuff
        # split tally region from file
        mcnp_tally = mcnp_file.split('1tally       21')[1].split('1tally      111')[0]

        # use regular expression to grab all data
        pattern = re.compile(r'    \d.\d\d\d\dE[+-]\d\d   \d.\d\d\d\d\dE[+-]\d\d \d.\d\d\d\d')
        results = re.findall(pattern, mcnp_tally)

        # create empty array to house data
        data_lo = np.empty((len(results), 2))

        # loop through
        for i, line in enumerate(results):

            # break line at spaces
            line = line.split()

            # put data into numpy array
            data_lo[i] = float(line[1]), float(line[2])

        # get rid of the positive component of data
        data_lo = data_lo[len(data_lo) // 2:]

        # convert error to absolute
        data_lo[:, 1] = data_lo[:, 0] * data_lo[:, 1]

        # reshape to follow bin structure
        data_lo = data_lo.reshape((*data_shape_lo, 2))

        # convert rows to pdfs and change error accordingly
        for i in range(data_shape_lo[1]):
            row_mag = np.sum(data_lo[:, i, 0])
            data_lo[:, i, :] = data_lo[:, i, :] / row_mag if row_mag else data_lo[:, i, :] * 0

        # collect index matches for energy groups
        indices = []
        for edge in erg_struct_lo:
            indices.append(np.where(erg_struct_hi == edge)[0][0])
        indices = np.array(indices).astype(int)

        # create final data structure
        data = np.empty((*data_shape_hi, 2))

        # extend pdfs to fill data structure
        for i in range(data_shape_hi[1]):
            data[:, i, :] = data_lo[:, indices.searchsorted(i), :]

        # calculate relative errors by root sum square each of the relative errors
        data[:, :, 1] = np.sqrt((data[:, :, 1] / data[:, :, 0])**2 + (data_hi[:, 1] / data_hi[:, 0])**2)

        # calculate the data
        data[:, :, 0] = data[:, :, 0] * data_hi[:, 0]

        # convert to absolute error
        data[:, :, 1] = data[:, :, 0] * data[:, :, 1]

        data[np.isnan(data)] = 0

        return data, erg_struct_hi, cos_struct

    # gammas not currently implemented
    elif par == 'g':

        # raise an error
        assert False, 'Gamma data is not implemented yet.'

    return


def rebin(array, error, old_struct, new_struct):
    """Utility that rebins integral data into a new structure."""

    # check input
    assert len(array) == len(old_struct), "Length of data doesn't match length of input data structure."
    assert len(array) == len(error), "Length of data doesn't match length of error."

    # create new array
    new_array = np.zeros(len(new_struct))
    new_error = np.zeros(len(new_struct))

    # loop through new structure
    for i in range(len(new_struct)):

        # create rolling window for new structure
        new_window = (-1, new_struct[i]) if not i else (new_struct[i - 1], new_struct[i])

        # initialize array that will contain bin fractions
        fractions = np.zeros(len(old_struct))

        # loop through old structure
        for j in range(len(old_struct)):

            # create rolling window for old structure
            old_window = (-1, old_struct[j]) if not j else (old_struct[j - 1], old_struct[j])

            # update bin fractions
            numerator = np.min([old_window[1], new_window[1]]) - np.max([old_window[0], new_window[0]])
            denomerator = (old_window[1] - old_window[0])
            fractions[j] = numerator / denomerator

            # negative fractions are set to zero
            fractions[j] = 0 if fractions[j] < 0 else fractions[j]

        # sum fractions with old array
        new_array[i] = np.sum(array * fractions)

        # set all non-zero fractions to one for error calc
        fractions = np.array([1 if fractions[k] else 0 for k in range(len(fractions))])
        new_error[i] = np.sqrt(np.sum((error * fractions)**2))

    return new_array, new_error


def nebp_flux(par, flux_type, erg_struct, cos_struct, power):
    """Grabs the flux."""

    # check input
    assert par in ('n', 'g'), 'Incorrect particle type.'
    assert flux_type in ('tot', 'total', 'erg', 'cos', 'cos_erg'), 'Please specify a correct flux type.'
    assert type(erg_struct) in (str, np.ndarray), 'Energy structure must be either a string or a numpy ndarray.'
    assert type(cos_struct) in (str, np.ndarray), 'Angular structure must be either a string or a numpy ndarray.'
    assert type(power) in (int, float), 'Power W(th) must be of type int or float.'

    # check if cos bins in deg or cos theta
    if True in (cos_struct > 1):
        cos_struct = np.cos(cos_struct * (np.pi / 180))

        # change extremly small numbers to zero
        cos_struct[cos_struct < 1e-12] = 0

    # if erg groups given as a name, get the actual bin values
    if type(erg_struct) == str:
        erg_struct = energy_groups(erg_struct)

    # grab data
    data, old_erg_struct, old_cos_struct = extract_mcnp(par)

    # first convert angular distribution
    # start by creating a new array
    new_data = np.zeros((len(cos_struct), data.shape[1], 2))

    # update cosine structure
    for i in range(data.shape[1]):
        new_data[:, i, 0], new_data[:, i, 1] = rebin(data[:, i, 0], data[:, i, 1], old_cos_struct, cos_struct)

    # variable change
    data = new_data

    # then convert energy distribution
    # start by creating a new array
    new_data = np.zeros((data.shape[0], len(erg_struct), 2))

    # update cosine structure
    for i in range(data.shape[0]):
        new_data[i, :, 0], new_data[i, :, 1] = rebin(data[i, :, 0], data[i, :, 1], old_erg_struct, erg_struct)

    # variable change
    data = new_data

    # break apart flux and error at this point
    flux = data[:, :, 0]
    error = data[:, :, 1]

    # scale by power level
    scaling_constant = (2.54 * power) / (200 * 1.60218e-13)
    flux *= scaling_constant
    error *= scaling_constant

    # if erg type
    if flux_type == 'erg':

        # integration is just a sum
        flux = np.sum(flux, axis=0)
        error = np.sqrt(np.sum(error**2, axis=0))

        return Spectrum(erg_struct, flux, error)

    # if cosine type
    elif flux_type == 'cos':

        # integrate across other axis
        flux = np.sum(flux, axis=1)
        error = np.sqrt(np.sum(error**2, axis=1))

        return Spectrum(cos_struct, flux, error, floor=-1)

    # total type
    elif flux_type in ('tot', 'total'):

        # integrate over one axis then the other
        flux = np.sum(flux)
        error = np.sqrt(np.sum(error**2))

        # this option only returns the total value and its error
        return flux, error

    # if anyone really ever wanted this
    elif flux_type == 'cos_erg':

        return Spectrum2D(cos_struct, erg_struct, flux, error, floor=(-1, 1e-11))

    return


def test_rebin():
    """A small utility that tests rebin()."""

    old_data = np.array([6, 6, 6, 6, 6])
    old_error = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    old_bin = np.array([1, 2, 3, 4, 5])
    new_bin = np.array([1, 2.5, 4.5, 5])
    new_data, new_error = rebin(old_data, old_error, old_bin, new_bin)
    plt.step(old_bin, old_data)
    plt.step(new_bin, new_data)
    print(new_data)
    print(new_error)


def test_nebp_flux():
    """A small utility that tests nebp_flux()."""

    # cosine bins
    cos_struct = np.array([90, 10, 5, 0])

    spec = nebp_flux('n', 'erg', 'scale252', cos_struct, 1)
    plt.figure(figsize=(10, 10))
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(*spec.plot('plot', 'diff'), 'k')
    plt.errorbar(*spec.plot('errorbar', 'diff'), ls='None', c='k')

    # cos_erg
    spec = nebp_flux('n', 'cos_erg', 'scale56', cos_struct, 1)

    return


if __name__ == '__main__':
    test_nebp_flux()
