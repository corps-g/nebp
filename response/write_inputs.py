import numpy as np
from template import mcnp_template
import sys
sys.path.insert(0, '../')
import paths
from nebp_flux import nebp_flux
from energy_groups import energy_groups


def card_writer(card, data, elements):
    """This will write multiline cards for SI and SP distributions for mcnp inputs.

    Input Data:
        card - name and number of the card
        data array - a numpy array containing the data you'd like placed in the card.
        Outputs:
            a string that can be copied and pasted into an mcnp input file"""
    s = '{}   '.format(card)
    empty_card = '   ' + ' ' * len(card)
    elements_per_row = elements
    row_counter = 0
    element = '{:6}  ' if data.dtype in ['int32', 'int64'] else '{:14.6e}  '
    for i, d in enumerate(data):
        s += element.format(d)
        row_counter += 1
        if row_counter == elements_per_row and i + 1 != len(data):
            row_counter = 0
            s += '\n{}'.format(empty_card)
    s += '\n'
    return s


def source_writer(erg_struct):
    """Writes the triga nebp mcnp source."""

    # first, match the cosine bin structure of the original mcnp
    cos_struct = np.array([0.000000e+00, 1.736482e-01, 3.420201e-01, 5.000000e-01, 6.427876e-01,
                           7.660444e-01, 8.660254e-01, 9.396926e-01, 9.848078e-01, 9.961947e-01,
                           9.993908e-01, 9.998477e-01, 9.999619e-01, 1.000000e+00])

    # this is the energy dependent flux spectrum for the first distribution
    erg_flux_spectrum = nebp_flux('n', 'erg', erg_struct, cos_struct, 1)

    # add the erg dependent distribution
    source = card_writer('SI2 H', erg_flux_spectrum.edges[1:], 4)

    # mcnp requires the first bin in a distribution be zero
    dist = np.zeros(len(erg_flux_spectrum.int))
    dist[1:] = erg_flux_spectrum.int[1:]
    source += card_writer('SP2 D', dist, 4)

    # add dependent distribution
    flux_spectrum = nebp_flux('n', 'cos_erg', erg_struct, cos_struct, 1)

    # add distribution
    shift = 4
    dist_nums = np.array(range(len(flux_spectrum.yedges) - 2)).astype(int)
    source += card_writer('DS3 S', dist_nums + shift, 5)

    for i in dist_nums:
        source += card_writer('SI{} H'.format(i + shift), flux_spectrum.xedges[1:], 4)
        dist = np.concatenate((np.array([0]), flux_spectrum.int[1:, i + 1]))
        source += card_writer('SP{} D'.format(i + shift), dist, 4)

    return source


def write_input(det, bonner_size=12):
    """Utility that writes two mcnp inputs (response function and
    integrated response) given a detector type."""

    # check input
    message = "Detector must be of type 'empty', 'bs, 'ft' or 'wt'."
    assert det in ('empty', 'bs', 'ft', 'wt'), message

    # choose erg bin structure
    erg_struct = 'scale252'

    # first, grab the empty bp geometry template
    mcnp_input = mcnp_template

    # calculate some things
    # convert bonner size from diameter in inches to radius in cm
    bonner_size = (bonner_size / 2) * 2.54

    # select mcnp fill
    if det == 'empty':
        fill = ('      ', '      ')
    elif det == 'bs':
        fill = ('      ', 'FILL=1')
    elif det == 'ft':
        raise NotImplementedError
    elif det == 'wt':
        raise NotImplementedError

    # grab the source term
    source = source_writer(erg_struct)

    # grab tally erg bins
    erg_bins = energy_groups(erg_struct)
    tally = card_writer('E114', erg_bins, 4)

    # format the mcnp
    mcnp_input = mcnp_input.format(*fill, bonner_size, source, tally)

    # write to file
    with open(det + '.i', 'w+') as F:
        F.write(mcnp_input)

    return


def test_write_input():
    """A small utility used to test write_input()."""

    # test empty case
    write_input('empty')
    write_input('bs')


if __name__ == '__main__':
    test_write_input()
