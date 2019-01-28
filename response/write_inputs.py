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


def foil_tube_geometry(foil_type, foil_mass, erg_bins):
    """Writes the cells, surfs and tallies for the foil tube."""

    # create bonner tube cells and surfs
    ft_cells = ''
    ft_surfs = ''
    ft_tally = ''

    # foil mat
    foil_mat = {'in': '5 -7.310', 'au': '9 -19.30'}

    # calculate foil thickness (convert to mg)
    rho = float(foil_mat[foil_type].split()[1][1:])
    radius = 0.25
    foil_thickness = (foil_mass * 0.001) / (rho * np.pi * radius**2)
    print(foil_thickness)

    # loop through each section of the foil tube
    for l in np.linspace(0, 11, 12).astype(int):

        # foil tube cells
        p1, p2, p3, p4 = 211 + l, 231 + l, 251 + l, 212 + l
        ft_cells += '{} 2 -1.300  ({} -{} -201):({} -{} 203 -201)  U=2 IMP:N=1\n'.format(p1, p1, p2, p2, p4)
        ft_cells += '{} {}  ({} -{} -203)                      U=2 IMP:N=1\n'.format(p2, foil_mat[foil_type], p2, p3)
        ft_cells += '{} 0         ({} -{} -203)                      U=2 IMP:N=1\n'.format(p3, p3, p4)

        # foil tube surfaces
        ft_surfs += '{} 2  PX  {}\n'.format(p1, l * 2.54)
        ft_surfs += '{} 2  PX  {}\n'.format(p2, (l * 2.54) + 2.44)
        ft_surfs += '{} 2  PX  {}\n'.format(p3, (l * 2.54) + 2.44 + foil_thickness)

        # foil tube tallies
        ft_tally += 'F{}4:N {}\n'.format(13 + l, p2)
        ft_tally += 'FM{}4: 1 {} 102\n'.format(13 + l, foil_mat[foil_type][0])
        ft_tally += card_writer('E{}4'.format(13 + l), erg_bins, 4)

        # scx tally for foil tube
        ft_tally += 'F{}4:N {}\n'.format(25 + l, p2)
        ft_tally += 'FM{}4: 1 {} 102\n'.format(25 + l, foil_mat[foil_type][0])
        ft_tally += 'FT{}4  SCX 2\n'.format(25 + l)

    # add a final surf
    ft_surfs += '{} 2  PX  {}\n'.format(223, 12 * 2.54)

    return ft_cells, ft_surfs, ft_tally


def write_input(det, foil_type='in', foil_mass=2.1, bonner_size=12):
    """Utility that writes two mcnp inputs (response function and
    integrated response) given a detector type."""

    # check input
    message = "Detector must be of type 'empty', 'bs, 'ft' or 'wt'."
    assert det in ('empty', 'bs', 'ft', 'wt'), message

    message = "Foil type must be literal 'in' or 'au'."
    assert foil_type in ('in', 'au'), message

    # choose erg bin structure
    erg_struct = 'scale252'
    erg_bins = energy_groups(erg_struct)

    # first, grab the empty bp geometry template
    mcnp_input = mcnp_template

    # calculate some things
    # convert bonner size from diameter in inches to radius in cm
    bonner_size = (bonner_size / 2) * 2.54

    # select mcnp fill
    if det == 'empty':
        fill = ('      ', '      ')
        fname = 'empty.i'
    elif det == 'bs':
        fill = ('      ', 'FILL=1')
        fname = 'bs.i'
    elif det == 'ft':
        fill = ('FILL=2', 'FILL=4')
        fname = 'ft_{}.i'.format(foil_type)
    elif det == 'wt':
        fill = ('FILL=3', 'FILL=4')
        fname = 'wt.i'

    # produce foil tube geometry
    ft_cells, ft_surfs, ft_tally = foil_tube_geometry(foil_type, foil_mass, erg_bins)

    # grab the source term
    source = source_writer(erg_struct)

    # grab tally erg bins
    tally = card_writer('E114', erg_bins, 4)

    # format the mcnp
    mcnp_input = mcnp_input.format(*fill, ft_cells, bonner_size, ft_surfs, source, tally, ft_tally)

    # write to file
    with open('mcnp/' + fname, 'w+') as F:
        F.write(mcnp_input)

    return


def test_write_input():
    """A small utility used to test write_input()."""

    # test empty case
    write_input('empty')
    write_input('bs')
    write_input('ft', 'in', 20.0)
    write_input('ft', 'au', 45.0)


if __name__ == '__main__':
    test_write_input()
