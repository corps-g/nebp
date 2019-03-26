import numpy as np
from template import mcnp_template
import sys
sys.path.insert(0, '../')
import paths
from nebp_flux import extract_mcnp
from group_structures import energy_groups, cosine_groups, radial_groups


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


def source_writer(erg_struct, source_region, source_bounds):
    """Writes the triga nebp mcnp source."""

    # first, match the cosine bin structure of the original mcnp
    cos_struct = cosine_groups('fine')
    erg_struct = energy_groups(erg_struct)

    # this is the energy dependent flux spectrum for the first distribution

    # add the erg dependent distribution
    source = card_writer('SI2 H', erg_struct, 4)

    # mcnp requires the first bin in a distribution be zero
    dist = np.ones(len(erg_struct))
    dist[0] = 0
    source += card_writer('SP2 D', dist, 4)

    # add dependent distribution
    flux = extract_mcnp('n', 1)[source_region]

    # add distribution
    shift = 4
    dist_nums = np.array(range(len(erg_struct) - 1)).astype(int)
    source += card_writer('DS3 S', dist_nums + shift, 5)

    for i in dist_nums:
        source += card_writer('SI{} H'.format(i + shift), cos_struct[1:], 4)
        dist = np.concatenate((np.array([0]), flux[1:, i + 1, 0]))

        # fix distribution if zero to avoid mcnp fatal error
        if np.all(dist == 0):
            dist[-1] = 1

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

        # scx tally for foil tube
        ft_tally += 'F{}4:N {}\n'.format(13 + l, p2)
        ft_tally += 'FM{}4: 1 {} 102\n'.format(13 + l, foil_mat[foil_type][0])
        ft_tally += 'FT{}4  SCX 2\n'.format(13 + l)

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

    # make one for each source region
    source_regions = zip(radial_groups('nebp')[1:], radial_groups('nebp')[:-1])

    for i, source_bounds in enumerate(source_regions):

        # choose erg bin structure
        erg_struct = 'scale252'
        erg_bins = energy_groups(erg_struct)

        # first, grab the empty bp geometry template
        mcnp_input = mcnp_template

        # select mcnp fill
        if det == 'empty':
            fill = ('      ', '      ')
            fname = 'empty{}.inp'.format(i)
        elif det == 'bs':
            fill = ('      ', 'FILL=1')
        fname = 'bs{}.inp'.format(str(int(bonner_size)))
        elif det == 'ft':
        fill = ('FILL=2', 'FILL=2')
            fname = 'ft_{}{}.inp'.format(foil_type, i)
        elif det == 'wt':
            fill = ('FILL=3', 'FILL=4')
            fname = 'wt{}.inp'.format(i)

    # calculate some things
    # convert bonner size from diameter in inches to radius in cm
    bonner_size = (bonner_size / 2) * 2.54

        # produce foil tube geometry
        ft_cells, ft_surfs, ft_tally = foil_tube_geometry(foil_type, foil_mass, erg_bins)

        # grab the source term
        source = source_writer(erg_struct, i, source_bounds)

        # format the mcnp
    mcnp_input = mcnp_input.format(*fill, ft_cells, bonner_size, ft_surfs, source, ft_tally)

        # write to file
        with open('mcnp/' + fname, 'w+') as F:
            F.write(mcnp_input)

    return


def write_all_inputs():
    """A small utility used to test write_input()."""

    for bonner_size in [0, 2, 3, 5, 8, 10, 12]:
        write_input('bs', bonner_size=bonner_size)
    write_input('ft', 'in', 20.0)
    write_input('ft', 'au', 45.0)


if __name__ == '__main__':
    write_all_inputs()
