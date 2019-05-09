import sys
sys.path.insert(0, '../')
import paths
import numpy as np
import matplotlib.pyplot as plt
from group_structures import energy_groups


class Fuel_Element(object):

    """Stores and calculates individual fuel element fission rate data."""

    def __init__(self, name, rr_abs, rr_rel_error, n_ax, n_rad, ax_dims, rad_dims):
        """Initialize with the following parameters:
            name - the identifying name of the fuel element (str)
            rr_abs - the absolute reaction rate in each cell (40x5 numpy array)
            rr_rel_error - the relative errors associated with rr_abs (40x5 numpy array)
            n_ax - the number of axial divisions (int)
            n_rad - the number of radial divisions (int)
            ax_dims - the highest and lowest dimension of the fuel (len 2 tuple of floats)
            rad_dims - the inner and outer diameter of the fuel (len 2 tuple of floats)"""
        self.name = name
        self.rr_abs = rr_abs
        self.rr_rel_error = rr_rel_error
        self.rr_abs_error = self.rr_rel_error * self.rr_abs
        self.n_ax = n_ax
        self.n_rad = n_rad
        self.ax_dims = ax_dims
        self.rad_dims = rad_dims
        self.calc_geometric_values()
        self.calc_rr_density()
        self.calc_integrated_values()

    def calc_geometric_values(self):
        """Will calculate and store the location of the geometric divisions, the
        midpoints within those divisions, the difference values between those
        divisions, and the volume of each cell in the fuel element."""

        # compute radial and axial bounding plane locations
        self.ax_divs = np.linspace(*self.ax_dims, self.n_ax + 1)
        self.rad_divs = np.linspace(*self.rad_dims, self.n_rad + 1)

        # compute radial and axial mid-plane locations
        self.ax_mps = (self.ax_divs[1:] + self.ax_divs[:-1]) / 2
        self.rad_mps = (self.rad_divs[1:] + self.rad_divs[:-1]) / 2

        # calculate cell volumes
        self.ax_diff = self.ax_divs[1:] - self.ax_divs[:-1]
        self.rad_diff = (self.rad_divs[1:]**2 - self.rad_divs[:-1]**2) * np.pi

        # make a volume matrix
        self.volumes = np.array([[self.rad_diff[i] * self.ax_diff[j] for i in range(self.n_rad)] for j in range(self.n_ax)])

    def calc_rr_density(self):
        """Uses the volumes to calculate reaction rate density."""

        # divide by the volumes to get the density
        self.rr_density = self.rr_abs / self.volumes
        self.rr_density_error = self.rr_abs_error / self.volumes

    def calc_integrated_values(self):
        """Compute integrated values and profiles for comparison against other elements."""

        # first, compute the total fission rate
        self.total_fission_rate = np.sum(self.rr_abs)

        # then, compute the axial and radial integrated fission rates
        self.rr_ax = np.sum(self.rr_abs, axis=1)
        self.rr_rad = np.sum(self.rr_abs, axis=0)

        # do the same for reaction rate density
        self.rr_density_ax = np.sum(self.rr_density, axis=1)
        self.rr_density_rad = np.sum(self.rr_density, axis=0)


class Triga_Core(object):

    """Calculates and stores a bunch of fission rate values associated with
    the fuel elements in the core."""

    def __init__(self, fuel_data):
        """Initialize the object with a dictionary of Fuel_Element objects."""
        self.fuel = fuel_data
        self.calc_extrema()
        self.calc_core_averages()

    def calc_extrema(self):
        """Calculates the min and max rr densities"""
        self.max_rr_density = np.max([np.max(element.rr_density) for element in self.fuel.values()])
        self.min_rr_density = np.min([np.min(element.rr_density) for element in self.fuel.values()])

    def calc_core_averages(self):
        """Calculates the core average for axial and radial distrubtions."""
        self.ax_avg = np.zeros(self.fuel['201'].n_ax)
        self.rad_avg = np.zeros(self.fuel['201'].n_rad)
        for i, element in enumerate(self.fuel.values()):
            self.ax_avg += (element.rr_ax / (np.sum(element.rr_ax) * len(element.rr_ax)))
            self.rad_avg += (element.rr_rad / (np.sum(element.rr_rad) * len(element.rr_rad)))


def extract_fission_data():
    """Utility that takes the ksu-triga mcnp output file, parses the fission
    tally data and stores it within the Fuel_Element and Triga_Core containers."""

    # initialize container
    cell_data = {}

    # load in the file
    with open(paths.main_path + '/flux/mcnp/ksu.inpo') as F:
        output = F.read()

    # grab the chunk containing the tally data
    chunk = output.split('1tally')[1].split('\n\n')[1].split('\n \n')[1:]

    # loop through chunk and extract cell numbers and fission data
    for cell in chunk:
        cell = cell.split()
        cell_data[cell[1]] = float(cell[-2]), float(cell[-1])

    # number of divisions
    n_axial = 40
    n_radial = 5

    # plot the axial profile
    axial_dims = -19.05, 19.05
    radial_dims = 0.2286, 1.8161

    # create a list of all fuel element in the current model
    elements = sorted(list(set([keys[1:4] for keys in cell_data.keys()])))
    fuel_data = {}

    # loop through each fuel element
    for e, element in enumerate(elements):

        # initialize some matrices
        element_data = np.empty(n_axial * n_radial)
        element_err = np.empty(n_axial * n_radial)

        # convert the data to a matrix
        for i in range(n_axial * n_radial):
            cell = '1' + element + '{0:04}'.format(i)
            element_data[i] = cell_data[cell][0]
            element_err[i] = cell_data[cell][1]

        # reshape the matrix into axial by radial
        element_data = element_data.reshape(n_axial, n_radial)
        element_err = element_err.reshape(n_axial, n_radial)

        # add to dict
        fuel_data[element] = Fuel_Element(element, element_data, element_err, n_axial, n_radial, axial_dims, radial_dims)

    # return the dictionary containing the fission rates
    return Triga_Core(fuel_data)


def card_writer(card, data, elements):
    """This will write multiline cards for SI and SP distributions for mcnp inputs

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


def write_fission_sdef():
    """Writes the SDEF cards (and a few others) for the ksu-triga nebp
    transport problem."""

    # grab data
    core = extract_fission_data()

    # dealing with locations
    radii = np.array([0, 3.192, 6.284, 9.406, 12.532, 15.660]) * 2.54
    rotationAngle = 50 * (np.pi / 180)
    locations = ['{}{:02d}'.format(p[1], int(n)) for p in enumerate('123456') for n in range(1, max(2, 1 + 6 * p[0]))]

    coord = []
    for ring in range(len(radii)):
        for theta in np.linspace(0, 2 * np.pi, max(1, 6 * ring), endpoint=False):
            coord.append((radii[ring] * np.sin(theta + rotationAngle), radii[ring] * np.cos(theta + rotationAngle)))

    # Create the dictionary by combining locations and coordinates
    coord = dict(zip(locations, coord))
    del locations[0]
    del coord['101']

    # create nps and turn off additional fissions
    s = 'NPS 1E8\n'
    s += 'NONU\n'

    # create sdef
    s += 'SDEF ERG=D1 RAD=D2  AXS=0 0 1  POS=D3  EXT=FPOS=D4 \n'

    # write energy distribution (watt spectrum)
    s += 'SP1  -3\n'

    # write radial dependence
    s += card_writer('SI2   ', core.fuel['201'].rad_divs, 3)
    s += card_writer('SP2   ', np.insert(core.rad_avg, 0, 0), 3)

    # loop through locations and if it's occupied by an element, store its absolute position and integrated fission rate
    pos_card = []
    mag_card = []
    dist_card = []
    for loc in locations:
        if loc in core.fuel:
            dist_card += [int(loc)]
            pos_card += [*coord[loc], 0]
            mag_card += [core.fuel[loc].total_fission_rate]

    # write those positions and fission rates into mcnp cards
    s += card_writer('SI3  L', np.array(pos_card), 3)
    s += card_writer('SP3   ', np.array(mag_card), 4)

    # write distributed source distribution numbers
    s += card_writer('DS4  S', np.array(dist_card), 8)

    # for each position, write the axial distribution cards
    for loc in dist_card:
        s += card_writer('SI{}  H'.format(loc), core.fuel[str(loc)].ax_divs, 4)
        s += card_writer('SP{}  D'.format(loc), np.insert(core.fuel[str(loc)].rr_ax, 0, 0), 4)

    # write the tally cards
    cos_struct = np.array([0.000000e+00, 1.736482e-01, 3.420201e-01, 5.000000e-01, 6.427876e-01,
                           7.660444e-01, 8.660254e-01, 9.396926e-01, 9.848078e-01, 9.961947e-01,
                           9.993908e-01, 9.998477e-01, 9.999619e-01, 1.000000e+00])

    # store surface numbers
    surface_outer = 40000000 + 15
    surface_inner = 40000000 + 13
    surface_radius = 40000000 + 10

    # fine energy resolution
    t = 'F11:N {}\n'.format(surface_outer)
    t += 'FS11  {}\n'.format(surface_radius)
    t += card_writer('E11  ', energy_groups('scale252'), 4)

    # course energy resolution w/ cosine bins
    t += 'F21:N {}\n'.format(surface_outer)
    t += 'FS21  {}\n'.format(surface_radius)
    t += card_writer('E21  ', energy_groups('hr6'), 4)
    t += card_writer('C21  ', cos_struct, 4)

    # identical tallys within beam port
    t += 'F111:N {}\n'.format(surface_inner)
    t += 'FS111  {}\n'.format(surface_radius)
    t += card_writer('E111  ', energy_groups('scale252'), 4)

    # copy
    t += 'F121:N {}\n'.format(surface_inner)
    t += 'FS121  {}\n'.format(surface_radius)
    t += card_writer('E121  ', energy_groups('hr6'), 4)
    t += card_writer('C121  ', cos_struct, 4)

    # import template
    with open('mcnp/template.inp', 'r') as F:
        template = F.read()

    # replace the flag in the template with the new source term
    ksun = template.replace('*SOURCE_FLAG*', s).replace('*TALLY_FLAG*', t)

    # write the new neutron input to a file
    with open('mcnp/ksun.inp', 'w+') as F:
        F.write(ksun)

    return


if __name__ == '__main__':
    pass
