import numpy as np
import matplotlib.pyplot as plt


class Fuel_Element(object):

    """Stores and calculates relevent fuel element fission rate data."""

    def __init__(self, name, rr_abs, rr_rel_error, n_ax, n_rad, ax_dims, rad_dims):
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
        """Calculate some geometrically significant parameters."""

        # compute radial and axial plane locations
        self.ax_divs = np.linspace(*self.ax_dims, self.n_ax + 1)
        self.rad_divs = np.linspace(*self.rad_dims, self.n_rad + 1)

        # compute radial and axial plane locations
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


class Triga_Core(object):

    """Calculates and stores a bunch of fission rate values associated with
    the fuel elements in the core."""

    def __init__(self, fuel_data):
        self.fuel = fuel_data
        self.calc_extrema()

    def calc_extrema(self):
        """Calculates the min and max rr densities"""
        self.max_rr_density = np.max([np.max(element.rr_density) for element in self.fuel.values()])
        self.min_rr_density = np.min([np.min(element.rr_density) for element in self.fuel.values()])


def extract_fission_data():
    """This function pulls the fission data from the ksu_triga fuel."""

    # initialize container
    cell_data = {}

    # load in the file
    with open('mcnp/ksu.inpo') as F:
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


def mirror_element(rr_density):
    """Given a fission rate density map, creates a symmetric mirror with an empty
    middle channel to more closely resemble an actual element."""

    n_ax, n_rad = rr_density.shape
    new_map = np.zeros((n_ax, n_rad * 2 + 1))

    new_map[:, 0:n_rad] = rr_density[:, ::-1]
    new_map[:, n_rad + 1:] = rr_density[:, ]

    return new_map


def plot_fission_rates():
    """A utility to visualize the fission data from the ksu-triga core."""

    # grab data
    core = extract_fission_data()

    # ---------------------------------- plot the heatmaps of the in-element fission rate densities
    # initalize plotting environment
    fig = plt.figure(0, figsize=(2, 10))

    # grab an element
    element = core.fuel['201']

    # plot the reaction rate density map
    ax = fig.add_subplot(111)
    ext = [*element.rad_dims, *element.ax_dims]
    ax.imshow(mirror_element(element.rr_density), vmin=core.min_rr_density, vmax=core.max_rr_density, extent=ext, cmap='viridis')

    # save the elements plot
    fig.savefig('plot/rr_dist_B1.png', dpi=300)
    plt.close(fig)

    # ---------------------------------- plot the axial and radial distributions for each ring
    # loop through each ring
    for i in range(2, 7):

        # make a list of each element in that ring
        ring_elements = []
        for element_id in core.fuel.keys():
            if str(i) == element_id[0]:
                ring_elements.append(element_id)

        # set up plotting environment
        fig0 = plt.figure(i, figsize=(3, 10))
        ax0 = fig0.add_subplot(111)
        fig1 = plt.figure(i + 10, figsize=(10, 3))
        ax1 = fig1.add_subplot(111)

        # axial plot
        for element_id in ring_elements:

            # grab the individual element
            element = core.fuel[element_id]

            # plot the axial reaction rate
            ax0.plot(element.rr_ax, element.ax_mps, label=element_id)
            ax1.plot(element.rad_mps, element.rr_rad, label=element_id)

        # save the figure
        ax0.legend()
        ax1.legend()
        fig0.savefig('plot/axial_rr_density_{}'.format(i), dpi=300)
        fig1.savefig('plot/radial_rr_density_{}'.format(i), dpi=300)
        plt.close(fig0)
        plt.close(fig1)

    return


if __name__ == '__main__':
    plot_fission_rates()
