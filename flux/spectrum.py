import numpy as np


class Spectrum(object):

    """A useful container for nuclear engineering applications.
    This description will be updated in the future."""

    def __init__(self, edges, values, error, form='int', floor=0):
        """Create a spectrum object and its associated values.
        Also calculates some useful things."""

        # check array inputs
        assert type(edges) in (list, tuple, np.ndarray), "Bin edges must be of type list, tuple, or ndarray."
        assert type(values) in (list, tuple, np.ndarray), "Values must be of type list, tuple, or ndarray."
        assert type(error) in (int, list, tuple, np.ndarray), "Error must be of type int, list, tuple, or ndarray."
        assert type(floor) in (int, float), "Floor must be either int or float."

        # check spectrum form
        message = "Spectrum form option must be literal in ('int', 'integral', 'dif', 'diff', 'differential')."
        assert form in ('int', 'integral', 'dif', 'diff', 'differential'), message

        # allow error to be input as 0, which implies that all error is zero
        if type(error) is int:
            assert error is 0, "Error integer input only allows 0."

            # then rewrite error
            error = np.zeros(len(values))

        # check val and err lengths
        message = "Values and error must have same length."
        assert len(values) == len(error), message

        # check edges and val lengths
        message = "Inconsistency in number of edges/values."
        assert len(edges) in (len(values), len(values) + 1), message

        # guarantee that bin edges are unique and increasing or decreasing
        message = "Bin edge values must be unique and strictly increasing or decreasing."
        assert all(x < y for x, y in zip(edges, edges[1:])) or all(x > y for x, y in zip(edges, edges[1:])), message

        # convert all data types to numpy arrays
        values = np.array(values)
        error = np.array(error)

        # special handling for different size edges
        if len(edges) == len(values):
            edges = np.concatenate(np.array([floor]), np.array(edges))
        elif len(edges) == len(values) + 1:
            edges = np.array(edges)

        # calculate midpoints and bin edges
        self.widths = edges[1:] - edges[:-1]
        self.midpoints = (edges[1:] + edges[:-1]) / 2

        # store other primary values
        self.edges = edges

        # store values and error in differential and integral forms
        if form in ('int', 'integral'):
            self.int = values
            self.int_error = error
            self.diff = self.int / self.widths
            self.diff_error = self.int_error / self.widths

        elif form in ('dif', 'diff', 'differential'):
            self.diff = values
            self.diff_error = error
            self.int = self.diff * self.widths
            self.int_error = self.int_error * self.widths

        return

    def __add__(self):
        raise NotImplementedError

    def __radd__(self):
        raise NotImplementedError

    def __sub__(self):
        raise NotImplementedError

    def __rsub__(self):
        raise NotImplementedError

    def __mul__(self):
        raise NotImplementedError

    def __rmul__(self):
        raise NotImplementedError

    def __div__(self):
        raise NotImplementedError

    def __rdiv__(self):
        raise NotImplementedError

    def plot(self, plot_type, form):
        """This function will return the arguments necessary for plotting
        the data with both plt.plot and plt.errorbar in both integral and
        differential form."""

        # check inputs
        message = "Plot type must be literal of either plot or errorbar."
        assert plot_type in ('plot', 'errorbar'), message
        message = "Spectrum form option must be literal in ('int', 'integral', 'dif', 'diff', 'differential')."
        assert form in ('int', 'integral', 'dif', 'diff', 'differential'), message

        # step looking data on a plt.plot
        if plot_type == 'plot':

            # create doubles of bin edges
            X = np.array([[xx, xx] for xx in np.array(self.edges)]).flatten()[1:-1]

            # plotting the integral form
            if form in ('int', 'integral'):
                Y = np.array([[yy, yy] for yy in np.array(self.int)]).flatten()

            # plotting the differential form
            elif form in ('dif', 'diff', 'differential'):
                Y = np.array([[yy, yy] for yy in np.array(self.diff)]).flatten()

            # return the x and y points
            return X, Y

        # for errorbars in the appropriate locations
        elif plot_type == 'errorbar':

            # errorbars on the integral form
            if form in ('int', 'integral'):
                return self.midpoints, self.int, self.int_error

            # errorbars on the differential form
            elif form in ('dif', 'diff', 'differential'):
                return self.midpoints, self.diff, self.diff_error
