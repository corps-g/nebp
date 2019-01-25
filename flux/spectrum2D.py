import numpy as np


class Spectrum2D(object):

    """A useful container for nuclear engineering applications.
    This description will be updated in the future."""

    def __init__(self, xedges, yedges, values, error, form='int', floor=(0, 0)):
        """Create a 2D verion of the spectrum object and its associated values.
        Also calculates some useful things."""

        # check array inputs
        assert type(xedges) in (list, tuple, np.ndarray), "Bin edges must be of type list, tuple, or ndarray."
        assert type(yedges) in (list, tuple, np.ndarray), "Bin edges must be of type list, tuple, or ndarray."
        assert type(values) in (list, tuple, np.ndarray), "Values must be of type list, tuple, or ndarray."
        assert type(error) in (int, list, tuple, np.ndarray), "Error must be of type int, list, tuple, or ndarray."
        assert type(floor) in (list, list, tuple, np.ndarray), "Floor must be either int or float."

        # check spectrum form
        message = "Spectrum form option must be literal in ('int', 'integral', 'dif', 'diff', 'differential')."
        assert form in ('int', 'integral', 'dif', 'diff', 'differential'), message

        # allow error to be input as 0, which implies that all error is zero
        if type(error) is int:
            assert error is 0, "Error integer input only allows 0."

            # then rewrite error
            error = np.zeros((len(values), len(values[0])))

        # check
        if type(values) in (list, tuple):
            message = "Issue with shape of values."
            assert all([len(values[i]) == len(values[i + 1]) for i in range(len(values) - 1)]), message

        # check val and err shapes
        message = "Values and error must have same shape."
        assert len(values) == len(error), message
        assert len(values[0]) == len(error[0]), message

        # check edges and val shapes
        message = "Inconsistency in number of x edges/values."
        assert len(xedges) in (len(values), len(values) + 1), message
        message = "Inconsistency in number of y edges/values."
        assert len(yedges) in (len(values[0]), len(values[0]) + 1), message

        # guarantee that bin edges are unique and increasing or decreasing
        message = "Bin edge values must be unique and strictly increasing or decreasing."
        assert all(x < y for x, y in zip(xedges, xedges[1:])) or all(x > y for x, y in zip(xedges, xedges[1:])), message
        assert all(x < y for x, y in zip(yedges, yedges[1:])) or all(x > y for x, y in zip(yedges, yedges[1:])), message

        # convert all data types to numpy arrays
        values = np.array(values)
        error = np.array(error)

        # special handling for different size edges
        if len(xedges) == values.shape[0]:
            xedges = np.concatenate(np.array([floor[0]]), np.array(xedges))
        elif len(xedges) == len(values) + 1:
            xedges = np.array(xedges)

        # do the same for y edges
        if len(yedges) == values.shape[1]:
            yedges = np.concatenate(np.array([floor[1]]), np.array(yedges))
        elif len(yedges) == len(values) + 1:
            yedges = np.array(yedges)

        # calculate bin widths
        self.xwidths = xedges[1:] - xedges[:-1]
        self.ywidths = yedges[1:] - yedges[:-1]
        self.areas = np.matrix(self.xwidths).T * np.matrix(self.ywidths)

        # calculate midpoints
        self.xmidpoints = (xedges[1:] + xedges[:-1]) / 2
        self.ymidpoints = (yedges[1:] + yedges[:-1]) / 2
        self.midpoints = np.meshgrid(self.xmidpoints, self.ymidpoints)

        # store other primary values
        self.xedges = xedges
        self.yedges = yedges

        # store values and error in differential and integral forms
        if form in ('int', 'integral'):
            self.int = values
            self.int_error = error
            self.diff = self.int / self.areas
            self.diff_error = self.int_error / self.areas

        elif form in ('dif', 'diff', 'differential'):
            self.diff = values
            self.diff_error = error
            self.int = self.diff * self.areas
            self.int_error = self.int_error * self.areas

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
        """This will return arguments needed for some 3D plotting."""
        raise NotImplementedError
