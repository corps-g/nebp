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
