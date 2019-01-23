import numpy as np


class Spectrum(object):

    """A useful container for nuclear engineering applications.
    This description will be updated in the future."""

    def __init__(self, edges, values, error, form='int'):
        """Create a spectrum object and its associated values.
        Also calculates some useful things."""

        # check array inputs
        assert type(edges) in (list, tuple, np.ndarray), "Bin edges must be of type list, tuple, or ndarray."
        assert type(values) in (list, tuple, np.ndarray), "Values must be of type list, tuple, or ndarray."
        assert type(error) in (list, tuple, np.ndarray), "Error must be of type list, tuple, or ndarray."

        # check spectrum form
        message = "Spectrum form option must be literal in ('int', 'integral', 'dif', 'diff', 'differential')."
        assert form in ('int', 'integral', 'dif', 'diff', 'differential'), message

        # guarantee that bin edges are unique and increasing or decreasing
        message = "Bin edge values must be unique and strictly increasing or decreasing."
        assert all(x < y for x, y in zip(edges, edges[1:])) or all(x > y for x, y in zip(edges, edges[1:])), message
