import numpy as np


class Spectrum(object):

    """A useful container for nuclear engineering applications.
    This description will be updated in the future."""

    def __init__(self, xedges, yedges, values, error, form='int', floor=0):
        """Create a 2D verion of the spectrum object and its associated values.
        Also calculates some useful things."""
