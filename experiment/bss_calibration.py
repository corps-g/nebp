import numpy as np
import sys
sys.path.insert(0, '../')
import paths
from cf252 import cf252_source
from response import response_data
import matplotlib.pyplot as plt


class BSS_Calibration(object):

    """Docstring."""

    def __init__(self):
        """Docstring."""

        # sizes
        self.sizes = np.array([0, 2, 3, 5, 8, 10, 12])

        # store the experiment that we're comparing with
        self.experiment = self.process_experiment()

        # calculate the theoretical saturation activities
        self.calc_responses()

        # compute the efficiency
        self.correction_factors = self.experiment / self.responses[1:]
        self.efficiency = np.average(self.experiment[-3:] / self.responses[-3:])

        # corrected
        self.corrected = self.experiment / self.efficiency

        return

    def process_experiment(self):
        """Docstring."""

        # LLD channel
        lld = 400

        # initialize array
        counts = np.zeros(len(self.sizes[1:]))

        # loop through each size
        for i, size in enumerate(self.sizes[1:]):

            #
            filename = '4_17_19/cf' + str(size) + '.Spe'

            # grab the data
            with open(filename, 'r') as F:
                lines = F.readlines()

            # extract time
            t = int(lines[1041])

            # extract channel data
            data = np.array([int(l) for l in lines[12:1036]])

            # sum counts beyond lld, convert to rate, and store
            counts[i] = np.sum(data[lld:]) / t

        return counts

    def calc_responses(self):
        """Docstring."""

        # get the flux data at 100kW
        flux = cf252_source()

        # get response functions
        responses = response_data()

        # this pulls only the rfs for the bonner spheres
        response_functions = []
        for name, response in responses.items():
            if 'pbs' in name:
                response_functions.append(response.int)
        response_functions = np.array(response_functions)

        # fold the rfs and the flux together, convert to uCi / atom
        self.responses = np.sum(response_functions * flux, axis=1)

        return


if __name__ == '__main__':
    experimental_data = BSS_Calibration()
