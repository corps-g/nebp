import numpy as np
import sys
sys.path.insert(0, '../')
import paths
from cf252 import cf252_source
from response import response_data


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
        self.efficiency = np.average(self.experiment / self.responses)

        # corrected
        self.corrected = self.experiment / self.efficiency

        return

    def process_experiment(self):
        """Docstring."""

        # grab the data
        data = np.loadtxt('4_16_19/cf252_calibration.txt')

        # convert to rate
        t = 5 * 60
        data = data / (t)

        # TODO: subtract background
        pass

        return data

    def calc_responses(self):
        """Docstring."""

        # get the flux data at 100kW
        flux = cf252_source()

        # get response functions
        responses = response_data()

        # this pulls only the rfs for the bonner spheres
        response_functions = []
        for name, response in responses.items():
            if 'bs' in name:
                response_functions.append(response.int)
        response_functions = np.array(response_functions)

        # fold the rfs and the flux together, convert to uCi / atom
        self.responses = np.sum(response_functions * flux, axis=1)

        return


if __name__ == '__main__':
    experimental_data = BSS_Calibration()
