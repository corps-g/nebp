import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')
import paths
from nebp_flux import extract_mcnp
from response import response_data
from bss_calibration import BSS_Calibration
from theoretical_activities import Au_Foil_Theoretical
from process_activities import Au_Foil_Data


class BSS_Data(object):

    """Docstring."""

    def __init__(self):
        """Docstring."""

        # nominal power level kW(th)
        self.P = 1000

        # sizes
        self.sizes = np.array([0, 2, 3, 5, 8, 10, 12])

        # store calibration data
        self.calibration = BSS_Calibration()

        # store the experiment that we're comparing with
        self.experiment = self.process_experiment()

        # grab the fudge factor
        foil_experiment = Au_Foil_Data()
        foil_data = Au_Foil_Theoretical(foil_experiment)
        self.nebp_fudge_factor = foil_data.nebp_fudge_factor

        # calculate the theoretical saturation activities
        self.calc_responses()

        return

    def process_experiment(self):
        """Implement after experiment."""
        # LLD channel
        lld = 400

        # channels surrounding peak
        L = 1000
        R = 1800
        w = 10

        # initialize array
        counts = np.zeros(len(self.sizes))

        # model
        def model(x, A, B, C, D, E):
            """Docstring."""
            return A * np.exp(-B * x) + C * (1 / np.sqrt(2 * np.pi * D**2)) * np.exp(-(x - E)**2 / (2 * D**2))

        # loop through each size
        for i, size in enumerate(self.sizes):

            #
            filename = '4_18_19/bss' + str(size) + '.Spe'

            # grab the data
            with open(filename, 'r') as F:
                lines = F.readlines()

            # extract time
            t = int(lines[2065])

            # extract channel data
            data = np.array([int(l) for l in lines[12:2059]])

            # get average backgrounds
            L_avg = np.average(data[L - w // 2:L + w // 2])
            R_avg = np.average(data[R - w // 2:R + w // 2])

            # compute bg area
            bg = (R - L) * (min([L_avg, R_avg]) + 0.5 * (max([L_avg, R_avg]) - min([L_avg, R_avg])))

            # sum counts beyond lld, convert to rate, and store
            counts[i] = (np.sum(data[L:R]) - bg) / t

            plt.figure(i)
            plt.plot(data)
            plt.plot([L, R], [L_avg, R_avg])
            #plt.yscale('log')

        return counts

    def calc_responses(self):
        """Docstring."""

        # get the flux data at 100kW
        flux_data = extract_mcnp('n', self.P)

        # sum to only energy dependent (exclude the first cos group)
        flux = np.sum(flux_data[:, 1:, 1:, 0], axis=(0, 1))

        # get response functions
        responses = response_data()

        # this pulls only the rfs for the bonner spheres
        response_functions = []
        for name, response in responses.items():
            if 'bs' in name and 'p' not in name:
                response_functions.append(response.int)
        response_functions = np.array(response_functions)

        # fold the rfs and the flux together, convert to uCi / atom
        self.responses = np.sum(response_functions * flux, axis=1)

        # apply calibration efficiency
        self.responses *= self.calibration.efficiency

        return


if __name__ == '__main__':
    experimental_data = BSS_Data()
