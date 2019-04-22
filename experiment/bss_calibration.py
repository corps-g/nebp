import numpy as np
import scipy as sp
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

        # model
        def model(x, A, B, C, D, E):
            """Docstring."""
            return A * np.exp(-B * x) + C * (1 / np.sqrt(2 * np.pi * D**2)) * np.exp(-(x - E)**2 / (2 * D**2))

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
            ydata = np.array([int(l) for l in lines[12:1036]])

            # trim up to lld
            ydata = ydata[lld:]
            ydata = ydata[:600]

            #
            xdata = range(len(ydata))

            # fit the curve
            popt, pcov = sp.optimize.curve_fit(model, xdata, ydata, p0=[1, 1, 1, 1, 300])

            # sum counts beyond lld, convert to rate, and store
            counts[i] = popt[2] / t

            # plot fit
            fig = plt.figure(i + 300)
            ax = fig.add_subplot(111)
            ax.set_xlabel('Channel')
            ax.set_ylabel('Counts')
            ax.plot(xdata, ydata, color='navy', ls='None', marker='.', markersize=0.3, label='Data')
            ax.plot(xdata, model(xdata, *popt), color='seagreen', label='Model')
            ax.legend()
            fig.savefig('plot/bs{}_calibration_spectrum.png'.format(i + 1), dpi=300)
            fig.clear()

            # sum counts beyond lld, convert to rate, and store
            counts[i] = popt[2] / t

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
        self.responses = np.sum(response_functions * flux[1:], axis=1)

        return


if __name__ == '__main__':
    experimental_data = BSS_Calibration()
