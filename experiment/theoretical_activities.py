import numpy as np
from process_activities import Au_Foil_Data
import sys
sys.path.insert(0, '../')
import paths
from nebp_flux import extract_mcnp
from response import response_data


class Au_Foil_Theoretical(object):

    """This object folds and holds activities based on the experimental data
    given to it so it can be used as a direct comparison."""

    def __init__(self, experiment):
        """Upon initialization, this calculates the saturation activities after
        storing the experimental data."""

        # store the experiment that we're comparing with
        self.experiment = experiment

        # calculate the theoretical saturation activities
        self.calc_a_sat()

        # calculate saturation per atom
        self.calc_a_sat_atom()

        return

    def calc_a_sat(self):
        """This utility folds the flux with the response functions to calculate
        the saturation activities and sat. act. per sample atom."""

        # get the flux data at 100kW
        flux_data = extract_mcnp('n', self.experiment.P)

        # sum to only energy dependent
        flux = np.sum(flux_data[:, :, :, 0], axis=(0, 1))

        # get response functions
        responses = response_data()

        # this pulls only the rfs for the gold foil tube
        response_functions = []
        for name, response in responses.items():
            if 'au' in name:
                response_functions.append(response.int)
        response_functions = np.array(response_functions)

        # fold the rfs and the flux together, convert to uCi
        self.a_sat = np.sum(response_functions * flux, axis=1) * (1 / 3.7E4)

        # calculate saturation activities per atom
        self.a_sat_atom / self.experiment.atoms

        return


if __name__ == '__main__':
    experimental_data = Au_Foil_Data()
    theoretical_data = Au_Foil_Theoretical(experimental_data)
