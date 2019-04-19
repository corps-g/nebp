import numpy as np
import sys
sys.path.insert(0, '../')
import paths
from nebp_flux import extract_mcnp
from response import response_data
from process_activities import Au_Foil_Data
from origami import unfold


class Unfold_NEBP(object):

    """Docstring."""

    def __init__(self):
        """Docstring."""

        # power level
        self.P = 1E5

        # unfolding parameters
        self.params = {'tol': 1E-20, 'max_iter': 300}

        # number of foils used
        self.num_foils = 9

        # get default spectrum
        self.ds = self.prepare_default_spectrum()

        # prep response matrix
        self.R, self.eb = self.prepare_response_matrix()

        # get responses
        self.N = self.prepare_responses()

        # unfold
        self.sol = self.unfold()

        return

    def prepare_default_spectrum(self):
        """Docstring."""

        # get the flux data at 100kW
        flux_data = extract_mcnp('n', self.P)

        # sum to only energy dependent (exclude the first cos group)
        flux = np.sum(flux_data[:, 1:, :, 0], axis=(0, 1))

        return flux

    def prepare_response_matrix(self):
        """Docstring."""

        # get response functions
        responses = response_data()

        # this pulls only the rfs for the bonner spheres
        response_functions = []
        for name, response in responses.items():
            if 'au' in name:
                response_functions.append(response.int)
                eb = response.edges
        response_functions = np.array(response_functions)

        #
        response_functions = response_functions[:self.num_foils]

        return response_functions, eb

    def prepare_responses(self):
        """Docstring."""

        #
        experimental_data = Au_Foil_Data()

        #
        responses = experimental_data.a_sat_atom

        return responses

    def unfold(self):
        """Docstring."""

        # unfold
        solution = unfold(self.N, self.N * 0.05, self.R, self.ds, method='Gravel', params=self.params)

        return solution

if __name__ == '__main__':
    unfolded_nebp = Unfold_NEBP()
