import re
import numpy as np
import matplotlib.pyplot as plt
from energy_groups import energy_groups
import sys
sys.path.insert(0, '../')
import paths


def extract_mcnp(par):
    """Utility that grabs the flux data from an mcnp output file."""

    # open file w/ neutron data
    with open(paths.main_path + '/flux/mcnp/ksuna.out') as F:
        mcnp_file = F.read()

    # grab all tally data
    pattern = re.compile(r'    \d.\d\d\d\dE[+-]\d\d   \d.\d\d\d\d\dE[+-]\d\d \d.\d\d\d\d')
    results = re.findall(pattern, mcnp_file)

    # convert to np array
    results = np.array([[float(r.split()[1]), float(r.split()[2])] for r in results])

    # convert error to absolute
    results[:, 1] = results[:, 0] * results[:, 1]

    # reshape to fit data structure
    results = results.reshape(8, -1, 253, 2)

    return



def test_nebp_flux():
    """A small utility that tests nebp_flux()."""

    nebp = extract_mcnp('n')

    return


if __name__ == '__main__':
    test_nebp_flux()
