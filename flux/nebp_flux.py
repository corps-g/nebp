import re
import numpy as np
import sys
sys.path.insert(0, '../')
import paths


def extract_mcnp(par, power):
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

    # scale
    scaling_constant = (2.54 * power) / (200 * 1.60218e-13)
    results *= scaling_constant

    return results
