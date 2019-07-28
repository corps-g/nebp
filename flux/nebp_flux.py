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

    # the following values were manually pulled from the file used above
    k_eff = 1.09946
    nu_bar = 2.438

    # scale
    scaling_constant = (nu_bar * power) / (200 * 1.60218e-13 * k_eff)
    results *= scaling_constant

    return results


def extract_mcnp(par, power):
    """Utility that grabs the flux data from an mcnp output file.
    Replaces the function above, but is used for the prdmp version of the
    flux."""

    # open file w/ neutron data
    with open(paths.main_path + '/flux/mcnp/ksuna_prdmp.out') as F:
        mcnp_file = F.read()

    # grab all tally data
    pattern = re.compile(r'    \d.\d\d\d\dE[+-]\d\d   \d.\d\d\d\d\dE[+-]\d\d \d.\d\d\d\d')
    results = re.findall(pattern, mcnp_file)
    
    # because of prdmp, we need to grab only the last dump of data
    results = results[-220616:]

    # convert to np array
    results = np.array([[float(r.split()[1]), float(r.split()[2])] for r in results])
    
    # convert error to absolute
    results[:, 1] = results[:, 0] * results[:, 1]

    # reshape to fit data structure
    results = results.reshape(8, -1, 253, 2)

    # the following values were manually pulled from the file used above
    k_eff = 1.09946
    nu_bar = 2.438

    # scale
    scaling_constant = (nu_bar * power) / (200 * 1.60218e-13 * k_eff)
    results *= scaling_constant

    return results