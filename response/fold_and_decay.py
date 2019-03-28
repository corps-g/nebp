import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')
import paths
from spectrum import Spectrum
from nebp_flux import extract_mcnp
from response import response_data


def fold_and_decay():
    """For gold mass specs."""

    # get the data
    flux_data = extract_mcnp('n', 1e5)
    flux = np.sum(flux_data[:,:,:,0], axis=(0, 1))

    responses = response_data()
    
    response_functions = []
    
    for name, response in responses.items():
        if 'au' in name:
            response_functions.append(response.int)
    
    response_functions = np.array(response_functions)
    
    # fold
    sat_act_mass = np.sum(response_functions * flux, axis=1) * (1 / 3.7E4)
    
    sat_act = sat_act_mass * 37
    
    act = sat_act * (1 - np.exp(-(np.log(2) / (2.7 * 24 * 3600)) * 3600 * 3))
    print(act)
    
    plt.plot(act, ls='None', marker='o')
    plt.yscale('log')
    
    

if __name__ == '__main__':
    fold_and_decay()
    