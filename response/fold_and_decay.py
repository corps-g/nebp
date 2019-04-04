import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')
import paths
from nebp_flux import extract_mcnp
from response import response_data


def fold_and_decay():
    """For gold mass specs."""

    # get the flux data at 100kW
    flux_data = extract_mcnp('n', 1e5)

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

    # gold the rfs and the flux together, convert to uCi/g
    sat_act_mass = np.sum(response_functions * flux, axis=1) * (1 / 3.7E4)

    # multiply by mass to get in uCi
    sat_act = sat_act_mass * 37

    # irradiate for 3 hours, getting to ~3% of saturation activity
    act = sat_act * (1 - np.exp(-(np.log(2) / (2.7 * 24 * 3600)) * 3600 * 3))

    # output the activities at irradiation cessation
    for a in act:
        print('{:6.4f}'.format(a))

    # plotting environment
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Foil')
    ax.set_ylabel('Activity $\mu Ci$')
    ax.set_yscale('log')

    # plot the actual data
    ax.plot(act, ls='None', marker='o', markersize=4.0, color='darkblue')

    # save
    plt.savefig('plot/ft_au_theoretical_responses.png', dpi=300)

    return


if __name__ == '__main__':
    fold_and_decay()
