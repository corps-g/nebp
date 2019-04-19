import matplotlib.pyplot as plt
from bss_calibration import BSS_Calibration
from bss_in_beam import BSS_Data


def plot_calibration():
    """Docstring."""

    # pull in calibration data
    data = BSS_Calibration()

    # response comparison -----------------------------------------------------
    # set up plotting environment
    fig = plt.figure(100)
    ax = fig.add_subplot(111)
    ax.set_xticks(data.sizes)
    ax.set_xticklabels(['Bare'] + list(data.sizes[1:]))
    ax.set_yscale('log')

    # plot the data
    style = {'color': 'red', 'marker': 'x', 'markerfacecolor': 'None',
             'markeredgecolor': 'red', 'linestyle': 'None', 'label': 'Experimental',
             'mew': 0.5, 'ms': 6}
    ax.plot(data.sizes[1:], data.experiment, **style)
    style = {'color': 'blue', 'marker': 'o', 'markerfacecolor': 'None',
             'markeredgecolor': 'blue', 'linestyle': 'None', 'label': 'Theoretical',
             'mew': 0.5, 'ms': 6}
    ax.plot(data.sizes, data.responses, **style)

    # add legend and save
    ax.legend()
    plt.savefig('plot/bss_calibration.png', dpi=300)
    fig.clear()

    # print efficiency
    print('Efficiency: ', data.efficiency)

    # correction factors ------------------------------------------------------
    # set up plotting environment
    fig = plt.figure(101)
    ax = fig.add_subplot(111)
    ax.set_xticks(data.sizes)
    ax.set_xticklabels(['Bare'] + list(data.sizes[1:]))

    # plot the data
    style = {'color': 'green', 'marker': '^', 'markerfacecolor': 'None',
             'markeredgecolor': 'green', 'linestyle': 'None', 'label': 'Correction Factors',
             'mew': 0.5, 'ms': 6}
    ax.plot(data.sizes[1:], data.correction_factors, **style)

    # add legend and save
    ax.legend()
    plt.savefig('plot/bss_calibration_correction_factors.png', dpi=300)
    fig.clear()

    # print efficiency
    print('Correction Factors: ', data.correction_factors)

    return


def plot_experiment():
    """Docstring."""

    # pull in calibration data
    data = BSS_Data()

    # response comparison -----------------------------------------------------
    # set up plotting environment
    fig = plt.figure(103)
    ax = fig.add_subplot(111)
    ax.set_xticks(data.sizes)
    ax.set_xticklabels(['Bare'] + list(data.sizes[1:]))
    ax.set_yscale('log')

    # plot the data
    # plot the data
    style = {'color': 'red', 'marker': 'x', 'markerfacecolor': 'None',
             'markeredgecolor': 'red', 'linestyle': 'None', 'label': 'Experimental',
             'mew': 0.5, 'ms': 6}
    ax.plot(data.sizes, data.experiment, **style)
    style = {'color': 'blue', 'marker': 'o', 'markerfacecolor': 'None',
             'markeredgecolor': 'blue', 'linestyle': 'None', 'label': 'Theoretical',
             'mew': 0.5, 'ms': 6}
    ax.plot(data.sizes, data.responses, **style)

    # add legend and save
    ax.legend()
    plt.savefig('plot/bss_response.png', dpi=300)
    fig.clear()

    return


if __name__ == '__main__':
    plot_calibration()
    plot_experiment()
